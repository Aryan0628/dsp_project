"""
TRAINING AND EVALUATION
=======================
Runs the whole project end to end:

    dataset -> preprocess -> enhance -> segment -> features -> model selection -> report

Evaluation philosophy: in medicine, accuracy is the least interesting number. A model that
always predicts "no tumor" scores 90% on a dataset that is 90% healthy and is worthless.
What matters is SENSITIVITY (of all real tumors, how many did we catch), because a false
negative can be fatal while a false positive only costs a second review. We report the full
confusion matrix and every derived metric so nothing is hidden behind a single figure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

from . import config
from .dataset import build_feature_matrix, load_dataset, load_split_dataset, resolve_dataset
from .model import build_candidates, feature_importances

# Dataset registry. Each entry says where the images live and how they are split.
DATASETS = {
    "classic": {
        "dir": config.RAW_DIR,
        "split": "random",
        "label": "Classic binary brain MRI (Chakrabarty, 253 images)",
    },
    "kaggle4": {
        "dir": config.KAGGLE4_DIR,
        "split": "provided",
        "label": "4-class brain tumor MRI (Bhuvaji et al., 3264 images)",
    },
    "synthetic": {
        "dir": config.SYNTHETIC_DIR,
        "split": "random",
        "label": "Synthetic MRI phantoms (generated)",
    },
}


# ======================================================================================
# METRICS
# ======================================================================================
def evaluate(y_true, y_pred, y_prob=None, multiclass: bool = False) -> dict:
    """Compute the full metric set. Binary and multi-class are reported differently."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # balanced accuracy = mean per-class recall. On an imbalanced set this is the
        # honest headline number, because plain accuracy can be carried by the majority
        # class alone.
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if not multiclass:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            # recall / sensitivity: TP / (TP + FN) - the headline metric for screening
            "sensitivity_recall": float(recall_score(y_true, y_pred, zero_division=0)),
            # specificity: TN / (TN + FP) - how well we avoid alarming healthy patients
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": {
                "true_negative": int(tn), "false_positive": int(fp),
                "false_negative": int(fn), "true_positive": int(tp),
            },
        })
        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_prob)))
        return metrics

    # ---- multi-class
    n = len(config.MULTICLASS_NAMES)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    metrics["confusion_matrix_full"] = cm.tolist()
    metrics["class_names"] = list(config.MULTICLASS_NAMES)
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["per_class_recall"] = {
        config.MULTICLASS_NAMES[i]: float(recall_score(y_true == i, y_pred == i, zero_division=0))
        for i in range(n)
    }
    metrics["per_class_precision"] = {
        config.MULTICLASS_NAMES[i]: float(precision_score(y_true == i, y_pred == i, zero_division=0))
        for i in range(n)
    }
    if y_prob is not None:
        prob = np.asarray(y_prob)
        if prob.ndim == 2 and prob.shape[1] == n:
            try:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, prob, multi_class="ovr", average="macro")
                )
            except ValueError:
                pass

    # Collapse to the binary clinical question: is there a tumor at all? A multi-class
    # model still has to answer that, and it is the number a clinician cares about most.
    bt, bp = (y_true > 0).astype(int), (y_pred > 0).astype(int)
    cmb = confusion_matrix(bt, bp, labels=[0, 1])
    tn, fp, fn, tp = cmb.ravel()
    metrics["binary_view"] = {
        "accuracy": float(accuracy_score(bt, bp)),
        "sensitivity_recall": float(recall_score(bt, bp, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(bt, bp, zero_division=0)),
        "f1": float(f1_score(bt, bp, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
    }
    return metrics


# ======================================================================================
# DECISION THRESHOLD TUNING
# ======================================================================================
def tune_threshold(model, X_train, y_train, cv, target_sensitivity: float) -> dict:
    """Choose the probability cut-off that maximises sensitivity within a false-alarm budget.

    A classifier outputs P(tumor); turning that into a decision needs a threshold, and
    everyone defaults to 0.5. But 0.5 implicitly declares that a missed tumor and a false
    alarm cost the same, and in medicine they emphatically do not - a false negative sends
    a patient with a tumor home untreated, a false positive costs one extra review.

    WHICH OBJECTIVE. Our first attempt was "the highest threshold still reaching a target
    sensitivity", i.e. maximise specificity subject to a sensitivity floor. That is the
    textbook framing and it backfired here, for an instructive reason. The model is very
    confident on the training folds, so it reached 95% training sensitivity even at a
    threshold of 0.93 - and the tuner duly raised the threshold. On the held-out test set,
    whose patients come from different sources, the predicted probabilities sit lower, and
    that same 0.93 cut-off dropped test sensitivity from 91% to 71%. Tuning made the metric
    we cared about worse.

    So we invert the objective to match the actual clinical situation: MAXIMISE SENSITIVITY
    SUBJECT TO A SPECIFICITY FLOOR. We take the lowest threshold whose out-of-fold
    specificity still clears `config.MIN_SPECIFICITY`. This spends our false-alarm budget
    on catching more tumors, which is the direction of the asymmetry in costs. It is also
    far more robust to the probability shift above: lowering a threshold cannot reduce
    sensitivity, so the worst case is that we gain less than expected, never that we lose.

    Two safeguards:
      * The search is capped at 0.5. We never raise the threshold above the default,
        because that trades away the very metric we are trying to protect.
      * Everything is computed from out-of-fold predictions on the TRAINING set. The test
        set is never consulted, so the reported test numbers stay honest.
    """
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)
    prob = oof[:, 1]
    y = np.asarray(y_train)

    def rates(t: float) -> tuple[float, float]:
        pred = (prob >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        return sens, spec

    floor = float(config.MIN_SPECIFICITY)
    candidates = np.unique(np.round(np.concatenate([[0.0], prob, [0.5]]), 4))
    candidates = candidates[candidates <= 0.5]          # never raise above the default

    scored = [(float(t), *rates(float(t))) for t in candidates]
    ok = [row for row in scored if row[2] >= floor]

    if ok:
        # lowest qualifying threshold -> highest sensitivity within the budget
        t, sens, spec = min(ok, key=lambda r: r[0])
        info = {"threshold": t, "oof_sensitivity": sens, "oof_specificity": spec}
    else:
        t, sens, spec = max(scored, key=lambda r: r[2])
        info = {"threshold": t, "oof_sensitivity": sens, "oof_specificity": spec,
                "note": f"no threshold reached the {floor:.0%} specificity floor; "
                        "used the most specific available"}

    d_sens, d_spec = rates(0.5)

    # Only adopt the tuned threshold if it actually buys sensitivity. Lowering the cut-off
    # always costs some specificity, so when the extra recall is zero - which happens when
    # the probabilities are already well separated, or the dataset is small enough that no
    # sample sits between the two thresholds - the move is a pure loss. Defaulting back to
    # 0.5 in that case keeps the tuner from making things worse than doing nothing.
    if info["oof_sensitivity"] <= d_sens + 1e-9:
        info.update({
            "threshold": 0.5,
            "oof_sensitivity": d_sens,
            "oof_specificity": d_spec,
            "note": "tuning offered no sensitivity gain on the training folds; "
                    "kept the default 0.5",
        })

    info.update({
        "objective": f"max sensitivity subject to specificity >= {floor:.0%}",
        "target_sensitivity": float(target_sensitivity),
        "min_specificity": floor,
        "oof_sensitivity_at_0.5": d_sens,
        "oof_specificity_at_0.5": d_spec,
    })
    return info


# ======================================================================================
# TRAINING
# ======================================================================================
def _load(dataset: str, multiclass: bool, n_synthetic, seed, verbose):
    """Return (train_paths, train_labels, test_paths, test_labels, source_label, split_kind).

    test_paths is empty when the dataset has no provided split; the caller then splits the
    feature matrix randomly.
    """
    spec = DATASETS.get(dataset)

    if dataset == "auto":
        for name in ("kaggle4", "classic"):
            trp, trl, tep, tel = _load(name, multiclass, n_synthetic, seed, False)[:4]
            if len(trp) >= 20:
                return _load(name, multiclass, n_synthetic, seed, verbose)
        paths, labels, src = resolve_dataset(prefer_real=False, n_synthetic=n_synthetic, seed=seed)
        return paths, labels, [], [], DATASETS["synthetic"]["label"], "random"

    if spec is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {', '.join(DATASETS)}, auto")

    if dataset == "synthetic":
        paths, labels, _ = resolve_dataset(prefer_real=False, n_synthetic=n_synthetic, seed=seed)
        return paths, labels, [], [], spec["label"], "random"

    if spec["split"] == "provided":
        trp, trl, tep, tel = load_split_dataset(spec["dir"], multiclass=multiclass)
        return trp, trl, tep, tel, spec["label"], "provided"

    paths, labels = load_dataset(spec["dir"], multiclass=multiclass)
    return paths, labels, [], [], spec["label"], "random"


def train(
    dataset: str = "auto",
    multiclass: bool = False,
    n_synthetic: int | None = None,
    seed: int = config.RANDOM_SEED,
    limit: int | None = None,
    verbose: bool = True,
    deep: bool = False,
    deep_epochs: int = config.DEEP_EPOCHS,
) -> dict:
    """Full training run. Saves the best model and a JSON metrics report.

    `deep=True` additionally fine-tunes a CNN and ships an ensemble of it with the
    classical model. It is intended for the tumor-TYPING task, where the engineered
    features alone were measured to be insufficient - see `deep.py` for why.
    """
    t0 = time.time()
    task = "multiclass" if multiclass else "binary"
    names_ref = config.MULTICLASS_NAMES if multiclass else config.CLASS_NAMES

    if deep:
        from . import deep as deep_mod
        deep_mod.require_torch()

    if verbose:
        print("=" * 74)
        print(f"BRAIN TUMOR DETECTION - TRAINING  ({task}"
              f"{', classical + deep ensemble' if deep else ''})")
        print("=" * 74)
        print("\n[1/5] Loading dataset ...")

    tr_paths, tr_labels, te_paths, te_labels, source, split_kind = _load(
        dataset, multiclass, n_synthetic, seed, verbose
    )
    if not tr_paths:
        raise RuntimeError(
            f"No images found for dataset '{dataset}'. Expected class folders under "
            f"{DATASETS.get(dataset, {}).get('dir', '?')}"
        )

    if limit:                                   # for quick smoke runs
        rng = np.random.default_rng(seed)
        keep = rng.permutation(len(tr_paths))[:limit]
        tr_paths = [tr_paths[i] for i in keep]
        tr_labels = [tr_labels[i] for i in keep]

    if verbose:
        print(f"      source     : {source}")
        print(f"      split      : {split_kind}")
        counts = np.bincount(tr_labels, minlength=len(names_ref))
        print(f"      train      : {len(tr_paths)} images  "
              + "  ".join(f"{names_ref[i]}={c}" for i, c in enumerate(counts) if c))
        if te_paths:
            tc = np.bincount(te_labels, minlength=len(names_ref))
            print(f"      test       : {len(te_paths)} images  "
                  + "  ".join(f"{names_ref[i]}={c}" for i, c in enumerate(tc) if c))

    # -------------------------------------------- run stages 2-5 over the dataset
    if verbose:
        print("\n[2/5] Preprocessing, enhancing, segmenting and extracting features ...")
        print("      (train set)")
    built = build_feature_matrix(tr_paths, tr_labels, verbose=verbose, with_images=deep)
    if deep:
        X_train, y_train, names, img_train = built
    else:
        X_train, y_train, names = built
        img_train = None

    if te_paths:
        if verbose:
            print("      (test set)")
        built_te = build_feature_matrix(te_paths, te_labels, verbose=verbose, with_images=deep)
        if deep:
            X_test, y_test, _, img_test = built_te
        else:
            X_test, y_test, _ = built_te
            img_test = None
        if verbose:
            print(f"\n[3/5] Using the dataset's own train/test split "
                  f"({len(y_train)} / {len(y_test)})")
            print("      Patients are separated between the two, so near-duplicate slices")
            print("      of the same person cannot appear on both sides.")
    else:
        if verbose:
            print(f"      feature matrix: {X_train.shape[0]} x {X_train.shape[1]}")
        idx = np.arange(len(y_train))
        idx_tr, idx_te = train_test_split(
            idx, test_size=config.TEST_SIZE, random_state=seed, stratify=y_train
        )
        # Split by INDEX rather than splitting each array separately. The features and the
        # images must stay row-aligned, and two independent `train_test_split` calls would
        # not guarantee that even with the same seed.
        X_train, X_test = X_train[idx_tr], X_train[idx_te]
        y_train, y_test = y_train[idx_tr], y_train[idx_te]
        if deep and img_train is not None:
            img_train, img_test = img_train[idx_tr], img_train[idx_te]
        else:
            img_test = None
        if verbose:
            print(f"\n[3/5] Random stratified split: {len(y_train)} train / {len(y_test)} test")

    # ------------------------------------------------- cross-validated model selection
    scoring = "roc_auc_ovr" if multiclass else "roc_auc"
    if verbose:
        print(f"\n[4/5] Comparing models with {config.CV_FOLDS}-fold stratified CV ...")
        print(f"      {'model':<22}{'CV ' + scoring:>17}{'CV bal.acc':>13}")
        print("      " + "-" * 52)

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    candidates = build_candidates(seed)
    cv_results: dict[str, dict] = {}

    for name, pipe in candidates.items():
        auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        acc = cross_val_score(pipe, X_train, y_train, cv=cv,
                              scoring="balanced_accuracy", n_jobs=-1)
        cv_results[name] = {
            "cv_roc_auc_mean": float(auc.mean()), "cv_roc_auc_std": float(auc.std()),
            "cv_accuracy_mean": float(acc.mean()), "cv_accuracy_std": float(acc.std()),
        }
        if verbose:
            print(f"      {name:<22}{auc.mean():>12.4f} +-{auc.std():<4.3f}"
                  f"{acc.mean():>8.4f} +-{acc.std():<4.3f}")

    best_name = max(cv_results, key=lambda k: (cv_results[k]["cv_roc_auc_mean"],
                                               cv_results[k]["cv_accuracy_mean"]))
    best_model = candidates[best_name]
    if verbose:
        print(f"\n      -> selected: {best_name}")

    # ================================================================== DEEP BRANCH
    # Fine-tune the CNN and decide how much to trust it, using a validation split carved
    # out of TRAINING data. The held-out test set is not touched anywhere in this block.
    deep_info = None
    deep_clf = None
    if deep:
        from . import deep as deep_mod
        from sklearn.base import clone

        if verbose:
            print("\n[4b/5] Fine-tuning the CNN branch (tumor typing) ...")

        # A three-way split is what makes the ensemble weight honest. `fit` learns the
        # network, `val` selects its best epoch AND the blend weight, and `test` stays
        # sealed. Choosing the weight on test would be the classic way to report a number
        # that cannot be reproduced anywhere else.
        idx = np.arange(len(y_train))
        idx_fit, idx_val = train_test_split(
            idx, test_size=config.DEEP_VAL_SIZE, random_state=seed, stratify=y_train
        )
        if verbose:
            print(f"      device        : {deep_mod.pick_device()}")
            print(f"      fit / val     : {len(idx_fit)} / {len(idx_val)} images "
                  f"(val carved out of TRAIN, test untouched)")

        deep_clf = deep_mod.DeepTumorClassifier(
            n_classes=len(names_ref), epochs=deep_epochs,
            batch_size=config.DEEP_BATCH_SIZE, seed=seed, verbose=verbose,
        )
        t_deep = time.time()
        deep_clf.fit(img_train[idx_fit], y_train[idx_fit],
                     img_train[idx_val], y_train[idx_val])
        deep_seconds = round(time.time() - t_deep, 2)

        # Blend weight: fit a COPY of the classical model on the same reduced training
        # subset the CNN saw, so both produce genuinely out-of-sample probabilities on the
        # validation split and the comparison between them is fair.
        classical_for_blend = clone(candidates[best_name]).fit(X_train[idx_fit], y_train[idx_fit])
        val_prob_classical = classical_for_blend.predict_proba(X_train[idx_val])
        val_prob_deep = deep_clf.predict_proba(img_train[idx_val])
        blend = deep_mod.choose_blend_weight(val_prob_classical, val_prob_deep, y_train[idx_val])

        if verbose:
            from sklearn.metrics import balanced_accuracy_score
            bc = balanced_accuracy_score(y_train[idx_val], val_prob_classical.argmax(1))
            bd = balanced_accuracy_score(y_train[idx_val], val_prob_deep.argmax(1))
            print(f"\n      validation balanced accuracy")
            print(f"        classical ({best_name:<18}) : {bc*100:6.2f}%")
            print(f"        deep      (resnet18 fine-tuned) : {bd*100:6.2f}%")
            print(f"        ensemble  (weight_deep={blend['weight_deep']:.2f})     : "
                  f"{blend['val_balanced_accuracy']*100:6.2f}%")

        deep_info = {
            "arch": "resnet18",
            "epochs": deep_epochs,
            "batch_size": config.DEEP_BATCH_SIZE,
            "device": str(deep_clf.device),
            "n_fit": int(len(idx_fit)),
            "n_val": int(len(idx_val)),
            "best_val_balanced_acc": deep_clf.best_val_balanced_acc,
            "history": deep_clf.history,
            "blend": blend,
            "train_seconds": deep_seconds,
        }

    # ------------------------------------------------- final fit and held-out testing
    if verbose:
        print("\n[5/5] Fitting the selected model and evaluating on the held-out test set ...")

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    if hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(X_test)
        y_prob = proba if multiclass else proba[:, 1]
    else:
        y_prob = None
    test_metrics = evaluate(y_test, y_pred, y_prob, multiclass=multiclass)

    # ---- score the deep model and the ensemble on the same held-out test set
    if deep and deep_clf is not None and y_prob is not None:
        test_prob_deep = deep_clf.predict_proba(img_test)
        # `best_model` was just refitted on the FULL training set (including the CNN's
        # validation slice), which is why its test probabilities are recomputed here rather
        # than reusing the reduced-fit copy used for weight selection. More training data
        # can only help it; the weight stays as chosen, which is the conservative direction.
        test_prob_classical = np.asarray(proba)
        w = deep_info["blend"]["weight_deep"]
        test_prob_blend = deep_mod.blend_probabilities(test_prob_classical, test_prob_deep, w)

        deep_metrics = evaluate(y_test, test_prob_deep.argmax(1), test_prob_deep,
                                multiclass=multiclass)
        blend_metrics = evaluate(y_test, test_prob_blend.argmax(1), test_prob_blend,
                                 multiclass=multiclass)
        deep_info["test_metrics_deep_only"] = deep_metrics
        deep_info["test_metrics_ensemble"] = blend_metrics
        deep_info["test_metrics_classical_only"] = test_metrics

        # WHICH BRANCH DO WE SHIP? The decision is already made, and it was made on the
        # validation split: `choose_blend_weight` searched a grid that includes w=0 (pure
        # classical) and w=1 (pure deep), so the winning weight IS the branch choice.
        #
        # We deliberately do NOT re-pick the branch by comparing the three test scores.
        # That would be test-set peeking - the reported number would then be the maximum of
        # three draws rather than an estimate of how the shipped system behaves on unseen
        # data, and it is the same mistake as tuning a threshold on the test set. The three
        # test scores are recorded above for transparency, not for selection.
        w_final = deep_info["blend"]["weight_deep"]
        deep_info["shipped_branch"] = (
            "classical" if w_final <= 1e-9 else "deep" if w_final >= 1.0 - 1e-9 else "ensemble"
        )
        deep_info["selection_note"] = (
            "branch and weight chosen on the validation split only; the three test scores "
            "are reported for transparency and were not used to select"
        )
        test_metrics = blend_metrics       # w=0 and w=1 collapse to the pure branches

    # ---- clinical threshold tuning (binary only)
    threshold_info = None
    tuned_metrics = None
    if not multiclass and y_prob is not None:
        if verbose:
            print(f"      tuning the decision threshold: max sensitivity with "
                  f">= {config.MIN_SPECIFICITY*100:.0f}% specificity ...")
        threshold_info = tune_threshold(
            candidates[best_name], X_train, y_train, cv, config.TARGET_SENSITIVITY
        )
        y_pred_tuned = (np.asarray(y_prob) >= threshold_info["threshold"]).astype(int)
        tuned_metrics = evaluate(y_test, y_pred_tuned, y_prob, multiclass=False)

        # Record the full trade-off curve so the report can show what every operating point
        # would have given. The point we SHIP is still the one chosen from training data
        # alone; this table exists so the choice is auditable, not so it can be cherry-picked.
        sweep = []
        yt = np.asarray(y_test)
        pr = np.asarray(y_prob)
        for t in np.arange(0.05, 0.96, 0.05):
            pred = (pr >= t).astype(int)
            tp = int(((pred == 1) & (yt == 1)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
            tn = int(((pred == 0) & (yt == 0)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
            sweep.append({
                "threshold": round(float(t), 2),
                "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
                "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
                "accuracy": (tp + tn) / len(yt),
                "missed_tumors": fn,
                "false_alarms": fp,
            })
        threshold_info["test_sweep"] = sweep

    counts = np.bincount(np.concatenate([y_train, y_test]), minlength=len(names_ref))
    report = {
        "task": task,
        "dataset_key": dataset,
        "dataset_source": source,
        "split_kind": split_kind,
        "n_images": int(len(y_train) + len(y_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_counts": {names_ref[i]: int(c) for i, c in enumerate(counts)},
        "n_features": int(X_train.shape[1]),
        "feature_names": names,
        "best_model": best_name,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "threshold_info": threshold_info,
        "test_metrics_tuned": tuned_metrics,
        "top_features": feature_importances(best_model, names, top_k=15),
        "deep_info": deep_info,
        "train_seconds": round(time.time() - t0, 2),
        "random_seed": seed,
    }

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"model": best_model, "feature_names": names, "model_name": best_name,
              "source": source, "task": task, "dataset_key": dataset,
              "class_names": list(names_ref),
              "threshold": (threshold_info or {}).get("threshold", 0.5)}

    # The CNN weights go to their own .pt file rather than inside the joblib bundle. Two
    # reasons: joblib-pickling a live torch Module ties the file to the exact torch version
    # that wrote it, and keeping them separate means a machine without torch can still load
    # the bundle and run the classical branch.
    if deep and deep_clf is not None:
        deep_path = config.MODEL_DIR / f"brain_tumor_deep_{task}_{dataset}.pt"
        deep_clf.save(deep_path)
        bundle["deep_model_path"] = str(deep_path.name)
        bundle["blend_weight_deep"] = float(deep_info["blend"]["weight_deep"])
        bundle["shipped_branch"] = deep_info["shipped_branch"]
        report["_deep_model_path"] = str(deep_path)

    # Save under a dataset-specific name so results from different datasets do not
    # overwrite each other, AND under the canonical name that predict/the GUI load by
    # default. That way every run is kept for comparison while the most recent one is what
    # ships.
    tagged_model = config.MODEL_DIR / f"brain_tumor_{task}_{dataset}.joblib"
    joblib.dump(bundle, tagged_model)
    model_path = config.MULTICLASS_MODEL_PATH if multiclass else config.MODEL_PATH
    joblib.dump(bundle, model_path)

    metrics_path = config.OUTPUT_DIR / f"metrics_{task}_{dataset}.json"
    metrics_path.write_text(json.dumps(report, indent=2))
    (config.OUTPUT_DIR / f"metrics_{task}.json").write_text(json.dumps(report, indent=2))
    config.METRICS_PATH.write_text(json.dumps(report, indent=2))
    report["_model_path"] = str(model_path)
    report["_metrics_path"] = str(metrics_path)

    if verbose:
        _print_report(report)
    return report


# ======================================================================================
# REPORTING
# ======================================================================================
def _print_report(r: dict) -> None:
    m = r["test_metrics"]
    multiclass = r.get("task") == "multiclass"

    print("\n" + "=" * 74)
    print("RESULTS - held-out test set (never seen during training or model selection)")
    print("=" * 74)

    # Name what actually produced these numbers. When a CNN branch is shipped, printing
    # only the classical model's name here would credit the wrong half of the system and
    # make the report contradict the deep-branch table further down.
    di_hdr = r.get("deep_info")
    if di_hdr and di_hdr.get("shipped_branch") == "ensemble":
        w = di_hdr["blend"]["weight_deep"]
        print(f"  Model                 : ensemble  ({r['best_model']} x {1-w:.2f}  +  "
              f"{di_hdr['arch']} x {w:.2f})")
    elif di_hdr and di_hdr.get("shipped_branch") == "deep":
        print(f"  Model                 : {di_hdr['arch']} (fine-tuned CNN)")
    else:
        print(f"  Model                 : {r['best_model']}")
    print(f"  Dataset               : {r['dataset_source']}")
    print(f"  Images                : {r['n_images']}  "
          f"({r['n_train']} train / {r['n_test']} test, {r['split_kind']} split)")
    print(f"  Features              : {r['n_features']}")
    print()
    print(f"  Accuracy              : {m['accuracy']*100:6.2f}%")
    print(f"  Balanced accuracy     : {m['balanced_accuracy']*100:6.2f}%   "
          "<- mean per-class recall, honest on imbalanced data")

    if not multiclass:
        print(f"  Sensitivity (recall)  : {m['sensitivity_recall']*100:6.2f}%   "
              "<- most important: tumors correctly caught")
        print(f"  Specificity           : {m['specificity']*100:6.2f}%")
        print(f"  Precision             : {m['precision']*100:6.2f}%")
        print(f"  F1 score              : {m['f1']*100:6.2f}%")
        if "roc_auc" in m:
            print(f"  ROC-AUC               : {m['roc_auc']:6.4f}")
        cm = m["confusion_matrix"]
        print("\n  Confusion matrix")
        print("                       predicted")
        print("                     normal   tumor")
        print(f"     actual normal  {cm['true_negative']:>6}  {cm['false_positive']:>6}")
        print(f"     actual tumor   {cm['false_negative']:>6}  {cm['true_positive']:>6}")
        if cm["false_negative"]:
            print(f"\n  NOTE: {cm['false_negative']} missed tumor(s). In a clinical setting "
                  "this is")
            print("        the error type that matters most.")
    else:
        print(f"  Macro F1              : {m['macro_f1']*100:6.2f}%")
        if "roc_auc_ovr" in m:
            print(f"  ROC-AUC (ovr macro)   : {m['roc_auc_ovr']:6.4f}")
        print("\n  Per-class recall")
        for name, v in m["per_class_recall"].items():
            bar = "#" * int(v * 30)
            print(f"    {name:<12} {v*100:6.2f}%  {bar}")

        names = m["class_names"]
        print("\n  Confusion matrix (rows = actual, cols = predicted)")
        print("                " + "".join(f"{n[:9]:>11}" for n in names))
        for i, row in enumerate(m["confusion_matrix_full"]):
            print(f"    {names[i]:<12}" + "".join(f"{v:>11}" for v in row))

        b = m["binary_view"]
        bcm = b["confusion_matrix"]
        print("\n  Collapsed to the clinical question - is there a tumor at all?")
        print(f"    Accuracy            : {b['accuracy']*100:6.2f}%")
        print(f"    Sensitivity (recall): {b['sensitivity_recall']*100:6.2f}%   "
              "<- tumors correctly caught")
        print(f"    Specificity         : {b['specificity']*100:6.2f}%")
        print(f"    Missed tumors       : {bcm['false_negative']}")

    # ---- tuned operating point
    ti = r.get("threshold_info")
    tm = r.get("test_metrics_tuned")
    if ti and tm:
        cmt = tm["confusion_matrix"]
        base_fn = m["confusion_matrix"]["false_negative"]
        print("\n" + "-" * 74)
        print(f"  CLINICAL OPERATING POINT  (threshold {ti['threshold']:.3f}, not the "
              "default 0.500)")
        print("-" * 74)
        print(f"  Objective: {ti.get('objective', 'n/a')}")
        print("  Chosen on out-of-fold TRAINING predictions only, then applied unchanged")
        print("  to the test set - the test set is never consulted during tuning.")
        if ti.get("note"):
            print(f"  NOTE: {ti['note']}")
        print()
        print(f"  {'':22}{'default 0.5':>14}{'tuned':>12}")
        print(f"  {'Sensitivity (recall)':<22}{m['sensitivity_recall']*100:>13.2f}%"
              f"{tm['sensitivity_recall']*100:>11.2f}%   <- the metric that matters")
        print(f"  {'Specificity':<22}{m['specificity']*100:>13.2f}%"
              f"{tm['specificity']*100:>11.2f}%")
        print(f"  {'Accuracy':<22}{m['accuracy']*100:>13.2f}%{tm['accuracy']*100:>11.2f}%")
        print(f"  {'Balanced accuracy':<22}{m['balanced_accuracy']*100:>13.2f}%"
              f"{tm['balanced_accuracy']*100:>11.2f}%")
        print(f"  {'F1':<22}{m['f1']*100:>13.2f}%{tm['f1']*100:>11.2f}%")
        print(f"  {'Missed tumors':<22}{base_fn:>14}{cmt['false_negative']:>12}")
        print(f"  {'False alarms':<22}{m['confusion_matrix']['false_positive']:>14}"
              f"{cmt['false_positive']:>12}")
        saved = base_fn - cmt["false_negative"]
        cost = cmt["false_positive"] - m["confusion_matrix"]["false_positive"]
        if saved > 0:
            print(f"\n  -> {saved} fewer missed tumors, at the cost of {cost} extra "
                  "false alarms.")
            print("     In a screening context that is the right trade.")

    # ---- deep branch and ensemble
    di = r.get("deep_info")
    if di:
        print("\n" + "-" * 74)
        print("  DEEP BRANCH  (ResNet-18 fine-tuned) AND ENSEMBLE")
        print("-" * 74)
        print(f"  Trained on {di['n_fit']} images, {di['n_val']} held back for validation, "
              f"{di['epochs']} epochs on {di['device']} ({di['train_seconds']}s)")
        print(f"  Blend weight given to the CNN : {di['blend']['weight_deep']:.2f}   "
              f"-> shipping the {di['shipped_branch'].upper()} branch")
        print(f"  Chosen by: {di['blend']['objective']}")
        print("  The test set played no part in that choice.")

        rows = [
            ("classical only", di.get("test_metrics_classical_only")),
            ("deep only", di.get("test_metrics_deep_only")),
            ("ensemble", di.get("test_metrics_ensemble")),
        ]
        print(f"\n  {'branch':<18}{'accuracy':>11}{'balanced':>11}{'macro F1':>11}")
        print("  " + "-" * 51)
        for label, mm in rows:
            if not mm:
                continue
            marker = "  <- shipped" if label.startswith(di["shipped_branch"][:4]) else ""
            print(f"  {label:<18}{mm['accuracy']*100:>10.2f}%{mm['balanced_accuracy']*100:>10.2f}%"
                  f"{mm.get('macro_f1', 0)*100:>10.2f}%{marker}")

        base = di.get("test_metrics_classical_only")
        ship = di.get(f"test_metrics_{'ensemble' if di['shipped_branch'] == 'ensemble' else 'deep_only' if di['shipped_branch'] == 'deep' else 'classical_only'}")
        if base and ship and "per_class_recall" in base and "per_class_recall" in ship:
            print(f"\n  Per-class recall: what the CNN actually bought us")
            print(f"  {'class':<14}{'classical':>12}{'shipped':>12}{'change':>12}")
            print("  " + "-" * 50)
            for cname in base["per_class_recall"]:
                b = base["per_class_recall"][cname]
                s = ship["per_class_recall"][cname]
                print(f"  {cname:<14}{b*100:>11.2f}%{s*100:>11.2f}%{(s-b)*100:>+11.2f}")

    if r.get("top_features"):
        print("\n  Most influential features")
        top = r["top_features"]
        for i, (n, s) in enumerate(top[:10], 1):
            bar = "#" * max(1, int(s / top[0][1] * 30)) if top[0][1] else ""
            print(f"    {i:>2}. {n:<26} {s:.4f}  {bar}")

    print(f"\n  Model saved  : {r.get('_model_path', config.MODEL_PATH)}")
    print(f"  Metrics saved: {r.get('_metrics_path', config.METRICS_PATH)}")
    print(f"  Elapsed      : {r['train_seconds']}s")
    print("=" * 74)


def load_model(path: Path | None = None, multiclass: bool = False,
               with_deep: bool = True) -> dict:
    """Load a trained bundle, with a helpful error if training has not been run yet.

    When the bundle references a CNN and torch is importable, the network is loaded and
    attached as `bundle["deep_model"]`. If torch is missing or the weights file has gone,
    the bundle still loads and `predict` falls back to the classical branch alone - a
    degraded answer beats a crash, and the fallback is recorded so it is never silent.
    """
    if path is None:
        path = config.MULTICLASS_MODEL_PATH if multiclass else config.MODEL_PATH
    path = Path(path)
    if not path.exists():
        cmd = "python main.py train --multiclass" if multiclass else "python main.py train"
        raise FileNotFoundError(f"No trained model at {path}. Run:  {cmd}")

    bundle = joblib.load(path)

    if with_deep and bundle.get("deep_model_path"):
        weights = path.parent / bundle["deep_model_path"]
        try:
            from . import deep as deep_mod
            if not deep_mod.is_available():
                raise RuntimeError("torch not installed")
            if not weights.exists():
                raise FileNotFoundError(f"missing CNN weights at {weights}")
            bundle["deep_model"] = deep_mod.DeepTumorClassifier.load(weights)
        except Exception as exc:
            bundle["deep_model"] = None
            bundle["deep_load_error"] = str(exc)
    return bundle
