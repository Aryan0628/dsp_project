"""
STAGE 7 - RESULT VISUALISATION
==============================
The figures produced here are the deliverable a human actually looks at, and the main
artefact to show during the demo. Every intermediate stage is displayed, so the system's
reasoning is inspectable rather than a black box.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")            # no GUI backend needed - we render straight to PNG
import matplotlib.pyplot as plt
import numpy as np

from . import config


def _show(ax, image, title, cmap="gray"):
    ax.imshow(image, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")


def pipeline_figure(result: dict, save_path: Path | str | None = None):
    """The headline 8-panel figure: every pipeline stage plus the verdict card."""
    seg = result["segmentation"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    fig.suptitle(
        "Brain Tumor Detection - Medical Image Enhancement & Diagnostic Assistance Pipeline",
        fontsize=13, fontweight="bold",
    )

    pre_disp = cv2.resize(result["original"], config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    _show(axes[0, 0], pre_disp, "1. Original MRI\n(as acquired)")
    _show(axes[0, 1], result["preprocessed"], "2. Preprocessed\n(median + bilateral denoise)")
    _show(axes[0, 2], result["enhanced"], "3. Enhanced\n(CLAHE + unsharp mask)")

    # histogram panel - visual proof that enhancement widened the tonal range
    ax = axes[0, 3]
    ax.hist(result["preprocessed"].ravel(), bins=64, range=(0, 255),
            alpha=0.55, label="before", color="#6b7280")
    ax.hist(result["enhanced"].ravel(), bins=64, range=(0, 255),
            alpha=0.55, label="after", color="#2563eb")
    ax.set_title("Intensity histogram\nbefore vs after enhancement",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_yticks([])
    ax.tick_params(labelsize=7)

    _show(axes[1, 0], seg["masked"], "4a. Skull stripped\n(brain tissue only)")
    _show(axes[1, 1], seg["binary"], "4b. Segmentation mask\n(threshold + morphology)")
    _show(axes[1, 2], cv2.cvtColor(result["overlay"], cv2.COLOR_BGR2RGB),
          "5. Detected region\n(contour + bounding box)", cmap=None)

    # ---- verdict card
    ax = axes[1, 3]
    ax.axis("off")
    is_tumor = result.get("is_tumor", result["label"] == 1)
    colour = "#dc2626" if is_tumor else "#16a34a"
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                               facecolor=colour, alpha=0.10,
                               edgecolor=colour, linewidth=2.5))
    ax.text(0.5, 0.90, "DIAGNOSTIC ASSISTANCE", ha="center", va="top",
            transform=ax.transAxes, fontsize=8.5, color="#374151", fontweight="bold")
    ax.text(0.5, 0.74, result["label_name"].upper(), ha="center", va="center",
            transform=ax.transAxes, fontsize=21, color=colour, fontweight="bold")
    ax.text(0.5, 0.60, f"decision confidence {result['confidence']*100:.1f}%", ha="center",
            va="center", transform=ax.transAxes, fontsize=12, color="#111827")

    # Probability bar with the decision threshold marked. Showing the threshold matters:
    # the model does not decide at 0.5, so a bar without the marker would look wrong
    # whenever the probability sits between the threshold and 0.5.
    p = result["tumor_probability"]
    thr = float(result.get("threshold", 0.5))
    ax.add_patch(plt.Rectangle((0.12, 0.47), 0.76, 0.055, transform=ax.transAxes,
                               facecolor="#e5e7eb"))
    ax.add_patch(plt.Rectangle((0.12, 0.47), 0.76 * p, 0.055, transform=ax.transAxes,
                               facecolor=colour))
    ax.plot([0.12 + 0.76 * thr] * 2, [0.455, 0.537], transform=ax.transAxes,
            color="#111827", linewidth=1.6)
    ax.text(0.12 + 0.76 * thr, 0.545, "threshold", ha="center", va="bottom",
            transform=ax.transAxes, fontsize=6, color="#111827")
    ax.text(0.5, 0.435, f"tumor probability {p*100:.1f}%   (decides at {thr:.2f})",
            ha="center", va="top", transform=ax.transAxes, fontsize=7.5, color="#374151")

    if seg.get("tumor_found"):
        pr = seg["properties"]
        info = (f"region area     {pr['area']:.0f} px\n"
                f"brain fraction  {pr.get('area_ratio', 0)*100:.2f} %\n"
                f"circularity     {pr['circularity']:.3f}\n"
                f"solidity        {pr['solidity']:.3f}\n"
                f"brightness      {pr.get('brightness_ratio', 0):.2f}x brain")
    else:
        info = "no suspicious region\nabove threshold"
    ax.text(0.5, 0.33, info, ha="center", va="top", transform=ax.transAxes,
            fontsize=8, family="monospace", color="#111827")

    ax.text(0.5, 0.055, "Assistive tool - not a clinical diagnosis.\nAlways confirm with a "
            "qualified radiologist.", ha="center", va="bottom",
            transform=ax.transAxes, fontsize=6.8, style="italic", color="#6b7280")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return Path(save_path)
    return fig


def enhancement_comparison(pre: np.ndarray, save_path: Path | str) -> Path:
    """Side-by-side of the enhancement options - shows WHY we chose CLAHE over global HE."""
    from .enhance import clahe, histogram_equalization, unsharp_mask

    he = histogram_equalization(pre)
    cl = clahe(pre)
    full = unsharp_mask(cl)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    fig.suptitle("Stage 3 - why CLAHE, not global histogram equalisation",
                 fontsize=13, fontweight="bold")

    for ax, img, title in zip(
        axes[0],
        [pre, he, cl, full],
        ["Preprocessed input", "Global histogram equalisation\n(skull saturates, noise amplified)",
         "CLAHE\n(local contrast, noise controlled)", "CLAHE + unsharp mask\n(final output)"],
    ):
        _show(ax, img, title)

    for ax, img, title in zip(axes[1], [pre, he, cl, full],
                              ["histogram", "histogram", "histogram", "histogram"]):
        ax.hist(img.ravel(), bins=64, range=(0, 255), color="#2563eb", alpha=0.8)
        ax.set_title(f"{title}   std={img.std():.1f}", fontsize=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return Path(save_path)


def results_dashboard(report: dict, save_path: Path | str) -> Path:
    """Training summary. Adapts to the task: binary shows the sensitivity/specificity
    trade-off at the tuned operating point; multi-class shows the per-class breakdown."""
    multiclass = report.get("task") == "multiclass"
    m = report["test_metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    fig.suptitle(
        f"Training report - {report['dataset_source']}\n"
        f"{report['n_images']} images ({report['n_train']} train / {report['n_test']} test, "
        f"{report.get('split_kind','random')} split), {report['n_features']} features",
        fontsize=12, fontweight="bold",
    )

    # ------------------------------------------------------------------ model comparison
    ax = axes[0, 0]
    names = list(report["cv_results"].keys())
    aucs = [report["cv_results"][n]["cv_roc_auc_mean"] for n in names]
    errs = [report["cv_results"][n]["cv_roc_auc_std"] for n in names]
    colours = ["#dc2626" if n == report["best_model"] else "#93c5fd" for n in names]
    ax.barh(names, aucs, xerr=errs, color=colours)
    lo = max(0.4, min(aucs) - 0.06)
    ax.set_xlim(lo, 1.02)
    ax.set_xlabel("cross-validated ROC-AUC" + (" (one-vs-rest)" if multiclass else ""))
    ax.set_title("Model comparison (red = selected)", fontsize=10, fontweight="bold")
    for i, v in enumerate(aucs):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=8)
    ax.tick_params(labelsize=8)

    # ------------------------------------------------------------------ confusion matrix
    ax = axes[0, 1]
    if multiclass:
        cm = np.array(m["confusion_matrix_full"])
        labels = m["class_names"]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), [l[:10] for l in labels], fontsize=8, rotation=30)
        ax.set_yticks(range(len(labels)), [l[:10] for l in labels], fontsize=8)
        ax.set_xlabel("predicted", fontsize=9)
        ax.set_ylabel("actual", fontsize=9)
        ax.set_title("Confusion matrix (held-out test set)", fontsize=10, fontweight="bold")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                        fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "#111827")
    else:
        cmd = m["confusion_matrix"]
        mat = np.array([[cmd["true_negative"], cmd["false_positive"]],
                        [cmd["false_negative"], cmd["true_positive"]]])
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks([0, 1], ["predicted\nnormal", "predicted\ntumor"], fontsize=9)
        ax.set_yticks([0, 1], ["actual\nnormal", "actual\ntumor"], fontsize=9)
        ax.set_title("Confusion matrix (default threshold 0.5)",
                     fontsize=10, fontweight="bold")
        tags = [["TN", "FP"], ["FN (worst)", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]}\n{tags[i][j]}", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if mat[i, j] > mat.max() / 2 else "#111827")

    # ------------------------------------------------------------------------- metrics
    ax = axes[1, 0]
    if multiclass:
        recalls = m["per_class_recall"]
        labels = list(recalls.keys())
        vals = [recalls[k] * 100 for k in labels]
        bars = ax.bar([l[:10] for l in labels], vals, color="#93c5fd")
        ax.set_ylim(0, 108)
        ax.set_ylabel("recall %")
        ax.set_title(f"Per-class recall   (macro F1 {m['macro_f1']*100:.1f}%)",
                     fontsize=10, fontweight="bold")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                    ha="center", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=8)
    else:
        tuned = report.get("test_metrics_tuned")
        keys = ["accuracy", "balanced_accuracy", "sensitivity_recall", "specificity", "f1"]
        labels = ["Accuracy", "Balanced\naccuracy", "Sensitivity\n(recall)",
                  "Specificity", "F1"]
        base = [m[k] * 100 for k in keys]
        x = np.arange(len(keys))
        if tuned:
            tv = [tuned[k] * 100 for k in keys]
            ax.bar(x - 0.2, base, 0.4, label="threshold 0.5", color="#93c5fd")
            ax.bar(x + 0.2, tv, 0.4, label="tuned threshold", color="#dc2626")
            ax.legend(fontsize=8)
            for xi, v in zip(x - 0.2, base):
                ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5)
            for xi, v in zip(x + 0.2, tv):
                ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5, fontweight="bold")
        else:
            ax.bar(x, base, 0.5, color="#93c5fd")
            for xi, v in zip(x, base):
                ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0, 112)
        ax.set_ylabel("%")
        ax.set_title("Test metrics", fontsize=10, fontweight="bold")

    # ------------------------------------------------------------- feature importances
    ax = axes[1, 1]
    top = report.get("top_features") or []
    if top:
        top = top[:12][::-1]
        ax.barh([t[0] for t in top], [t[1] for t in top], color="#2563eb")
        ax.set_title("Most influential features", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("importance")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{report['best_model']} does not expose\n"
                          "per-feature importances", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#6b7280")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return Path(save_path)


def sample_grid(save_path: Path | str, n: int = 4, seed: int = 3) -> Path:
    """A grid of example scans through the pipeline - useful as a poster/report figure."""
    from .dataset import generate_brain_mri
    from .enhance import enhance as _enh
    from .preprocess import preprocess as _pre
    from .segment import draw_overlay, segment_tumor

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3.1 * n))
    fig.suptitle("Sample scans through the pipeline", fontsize=13, fontweight="bold")

    for r in range(n):
        tumor = r % 2 == 0
        raw = generate_brain_mri(tumor, rng)
        pre = _pre(raw)
        enh = _enh(pre)
        seg = segment_tumor(enh)
        _show(axes[r, 0], pre, f"input ({'tumor' if tumor else 'healthy'})")
        _show(axes[r, 1], enh, "enhanced")
        _show(axes[r, 2], seg["binary"], "segmentation")
        _show(axes[r, 3], cv2.cvtColor(draw_overlay(enh, seg), cv2.COLOR_BGR2RGB),
              "detected region", cmap=None)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return Path(save_path)

def real_sample_grid(save_path: Path | str, bundle: dict, n: int = 5, seed: int = 3):
    """A grid of REAL scans through the pipeline, with the model's verdict on each.

    Returns None when no real dataset is installed, so the caller can fall back to the
    synthetic grid. Cases are drawn from both classes and the panel is annotated with both
    the prediction and the ground-truth label - including the misreads, because a figure
    that only shows successes is not evidence of anything.
    """
    import random

    from .predict import analyse_file

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    pool: list[tuple[Path, int]] = []
    for folder, label in ((config.RAW_DIR / "yes", 1), (config.RAW_DIR / "no", 0)):
        if folder.is_dir():
            pool += [(f, label) for f in sorted(folder.iterdir()) if f.suffix.lower() in exts]
    if not pool:
        return None

    random.seed(seed)
    tumors = [x for x in pool if x[1] == 1]
    healthy = [x for x in pool if x[1] == 0]
    picks = (random.sample(tumors, min(len(tumors), (n + 1) // 2))
             + random.sample(healthy, min(len(healthy), n // 2)))
    random.shuffle(picks)
    picks = picks[:n]

    fig, axes = plt.subplots(len(picks), 4, figsize=(12, 3.1 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    fig.suptitle("Real MRI scans through the pipeline (prediction vs ground truth)",
                 fontsize=13, fontweight="bold")

    for r, (path, label) in enumerate(picks):
        res = analyse_file(path, bundle)
        seg = res["segmentation"]
        truth = "TUMOR" if label else "HEALTHY"
        pred = "TUMOR" if res["is_tumor"] else "HEALTHY"
        ok = (truth == pred)

        _show(axes[r, 0], res["preprocessed"], f"input - truth: {truth}")
        _show(axes[r, 1], res["enhanced"], "enhanced (CLAHE)")
        _show(axes[r, 2], seg["binary"], "segmentation")
        _show(axes[r, 3], cv2.cvtColor(res["overlay"], cv2.COLOR_BGR2RGB),
              f"predicted: {pred}  p={res['tumor_probability']:.2f}"
              + ("" if ok else "   <-- MISREAD"), cmap=None)
        axes[r, 3].title.set_color("#16a34a" if ok else "#dc2626")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return Path(save_path)

