"""
INFERENCE - run the full pipeline on a single MRI and return a verdict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .enhance import enhance, enhancement_report
from .features import feature_vector
from .preprocess import load_image, preprocess
from .segment import draw_overlay, segment_tumor
from .train import load_model


def analyse_image(image: np.ndarray, bundle: dict | None = None) -> dict:
    """Stages 2-7 for one in-memory grayscale image.

    Returns every intermediate result, so the caller (CLI, GUI or report generator) can
    display whichever stages it wants without recomputing anything.
    """
    bundle = bundle or load_model()
    model = bundle["model"]
    expected_names = bundle["feature_names"]

    original = image
    pre = preprocess(original)
    enh = enhance(pre)
    seg = segment_tumor(enh)
    overlay = draw_overlay(enh, seg)

    values, names = feature_vector(enh, seg)

    # Guard against a feature-order mismatch between training and inference. If the feature
    # code ever changes, this fails loudly instead of silently predicting from garbage.
    if names != expected_names:
        lookup = dict(zip(names, values))
        missing = [n for n in expected_names if n not in lookup]
        if missing:
            raise ValueError(
                f"Model expects {len(expected_names)} features but {len(missing)} are "
                f"missing (e.g. {missing[:3]}). Retrain with: python main.py train"
            )
        values = np.array([lookup[n] for n in expected_names], dtype=np.float64)

    x = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)

    # Apply the tuned clinical decision threshold rather than the model's built-in 0.5.
    # The threshold was chosen during training to reach the target sensitivity; using
    # model.predict() here would silently fall back to 0.5 and undo that.
    threshold = float(bundle.get("threshold", 0.5))
    class_names = bundle.get("class_names", list(config.CLASS_NAMES))
    is_binary = len(class_names) == 2

    # ---- deep branch. When the bundle carries a fine-tuned CNN, its opinion is blended in
    # with the weight chosen during training. `branch` records which opinion actually
    # produced the answer, so the UI can say so rather than implying one model did it all.
    deep_model = bundle.get("deep_model")
    blend_weight = float(bundle.get("blend_weight_deep", 0.0))
    branch = "classical"
    deep_proba = None
    if deep_model is not None and blend_weight > 0.0:
        try:
            deep_proba = deep_model.predict_proba(enh[None, ...])[0]
            branch = bundle.get("shipped_branch", "ensemble")
        except Exception:
            # Inference must not die because the accelerator is unavailable on this host.
            deep_proba = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        if deep_proba is not None and len(proba) == len(deep_proba):
            from .deep import blend_probabilities
            classical_proba = proba
            proba = blend_probabilities(classical_proba, deep_proba, blend_weight)
        if is_binary:
            tumor_probability = float(proba[1])
            label = int(tumor_probability >= threshold)
            # Confidence must be measured against the threshold we actually decide with,
            # not against 0.5. With a tuned threshold of 0.14, a probability of 0.23 is a
            # comfortable "tumor" call - reporting it as "22.9% confidence" would read as
            # the opposite of what the model means. So we rescale: 0.5 sits exactly on the
            # decision boundary and 1.0 is maximum certainty in whichever direction we went.
            if tumor_probability >= threshold:
                span = max(1.0 - threshold, 1e-9)
                confidence = 0.5 + 0.5 * (tumor_probability - threshold) / span
            else:
                span = max(threshold, 1e-9)
                confidence = 0.5 + 0.5 * (threshold - tumor_probability) / span
            confidence = float(min(max(confidence, 0.0), 1.0))
        else:
            label = int(np.argmax(proba))
            tumor_probability = float(1.0 - proba[0])   # P(any tumor type)
            confidence = float(proba[label])
    else:
        label = int(model.predict(x)[0])
        confidence = float("nan")
        tumor_probability = float(label > 0)

    return {
        "label": label,
        "label_name": class_names[label],
        "class_names": class_names,
        # Full probability vector. For the 4-class model this is the actual clinical
        # payload - "probably meningioma, but glioma is not far behind" is a more useful
        # thing to hand a radiologist than a bare winning label.
        "probabilities": [float(v) for v in proba] if hasattr(model, "predict_proba") else None,
        "threshold": threshold,
        "is_tumor": label > 0,
        "confidence": confidence,
        "tumor_probability": tumor_probability,
        "model_name": bundle.get("model_name", "unknown"),
        "branch": branch,
        "blend_weight_deep": blend_weight if deep_proba is not None else 0.0,
        "deep_probabilities": [float(v) for v in deep_proba] if deep_proba is not None else None,
        "original": original,
        "preprocessed": pre,
        "enhanced": enh,
        "segmentation": seg,
        "overlay": overlay,
        "enhancement_report": enhancement_report(pre, enh),
        "features": dict(zip(names, values)),
    }


def analyse_file(path: Path | str, bundle: dict | None = None) -> dict:
    """Same as analyse_image but reads from disk first."""
    result = analyse_image(load_image(path), bundle)
    result["path"] = str(path)
    return result


def verdict_text(result: dict) -> str:
    """A short human-readable summary suitable for the terminal or a report."""
    seg = result["segmentation"]
    lines = [
        f"Prediction        : {result['label_name']}",
        f"Confidence        : {result['confidence']*100:.1f}%",
        f"Tumor probability : {result['tumor_probability']*100:.1f}%",
        f"Decision threshold: {result.get('threshold', 0.5):.3f}"
        f"{'  (tuned for sensitivity)' if abs(result.get('threshold', 0.5) - 0.5) > 1e-6 else ''}",
        f"Model             : {result['model_name']}",
    ]

    branch = result.get("branch", "classical")
    if branch != "classical":
        lines.append(
            f"Decided by        : {branch} "
            f"(CNN weight {result.get('blend_weight_deep', 0.0):.2f})"
        )

    # For the 4-class model, the runner-up matters. A 55/40 split between meningioma and
    # glioma is a materially different message from a 95/2, and collapsing both to a bare
    # label would hide exactly the cases a radiologist should look at first.
    probs = result.get("probabilities")
    names = result.get("class_names") or []
    if probs and len(names) == len(probs) and len(names) > 2:
        lines.append("")
        lines.append("Class probabilities")
        for name, p in sorted(zip(names, probs), key=lambda kv: -kv[1]):
            bar = "#" * int(round(p * 30))
            lines.append(f"  {name:<12} {p*100:6.2f}%  {bar}")
    if seg.get("tumor_found"):
        p = seg["properties"]
        lines += [
            f"Region located    : yes",
            f"  centroid        : {p['centroid']}",
            f"  area            : {p['area']:.0f} px ({p.get('area_ratio', 0)*100:.2f}% of brain)",
            f"  circularity     : {p['circularity']:.3f}  (1.0 = perfect circle)",
            f"  solidity        : {p['solidity']:.3f}",
            f"  brightness vs brain: {p.get('brightness_ratio', 0):.2f}x",
        ]
    else:
        lines.append("Region located    : no suspicious region above threshold")

    er = result["enhancement_report"]
    lines += [
        "",
        "Enhancement gain (Stage 3)",
        f"  contrast (std)  : {er['contrast_std_before']:.1f} -> {er['contrast_std_after']:.1f}",
        f"  entropy (bits)  : {er['entropy_before']:.2f} -> {er['entropy_after']:.2f}",
        f"  sharpness       : {er['sharpness_before']:.1f} -> {er['sharpness_after']:.1f}",
    ]
    return "\n".join(lines)
