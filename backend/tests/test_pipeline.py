"""
Test suite - also serves as the demo-day pre-flight check (`./run.sh check`).

Each test maps to one pipeline stage, so a failure tells you exactly which stage broke.
"""

import numpy as np
import pytest

from braintumor import config
from braintumor.dataset import generate_brain_mri
from braintumor.enhance import clahe, enhance, enhancement_report, unsharp_mask
from braintumor.features import extract_features, feature_vector
from braintumor.preprocess import load_image, preprocess
from braintumor.segment import brain_mask, segment_tumor


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(config.RANDOM_SEED)


@pytest.fixture(scope="module")
def tumor_image(rng):
    return generate_brain_mri(True, np.random.default_rng(1))


@pytest.fixture(scope="module")
def healthy_image(rng):
    return generate_brain_mri(False, np.random.default_rng(2))


# --------------------------------------------------------------- Stage 1: acquisition
def test_generator_shape_and_type(tumor_image):
    assert tumor_image.shape == (256, 256)
    assert tumor_image.dtype == np.uint8


def test_generator_is_not_degenerate(tumor_image, healthy_image):
    """A blank or saturated image would silently break everything downstream."""
    for img in (tumor_image, healthy_image):
        assert 5 < img.mean() < 250
        assert img.std() > 10


def test_tumor_images_are_brighter_at_the_top_end(rng):
    """Sanity check on the simulator: tumors are hyperintense by construction, so the
    99th percentile of a tumor scan should on average exceed that of a healthy scan."""
    r = np.random.default_rng(99)
    t = [np.percentile(generate_brain_mri(True, r), 99) for _ in range(10)]
    h = [np.percentile(generate_brain_mri(False, r), 99) for _ in range(10)]
    assert np.mean(t) > np.mean(h)


# --------------------------------------------------------------- Stage 2: preprocessing
def test_preprocess_standardises_output(tumor_image):
    out = preprocess(tumor_image)
    assert out.shape == config.IMAGE_SIZE[::-1]
    assert out.dtype == np.uint8
    assert out.ndim == 2


def test_preprocess_reduces_noise(tumor_image):
    """Denoising must lower the high-frequency energy."""
    import cv2

    before = cv2.Laplacian(tumor_image, cv2.CV_64F).var()
    after = cv2.Laplacian(preprocess(tumor_image), cv2.CV_64F).var()
    assert after < before


def test_preprocess_accepts_colour_input(tumor_image):
    import cv2

    colour = cv2.cvtColor(tumor_image, cv2.COLOR_GRAY2BGR)
    assert preprocess(colour).ndim == 2


# ---------------------------------------------------------------- Stage 3: enhancement
def test_enhancement_increases_contrast_and_entropy(tumor_image):
    pre = preprocess(tumor_image)
    rep = enhancement_report(pre, enhance(pre))
    assert rep["contrast_std_after"] > rep["contrast_std_before"]
    assert rep["entropy_after"] > rep["entropy_before"]


def test_enhancement_sharpens(tumor_image):
    pre = preprocess(tumor_image)
    rep = enhancement_report(pre, enhance(pre))
    assert rep["sharpness_after"] > rep["sharpness_before"]


def test_clahe_and_unsharp_stay_in_range(tumor_image):
    pre = preprocess(tumor_image)
    for img in (clahe(pre), unsharp_mask(pre), enhance(pre)):
        assert img.dtype == np.uint8
        assert img.min() >= 0 and img.max() <= 255


# --------------------------------------------------------------- Stage 4: segmentation
def test_brain_mask_excludes_the_skull(tumor_image):
    """The regression test for our worst bug. If the mask keeps the bright skull rim, the
    intensity statistics blow up and the tumor threshold exceeds 255, so nothing is ever
    detected. The mask must cover a sensible fraction of the image and exclude the rim."""
    enh = enhance(preprocess(tumor_image))
    mask = brain_mask(enh)
    frac = (mask > 0).mean()
    assert 0.15 < frac < 0.55, f"brain mask covers {frac:.0%} of the image"


def test_brain_mask_is_one_solid_region(tumor_image):
    """Regression test for the flood-fill bug: when the head touched the image border the
    background split in two and the mask became a crescent covering half the frame."""
    import cv2

    enh = enhance(preprocess(tumor_image))
    mask = brain_mask(enh)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
    assert areas, "mask is empty"
    assert areas[0] / max(mask.sum() / 255, 1) > 0.95, "mask is fragmented"


def test_segmentation_localises_tumors():
    """Across a batch, every tumor should be localised and most healthy scans should not
    produce a region. This guards the threshold rule against regressions."""
    r = np.random.default_rng(11)
    found_tumor = found_healthy = 0
    n = 20
    for i in range(2 * n):
        is_tumor = i % 2 == 0
        seg = segment_tumor(enhance(preprocess(generate_brain_mri(is_tumor, r))))
        if is_tumor and seg["tumor_found"]:
            found_tumor += 1
        if (not is_tumor) and seg["tumor_found"]:
            found_healthy += 1

    assert found_tumor >= 0.90 * n, f"only localised {found_tumor}/{n} tumors"
    assert found_healthy <= 0.30 * n, f"{found_healthy}/{n} false regions on healthy scans"


def test_segmentation_returns_complete_structure(tumor_image):
    seg = segment_tumor(enhance(preprocess(tumor_image)))
    for key in ("brain_mask", "masked", "binary", "threshold", "tumor_found", "tumor_mask"):
        assert key in seg
    assert 0 <= seg["threshold"] <= 255


# ------------------------------------------------------------ Stage 5: feature extraction
def test_feature_vector_is_fixed_length_and_finite(tumor_image, healthy_image):
    v1, n1 = feature_vector(enhance(preprocess(tumor_image)))
    v2, n2 = feature_vector(enhance(preprocess(healthy_image)))
    assert n1 == n2, "feature names must be identical and ordered across images"
    assert v1.shape == v2.shape
    assert len(n1) > 50
    assert np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))


def test_feature_families_are_all_present(tumor_image):
    feats = extract_features(enhance(preprocess(tumor_image)))
    for prefix in ("int_", "glcm_", "lbp_", "shape_", "hist_", "asym_", "reg_"):
        assert any(k.startswith(prefix) for k in feats), f"missing {prefix} features"


def test_features_separate_the_classes():
    """The whole project rests on this: tumor and healthy scans must differ measurably in
    the features we chose. We check the brightest-pixel feature, which is the one the
    trained model ranks most important."""
    r = np.random.default_rng(5)
    t, h = [], []
    for i in range(8):
        for lbl, bucket in ((True, t), (False, h)):
            f = extract_features(enhance(preprocess(generate_brain_mri(lbl, r))))
            bucket.append(f["int_p99"])
    assert np.mean(t) > np.mean(h)


# ------------------------------------------------------------------ real-data handling
REAL_DIR = config.RAW_DIR


def _real_images(limit=None):
    import itertools

    paths = []
    for cls, label in (("yes", 1), ("no", 0)):
        d = REAL_DIR / cls
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    paths.append((f, label))
    if limit:
        paths = list(itertools.islice(paths, 0, len(paths), max(1, len(paths) // limit)))
    return paths


def test_autocrop_removes_a_white_page_frame():
    """Regression test for the scanned-figure bug. Some real images are photographs of
    printed figures and carry a solid white border. Otsu then selects the frame as the
    brightest connected component, the frame is a closed ring, and hole-filling floods its
    interior - so the brain mask became the entire image."""
    from braintumor.preprocess import autocrop_border

    inner = np.random.default_rng(0).integers(0, 200, (100, 120), dtype=np.uint8)
    framed = np.full((140, 160), 255, dtype=np.uint8)
    framed[20:120, 20:140] = inner

    out = autocrop_border(framed)
    assert out.shape == (100, 120), f"expected the frame to be trimmed, got {out.shape}"


def test_autocrop_leaves_a_normal_image_alone():
    from braintumor.preprocess import autocrop_border

    img = np.random.default_rng(1).integers(0, 255, (128, 128), dtype=np.uint8)
    assert autocrop_border(img).shape == (128, 128)


@pytest.mark.skipif(not (REAL_DIR / "yes").is_dir(), reason="real dataset not installed")
def test_no_degenerate_brain_masks_on_real_data():
    """The brain mask must never cover the whole frame or collapse to nothing, whatever
    the input. Before the autocrop and the plausibility guards, 8 of 253 real images
    produced a mask covering 100% of the frame."""
    from braintumor.segment import brain_mask

    for path, _ in _real_images(limit=40):
        enh = enhance(preprocess(load_image(path)))
        cov = (brain_mask(enh) > 0).mean()
        assert config.MIN_BRAIN_COVERAGE <= cov <= config.MAX_BRAIN_COVERAGE, (
            f"{path.name}: brain mask covers {cov:.0%} of the frame"
        )


@pytest.mark.skipif(not (REAL_DIR / "yes").is_dir(), reason="real dataset not installed")
def test_pipeline_runs_on_every_real_image_without_error():
    """Real datasets are messy: mixed sizes from 150px to 1920px, 1/3/4 channels, scanned
    figures, burned-in annotations. Nothing may raise."""
    for path, _ in _real_images(limit=30):
        vec, names = feature_vector(enhance(preprocess(load_image(path))))
        assert np.all(np.isfinite(vec)), f"{path.name} produced non-finite features"
        assert len(names) > 100


@pytest.mark.skipif(not (REAL_DIR / "yes").is_dir(), reason="real dataset not installed")
def test_asymmetry_is_higher_for_tumor_scans():
    """The asymmetry family exists because a lesion breaks left-right symmetry. If that
    premise does not hold on real data, the features are useless - so we assert it."""
    from braintumor.features import asymmetry_features
    from braintumor.segment import brain_mask

    scores = {0: [], 1: []}
    for path, label in _real_images(limit=60):
        enh = enhance(preprocess(load_image(path)))
        feats = asymmetry_features(enh, brain_mask(enh))
        scores[label].append(feats["asym_blob_ratio"])

    assert scores[0] and scores[1]
    assert np.mean(scores[1]) > np.mean(scores[0]), (
        f"tumor asymmetry {np.mean(scores[1]):.4f} did not exceed "
        f"healthy {np.mean(scores[0]):.4f}"
    )


# ------------------------------------------------- Stage 6/7: model, inference, figures
def _load_bundle():
    from braintumor.train import load_model

    try:
        return load_model()
    except FileNotFoundError:
        pytest.skip("no trained model yet - run: python main.py train")


def test_trained_model_exists_and_predicts():
    from braintumor.predict import analyse_image

    bundle = _load_bundle()
    imgs = _real_images(limit=3)
    if not imgs:
        pytest.skip("real dataset not installed")

    result = analyse_image(load_image(imgs[0][0]), bundle)
    assert result["label"] in range(len(bundle.get("class_names", config.CLASS_NAMES)))
    assert 0.0 <= result["tumor_probability"] <= 1.0
    assert result["label_name"] in bundle.get("class_names", config.CLASS_NAMES)


def test_model_bundle_carries_a_tuned_threshold():
    """The clinical operating point must survive being saved and reloaded - if it does not,
    inference silently falls back to 0.5 and undoes the sensitivity tuning."""
    bundle = _load_bundle()
    if bundle.get("task") == "multiclass":
        pytest.skip("threshold tuning applies to the binary task")
    t = bundle.get("threshold")
    assert t is not None, "model bundle has no decision threshold"
    assert 0.0 <= float(t) <= 1.0


def test_inference_uses_the_stored_threshold():
    """A probability just above the threshold must classify as tumor, just below as normal.
    This catches the model.predict() fallback that would ignore the tuned threshold."""
    from braintumor.predict import analyse_image

    bundle = _load_bundle()
    if bundle.get("task") == "multiclass":
        pytest.skip("threshold tuning applies to the binary task")

    imgs = _real_images(limit=12)
    if not imgs:
        pytest.skip("real dataset not installed")

    t = float(bundle.get("threshold", 0.5))
    for path, _ in imgs:
        r = analyse_image(load_image(path), bundle)
        expected = int(r["tumor_probability"] >= t)
        assert r["label"] == expected, (
            f"{path.name}: p={r['tumor_probability']:.3f} threshold={t:.3f} "
            f"but label={r['label']}"
        )


def test_reported_metrics_are_sane():
    """Guards against a broken evaluation path silently reporting nonsense."""
    import json

    path = config.OUTPUT_DIR / "metrics_binary.json"
    if not path.exists():
        pytest.skip("no metrics yet - run: python main.py train")
    r = json.loads(path.read_text())
    m = r["test_metrics"]
    for k in ("accuracy", "balanced_accuracy", "sensitivity_recall", "specificity"):
        assert 0.0 <= m[k] <= 1.0, f"{k} out of range: {m[k]}"
    cm = m["confusion_matrix"]
    assert sum(cm.values()) == r["n_test"], "confusion matrix does not sum to the test size"


def test_pipeline_figure_renders(tmp_path):
    from braintumor.predict import analyse_image
    from braintumor.visualize import pipeline_figure

    bundle = _load_bundle()
    imgs = _real_images(limit=2)
    img = load_image(imgs[0][0]) if imgs else generate_brain_mri(True, np.random.default_rng(3))

    out = tmp_path / "fig.png"
    pipeline_figure(analyse_image(img, bundle), out)
    assert out.exists() and out.stat().st_size > 10_000


def test_results_dashboard_renders(tmp_path):
    import json

    from braintumor.visualize import results_dashboard

    path = config.OUTPUT_DIR / "metrics_binary.json"
    if not path.exists():
        pytest.skip("no metrics yet - run: python main.py train")
    out = tmp_path / "dash.png"
    results_dashboard(json.loads(path.read_text()), out)
    assert out.exists() and out.stat().st_size > 10_000
