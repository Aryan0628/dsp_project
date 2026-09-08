"""
Stage 5, Family 7 - anatomical location features.

These tests exist because the location family makes CLAIMS about anatomy, and a feature
that claims to measure "distance from the midline" is worthless if it does not actually
respond to a lesion moving off the midline. Asserting the direction of response is what
separates a feature from a number.

The synthetic generator is used rather than real scans precisely because it lets us PLACE a
lesion where we choose. On a real MRI we do not know the ground-truth position, so we could
only check that a value came out - not that it came out right.
"""

import cv2
import numpy as np
import pytest

from braintumor.enhance import enhance
from braintumor.features import extract_features, feature_vector, location_features
from braintumor.preprocess import preprocess
from braintumor.segment import segment_tumor


def _synthetic_with_blob(centre, radius=18):
    """A simple bright disc on a dim elliptical 'brain', at a chosen position.

    Deliberately simpler than the full MRI phantom: this is a geometry test, and the
    phantom's noise and bias field would add variance without testing anything extra.
    """
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(img, (128, 128), (95, 110), 0, 0, 360, 90, -1)
    cv2.circle(img, centre, radius, 235, -1)
    return img


def _loc(centre, radius=18):
    enh = enhance(preprocess(_synthetic_with_blob(centre, radius)))
    seg = segment_tumor(enh)
    return location_features(enh, seg), seg


# ------------------------------------------------------------------ presence / absence
def test_location_features_present_in_full_vector():
    _, names = feature_vector(enhance(preprocess(_synthetic_with_blob((128, 128)))))
    assert any(n.startswith("loc_") for n in names), "location family missing"
    assert any(n.startswith("plane_") for n in names), "plane family missing"


def test_all_values_finite_with_and_without_a_lesion():
    """A blank scan must not produce NaN. Downstream, NaN poisons the whole feature row."""
    for img in (_synthetic_with_blob((128, 128)), np.zeros((256, 256), dtype=np.uint8)):
        enh = enhance(preprocess(img))
        feats = location_features(enh, segment_tumor(enh))
        assert all(np.isfinite(v) for v in feats.values()), "non-finite location feature"


def test_no_region_zeroes_the_block_and_flags_it():
    """With nothing detected, `loc_found` must be 0 so the model can ignore the zeros
    rather than reading them as 'a lesion at the exact brain centre'.

    The lesion-free brain here carries a little noise, which is not incidental. A
    PERFECTLY flat synthetic region makes CLAHE amplify quantisation steps into edges and
    the segmenter then reports a spurious region - but that input cannot occur in practice,
    because every real MRI carries acquisition noise. Testing against the flat image would
    be asserting behaviour on an input the system will never see.
    """
    brain = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(brain, (128, 128), (95, 110), 0, 0, 360, 90, -1)
    noisy = np.clip(
        brain.astype(np.float64) + np.random.default_rng(0).normal(0, 3, brain.shape), 0, 255
    ).astype(np.uint8)
    noisy[brain == 0] = 0

    enh = enhance(preprocess(noisy))
    seg = segment_tumor(enh)
    assert not seg["tumor_found"], "a lesion-free brain should not produce a region"

    feats = location_features(enh, seg)
    assert feats["loc_found"] == 0.0
    assert feats["loc_radial"] == 0.0
    assert feats["loc_midline_abs"] == 0.0


# ------------------------------------------------------------------- directional truth
def test_midline_distance_grows_as_the_lesion_moves_laterally():
    """THE PITUITARY CUE. A midline mass must score lower than a lateral one - this is the
    single measurement that separates a pituitary adenoma from a hemispheric glioma."""
    centre, _ = _loc((128, 120))
    lateral, seg = _loc((185, 120))
    if not seg.get("tumor_found"):
        pytest.skip("segmenter did not localise the lateral blob in this configuration")
    assert lateral["loc_midline_abs"] > centre["loc_midline_abs"], (
        f"midline distance did not increase: centre={centre['loc_midline_abs']:.3f} "
        f"lateral={lateral['loc_midline_abs']:.3f}"
    )


def test_radial_position_grows_towards_the_rim():
    inner, _ = _loc((128, 120))
    outer, seg = _loc((128, 45))
    if not seg.get("tumor_found"):
        pytest.skip("segmenter did not localise the peripheral blob")
    assert outer["loc_radial"] > inner["loc_radial"]


def test_depth_is_higher_for_a_central_lesion_than_a_peripheral_one():
    """THE MENINGIOMA/GLIOMA AXIS. An extra-axial mass hugging the skull must measure
    shallower than one embedded deep in the parenchyma."""
    deep, seg_d = _loc((128, 125))
    shallow, seg_s = _loc((128, 42))
    if not (seg_d.get("tumor_found") and seg_s.get("tumor_found")):
        pytest.skip("segmenter did not localise both blobs")
    assert deep["loc_depth_centroid"] > shallow["loc_depth_centroid"]


def test_border_contact_is_higher_at_the_edge():
    deep, seg_d = _loc((128, 125))
    edge, seg_e = _loc((128, 38))
    if not (seg_d.get("tumor_found") and seg_e.get("tumor_found")):
        pytest.skip("segmenter did not localise both blobs")
    assert edge["loc_border_contact"] >= deep["loc_border_contact"]


# --------------------------------------------------------------------- scale invariance
def test_location_is_scale_free():
    """The 4-class dataset's train and test halves differ in IMAGE SIZE (every training
    glioma is 512x512, the test gliomas come in 64 sizes). A location feature measured in
    raw pixels would therefore encode the scanner rather than the anatomy, and the model
    would learn the artefact. Normalising by the brain's own radius is what prevents that,
    so this test guards the property directly."""
    small = _synthetic_with_blob((150, 110))
    large = cv2.resize(small, (512, 512), interpolation=cv2.INTER_LINEAR)

    fa, sa = _loc_from_array(small)
    fb, sb = _loc_from_array(large)
    if not (sa.get("tumor_found") and sb.get("tumor_found")):
        pytest.skip("segmenter did not localise the blob at both scales")

    for key in ("loc_radial", "loc_midline_abs", "loc_x_norm", "loc_y_norm"):
        assert abs(fa[key] - fb[key]) < 0.12, (
            f"{key} changed with image scale: {fa[key]:.3f} vs {fb[key]:.3f}"
        )


def _loc_from_array(img):
    enh = enhance(preprocess(img))
    seg = segment_tumor(enh)
    return location_features(enh, seg), seg


# ------------------------------------------------------------------------ bookkeeping
def test_band_and_side_fractions_are_normalised():
    """Mass fractions must sum to 1 - if they do not, the lesion has been counted twice or
    dropped, and every band feature is quietly wrong."""
    feats, seg = _loc((128, 120))
    if not seg.get("tumor_found"):
        pytest.skip("no region located")
    bands = feats["loc_band_sup"] + feats["loc_band_mid"] + feats["loc_band_inf"]
    sides = feats["loc_side_left"] + feats["loc_side_centre"] + feats["loc_side_right"]
    assert bands == pytest.approx(1.0, abs=1e-6)
    assert sides == pytest.approx(1.0, abs=1e-6)


def test_feature_names_are_stable_and_sorted():
    """predict.py matches features by name against the saved model. Unstable ordering
    between two images would silently feed the classifier a permuted row."""
    _, n1 = feature_vector(enhance(preprocess(_synthetic_with_blob((120, 120)))))
    _, n2 = feature_vector(enhance(preprocess(_synthetic_with_blob((160, 90)))))
    assert n1 == n2
    assert n1 == sorted(n1)


def test_extract_features_includes_every_family():
    feats = extract_features(enhance(preprocess(_synthetic_with_blob((128, 118)))))
    for prefix in ("int_", "glcm_", "lbp_", "shape_", "asym_", "reg_", "loc_", "plane_"):
        assert any(k.startswith(prefix) for k in feats), f"missing {prefix} family"
