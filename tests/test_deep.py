"""
Stage 6b - the deep tumor-typing branch and the ensemble.

Everything here skips cleanly when torch is absent. The deep branch is an OPTIONAL
dependency by design - the binary pipeline, the CLI and the API all work without it - and
a suite that fails on a torch-free machine would misrepresent that.

These tests check contracts and invariants, not accuracy. Accuracy is measured by the
training run against a held-out split; what a unit test can usefully guarantee is that
probabilities are probabilities, that the blend weight is chosen on validation data, and
that a missing CNN degrades rather than crashes.
"""

import numpy as np
import pytest

from braintumor import config

deep = pytest.importorskip("braintumor.deep")
pytestmark = pytest.mark.skipif(not deep.is_available(), reason="torch not installed")


@pytest.fixture(scope="module")
def toy_data():
    """A tiny, deliberately learnable 3-class set: brightness encodes the class.

    Real MRI is not needed to test the plumbing, and using it would make these tests slow
    enough that people stop running them.
    """
    rng = np.random.default_rng(0)
    imgs, labels = [], []
    for cls, level in enumerate((40, 130, 220)):
        for _ in range(12):
            base = np.full((256, 256), level, dtype=np.float64)
            base += rng.normal(0, 12, base.shape)
            imgs.append(np.clip(base, 0, 255).astype(np.uint8))
            labels.append(cls)
    return np.stack(imgs), np.array(labels)


# ================================================================== availability / device
def test_device_selection_returns_a_real_torch_device():
    dev = deep.pick_device()
    assert dev.type in ("mps", "cuda", "cpu")


def test_require_torch_is_a_no_op_when_available():
    deep.require_torch()          # must not raise


# ============================================================================== backbone
def test_backbone_head_matches_the_requested_class_count():
    import torch

    for n in (2, 4):
        net = deep.build_backbone(n, pretrained=False)
        out = net(torch.zeros(2, 3, deep.CNN_INPUT_SIZE, deep.CNN_INPUT_SIZE))
        assert out.shape == (2, n)


# ============================================================================ predictions
def test_predict_proba_returns_a_valid_distribution(toy_data):
    imgs, labels = toy_data
    clf = deep.DeepTumorClassifier(n_classes=3, epochs=1, batch_size=8, verbose=False,
                                   pretrained=False, tta=False)
    clf.fit(imgs, labels)

    p = clf.predict_proba(imgs[:6])
    assert p.shape == (6, 3)
    assert np.all(p >= 0) and np.all(p <= 1)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    # predict() must agree with the argmax of predict_proba(), or the label shown to a user
    # would not correspond to the probabilities shown next to it.
    np.testing.assert_array_equal(clf.predict(imgs[:6]), p.argmax(1))


def test_tta_preserves_the_probability_simplex(toy_data):
    """TTA averages several forward passes; the result must still be a distribution."""
    imgs, labels = toy_data
    clf = deep.DeepTumorClassifier(n_classes=3, epochs=1, batch_size=8, verbose=False,
                                   pretrained=False, tta=True)
    clf.fit(imgs, labels)
    p = clf.predict_proba(imgs[:4])
    assert p.shape == (4, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_augmentation_changes_the_image_but_keeps_the_dtype(toy_data):
    """Augmentation must actually perturb the input - a silently identity transform would
    leave the model exposed to the scale shift these tests exist to guard against."""
    imgs, _ = toy_data
    ds = deep._MRIDataset(imgs, None, train=True, seed=0)
    a = ds._augment(imgs[0])
    assert a.shape == imgs[0].shape
    assert a.dtype == np.uint8
    assert not np.array_equal(a, imgs[0]), "augmentation was a no-op"


def test_validation_selects_the_best_epoch(toy_data):
    imgs, labels = toy_data
    clf = deep.DeepTumorClassifier(n_classes=3, epochs=2, batch_size=8, verbose=False,
                                   pretrained=False, tta=False)
    clf.fit(imgs, labels, imgs[:9], labels[:9])
    assert clf.best_val_balanced_acc is not None
    assert len(clf.history) == 2
    assert all("val_balanced_acc" in h for h in clf.history)


# =============================================================================== ensemble
def test_blend_is_a_proper_convex_combination():
    a = np.array([[0.8, 0.1, 0.1]])
    b = np.array([[0.0, 0.9, 0.1]])

    np.testing.assert_allclose(deep.blend_probabilities(a, b, 0.0), a)
    np.testing.assert_allclose(deep.blend_probabilities(a, b, 1.0), b)
    mid = deep.blend_probabilities(a, b, 0.5)
    np.testing.assert_allclose(mid, (a + b) / 2)
    # A blend of two distributions must remain one, otherwise every downstream confidence
    # figure is meaningless.
    np.testing.assert_allclose(mid.sum(), 1.0, atol=1e-9)


def test_blend_weight_is_clipped_to_the_unit_interval():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    np.testing.assert_allclose(deep.blend_probabilities(a, b, 5.0), b)
    np.testing.assert_allclose(deep.blend_probabilities(a, b, -3.0), a)


def test_weight_search_prefers_the_branch_that_is_actually_right():
    """The classical vector is wrong on every sample and the deep one is right, so the
    search must land on a weight that fixes the predictions.

    We assert the OUTCOME (the chosen weight classifies correctly) rather than a specific
    number. `choose_blend_weight` breaks ties towards 0.5, so on a symmetric setup like
    this one the correct answer genuinely can be 0.5 - pinning the weight itself would be
    testing the tie-break rule instead of the behaviour that matters.
    """
    y = np.array([0, 1, 2, 0, 1, 2])
    classical = np.full((6, 3), 1 / 3)
    classical[np.arange(6), (y + 1) % 3] = 0.9
    deep_p = np.full((6, 3), 0.05)
    deep_p[np.arange(6), y] = 0.9

    info = deep.choose_blend_weight(classical, deep_p, y)
    assert info["weight_deep"] >= 0.5, "search leaned towards the branch that is wrong"
    assert info["val_balanced_accuracy"] == pytest.approx(1.0)
    assert len(info["curve"]) == 21

    pred = deep.blend_probabilities(classical, deep_p, info["weight_deep"]).argmax(1)
    np.testing.assert_array_equal(pred, y)


def test_weight_search_keeps_the_classical_branch_when_the_cnn_adds_nothing():
    """The mirror case. If the CNN is pure noise the search must not drift towards it just
    because blending is the more elaborate option."""
    y = np.array([0, 1, 2, 0, 1, 2])
    classical = np.full((6, 3), 0.05)
    classical[np.arange(6), y] = 0.9
    noise = np.full((6, 3), 1 / 3)

    info = deep.choose_blend_weight(classical, noise, y)
    pred = deep.blend_probabilities(classical, noise, info["weight_deep"]).argmax(1)
    np.testing.assert_array_equal(pred, y)


# ============================================================================ persistence
def test_save_and_load_round_trip_preserves_predictions(toy_data, tmp_path):
    """A reloaded model must predict identically. If it does not, the numbers in the
    training report describe something other than the artefact that ships."""
    imgs, labels = toy_data
    clf = deep.DeepTumorClassifier(n_classes=3, epochs=1, batch_size=8, verbose=False,
                                   pretrained=False, tta=False)
    clf.fit(imgs, labels)
    before = clf.predict_proba(imgs[:5])

    path = tmp_path / "cnn.pt"
    clf.save(path)
    assert path.exists()

    restored = deep.DeepTumorClassifier.load(path)
    restored.tta = False
    np.testing.assert_allclose(restored.predict_proba(imgs[:5]), before, atol=1e-5)


# ======================================================== graceful degradation in loading
def test_missing_weights_file_degrades_instead_of_crashing(tmp_path):
    """load_model must survive a bundle whose .pt file has been deleted. A degraded
    classical-only answer is better than a 500, and the reason is recorded rather than
    swallowed silently."""
    import joblib

    from braintumor.train import load_model

    bundle = {
        "model": None, "feature_names": [], "model_name": "stub",
        "class_names": list(config.MULTICLASS_NAMES), "threshold": 0.5,
        "deep_model_path": "does_not_exist.pt", "blend_weight_deep": 0.5,
    }
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)

    loaded = load_model(path)
    assert loaded["deep_model"] is None
    assert "deep_load_error" in loaded
