"""
Stage 7b - the REST API.

These tests drive the app through FastAPI's TestClient, which exercises the real routing,
validation and serialisation stack in-process. No server is started and no port is bound,
so the suite stays runnable in CI and cannot collide with a backend the user already has
running on port 3000.

Tests that need a trained model skip rather than fail when none exists. A fresh clone has
no `models/*.joblib`, and a test suite that goes red on a clean checkout trains people to
ignore red.
"""

import io

import cv2
import numpy as np
import pytest

from braintumor import config

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture(scope="module")
def client():
    from braintumor.api import app

    return fastapi_testclient.TestClient(app)


@pytest.fixture(scope="module")
def mri_png() -> bytes:
    """A synthetic brain MRI encoded as PNG bytes, ready to upload."""
    from braintumor.dataset import generate_brain_mri

    img = generate_brain_mri(True, np.random.default_rng(7))
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _has_model(multiclass: bool) -> bool:
    path = config.MULTICLASS_MODEL_PATH if multiclass else config.MODEL_PATH
    return path.exists()


# ============================================================================ meta routes
def test_root_lists_the_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "/analyze" in body["endpoints"]
    # The disclaimer is a product requirement, not a nicety: this API returns
    # clinical-sounding verdicts and every response path must carry the caveat.
    assert "not a medical device" in body["disclaimer"].lower()


def test_health_answers_even_with_no_model(client):
    """/health must never 503. Its whole job is to report what is wrong, which it cannot
    do if it refuses to answer when something is."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "binary" in body["models"] and "multiclass" in body["models"]
    assert isinstance(body["torch_available"], bool)


def test_classes_describes_every_label(client):
    r = client.get("/classes")
    assert r.status_code == 200
    body = r.json()
    assert body["multiclass"] == list(config.MULTICLASS_NAMES)
    for name in config.MULTICLASS_NAMES:
        assert name in body["descriptions"], f"no description for {name}"


def test_cors_allows_the_client_origin(client):
    """The frontend runs on a different port, so without this header every browser request
    fails - and it fails in a way that looks like a backend outage rather than a policy."""
    r = client.options(
        "/analyze",
        headers={
            "Origin": f"http://localhost:{config.CLIENT_PORT}",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code in (200, 204)
    allowed = r.headers.get("access-control-allow-origin")
    assert allowed == f"http://localhost:{config.CLIENT_PORT}"


# ======================================================================= upload validation
def test_rejects_an_undecodable_file(client):
    r = client.post(
        "/analyze",
        files={"file": ("notanimage.png", b"this is not a PNG", "image/png")},
    )
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "decode_failed"


def test_rejects_an_empty_file(client):
    r = client.post("/analyze", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "empty_file"


def test_rejects_an_unsupported_content_type(client):
    r = client.post(
        "/analyze",
        files={"file": ("notes.pdf", b"%PDF-1.4 ...", "application/pdf")},
    )
    assert r.status_code == 415
    assert r.json()["detail"]["error"] == "unsupported_type"


def test_rejects_an_oversized_upload(client):
    """Size is checked BEFORE decoding. cv2.imdecode allocates from the header's declared
    dimensions, so validating afterwards would be validating too late to help."""
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * (config.MAX_UPLOAD_BYTES + 1024)
    r = client.post("/analyze", files={"file": ("huge.png", payload, "image/png")})
    assert r.status_code == 413
    assert r.json()["detail"]["error"] == "file_too_large"


def test_missing_file_field_is_a_validation_error(client):
    assert client.post("/analyze").status_code == 422


# ============================================================================= analysis
@pytest.mark.skipif(not _has_model(False), reason="binary model not trained")
def test_analyze_binary_returns_a_complete_payload(client, mri_png):
    r = client.post("/analyze", files={"file": ("scan.png", mri_png, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()

    p = body["prediction"]
    assert p["label"] in (0, 1)
    assert p["label_name"] in config.CLASS_NAMES
    assert 0.0 <= p["confidence"] <= 1.0
    assert 0.0 <= p["tumor_probability"] <= 1.0
    assert p["is_tumor"] == (p["label"] > 0)

    # Every stage image must be present: the auditability claim is the product, and a
    # response carrying only a verdict silently drops it.
    for key in ("original", "preprocessed", "enhanced", "brain_mask",
                "segmentation_mask", "overlay"):
        assert body["images"][key].startswith("data:image/png;base64,")

    assert body["enhancement"]["contrast_after"] > 0
    assert body["meta"]["task"] == "binary"


@pytest.mark.skipif(not _has_model(True), reason="multiclass model not trained")
def test_analyze_multiclass_returns_a_normalised_distribution(client, mri_png):
    r = client.post(
        "/analyze?multiclass=true", files={"file": ("scan.png", mri_png, "image/png")}
    )
    assert r.status_code == 200, r.text
    p = r.json()["prediction"]

    assert p["class_names"] == list(config.MULTICLASS_NAMES)
    probs = p["probabilities"]
    assert probs is not None and len(probs) == 4
    assert sum(probs) == pytest.approx(1.0, abs=1e-4), "probabilities must sum to 1"
    # The reported label must be the argmax of the vector shown to the user. If these ever
    # disagree, the interface is displaying evidence for a different conclusion than the
    # one it prints.
    assert p["label"] == max(range(4), key=lambda i: probs[i])


@pytest.mark.skipif(not _has_model(False), reason="binary model not trained")
def test_include_images_false_drops_the_payload(client, mri_png):
    """The images dominate the response size, so a client that only needs the verdict must
    be able to opt out."""
    small = client.post(
        "/analyze?include_images=false", files={"file": ("s.png", mri_png, "image/png")}
    )
    full = client.post("/analyze", files={"file": ("s.png", mri_png, "image/png")})
    assert small.status_code == full.status_code == 200
    assert "images" not in small.json()
    assert len(small.content) < len(full.content)


@pytest.mark.skipif(not _has_model(False), reason="binary model not trained")
def test_feature_endpoint_groups_every_family(client, mri_png):
    r = client.post("/analyze/features", files={"file": ("s.png", mri_png, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["n_features"] > 150
    for family in ("intensity", "texture_glcm", "shape", "asymmetry", "location", "plane"):
        assert body["family_sizes"][family] > 0, f"{family} family is empty"


@pytest.mark.skipif(not _has_model(False), reason="binary model not trained")
def test_metrics_endpoint_returns_the_training_report(client):
    r = client.get("/metrics")
    if r.status_code == 404:
        pytest.skip("no training report on disk")
    body = r.json()
    assert "test_metrics" in body
    # Stripped deliberately: a 171-element list no client renders, roughly doubling the
    # response size.
    assert "feature_names" not in body


@pytest.mark.skipif(not _has_model(False), reason="binary model not trained")
def test_models_endpoint_describes_what_is_loaded(client):
    body = client.get("/models").json()
    assert body["binary"]["trained"] is True
    assert body["binary"]["n_features"] > 150
