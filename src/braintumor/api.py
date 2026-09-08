"""
STAGE 7b - REST API
===================
The HTTP surface the Next.js client in `client/` talks to. It is a thin layer: every
endpoint here delegates to the same pipeline functions the CLI uses, so the web app and the
terminal cannot drift apart in what they report.

Design notes:

  * MODELS ARE LOADED ONCE, LAZILY, AND CACHED. Loading a joblib bundle plus a ResNet-18
    takes a couple of seconds; doing that per request would dominate the response time.
    Loading them at import time instead would mean the server refuses to start when no
    model has been trained yet, which is a worse failure - /health should still answer and
    tell you what is missing.

  * IMAGES ARE RETURNED INLINE AS BASE64 DATA URIs. The alternative - writing files to a
    static directory and returning URLs - needs a cleanup story, a collision-free naming
    scheme, and leaves per-patient scans on disk after the request. Inline bytes keep the
    analysis stateless: nothing about an uploaded scan outlives the response.

  * ERRORS ARE TYPED. A missing model, an undecodable upload, and an oversized file are
    three different problems with three different fixes, so the client is told which.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import config

app = FastAPI(
    title="Brain MRI Diagnostic Assistance API",
    description=(
        "Enhances a brain MRI, segments the suspicious region, and classifies it - "
        "either tumor / no tumor, or by tumor type. Assistive only: not a medical device."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ======================================================================================
# MODEL CACHE
# ======================================================================================
_BUNDLES: dict[str, Any] = {}


def get_bundle(multiclass: bool):
    """Load and cache a model bundle. Raises a 503 when it has not been trained yet."""
    from .train import load_model

    key = "multiclass" if multiclass else "binary"
    if key not in _BUNDLES:
        try:
            _BUNDLES[key] = load_model(multiclass=multiclass)
        except FileNotFoundError as exc:
            cmd = ("python main.py train --dataset kaggle4 --multiclass --deep"
                   if multiclass else "python main.py train --dataset kaggle4")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "model_not_trained",
                    "message": str(exc),
                    "fix": f"Run: {cmd}",
                },
            ) from exc
    return _BUNDLES[key]


# ======================================================================================
# HELPERS
# ======================================================================================
def _to_data_uri(image: np.ndarray, fmt: str = ".png") -> str:
    """Encode an OpenCV image (grayscale or BGR) as a base64 data URI."""
    ok, buf = cv2.imencode(fmt, image)
    if not ok:
        raise HTTPException(status_code=500, detail={"error": "encode_failed",
                                                     "message": "Could not encode image"})
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/{fmt.lstrip('.')};base64,{b64}"


async def _read_upload(file: UploadFile) -> np.ndarray:
    """Validate an uploaded file and decode it to a grayscale array.

    We check the SIZE before decoding, not after. cv2.imdecode on a hostile or corrupt
    file allocates based on the header's declared dimensions, so validating afterwards is
    validating too late.
    """
    if file.content_type and file.content_type not in config.ALLOWED_UPLOAD_TYPES:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "unsupported_type",
                "message": f"'{file.content_type}' is not an accepted image type.",
                "accepted": sorted(config.ALLOWED_UPLOAD_TYPES),
            },
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail={"error": "empty_file",
                                                     "message": "The uploaded file is empty."})
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": f"File is {len(raw)/1e6:.1f} MB; the limit is "
                           f"{config.MAX_UPLOAD_BYTES/1e6:.0f} MB.",
            },
        )

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "decode_failed",
                "message": "The file could not be decoded as an image. MRI slices should "
                           "be PNG, JPEG, BMP or TIFF.",
            },
        )
    return img


def _serialise(result: dict, include_images: bool = True) -> dict:
    """Turn the pipeline's result dict into something JSON can carry.

    The raw result holds numpy arrays for every stage. We keep the numbers, convert the
    pictures to data URIs, and drop the 171-value feature vector by default - it is large,
    and no UI shows all of it. `/analyze/features` returns it for anyone who wants it.
    """
    seg = result["segmentation"]
    er = result["enhancement_report"]

    payload: dict[str, Any] = {
        "prediction": {
            "label": int(result["label"]),
            "label_name": result["label_name"],
            "is_tumor": bool(result["is_tumor"]),
            "confidence": float(result["confidence"]),
            "tumor_probability": float(result["tumor_probability"]),
            "class_names": list(result["class_names"]),
            "probabilities": result.get("probabilities"),
            "threshold": float(result.get("threshold", 0.5)),
        },
        "model": {
            "name": result.get("model_name", "unknown"),
            "branch": result.get("branch", "classical"),
            "blend_weight_deep": float(result.get("blend_weight_deep", 0.0)),
            "deep_probabilities": result.get("deep_probabilities"),
        },
        "segmentation": {
            "tumor_found": bool(seg.get("tumor_found")),
            "n_candidates": int(seg.get("n_candidates", 0)),
            "threshold": float(seg.get("threshold", 0)),
            "properties": None,
        },
        "enhancement": {
            "contrast_before": float(er["contrast_std_before"]),
            "contrast_after": float(er["contrast_std_after"]),
            "entropy_before": float(er["entropy_before"]),
            "entropy_after": float(er["entropy_after"]),
            "sharpness_before": float(er["sharpness_before"]),
            "sharpness_after": float(er["sharpness_after"]),
        },
    }

    if seg.get("tumor_found") and seg.get("properties"):
        p = seg["properties"]
        payload["segmentation"]["properties"] = {
            "centroid": [float(p["centroid"][0]), float(p["centroid"][1])],
            "bbox": [int(v) for v in p["bbox"]],
            "area": float(p["area"]),
            "area_ratio": float(p.get("area_ratio", 0.0)),
            "perimeter": float(p["perimeter"]),
            "circularity": float(p["circularity"]),
            "solidity": float(p["solidity"]),
            "extent": float(p["extent"]),
            "aspect_ratio": float(p["aspect_ratio"]),
            "mean_intensity": float(p["mean_intensity"]),
            "max_intensity": float(p["max_intensity"]),
            "brightness_ratio": float(p.get("brightness_ratio", 0.0)),
        }

    if include_images:
        # Every stage the pipeline produced, so the client can show the same 8-panel story
        # the CLI figure tells rather than just a verdict the user has to take on faith.
        payload["images"] = {
            "original": _to_data_uri(result["original"]),
            "preprocessed": _to_data_uri(result["preprocessed"]),
            "enhanced": _to_data_uri(result["enhanced"]),
            "brain_mask": _to_data_uri(seg["brain_mask"]),
            "segmentation_mask": _to_data_uri(seg["binary"]),
            "overlay": _to_data_uri(result["overlay"]),
        }

    # A few location features are worth surfacing on their own: they are the measurements
    # that justify a TYPE call, and they are readable by a clinician without a stats
    # background ("sits on the midline", "hugs the brain edge").
    feats = result.get("features", {})
    if feats:
        payload["location"] = {
            k: float(feats[k])
            for k in ("loc_found", "loc_radial", "loc_midline_abs", "loc_depth_centroid",
                      "loc_border_contact", "loc_band_sup", "loc_band_mid", "loc_band_inf")
            if k in feats
        }
    return payload


# ======================================================================================
# SCHEMAS
# ======================================================================================
class HealthResponse(BaseModel):
    # Pydantic resolves these annotations at runtime, so on Python 3.9 they must use
    # typing.Optional rather than the 3.10+ `str | None` syntax - `from __future__ import
    # annotations` defers the evaluation but does not make the operator legal.
    status: str
    version: str
    models: dict
    torch_available: bool
    device: Optional[str] = None


# ======================================================================================
# ENDPOINTS
# ======================================================================================
@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": "Brain MRI Diagnostic Assistance API",
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["/health", "/models", "/analyze", "/analyze/features", "/metrics",
                      "/classes"],
        "disclaimer": "Assistive tool only. Not a medical device. Not for clinical use.",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness plus a report of which models are actually on disk.

    Deliberately does not 503 when a model is missing: the point of a health endpoint is to
    tell you what is wrong, which it cannot do if it refuses to answer.
    """
    from . import deep as deep_mod

    models = {}
    for key, path in (("binary", config.MODEL_PATH),
                      ("multiclass", config.MULTICLASS_MODEL_PATH)):
        models[key] = {"trained": Path(path).exists(), "path": str(path)}

    device = None
    if deep_mod.is_available():
        try:
            device = str(deep_mod.pick_device())
        except Exception:
            device = None

    return HealthResponse(
        status="ok",
        version=app.version,
        models=models,
        torch_available=deep_mod.is_available(),
        device=device,
    )


@app.get("/models", tags=["meta"])
def models_info() -> dict:
    """Describe the trained models without loading images through the pipeline."""
    out: dict[str, Any] = {}
    for key, multiclass in (("binary", False), ("multiclass", True)):
        try:
            b = get_bundle(multiclass)
        except HTTPException:
            out[key] = {"trained": False}
            continue
        out[key] = {
            "trained": True,
            "model_name": b.get("model_name"),
            "task": b.get("task"),
            "dataset": b.get("dataset_key"),
            "source": b.get("source"),
            "class_names": b.get("class_names"),
            "n_features": len(b.get("feature_names", [])),
            "threshold": float(b.get("threshold", 0.5)),
            "has_deep_branch": b.get("deep_model") is not None,
            "blend_weight_deep": float(b.get("blend_weight_deep", 0.0)),
            "shipped_branch": b.get("shipped_branch", "classical"),
            "deep_load_error": b.get("deep_load_error"),
        }
    return out


@app.get("/classes", tags=["meta"])
def classes() -> dict:
    """The label sets, with a plain-language note on each tumor type.

    The descriptions are here rather than hardcoded in the frontend so that the API stays
    the single source of truth about what the labels mean.
    """
    return {
        "binary": list(config.CLASS_NAMES),
        "multiclass": list(config.MULTICLASS_NAMES),
        "descriptions": {
            "No Tumor": "No suspicious mass detected on this slice.",
            "Glioma": "Arises from glial cells inside the brain tissue itself "
                      "(intra-axial). Typically embedded in white matter.",
            "Meningioma": "Arises from the meninges, the membranes lining the skull "
                          "(extra-axial). Typically sits against the brain's outer edge.",
            "Pituitary": "Arises from the pituitary gland at the skull base. Typically "
                         "central and on the midline.",
        },
    }


@app.post("/analyze", tags=["analysis"])
async def analyze(
    file: UploadFile = File(..., description="A brain MRI slice (PNG/JPEG/BMP/TIFF)"),
    multiclass: bool = Query(False, description="Classify tumor TYPE instead of presence"),
    include_images: bool = Query(True, description="Return the per-stage images inline"),
) -> JSONResponse:
    """Run the full pipeline on one uploaded MRI and return the verdict plus every stage."""
    from .predict import analyse_image

    t0 = time.time()
    image = await _read_upload(file)
    bundle = get_bundle(multiclass)

    try:
        result = analyse_image(image, bundle)
    except ValueError as exc:
        # Raised by predict.py when the saved model's feature list does not match the code.
        raise HTTPException(status_code=500, detail={"error": "feature_mismatch",
                                                     "message": str(exc)}) from exc

    payload = _serialise(result, include_images=include_images)
    payload["meta"] = {
        "filename": file.filename,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "task": "multiclass" if multiclass else "binary",
    }
    payload["disclaimer"] = (
        "Assistive output only. This is not a medical device and must not be the sole "
        "basis for any clinical decision."
    )
    return JSONResponse(payload)


@app.post("/analyze/features", tags=["analysis"])
async def analyze_features(
    file: UploadFile = File(...),
    multiclass: bool = Query(False),
) -> dict:
    """The complete named feature vector for one image.

    Exposed because 'explainable' has to mean something concrete. A radiologist or reviewer
    can pull the exact 171 measurements the classifier saw, rather than being asked to
    trust a number.
    """
    from .predict import analyse_image

    image = await _read_upload(file)
    bundle = get_bundle(multiclass)
    result = analyse_image(image, bundle)

    feats = {k: float(v) for k, v in result["features"].items()}
    families = {
        "intensity": "int_", "texture_glcm": "glcm_", "texture_lbp": "lbp_",
        "shape": "shape_", "asymmetry": "asym_", "regional": "reg_",
        "location": "loc_", "plane": "plane_",
    }
    grouped = {
        family: {k: v for k, v in feats.items() if k.startswith(prefix)}
        for family, prefix in families.items()
    }
    return {
        "n_features": len(feats),
        "features": feats,
        "by_family": grouped,
        "family_sizes": {k: len(v) for k, v in grouped.items()},
    }


@app.get("/metrics", tags=["analysis"])
def metrics(
    multiclass: bool = Query(False, description="Return the tumor-typing report"),
) -> dict:
    """The last training report for the requested task.

    `feature_names` is stripped out: it is a 171-element list that no client renders, and
    it roughly doubles the response size.
    """
    import json

    task = "multiclass" if multiclass else "binary"
    path = config.OUTPUT_DIR / f"metrics_{task}.json"
    if not path.exists():
        path = config.METRICS_PATH
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail={"error": "no_metrics",
                    "message": "No training report found.",
                    "fix": "Run: python main.py train --dataset kaggle4"},
        )

    report = json.loads(path.read_text())
    report.pop("feature_names", None)
    return report
