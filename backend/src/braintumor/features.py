"""
STAGE 5 - FEATURE EXTRACTION
============================
Goal: turn a picture into a fixed-length row of numbers a classifier can learn from.

We deliberately use *interpretable* features rather than raw pixels, in four families:

  1. First-order intensity statistics  - how bright, how spread out, how skewed
  2. GLCM / Haralick texture           - how the brightness of NEIGHBOURING pixels relates
  3. LBP micro-texture                 - local patterns, robust to brightness shifts
  4. Shape / region descriptors        - geometry of the region found in Stage 4
  5. Bilateral asymmetry               - do the two halves of the brain disagree?
  6. Regional statistics               - a coarse 3x3 spatial map of the brain

Every feature has a name, so after training we can report exactly which measurements the
model relies on. That is something a raw-pixel CNN cannot give you.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from . import config
from .segment import segment_tumor


# --------------------------------------------------------------------------------------
# Family 1 - first order intensity statistics
# --------------------------------------------------------------------------------------
def intensity_features(image: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Statistics of the intensity distribution over brain tissue only.

    Restricting to the brain matters: if we included the black background, every image
    would be dominated by zeros and the statistics would be meaningless.
    """
    pixels = image[mask > 0].astype(np.float64) if mask is not None else image.ravel().astype(np.float64)
    if pixels.size == 0:
        pixels = image.ravel().astype(np.float64)

    mean = float(pixels.mean())
    std = float(pixels.std())
    # skewness  = asymmetry of the distribution (a bright lesion pulls the right tail out)
    # kurtosis  = "peakedness"/tail weight relative to a normal distribution
    skew = float(((pixels - mean) ** 3).mean() / (std ** 3)) if std > 1e-6 else 0.0
    kurt = float(((pixels - mean) ** 4).mean() / (std ** 4)) if std > 1e-6 else 0.0

    hist = cv2.calcHist([image], [0], mask, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1e-9)
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())   # Shannon entropy in bits

    feats = {
        "int_mean": mean,
        "int_std": std,
        "int_var": float(pixels.var()),
        "int_skew": skew,
        "int_kurtosis": kurt,
        "int_entropy": entropy,
        "int_min": float(pixels.min()),
        "int_max": float(pixels.max()),
        "int_median": float(np.median(pixels)),
        "int_range": float(pixels.max() - pixels.min()),
        "int_energy": float((pixels ** 2).mean()) / 1e4,
        "int_rms": float(np.sqrt((pixels ** 2).mean())),
        # upper percentiles capture "how bright are the brightest bits" - directly relevant
        # because tumors present as hyperintense regions
        "int_p90": float(np.percentile(pixels, 90)),
        "int_p95": float(np.percentile(pixels, 95)),
        "int_p99": float(np.percentile(pixels, 99)),
        # fraction of brain that is unusually bright
        "int_bright_frac": float((pixels > mean + 2 * std).mean()),
        "int_verybright_frac": float((pixels > mean + 3 * std).mean()),
    }

    # coarse normalised histogram - a compact shape descriptor of the whole distribution
    coarse = cv2.calcHist([image], [0], mask, [config.HIST_BINS], [0, 256]).ravel()
    coarse = coarse / max(coarse.sum(), 1e-9)
    for i, v in enumerate(coarse):
        feats[f"hist_{i:02d}"] = float(v)

    return feats


# --------------------------------------------------------------------------------------
# Family 2 - GLCM (Haralick) texture
# --------------------------------------------------------------------------------------
def glcm_features(image: np.ndarray) -> dict:
    """Grey Level Co-occurrence Matrix texture descriptors.

    The GLCM counts how often a pixel of grey level i sits at a given offset from a pixel
    of grey level j. It is a SECOND-ORDER statistic: it describes spatial relationships,
    which plain mean/variance cannot see.

    Tumor tissue is texturally heterogeneous and chaotic; healthy white matter is smooth
    and regular. That difference shows up strongly in these numbers.

    We quantise to 32 grey levels (a 32x32 matrix instead of 256x256) - this keeps the
    matrix dense enough to be statistically stable and 64x cheaper to compute.
    We average over four angles so the descriptor is rotation-robust.
    """
    levels = config.GLCM_LEVELS
    quantised = (image.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    quantised = np.clip(quantised, 0, levels - 1)

    angles = [np.deg2rad(a) for a in config.GLCM_ANGLES_DEG]
    glcm = graycomatrix(
        quantised,
        distances=config.GLCM_DISTANCES,
        angles=angles,
        levels=levels,
        symmetric=True,      # treat (i,j) and (j,i) as the same pairing
        normed=True,         # convert counts to probabilities
    )

    feats = {}
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    for prop in props:
        values = graycoprops(glcm, prop)          # shape: (n_distances, n_angles)
        for di, d in enumerate(config.GLCM_DISTANCES):
            row = values[di]
            feats[f"glcm_{prop}_d{d}_mean"] = float(row.mean())
            feats[f"glcm_{prop}_d{d}_range"] = float(row.max() - row.min())

    # GLCM entropy is not provided by graycoprops, so compute it directly
    eps = 1e-12
    ent = -(glcm * np.log2(glcm + eps)).sum(axis=(0, 1))
    feats["glcm_entropy_mean"] = float(ent.mean())
    feats["glcm_entropy_std"] = float(ent.std())
    return feats


# --------------------------------------------------------------------------------------
# Family 3 - Local Binary Patterns
# --------------------------------------------------------------------------------------
def lbp_features(image: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Local Binary Pattern histogram.

    For each pixel, compare it with P neighbours on a circle of radius R. Each comparison
    contributes one bit (1 if the neighbour is brighter), giving an integer code per pixel.
    The histogram of those codes is a micro-texture signature.

    We use the "uniform" variant: codes with at most two 0<->1 transitions around the
    circle are kept separately and all others collapse into one bin. Uniform patterns
    correspond to meaningful primitives (edges, corners, spots) and the reduced bin count
    makes the histogram far more stable. LBP is invariant to monotonic grey-level changes,
    so it survives brightness differences between scanners.
    """
    lbp = local_binary_pattern(image, config.LBP_POINTS, config.LBP_RADIUS, method="uniform")
    n_bins = config.LBP_POINTS + 2

    values = lbp[mask > 0] if mask is not None and mask.any() else lbp.ravel()
    hist, _ = np.histogram(values, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    hist /= max(hist.sum(), 1e-9)

    feats = {f"lbp_{i:02d}": float(v) for i, v in enumerate(hist)}
    nz = hist[hist > 0]
    feats["lbp_entropy"] = float(-(nz * np.log2(nz)).sum())
    feats["lbp_uniformity"] = float((hist ** 2).sum())
    return feats


# --------------------------------------------------------------------------------------
# Family 4 - shape / region descriptors from Stage 4
# --------------------------------------------------------------------------------------
def shape_features(seg: dict) -> dict:
    """Geometry of the suspicious region detected during segmentation.

    When no region was found, every value is zero and `shape_found` is 0 - which is itself
    a strong signal that the scan is normal.
    """
    if not seg.get("tumor_found") or seg.get("properties") is None:
        return {
            "shape_found": 0.0,
            "shape_area": 0.0,
            "shape_area_ratio": 0.0,
            "shape_perimeter": 0.0,
            "shape_circularity": 0.0,
            "shape_solidity": 0.0,
            "shape_extent": 0.0,
            "shape_aspect": 0.0,
            "shape_mean_int": 0.0,
            "shape_max_int": 0.0,
            "shape_brightness_ratio": 0.0,
            "shape_n_candidates": float(seg.get("n_candidates", 0)),
        }

    p = seg["properties"]
    return {
        "shape_found": 1.0,
        "shape_area": float(p["area"]),
        "shape_area_ratio": float(p.get("area_ratio", 0.0)),
        "shape_perimeter": float(p["perimeter"]),
        "shape_circularity": float(p["circularity"]),
        "shape_solidity": float(p["solidity"]),
        "shape_extent": float(p["extent"]),
        "shape_aspect": float(p["aspect_ratio"]),
        "shape_mean_int": float(p["mean_intensity"]),
        "shape_max_int": float(p["max_intensity"]),
        "shape_brightness_ratio": float(p.get("brightness_ratio", 0.0)),
        "shape_n_candidates": float(seg.get("n_candidates", 0)),
    }


# --------------------------------------------------------------------------------------
# Family 5 - bilateral asymmetry
# --------------------------------------------------------------------------------------
def asymmetry_features(image: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Left-right asymmetry of the brain about its mid-sagittal plane.

    This is one of the strongest and most clinically intuitive cues available. A healthy
    brain is approximately symmetric across the midline: whatever structure exists on the
    left has a counterpart on the right. A space-occupying lesion breaks that symmetry -
    it adds bright tissue on one side only, and often pushes the midline across (mass
    effect). Radiologists use exactly this cue when they scan an image quickly.

    It is also complementary to everything else we compute. Intensity and texture features
    describe WHAT is in the image; asymmetry describes whether the two halves DISAGREE,
    which stays informative even when the absolute brightness is unusual - and mixed MRI
    sequences (T1, T2, FLAIR) in one dataset make absolute brightness unreliable.

    Implementation detail that matters: we mirror about the brain's own centroid rather
    than the image centre, because scans are rarely perfectly centred and a few pixels of
    offset would register as large fake asymmetry everywhere.

    Caveat we accept knowingly: on sagittal slices a horizontal flip compares anterior with
    posterior, which is not a symmetry plane. The feature is then uninformative rather than
    misleading, and the classifier learns to down-weight it.
    """
    h, w = image.shape[:2]
    if mask is None or not mask.any():
        mask = np.ones((h, w), dtype=np.uint8) * 255

    # centre the brain horizontally on its centroid so the mirror axis is anatomical
    moments = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    cx = moments["m10"] / moments["m00"] if moments["m00"] > 0 else w / 2.0
    shift = int(round(w / 2.0 - cx))
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    centred = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    centred_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    mirrored = cv2.flip(centred, 1)
    # only compare pixels that are inside the brain on BOTH sides, otherwise the brain
    # outline itself dominates the difference
    both = cv2.bitwise_and(centred_mask, cv2.flip(centred_mask, 1))

    diff = cv2.absdiff(centred, mirrored)
    values = diff[both > 0].astype(np.float64)
    if values.size == 0:
        values = diff.ravel().astype(np.float64)

    feats = {
        "asym_mean": float(values.mean()),
        "asym_std": float(values.std()),
        "asym_max": float(values.max()),
        "asym_p90": float(np.percentile(values, 90)),
        "asym_p99": float(np.percentile(values, 99)),
        # fraction of the brain where the two halves disagree strongly
        "asym_frac_30": float((values > 30).mean()),
        "asym_frac_60": float((values > 60).mean()),
        # normalised so it does not simply track overall image brightness
        "asym_ratio": float(values.mean() / max(image[both > 0].mean(), 1e-6))
        if (both > 0).any() else 0.0,
    }

    # Largest CONNECTED asymmetric region. Diffuse noise produces many scattered
    # differences; a tumor produces one big coherent blob. That distinction is exactly what
    # separates a real lesion from acquisition noise, so it deserves its own feature.
    strong = ((diff > 40) & (both > 0)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, k, iterations=1)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
    largest = float(stats[1:, cv2.CC_STAT_AREA].max()) if n_labels > 1 else 0.0
    brain_area = float((both > 0).sum())
    feats["asym_blob_area"] = largest
    feats["asym_blob_ratio"] = largest / max(brain_area, 1.0)
    return feats


# --------------------------------------------------------------------------------------
# Family 6 - regional statistics
# --------------------------------------------------------------------------------------
def regional_features(image: np.ndarray, mask: np.ndarray | None = None,
                      grid: int = 3) -> dict:
    """Mean and standard deviation of intensity in each cell of a 3x3 grid over the brain.

    Global statistics say how bright the brain is overall; they cannot say that ONE corner
    of it is unusually bright. Splitting the brain's bounding box into a grid gives the
    classifier a coarse spatial map, so a localised hyperintensity becomes visible as a
    single anomalous cell rather than being averaged away across the whole organ.

    We also record the spread ACROSS cells: a healthy brain has fairly uniform regional
    means, while a lesion makes one cell stand out, which raises the spread.
    """
    if mask is None or not mask.any():
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    feats: dict = {}
    cell_means: list[float] = []
    for gy in range(grid):
        for gx in range(grid):
            ya = y0 + (y1 - y0) * gy // grid
            yb = y0 + (y1 - y0) * (gy + 1) // grid
            xa = x0 + (x1 - x0) * gx // grid
            xb = x0 + (x1 - x0) * (gx + 1) // grid

            cell = image[ya:yb, xa:xb]
            cell_mask = mask[ya:yb, xa:xb]
            px = cell[cell_mask > 0].astype(np.float64)
            m = float(px.mean()) if px.size else 0.0
            sd = float(px.std()) if px.size else 0.0
            feats[f"reg_{gy}{gx}_mean"] = m
            feats[f"reg_{gy}{gx}_std"] = sd
            cell_means.append(m)

    arr = np.array(cell_means, dtype=np.float64)
    feats["reg_spread"] = float(arr.std())
    feats["reg_max_over_mean"] = float(arr.max() / max(arr.mean(), 1e-6))
    return feats


# --------------------------------------------------------------------------------------
# Family 7 - anatomical location of the lesion
# --------------------------------------------------------------------------------------
# WHY THIS FAMILY EXISTS. Families 1-6 answer "what does the tissue look like". They are
# enough to answer "is there a tumor", and our binary model reaches 97.7% with them alone.
# They are NOT enough to answer "which KIND of tumor", and our first multi-class attempt
# proved it: 72.1% accuracy with glioma recall at 24%.
#
# The reason is that a radiologist does not name a tumor type primarily from texture. They
# name it from WHERE IT SITS, because the three types have different origins:
#
#   * Pituitary adenoma - grows from the pituitary gland, which sits on the midline at the
#     skull base. It is therefore central and inferior, essentially never out at the rim.
#   * Meningioma        - grows from the meninges, the membranes lining the inside of the
#     skull. It is EXTRA-AXIAL: it pushes on the brain from outside, so it hugs the border.
#   * Glioma            - grows from glial cells inside the brain itself. It is INTRA-AXIAL:
#     embedded in white matter, away from both the midline and the rim.
#
# Two of the three defining cues here are pure geometry - distance to the midline and
# distance to the brain boundary - and neither is recoverable from a global texture
# histogram, which is exactly why the texture-only model could not separate the classes.
# This family measures those cues directly.
def location_features(image: np.ndarray, seg: dict | None = None) -> dict:
    """Where the detected region sits inside the brain, in scale-free coordinates.

    Everything is normalised by the brain's own bounding box and radius, never by the image
    size. That matters because the 4-class dataset mixes image resolutions (its training
    gliomas are all 512x512 while its test gliomas come in 64 different sizes), so any
    feature measured in raw pixels would encode the scanner rather than the anatomy.
    """
    h, w = image.shape[:2]
    mask = seg.get("brain_mask") if seg else None
    if mask is None or not mask.any():
        mask = np.ones((h, w), dtype=np.uint8) * 255

    ys, xs = np.where(mask > 0)
    by0, by1 = float(ys.min()), float(ys.max() + 1)
    bx0, bx1 = float(xs.min()), float(xs.max() + 1)
    bw, bh = max(bx1 - bx0, 1.0), max(by1 - by0, 1.0)
    brain_cx, brain_cy = float(xs.mean()), float(ys.mean())
    brain_area = float((mask > 0).sum())
    # radius of the circle with the same area - a scale reference that does not depend on
    # the bounding box being axis-aligned with the head
    brain_radius = float(np.sqrt(brain_area / np.pi)) or 1.0

    # Distance from every brain pixel to the nearest non-brain pixel. Used to ask how deep
    # inside the parenchyma a point lies - the intra-axial vs extra-axial question.
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    max_depth = float(dist.max()) or 1.0

    # ---- shape of the head itself. This is a proxy for the ACQUISITION PLANE, and the
    # plane changes what a location means: a pituitary tumor looks central on an axial
    # slice but sits low and forward on a sagittal one. The 4-class dataset mixes axial,
    # sagittal and coronal slices, so without this the model would be reading one
    # coordinate system as if it were another.
    brain_contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if brain_contours:
        bc = max(brain_contours, key=cv2.contourArea)
        b_per = float(cv2.arcLength(bc, True))
        b_area = float(cv2.contourArea(bc))
        b_hull = cv2.convexHull(bc)
        b_hull_area = float(cv2.contourArea(b_hull)) or 1.0
        plane_circ = float(4 * np.pi * b_area / (b_per ** 2)) if b_per > 1e-6 else 0.0
        plane_solidity = float(b_area / b_hull_area)
    else:
        plane_circ, plane_solidity = 0.0, 0.0

    feats: dict = {
        "plane_brain_aspect": float(bw / bh),
        "plane_brain_fill": float(brain_area / max(bw * bh, 1.0)),
        "plane_brain_coverage": float(brain_area / float(h * w)),
        "plane_brain_circularity": plane_circ,
        "plane_brain_solidity": plane_solidity,
        # how far the head's centre of mass sits from the frame centre, in brain radii
        "plane_brain_offset_x": float((brain_cx - w / 2.0) / brain_radius),
        "plane_brain_offset_y": float((brain_cy - h / 2.0) / brain_radius),
    }

    if not seg or not seg.get("tumor_found") or seg.get("properties") is None:
        # No region was located. Zeroing every location feature is right: there is no
        # lesion to describe. `loc_found` tells the model to ignore the rest of the block
        # rather than reading the zeros as "a lesion at the exact brain centre".
        feats.update({
            "loc_found": 0.0,
            "loc_x_norm": 0.0, "loc_y_norm": 0.0,
            "loc_offset_x": 0.0, "loc_offset_y": 0.0,
            "loc_midline_abs": 0.0, "loc_radial": 0.0,
            "loc_angle_sin": 0.0, "loc_angle_cos": 0.0,
            "loc_depth_centroid": 0.0, "loc_depth_min": 0.0, "loc_depth_max": 0.0,
            "loc_border_contact": 0.0, "loc_rim_fraction": 0.0,
            "loc_core_fraction": 0.0, "loc_span_norm": 0.0,
            "loc_band_sup": 0.0, "loc_band_mid": 0.0, "loc_band_inf": 0.0,
            "loc_side_left": 0.0, "loc_side_centre": 0.0, "loc_side_right": 0.0,
            "loc_crosses_midline": 0.0,
        })
        return feats

    tmask = seg.get("tumor_mask")
    p = seg["properties"]
    cx, cy = (float(p["centroid"][0]), float(p["centroid"][1])) if "centroid" in p else (brain_cx, brain_cy)

    # ---- position of the lesion centroid, in three complementary coordinate systems
    dx = (cx - brain_cx) / brain_radius          # signed, in brain radii
    dy = (cy - brain_cy) / brain_radius
    radial = float(np.sqrt(dx * dx + dy * dy))   # 0 = dead centre, ~1 = at the rim
    angle = float(np.arctan2(dy, dx))

    feats.update({
        "loc_found": 1.0,
        # position within the brain's bounding box, 0..1 on each axis
        "loc_x_norm": float((cx - bx0) / bw),
        "loc_y_norm": float((cy - by0) / bh),
        "loc_offset_x": float(dx),
        "loc_offset_y": float(dy),
        # THE PITUITARY CUE. Distance from the mid-sagittal plane, unsigned because left
        # and right are equivalent here. A pituitary adenoma sits on the midline, so this
        # is near zero; a glioma in one hemisphere pushes it up.
        "loc_midline_abs": float(abs(dx)),
        # THE MENINGIOMA/GLIOMA AXIS. 0 = centre of the brain, 1 = out at the border.
        "loc_radial": radial,
        # angle as sin/cos rather than radians - otherwise a lesion at +179 deg and one at
        # -179 deg, which are neighbours, would look maximally far apart to the model
        "loc_angle_sin": float(np.sin(angle)),
        "loc_angle_cos": float(np.cos(angle)),
    })

    if tmask is not None and (tmask > 0).any():
        tm = tmask > 0
        depths = dist[tm]
        # depth = how far inside the parenchyma, as a fraction of the deepest point in this
        # brain. Extra-axial meningioma -> small. Intra-axial glioma -> large.
        feats["loc_depth_centroid"] = float(dist[int(np.clip(cy, 0, h - 1)),
                                                 int(np.clip(cx, 0, w - 1))] / max_depth)
        feats["loc_depth_min"] = float(depths.min() / max_depth)
        feats["loc_depth_max"] = float(depths.max() / max_depth)
        # fraction of the lesion lying within one "rim thickness" of the brain edge - the
        # direct measurement of "is this thing stuck to the inside of the skull"
        rim = max(max_depth * 0.15, 2.0)
        feats["loc_border_contact"] = float((depths <= rim).mean())
        feats["loc_rim_fraction"] = float((depths <= max_depth * 0.25).mean())
        feats["loc_core_fraction"] = float((depths >= max_depth * 0.60).mean())

        tys, txs = np.where(tm)
        feats["loc_span_norm"] = float(
            max(txs.max() - txs.min(), tys.max() - tys.min()) / max(bw, bh)
        )

        # ---- coarse anatomical bands. Which third of the head (superior / middle /
        # inferior) holds the lesion, and which side. Reported as mass fractions rather
        # than a single hard label so a lesion straddling a boundary is described honestly.
        ty_rel = (tys - by0) / bh
        feats["loc_band_sup"] = float((ty_rel < 1 / 3).mean())
        feats["loc_band_mid"] = float(((ty_rel >= 1 / 3) & (ty_rel < 2 / 3)).mean())
        feats["loc_band_inf"] = float((ty_rel >= 2 / 3).mean())

        tx_off = (txs - brain_cx) / brain_radius
        feats["loc_side_left"] = float((tx_off < -0.15).mean())
        feats["loc_side_centre"] = float((np.abs(tx_off) <= 0.15).mean())
        feats["loc_side_right"] = float((tx_off > 0.15).mean())
        # a mass spanning both hemispheres is a different animal from one confined to one
        feats["loc_crosses_midline"] = float(
            (tx_off.min() < -0.05) and (tx_off.max() > 0.05)
        )
    else:
        feats.update({
            "loc_depth_centroid": 0.0, "loc_depth_min": 0.0, "loc_depth_max": 0.0,
            "loc_border_contact": 0.0, "loc_rim_fraction": 0.0,
            "loc_core_fraction": 0.0, "loc_span_norm": 0.0,
            "loc_band_sup": 0.0, "loc_band_mid": 0.0, "loc_band_inf": 0.0,
            "loc_side_left": 0.0, "loc_side_centre": 0.0, "loc_side_right": 0.0,
            "loc_crosses_midline": 0.0,
        })
    return feats


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def extract_features(enhanced: np.ndarray, seg: dict | None = None) -> dict:
    """Build the complete named feature dictionary for one enhanced MRI slice."""
    if seg is None:
        seg = segment_tumor(enhanced)

    mask = seg.get("brain_mask")
    feats: dict = {}
    feats.update(intensity_features(enhanced, mask))
    feats.update(glcm_features(enhanced))
    feats.update(lbp_features(enhanced, mask))
    feats.update(shape_features(seg))
    feats.update(asymmetry_features(enhanced, mask))
    feats.update(regional_features(enhanced, mask))
    feats.update(location_features(enhanced, seg))
    return feats


def feature_vector(enhanced: np.ndarray, seg: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """Same as extract_features but returned as (values, names) with a deterministic order."""
    feats = extract_features(enhanced, seg)
    names = sorted(feats.keys())
    values = np.array([feats[n] for n in names], dtype=np.float64)
    return values, names
