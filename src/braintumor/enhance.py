"""
STAGE 3 - IMAGE ENHANCEMENT
===========================
Goal: make the invisible visible.

A tumor is often only slightly brighter than the tissue around it. Global histogram
equalisation fails here because the very bright skull dominates the histogram. CLAHE fixes
that by equalising locally, and unsharp masking then crisps up the boundaries.

This is the stage that produces the most visually striking before/after for the demo.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Classic global HE. Kept for the side-by-side comparison that shows why we use CLAHE
    instead: HE stretches contrast using ONE mapping for the entire image."""
    return cv2.equalizeHist(image)


def clahe(
    image: np.ndarray,
    clip_limit: float | None = None,
    tile_grid: tuple[int, int] | None = None,
) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation.

    - "Adaptive": the image is split into a grid of tiles (default 8x8) and each tile gets
      its own equalisation mapping, so a dark region brightens even inside a bright image.
    - "Contrast Limited": each tile histogram is clipped at `clip_limit` and the clipped
      mass is redistributed uniformly before equalising. Without this, near-uniform regions
      would have their noise amplified into convincing-looking fake structure.
    - Tile seams are removed by bilinear interpolation between neighbouring tile mappings.
    """
    op = cv2.createCLAHE(
        clipLimit=clip_limit or config.CLAHE_CLIP_LIMIT,
        tileGridSize=tile_grid or config.CLAHE_TILE_GRID,
    )
    return op.apply(image)


def unsharp_mask(
    image: np.ndarray,
    sigma: float | None = None,
    amount: float | None = None,
) -> np.ndarray:
    """Sharpen by amplifying the detail layer.

        blurred = G(image)                  <- low frequencies
        detail  = image - blurred           <- high frequencies (edges, texture)
        result  = image + amount * detail

    Implemented with addWeighted, which is the same expression rearranged:
        result = (1 + amount) * image - amount * blurred
    """
    sigma = sigma if sigma is not None else config.UNSHARP_SIGMA
    amount = amount if amount is not None else config.UNSHARP_AMOUNT
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def gamma_correct(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Non-linear brightness curve. gamma < 1 brightens dark tissue, gamma > 1 darkens it.
    Applied through a 256-entry lookup table, which is far faster than per-pixel maths."""
    if abs(gamma - 1.0) < 1e-6:
        return image
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def enhance(
    image: np.ndarray,
    clip_limit: float | None = None,
    amount: float | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Full Stage-3 chain: CLAHE -> unsharp mask -> optional gamma."""
    out = clahe(image, clip_limit=clip_limit)
    out = unsharp_mask(out, amount=amount)
    out = gamma_correct(out, gamma)
    return out


def enhancement_report(original: np.ndarray, enhanced: np.ndarray) -> dict:
    """Quantify the improvement so we can claim it with numbers, not just "it looks better".

    - contrast_std: standard deviation of intensities. Higher = wider tonal spread.
    - entropy: Shannon entropy of the histogram in bits. Higher = more information content.
    - sharpness: variance of the Laplacian, the standard no-reference focus measure.
    """

    def _entropy(img: np.ndarray) -> float:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
        p = hist / max(hist.sum(), 1e-9)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def _sharpness(img: np.ndarray) -> float:
        return float(cv2.Laplacian(img, cv2.CV_64F).var())

    return {
        "contrast_std_before": float(original.std()),
        "contrast_std_after": float(enhanced.std()),
        "entropy_before": _entropy(original),
        "entropy_after": _entropy(enhanced),
        "sharpness_before": _sharpness(original),
        "sharpness_after": _sharpness(enhanced),
    }
