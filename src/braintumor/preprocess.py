"""
STAGE 2 - PREPROCESSING
=======================
Goal: make every incoming MRI "speak the same language" before we analyse it.

Different scanners produce different sizes, bit depths and brightness levels. If we do not
standardise, the classifier learns the scanner rather than the disease.

Steps: grayscale -> resize -> median filter -> bilateral filter -> normalise.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """MRI encodes tissue density, not colour, so one channel carries all the information."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def autocrop_border(image: np.ndarray, tol: float = 6.0, max_passes: int = 4) -> np.ndarray:
    """Trim uniform padding from the edges of a real-world MRI image.

    Real datasets scraped from papers, textbooks and PACS exports frequently carry a solid
    border: a white page frame around a scanned figure, or thick black letterboxing. Both
    break skull stripping badly. A white frame is the worst case - Otsu selects the frame as
    the brightest connected component, the frame is a closed ring, and the hole-filling step
    then floods its entire interior, so the "brain mask" becomes the whole image. That was
    happening on 3% of the real dataset before this step existed.

    A border row or column is uniform padding when its standard deviation is below `tol`;
    real anatomy always varies. We trim rows and columns alternately for a few passes,
    because removing a white top and bottom edge is what exposes the left and right edges as
    uniform in turn.
    """
    out = image
    for _ in range(max_passes):
        h, w = out.shape[:2]
        if h < 32 or w < 32:
            break

        rows = out.std(axis=1) > tol
        cols = out.std(axis=0) > tol
        if not rows.any() or not cols.any():
            break                                  # image is entirely uniform - leave it

        y0, y1 = int(np.argmax(rows)), h - int(np.argmax(rows[::-1]))
        x0, x1 = int(np.argmax(cols)), w - int(np.argmax(cols[::-1]))
        if (y0, y1, x0, x1) == (0, h, 0, w):
            break                                  # nothing left to trim

        cropped = out[y0:y1, x0:x1]
        if cropped.size < 0.1 * out.size:
            break                                  # refuse to throw away most of the image
        out = cropped
    return out


def resize(image: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """INTER_AREA is the correct interpolation when shrinking - it averages, so it does
    not alias fine texture into false structure."""
    size = size or config.IMAGE_SIZE
    interp = cv2.INTER_AREA if image.shape[0] > size[1] else cv2.INTER_CUBIC
    return cv2.resize(image, size, interpolation=interp)


def denoise(image: np.ndarray) -> np.ndarray:
    """Two-stage denoising.

    1. Median filter: replaces each pixel with the median of its neighbourhood. Unlike a
       mean filter it is not dragged by extreme values, so it erases salt-and-pepper
       speckle without smearing it around.
    2. Bilateral filter: weights neighbours by spatial distance AND intensity similarity.
       Pixels on the far side of a strong edge get almost zero weight, so flat regions are
       smoothed while the tumor boundary stays sharp. This is why we do not use a plain
       Gaussian blur here.
    """
    out = cv2.medianBlur(image, config.MEDIAN_KERNEL)
    out = cv2.bilateralFilter(
        out,
        d=config.BILATERAL_D,
        sigmaColor=config.BILATERAL_SIGMA_COLOR,
        sigmaSpace=config.BILATERAL_SIGMA_SPACE,
    )
    return out


def normalize(image: np.ndarray) -> np.ndarray:
    """Min-max stretch to the full 0-255 range so brightness is comparable across scans."""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess(image: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """Run the full Stage-2 chain. Returns a uint8 grayscale image of standard size."""
    out = to_grayscale(image)
    if out.dtype != np.uint8:
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = autocrop_border(out)          # strip scanned page frames / letterboxing
    out = resize(out, size)
    out = denoise(out)
    out = normalize(out)
    return out


def load_image(path) -> np.ndarray:
    """Read an image from disk as grayscale, raising a clear error if the path is bad."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image
