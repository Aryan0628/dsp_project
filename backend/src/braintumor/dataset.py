"""
STAGE 1 - IMAGE ACQUISITION + DATASET HANDLING
==============================================
The project runs in two interchangeable modes:

  REAL MODE       data/raw/yes/*  and  data/raw/no/*   <- drop a Kaggle dataset here
  SYNTHETIC MODE  generated on the fly by the phantom simulator below

The synthetic generator exists so the whole project is reproducible on any machine with
zero downloads and no licensing problems. It is not a toy: it models the actual physics of
an MRI magnitude image, including Rician noise and B1 bias-field inhomogeneity.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from . import config
from .enhance import enhance
from .features import feature_vector
from .preprocess import load_image, preprocess
from .segment import segment_tumor


# ======================================================================================
# SYNTHETIC MRI PHANTOM SIMULATOR
# ======================================================================================
def _ellipse_mask(shape, centre, axes, angle=0.0) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(m, tuple(int(v) for v in centre), tuple(int(v) for v in axes),
                float(angle), 0, 360, 255, -1)
    return m


def _irregular_blob(shape, centre, radius, rng, irregularity=0.35, n_harmonics=5) -> np.ndarray:
    """Draw a closed blob whose radius varies with angle.

        r(theta) = radius * (1 + sum_k  a_k * cos(k*theta + phase_k))

    Summing a few random Fourier harmonics produces a lobed, organic outline rather than a
    circle - which is what real tumors look like, and what makes the circularity and
    solidity shape features meaningful.
    """
    thetas = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    r = np.full_like(thetas, float(radius))
    for k in range(2, 2 + n_harmonics):
        amp = rng.uniform(0, irregularity) / (k - 1)
        phase = rng.uniform(0, 2 * np.pi)
        r += radius * amp * np.cos(k * thetas + phase)
    r = np.clip(r, radius * 0.45, radius * 1.7)

    pts = np.stack([centre[0] + r * np.cos(thetas), centre[1] + r * np.sin(thetas)], axis=1)
    blob = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(blob, [pts.astype(np.int32)], 255)
    return blob


def _bias_field(shape, rng, strength=0.16) -> np.ndarray:
    """Simulate B1 inhomogeneity: a smooth low-frequency multiplicative brightness gradient.

    Real MRI coils do not illuminate the head evenly, so one side of the image is slightly
    brighter. This is exactly the artefact CLAHE is good at compensating for, so including
    it makes the enhancement stage genuinely useful rather than cosmetic.
    """
    small = rng.normal(0, 1, size=(4, 4))
    field = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=shape[1] / 6.0)
    field /= (np.abs(field).max() + 1e-9)
    return 1.0 + strength * field


def _rician_noise(signal: np.ndarray, sigma: float, rng) -> np.ndarray:
    """The correct noise model for an MRI magnitude image.

    Raw MRI data is complex-valued with independent Gaussian noise in the real and
    imaginary channels. The scanner displays the MAGNITUDE, and |complex Gaussian| follows
    a RICIAN distribution:

        observed = sqrt( (S + n_real)^2 + n_imag^2 ),   n ~ N(0, sigma)

    In dark regions this is non-zero-mean and biased upward, which is why MRI backgrounds
    look grainy grey rather than pure black. Plain additive Gaussian noise would be wrong.
    """
    n1 = rng.normal(0, sigma, signal.shape)
    n2 = rng.normal(0, sigma, signal.shape)
    return np.sqrt((signal + n1) ** 2 + n2 ** 2)


def generate_brain_mri(
    tumor: bool,
    rng: np.random.Generator,
    size: int = 256,
    noise_sigma: float | None = None,
) -> np.ndarray:
    """Generate one synthetic axial brain MRI slice, with or without a tumor.

    Tissue layers are built up in anatomical order, each with its own T1-like intensity:
        skull (very bright) > white matter > grey matter > CSF/ventricles (dark)
    """
    shape = (size, size)
    noise_sigma = config.SYNTH_NOISE_SIGMA if noise_sigma is None else noise_sigma
    img = np.zeros(shape, dtype=np.float64)

    # ---- global anatomy variability so no two subjects are identical
    cx = size / 2 + rng.uniform(-6, 6)
    cy = size / 2 + rng.uniform(-6, 6)
    head_rx = size * rng.uniform(0.36, 0.42)
    head_ry = size * rng.uniform(0.44, 0.49)
    tilt = rng.uniform(-12, 12)

    # ---- skull: bright outer ring (cortical bone + scalp fat)
    outer = _ellipse_mask(shape, (cx, cy), (head_rx, head_ry), tilt)
    inner_skull = _ellipse_mask(shape, (cx, cy), (head_rx * 0.92, head_ry * 0.92), tilt)
    img[outer > 0] = rng.uniform(195, 235)

    # ---- CSF gap between skull and brain (dark)
    img[inner_skull > 0] = rng.uniform(28, 45)

    # ---- brain parenchyma: grey matter shell
    brain = _ellipse_mask(shape, (cx, cy), (head_rx * 0.85, head_ry * 0.85), tilt)
    grey_level = rng.uniform(105, 128)
    img[brain > 0] = grey_level

    # ---- white matter core (brighter than grey matter on T1)
    wm = _ellipse_mask(shape, (cx, cy), (head_rx * 0.66, head_ry * 0.68), tilt)
    white_level = grey_level + rng.uniform(24, 38)
    img[wm > 0] = white_level

    # ---- cortical folding (gyri / sulci): dark sinusoidal grooves near the cortex
    yy, xx = np.mgrid[0:size, 0:size]
    ang = np.arctan2(yy - cy, xx - cx)
    rad = np.sqrt(((xx - cx) / head_rx) ** 2 + ((yy - cy) / head_ry) ** 2)
    folds = np.sin(ang * rng.integers(9, 15) + rng.uniform(0, 6.28))
    cortex_band = (rad > 0.62) & (rad < 0.86)
    img[cortex_band] -= 16 * np.clip(folds[cortex_band], 0, None)

    # ---- lateral ventricles: two dark butterfly-shaped CSF cavities
    v_off = size * rng.uniform(0.055, 0.085)
    v_rx, v_ry = size * rng.uniform(0.030, 0.045), size * rng.uniform(0.075, 0.105)
    for sign in (-1, 1):
        v = _ellipse_mask(shape, (cx + sign * v_off, cy - size * 0.02),
                          (v_rx, v_ry), tilt + sign * 16)
        img[v > 0] = rng.uniform(22, 38)

    # ---- fine tissue texture so the region is not flat (real tissue is granular)
    texture = cv2.GaussianBlur(rng.normal(0, 1, shape), (0, 0), sigmaX=1.4)
    img += 7.0 * texture * (brain > 0)

    # ---- THE TUMOR ------------------------------------------------------------------
    if tumor:
        # place it inside the brain but off-centre, avoiding the ventricles
        for _ in range(40):
            t_ang = rng.uniform(0, 2 * np.pi)
            t_dist = rng.uniform(0.22, 0.55)
            tx = cx + np.cos(t_ang) * head_rx * t_dist
            ty = cy + np.sin(t_ang) * head_ry * t_dist
            if brain[int(np.clip(ty, 0, size - 1)), int(np.clip(tx, 0, size - 1))] > 0:
                break

        t_radius = size * rng.uniform(0.045, 0.095)
        blob = _irregular_blob(shape, (tx, ty), t_radius, rng,
                               irregularity=rng.uniform(0.25, 0.45))
        blob = cv2.bitwise_and(blob, brain)   # a tumor cannot extend outside the brain

        # peritumoral edema: a diffuse, slightly bright halo surrounding the core
        halo = cv2.GaussianBlur(blob.astype(np.float64), (0, 0), sigmaX=t_radius * 0.55)
        halo /= (halo.max() + 1e-9)
        img += halo * rng.uniform(18, 30) * (brain > 0)

        # tumor core: hyperintense relative to white matter, with internal heterogeneity
        core_level = white_level + rng.uniform(35, 62)
        core_texture = cv2.GaussianBlur(rng.normal(0, 1, shape), (0, 0), sigmaX=2.0)
        img[blob > 0] = core_level + 11 * core_texture[blob > 0]

        # necrotic centre in some larger tumors - a darker core inside the bright rim
        if rng.random() < 0.35 and t_radius > size * 0.06:
            nec = _irregular_blob(shape, (tx, ty), t_radius * 0.42, rng, irregularity=0.30)
            nec = cv2.bitwise_and(nec, blob)
            img[nec > 0] = core_level - rng.uniform(45, 70)

    # ---- acquisition artefacts -------------------------------------------------------
    img *= _bias_field(shape, rng)                       # coil inhomogeneity
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=rng.uniform(0.6, 1.1))  # partial volume / PSF
    img = _rician_noise(img, noise_sigma, rng)           # MRI magnitude noise

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_synthetic_dataset(
    n_total: int | None = None,
    out_dir: Path | None = None,
    seed: int = config.RANDOM_SEED,
    clean: bool = True,
) -> tuple[list[Path], list[int]]:
    """Create a balanced synthetic dataset on disk: half tumor, half healthy."""
    n_total = n_total or config.SYNTH_DEFAULT_COUNT
    out_dir = Path(out_dir or config.SYNTHETIC_DIR)

    yes_dir, no_dir = out_dir / "yes", out_dir / "no"
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    yes_dir.mkdir(parents=True, exist_ok=True)
    no_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_tumor = n_total // 2
    n_healthy = n_total - n_tumor

    paths: list[Path] = []
    labels: list[int] = []

    for i in range(n_tumor):
        img = generate_brain_mri(True, rng)
        p = yes_dir / f"tumor_{i:04d}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
        labels.append(1)

    for i in range(n_healthy):
        img = generate_brain_mri(False, rng)
        p = no_dir / f"healthy_{i:04d}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
        labels.append(0)

    return paths, labels


# ======================================================================================
# DATASET LOADING
# ======================================================================================
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Folder-name aliases so common public dataset layouts work without renaming anything.
POSITIVE_NAMES = {"yes", "tumor", "tumour", "positive", "1", "brain_tumor"}
NEGATIVE_NAMES = {"no", "notumor", "no_tumor", "healthy", "negative", "0", "none"}

# Tumor-type folders, mapped to their index in config.MULTICLASS_NAMES.
TUMOR_TYPES = {
    "glioma": 1, "glioma_tumor": 1,
    "meningioma": 2, "meningioma_tumor": 2,
    "pituitary": 3, "pituitary_tumor": 3,
}


def _normalise(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _class_of(folder_name: str, multiclass: bool) -> int | None:
    """Map a class folder name to a label, or None if the folder is not a class folder."""
    name = _normalise(folder_name)
    if name in NEGATIVE_NAMES:
        return 0
    if name in TUMOR_TYPES:
        return TUMOR_TYPES[name] if multiclass else 1
    if name in POSITIVE_NAMES:
        if multiclass:
            return None          # a generic "yes" folder carries no type information
        return 1
    return None


def _images_in(folder: Path) -> list[Path]:
    return [f for f in sorted(folder.rglob("*")) if f.suffix.lower() in IMAGE_EXTS]


def load_split_dataset(root: Path | str, multiclass: bool = False) -> tuple[list, list, list, list]:
    """Load a dataset that ships its OWN train/ and test/ directories.

    Respecting a provided split matters more than it looks. These datasets contain several
    consecutive slices from the same patient, so a random split puts near-duplicate images
    of the same person on both sides and the reported score measures memorisation rather
    than generalisation. The curators separated patients between train and test, so using
    their split is the only way to get an honest number.

    Returns (train_paths, train_labels, test_paths, test_labels).
    """
    root = Path(root)
    out: dict[str, tuple[list, list]] = {"train": ([], []), "test": ([], [])}

    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        paths, labels = out[split]
        for sub in sorted(split_dir.iterdir()):
            if not sub.is_dir():
                continue
            label = _class_of(sub.name, multiclass)
            if label is None:
                continue
            for f in _images_in(sub):
                paths.append(f)
                labels.append(label)

    return out["train"][0], out["train"][1], out["test"][0], out["test"][1]


def load_dataset(root: Path | str, multiclass: bool = False) -> tuple[list[Path], list[int]]:
    """Scan `root` for class sub-folders and return (paths, labels).

    Label 1 = tumor, 0 = no tumor.
    """
    root = Path(root)
    if not root.exists():
        return [], []

    paths: list[Path] = []
    labels: list[int] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        label = _class_of(sub.name, multiclass)
        if label is None:
            continue
        for f in _images_in(sub):
            paths.append(f)
            labels.append(label)
    return paths, labels


def resolve_dataset(
    prefer_real: bool = True,
    n_synthetic: int | None = None,
    seed: int = config.RANDOM_SEED,
) -> tuple[list[Path], list[int], str]:
    """Use the real dataset if the user supplied one, otherwise generate synthetic data."""
    if prefer_real:
        paths, labels = load_dataset(config.RAW_DIR)
        if len(paths) >= 20 and len(set(labels)) == 2:
            return paths, labels, "real"

    paths, labels = load_dataset(config.SYNTHETIC_DIR)
    wanted = n_synthetic or config.SYNTH_DEFAULT_COUNT
    if len(paths) != wanted or len(set(labels)) != 2:
        paths, labels = generate_synthetic_dataset(wanted, seed=seed)
    return paths, labels, "synthetic"


# ======================================================================================
# FEATURE MATRIX CONSTRUCTION - runs Stages 2..5 over the whole dataset
# ======================================================================================
def process_single(path: Path | str) -> tuple[np.ndarray, list[str]]:
    """Stages 2 -> 3 -> 4 -> 5 for one image on disk."""
    raw = load_image(path)
    pre = preprocess(raw)
    enh = enhance(pre)
    seg = segment_tumor(enh)
    return feature_vector(enh, seg)


def process_single_with_image(path: Path | str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Same as `process_single`, but also returns the enhanced image itself.

    The deep tumor-typing branch needs the picture, the classical branch needs the numbers,
    and both want them derived from the SAME preprocessing. Computing them together means
    one pass over the dataset instead of two - segmentation is the expensive stage, and on
    3264 images running it twice costs minutes for nothing.
    """
    raw = load_image(path)
    pre = preprocess(raw)
    enh = enhance(pre)
    seg = segment_tumor(enh)
    values, names = feature_vector(enh, seg)
    return values, names, enh


def build_feature_matrix(
    paths: list[Path],
    labels: list[int],
    verbose: bool = True,
    with_images: bool = False,
):
    """Run the pipeline over every image and stack the results into X (n_samples, n_feats).

    With `with_images=True` the return grows to (X, y, names, images), where `images` is a
    uint8 array of the enhanced slices aligned row-for-row with X. Images that fail to
    process are dropped from BOTH, so the alignment holds.
    """
    rows: list[np.ndarray] = []
    kept: list[int] = []
    imgs: list[np.ndarray] = []
    names: list[str] = []

    total = len(paths)
    for i, (p, y) in enumerate(zip(paths, labels), start=1):
        try:
            if with_images:
                vec, names, enh = process_single_with_image(p)
                imgs.append(enh)
            else:
                vec, names = process_single(p)
        except Exception as exc:                     # a corrupt file must not kill the run
            if verbose:
                print(f"  ! skipping {p.name}: {exc}")
            continue
        rows.append(vec)
        kept.append(y)
        if verbose and (i % 50 == 0 or i == total):
            print(f"  processed {i}/{total} images")

    if not rows:
        raise RuntimeError("No images could be processed - check the dataset folder.")

    X = np.vstack(rows)
    y = np.array(kept, dtype=int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if with_images:
        return X, y, names, np.stack(imgs).astype(np.uint8)
    return X, y, names
