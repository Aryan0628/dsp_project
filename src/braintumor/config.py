"""Central configuration. Every tunable constant lives here, not scattered in the code."""

from pathlib import Path

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # binary dataset: raw/yes/*.jpg, raw/no/*.jpg
KAGGLE4_DIR = DATA_DIR / "kaggle4"    # 4-class dataset with a provided train/ + test/ split
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"

MODEL_PATH = MODEL_DIR / "brain_tumor_model.joblib"
MULTICLASS_MODEL_PATH = MODEL_DIR / "brain_tumor_multiclass.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

# ------------------------------------------------------------------- reproducibility
RANDOM_SEED = 42

# --------------------------------------------------------------- Stage 2 preprocessing
IMAGE_SIZE = (256, 256)      # every image is resized to this (w, h)
MEDIAN_KERNEL = 3            # salt-and-pepper noise removal
BILATERAL_D = 7              # neighbourhood diameter for edge-preserving smoothing
BILATERAL_SIGMA_COLOR = 50
BILATERAL_SIGMA_SPACE = 50

# ----------------------------------------------------------------- Stage 3 enhancement
CLAHE_CLIP_LIMIT = 2.0       # higher = more contrast but more amplified noise
CLAHE_TILE_GRID = (8, 8)     # image is split into 8x8 tiles, each equalised separately
UNSHARP_SIGMA = 1.5          # gaussian sigma for the blur used in unsharp masking
UNSHARP_AMOUNT = 1.2         # how strongly the detail layer is added back

# --------------------------------------------------------------- Stage 4 segmentation
SKULL_STRIP_FRACTION = 0.22    # keep pixels farther than this fraction of max distance-to-edge
MAX_HOLE_FILL_GROWTH = 3.0     # reject hole-filling that inflates the mask more than this
MIN_BRAIN_COVERAGE = 0.05      # below this the mask collapsed -> use the ellipse fallback
MAX_BRAIN_COVERAGE = 0.88      # above this no background was found -> use the ellipse fallback
MORPH_KERNEL_SIZE = 5
MORPH_OPEN_ITER = 2          # removes small bright specks
MORPH_CLOSE_ITER = 3         # fills holes inside the tumor blob
MIN_TUMOR_AREA = 120         # contours smaller than this (px) are rejected as noise
BOUNDARY_GUARD = 7           # extra erosion (px) applied before candidate detection
MAX_ASPECT_RATIO = 5.0       # reject long thin slivers - a mass is a blob, not a streak
TUMOR_SIGMA_K = 1.5          # suspicious = brighter than mean + k*sigma of normal tissue
ASYM_SIGMA_K = 2.0           # ...or brighter than its mirror counterpart by k*sigma

# ------------------------------------------------------------ Stage 5 feature extraction
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES_DEG = [0, 45, 90, 135]
GLCM_LEVELS = 32             # intensities quantised to 32 bins to keep the matrix small
LBP_RADIUS = 3
LBP_POINTS = 24              # 8 * radius is the usual choice
HIST_BINS = 16

# -------------------------------------------------------------- Stage 6 classification
TEST_SIZE = 0.25
CV_FOLDS = 5
CLASS_NAMES = ("No Tumor", "Tumor")

# Multi-class task. Index order IS the label order, so index 0 must stay "No Tumor" -
# that keeps "label > 0 means tumor" true and lets binary metrics be derived from a
# multi-class prediction without a lookup table.
MULTICLASS_NAMES = ("No Tumor", "Glioma", "Meningioma", "Pituitary")

# Real clinical datasets are imbalanced (the 4-class set is 85% tumor). Weighting classes
# inversely to their frequency stops the model from taking the "always predict tumor"
# shortcut, which would score well on accuracy while being clinically useless.
CLASS_WEIGHT = "balanced"

# Clinical operating point. A missed tumor is far more costly than a false alarm, so we do
# not accept the arbitrary 0.5 probability cut-off. The decision threshold is tuned on
# out-of-fold training predictions to reach at least this sensitivity.
TARGET_SENSITIVITY = 0.95

# False-alarm budget - a CLINICAL POLICY choice, not a fitted parameter. The threshold is
# lowered as far as it can go while out-of-fold specificity stays above this floor, which
# spends spare specificity on catching more tumors. 0.80 means we accept flagging up to one
# in five healthy scans for a second look; for a triage tool whose alternative is a missed
# tumor, that is a reasonable trade. Raise it if the reading department cannot absorb the
# extra reviews. See tune_threshold() for why this objective beats the textbook alternative.
MIN_SPECIFICITY = 0.80

# --------------------------------------------------- Stage 6b deep tumor-typing branch
# The CNN exists only because we measured the engineered features falling short on tumor
# TYPING (72.1% accuracy, 24% glioma recall) while being sufficient for tumor DETECTION.
# See deep.py for the full argument.
DEEP_EPOCHS = 14
DEEP_BATCH_SIZE = 32

# Average the CNN's prediction over 4 label-preserving views (mirror x mild zoom) at
# inference. Costs 4x forward passes, buys robustness to the scale/framing shift between
# this dataset's train and test halves. See DeepTumorClassifier.predict_proba.
DEEP_TTA = True

# Fraction of the TRAINING set held back to select the CNN's best epoch and the ensemble
# weight. It is carved out of train, never out of test, so the held-out test numbers stay
# untouched by any tuning decision.
DEEP_VAL_SIZE = 0.15

# -------------------------------------------------------------------- Stage 7b web API
# The backend serves the Next.js client in `client/`. Port 3000 is Next.js's own default,
# so the frontend is moved off it (see client/package.json) rather than the API - the API
# port is the one baked into deployment scripts and reverse-proxy configs, so it is the one
# worth keeping stable.
API_PORT = 3000
CLIENT_PORT = 5713

# Browsers block cross-origin requests unless the server opts in, and the client runs on a
# different port from the API, so every origin the client may be served from has to be
# listed. We enumerate them explicitly instead of allowing "*": the API accepts file
# uploads, and a wildcard would let any page on the internet drive it through a visitor's
# browser.
CORS_ORIGINS = [
    f"http://localhost:{CLIENT_PORT}",
    f"http://127.0.0.1:{CLIENT_PORT}",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

# Upload ceiling. Decoding an image allocates width x height x channels bytes, so an
# unbounded upload is a trivial way to exhaust the server's memory.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024      # 20 MB
ALLOWED_UPLOAD_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp",
                        "image/tiff", "image/webp"}

# ------------------------------------------------------- synthetic dataset generation
SYNTH_DEFAULT_COUNT = 400    # total images (half tumor, half healthy)
SYNTH_NOISE_SIGMA = 8.0      # Rician noise level

for _d in (DATA_DIR, RAW_DIR, KAGGLE4_DIR, SYNTHETIC_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
