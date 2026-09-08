# Medical Image Enhancement and Diagnostic Assistance System
## Brain Tumor Detection from MRI using OpenCV and Machine Learning

A hybrid computer-vision + machine-learning system that enhances brain MRI scans, locates
suspicious regions, and classifies them as **Tumor** or **No Tumor** with a confidence score
and a full visual explanation of how it reached that conclusion.

> **Assistive tool only.** This is a decision-support and triage system for radiologists.
> It is not a medical device, is not regulatory-approved, and must never be the sole basis
> for a clinical decision.

---

## Quick start

```bash
./run.sh check      # verify everything works        (~5 s)
./run.sh train      # train + evaluate the model     (~15 s)
./run.sh demo       # generate all figures           (~10 s)
./run.sh app        # launch the web interface
```

The first run creates a virtual environment and installs dependencies automatically.
No dataset download is required — the project generates its own realistic MRI data.

---

## Results

| metric | value |
|---|---|
| Accuracy | **98.00%** |
| Sensitivity (recall) | **96.00%** |
| Specificity | 100.00% |
| Precision | 100.00% |
| F1 score | 97.96% |
| ROC-AUC | 1.0000 |
| Training time | 15 s on CPU |

Selected model: **Logistic Regression**, chosen automatically by 5-fold cross-validated
ROC-AUC from four candidates. Full breakdown in [docs/05_RESULTS.md](docs/05_RESULTS.md).

---

## Pipeline

```
   [1] MRI IMAGE IN
        |
        v
   [2] PREPROCESSING          grayscale -> resize 256x256 -> median + bilateral -> normalise
        |                     "make every image speak the same language"
        v
   [3] ENHANCEMENT            CLAHE + unsharp masking
        |                     "make the invisible visible"
        v
   [4] SEGMENTATION           skull strip -> statistical threshold -> morphology -> contours
        |                     "find and isolate the suspicious blob"
        v
   [5] FEATURE EXTRACTION     intensity + GLCM texture + LBP + shape  = 111 numbers
        |                     "turn a picture into a row of numbers"
        v
   [6] CLASSIFICATION         LogReg / SVM / RandomForest / MLP -> best by CV ROC-AUC
        |
        v
   [7] VISUALISATION          8-panel figure: every stage + the verdict card
```

### Why hybrid, and not pure deep learning?

A CNN outputs a label a doctor cannot audit. Our OpenCV stages produce a **visible,
measurable region** — area, circularity, solidity, hyperintensity ratio — so the reasoning
is inspectable. Classical feature engineering also works on hundreds of images rather than
the thousands a CNN needs. Stages 1–5 are model-agnostic, so a CNN can be dropped into
Stage 6 later without touching anything else.

---

## Commands

| command | what it does |
|---|---|
| `./run.sh check` | run the 19-test suite (pre-flight check before a demo) |
| `./run.sh train` | full training run: generate data, extract features, compare 4 models, evaluate |
| `./run.sh demo` | produce every presentation figure into `outputs/` |
| `./run.sh predict <image>` | analyse a single MRI and save the 8-panel figure |
| `./run.sh generate` | (re)generate the synthetic dataset |
| `./run.sh metrics` | reprint the last training report |
| `./run.sh app` | launch the Streamlit web interface |

Equivalent direct form: `python main.py <command>` (see `python main.py --help`).

---

## Using a real dataset

The pipeline is dataset-agnostic. Drop any brain MRI dataset into:

```
data/raw/yes/     # scans WITH a tumor
data/raw/no/      # scans WITHOUT a tumor
```

then run `./run.sh train`. The loader auto-detects it (it also accepts folder names
`tumor`, `notumor`, `positive`, `negative`, `healthy`, `1`, `0`) and uses it instead of
synthetic data. **No code changes are needed.**

Force synthetic data even when a real dataset is present with `./run.sh train --synthetic`.

---

## The synthetic MRI simulator

Rather than requiring a dataset download, the project ships a physically-motivated MRI
phantom generator (`src/braintumor/dataset.py`). It builds anatomy in layers — skull, CSF,
grey matter, white matter, lateral ventricles, cortical folding — then optionally injects a
tumor with an irregular Fourier-perturbed boundary, peritumoral edema halo, and sometimes a
necrotic core. It then applies realistic acquisition artefacts:

- **B1 bias field** — the smooth brightness gradient real coils produce
- **Rician noise** — the mathematically correct model for an MRI *magnitude* image, since
  the magnitude of complex Gaussian noise is Rician, not Gaussian

This makes the project reproducible on any machine with zero downloads, and it means the
enhancement stage is correcting genuine simulated artefacts rather than doing cosmetic work.

---

## Project layout

```
dsp_project/
├── main.py                     entry point  (python main.py --help)
├── run.sh                      command shortcuts
├── requirements.txt
├── SYNOPSIS.md                 the project synopsis
│
├── docs/
│   ├── 01_WHAT_IS_THIS_PROJECT.md    <- READ THIS FIRST (speaking script)
│   ├── 02_VIVA_QUESTIONS_ANSWERS.md  <- 21 likely questions with answers
│   ├── 03_DEMO_SCRIPT_30MIN.md       <- minute-by-minute presentation plan
│   ├── 04_TECHNICAL_DEEP_DIVE.md     <- the maths behind every stage
│   └── 05_RESULTS.md                 <- measured results (auto-generated)
│
├── src/braintumor/
│   ├── config.py               every tunable constant in one place
│   ├── preprocess.py           STAGE 2  grayscale, resize, denoise, normalise
│   ├── enhance.py              STAGE 3  CLAHE, unsharp mask, gamma, metrics
│   ├── segment.py              STAGE 4  skull strip, threshold, morphology, contours
│   ├── features.py             STAGE 5  intensity + GLCM + LBP + shape  (111 features)
│   ├── dataset.py              STAGE 1  MRI simulator + real-dataset loader
│   ├── model.py                STAGE 6  the four candidate classifiers
│   ├── train.py                training, model selection, evaluation
│   ├── predict.py              single-image inference
│   ├── visualize.py            STAGE 7  all figures
│   └── cli.py                  command line interface
│
├── app/streamlit_app.py        web GUI
├── tests/test_pipeline.py      19 tests, one group per stage
├── data/                       raw/ (real datasets) and synthetic/ (generated)
├── models/                     trained model bundle
└── outputs/                    figures and metrics.json
```

---

## Generated figures

After `./run.sh demo`, `outputs/` contains:

| file | contents |
|---|---|
| `demo_tumor_pipeline.png` | the headline 8-panel figure for a tumor case |
| `demo_healthy_pipeline.png` | the same for a healthy case |
| `demo_enhancement.png` | Input vs global HE vs CLAHE vs CLAHE+unsharp, with histograms |
| `demo_samples.png` | a grid of several scans through the pipeline |
| `training_report.png` | model comparison, confusion matrix, metrics, feature importances |
| `metrics.json` | the full machine-readable report |

---

## Two real bugs, and how they were fixed

Documented in code comments and covered by regression tests — see
[docs/04_TECHNICAL_DEEP_DIVE.md](docs/04_TECHNICAL_DEEP_DIVE.md).

1. **The flood-fill split-background bug.** The head ellipse often touches the top and
   bottom image borders, which splits the background into two disconnected regions. A
   corner flood-fill reached only one, so the other half of the frame was misclassified as
   an interior hole and filled in — the brain mask became a crescent covering half the
   image. Fixed by padding a 1-pixel frame before flood filling.

2. **Otsu flags every healthy brain.** Otsu's method always splits a histogram into two
   classes, even when nothing is abnormal, so on a healthy brain it returned grey-vs-white
   matter as "lesion". Measured: **100% of healthy scans falsely flagged.** Replaced with a
   tissue-relative statistical threshold (`mean + 1.5σ` over pixels above the brain median,
   which excludes dark CSF): **100% of tumors detected, 3% false flags.**

---

## Limitations

- Performance depends on dataset quality and diversity; models degrade across scanners
  (domain shift).
- Binary classification only — tumor present/absent, not tumor *type*.
- 2D slice analysis, not 3D volumetric.
- Intensity-based segmentation can be misled by calcifications or imaging artefacts.
- Results reported here are on synthetic data and demonstrate pipeline correctness, not
  clinical validity. Clinical use would require real multi-centre validation and regulatory
  approval.

## Future scope

Multi-class tumor typing (glioma / meningioma / pituitary) · U-Net pixel-level segmentation
with Dice scoring · transfer learning with a pretrained CNN backbone · Grad-CAM heatmaps ·
3D tumor volume estimation across a full MRI stack · automated PDF reporting · extension to
CT and other modalities.

---

## Requirements

Python 3.9+. Dependencies in `requirements.txt` (OpenCV, scikit-learn, scikit-image, NumPy,
Matplotlib, Streamlit). No GPU required.
