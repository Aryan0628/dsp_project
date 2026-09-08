# Medical Image Enhancement and Diagnostic Assistance System
## Brain Tumor Detection and Typing from MRI using OpenCV and Machine Learning

A hybrid computer-vision + machine-learning system that enhances brain MRI scans, locates
suspicious regions, and answers two questions — **is there a tumor?** and **what kind?** —
with a confidence score and a full visual explanation of how it reached its conclusion.

Python/FastAPI backend, Next.js frontend.

> **Assistive tool only.** This is a decision-support and triage system for radiologists.
> It is not a medical device, is not regulatory-approved, and must never be the sole basis
> for a clinical decision.

---

## Quick start

```bash
./run.sh setup      # one-time: Python venv + client dependencies
```

Then two terminals:

```bash
./run.sh serve      # REST API      -> http://localhost:3000
```

```bash
./run.sh client     # web interface -> http://localhost:5713
```

Open **http://localhost:5713**, drop in an MRI slice, and pick Detection or Typing.
Sample scans live in `data/kaggle4/test/`.

---

## Results

Two models ship, answering two different questions. Both are measured on the dataset's own
held-out test split — slices from one patient never appear on both sides.

### Detection — is there a tumor?

| metric | value |
|---|---|
| Accuracy | **95.43%** |
| Sensitivity (recall) | **93.77%** |
| Specificity | **100.00%** |
| ROC-AUC | 0.9968 |
| Missed tumors | 18 of 289 |
| False alarms | 0 of 105 |
| Model | SVM-RBF on 171 features |

### Typing — which kind of tumor?

| metric | value |
|---|---|
| Accuracy | **80.20%** |
| Balanced accuracy | **79.48%** |
| Macro F1 | 78.10% |
| ROC-AUC (ovr) | 0.9610 |
| Model | ensemble: SVM-RBF × 0.05 + fine-tuned ResNet-18 × 0.95 |

Per-class recall: No Tumor 100% · Meningioma 99.1% · Pituitary 83.8% · **Glioma 35.0%**.

Glioma is the weak class and we report it as such — see
[docs/05_RESULTS.md](docs/05_RESULTS.md) for the measured reason (the dataset's own train
and test gliomas are separable at 0.858 AUC by image size alone).

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
   [4] SEGMENTATION           skull strip -> statistical threshold + contralateral
        |                     asymmetry -> morphology -> contours
        v
   [5] FEATURE EXTRACTION     intensity + GLCM + LBP + shape + asymmetry + regional
        |                     + anatomical LOCATION            = 171 numbers
        v
   [6] CLASSIFICATION         (a) LogReg / SVM / RandomForest / MLP, best by CV ROC-AUC
        |                     (b) fine-tuned ResNet-18  [typing only]
        |                     -> blended with a weight chosen on validation data
        v
   [7] PRESENTATION           8-panel figure (CLI) or the web interface
```

### Why hybrid, and where each half earns its place

The engineered features make the system **auditable**: they produce a visible, measurable
region — area, circularity, solidity, midline distance, depth in the parenchyma — that a
clinician can inspect and overrule. They are enough to answer *is there a tumor* at 93.8%
sensitivity.

They are **not** enough to answer *which tumor*, and we measured that rather than assuming
it: the classical model alone reaches 69.3% on typing. Tumor type depends on spatial
structure a global texture histogram averages away, so the typing model adds a fine-tuned
CNN and blends the two. Detection stays purely classical, because there the CNN adds
nothing worth the loss of interpretability.

---

## Commands

| command | what it does |
|---|---|
| `./run.sh serve` | start the REST API on port 3000 |
| `./run.sh client` | start the Next.js frontend on port 5713 |
| `./run.sh check` | run the 66-test suite (pre-flight check before a demo) |
| `./run.sh train-all` | retrain both shipped models |
| `./run.sh train --dataset kaggle4` | train detection only |
| `./run.sh train --dataset kaggle4 --multiclass --deep` | train typing, classical + CNN |
| `./run.sh predict <image>` | analyse a single MRI and save the 8-panel figure |
| `./run.sh demo` | produce every presentation figure into `outputs/` |
| `./run.sh metrics [--multiclass]` | reprint the last training report |

Equivalent direct form: `python main.py <command>` (see `python main.py --help`).

---

## The API

FastAPI on **port 3000**. Interactive docs at `http://localhost:3000/docs`.

| method | route | purpose |
|---|---|---|
| `GET` | `/health` | liveness, which models are trained, torch/device status |
| `GET` | `/models` | details of the loaded models, including the ensemble weight |
| `GET` | `/classes` | label sets with a plain-language description of each tumor type |
| `GET` | `/metrics?multiclass=` | the full training report for either task |
| `POST` | `/analyze?multiclass=&include_images=` | run the pipeline on an uploaded slice |
| `POST` | `/analyze/features` | the complete named 171-feature vector, grouped by family |

`/analyze` returns the verdict, per-class probabilities, region geometry, anatomical
location measurements, enhancement gain, and every pipeline stage as an inline base64
image. Uploaded scans are never written to disk — the analysis is stateless.

**Ports.** The API takes 3000 and the frontend 5713. Next.js defaults to 3000, so it is the
frontend that was moved: the API port is the one that ends up in deployment scripts and
proxy configs, so it is the one worth keeping conventional. Change either in
`src/braintumor/config.py` (`API_PORT`, `CLIENT_PORT`, `CORS_ORIGINS`) and
`client/package.json`.

---

## The web interface

`client/` is a Next.js 16 app (App Router, TypeScript, Tailwind v4).

- **Analyse** — drop in a slice, switch between Detection and Typing, and see the verdict,
  the full probability distribution, all six pipeline stages, and every measurement.
- **Performance** — held-out metrics for either model: confusion matrix, per-class recall,
  and a side-by-side of what the CNN branch contributed.

The UI is dark by design, not by fashion: MRI slices are grayscale and a bright surround
raises the viewer's adaptation level, flattening exactly the low-contrast detail Stage 3
recovers.

---

## Using a real dataset

The pipeline is dataset-agnostic. For binary data:

```
data/raw/yes/     # scans WITH a tumor
data/raw/no/      # scans WITHOUT a tumor
```

then `./run.sh train --dataset classic`. The loader also accepts `tumor`, `notumor`,
`positive`, `negative`, `healthy`, `glioma`, `meningioma`, `pituitary`, `1`, `0`.

For a dataset with its own split, mirror `data/kaggle4/{train,test}/<class>/` and use
`--dataset kaggle4`. Download both public datasets with:

```bash
.venv/bin/python scripts/download_datasets.py
```

`--dataset synthetic` needs no download at all — the project ships a physically-motivated
MRI phantom generator (skull, CSF, grey/white matter, ventricles, tumors with
Fourier-perturbed boundaries and necrotic cores, B1 bias field, Rician noise).

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
│   ├── 02_VIVA_QUESTIONS_ANSWERS.md  <- likely questions with answers
│   ├── 03_DEMO_SCRIPT_30MIN.md       <- minute-by-minute presentation plan
│   ├── 04_TECHNICAL_DEEP_DIVE.md     <- the maths behind every stage
│   ├── 05_RESULTS.md                 <- measured results
│   └── 06_DATASETS.md                <- the datasets, and what is wrong with them
│
├── src/braintumor/
│   ├── config.py               every tunable constant in one place
│   ├── preprocess.py           STAGE 2  grayscale, resize, denoise, normalise
│   ├── enhance.py              STAGE 3  CLAHE, unsharp mask, gamma, metrics
│   ├── segment.py              STAGE 4  skull strip, threshold, morphology, contours
│   ├── features.py             STAGE 5  7 feature families, 171 features
│   ├── dataset.py              STAGE 1  MRI simulator + real-dataset loaders
│   ├── model.py                STAGE 6  the four classical candidates
│   ├── deep.py                 STAGE 6b ResNet-18 fine-tuning + ensemble
│   ├── train.py                training, model selection, threshold tuning, evaluation
│   ├── predict.py              single-image inference
│   ├── visualize.py            STAGE 7  all figures
│   ├── api.py                  STAGE 7b FastAPI REST backend
│   └── cli.py                  command line interface
│
├── client/                     Next.js frontend (port 5713)
│   └── src/
│       ├── app/                pages: analyse, performance
│       ├── components/         upload, verdict, probabilities, stages, measurements
│       └── lib/                typed API client
│
├── scripts/
│   ├── download_datasets.py    fetch both public datasets from Hugging Face
│   └── train_all.sh            retrain both shipped models
│
├── tests/                      66 tests: pipeline, location features, deep branch, API
├── data/                       raw/ · kaggle4/ · synthetic/
├── models/                     trained bundles (.joblib) + CNN weights (.pt)
└── outputs/                    figures and metrics JSON
```

---

## Four real bugs, and how they were fixed

Documented in code comments and covered by regression tests — see
[docs/04_TECHNICAL_DEEP_DIVE.md](docs/04_TECHNICAL_DEEP_DIVE.md).

1. **The flood-fill split-background bug.** The head ellipse often touches the top and
   bottom image borders, splitting the background into two disconnected regions. A corner
   flood-fill reached only one, so the other half of the frame was misclassified as an
   interior hole and filled in. Fixed by padding a 1-pixel frame before flood filling.

2. **Otsu flags every healthy brain.** Otsu always splits a histogram into two classes,
   even when nothing is abnormal, so on a healthy brain it returned grey-vs-white matter as
   "lesion" — **100% of healthy scans falsely flagged.** Replaced with a tissue-relative
   statistical threshold (`mean + 1.5σ` over pixels above the brain median).

3. **Threshold tuning made results worse.** The first tuner maximised specificity subject to
   a sensitivity floor, and picked 0.93 because the model is over-confident on training
   folds. On the test set that dropped sensitivity from 91% to 71%. Inverting the objective
   — maximise sensitivity subject to a specificity floor — fixed it and is robust to the
   probability shift, because lowering a threshold cannot reduce sensitivity.

4. **Test-set peeking in the ensemble.** The first version of the deep branch picked
   whichever of classical / CNN / ensemble scored best *on the test set*. That makes the
   reported number the maximum of three draws rather than an estimate. Fixed by choosing
   the blend weight on a validation split carved out of training data, with 0.0 and 1.0 in
   the grid so the weight search *is* the branch choice.

---

## Limitations

- **Glioma typing is weak** (35% recall). Detection is strong; typing is partially solved.
- The CNN that supplies most of the typing accuracy is **not interpretable** — the audit
  trail comes from the classical branch only.
- Performance depends on dataset quality and diversity; models degrade across scanners.
- 2D slice analysis, not 3D volumetric.
- Results come from one curated public dataset. Clinical use would require multi-centre
  validation and regulatory approval.

## Future scope

Grad-CAM heatmaps over the CNN branch · U-Net pixel-level segmentation with Dice scoring ·
patient-level rather than slice-level splits · 3D tumor volume estimation across a full MRI
stack · automated PDF reporting · extension to CT and other modalities.

---

## Requirements

Python 3.9+ and Node 18+. Python dependencies in `requirements.txt` (OpenCV, scikit-learn,
scikit-image, NumPy, Matplotlib, FastAPI, PyTorch). **No GPU required** — PyTorch is
optional, and the whole binary pipeline, CLI and API run without it. On Apple Silicon the
CNN fine-tunes on Metal (MPS) in about 6 minutes.
