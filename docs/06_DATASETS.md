# 06 — THE DATASETS

The project trains on **real clinical brain MRI scans**, not simulated data.

---

## GETTING THEM

```bash
.venv/bin/python scripts/download_datasets.py
```

Both datasets are public and are fetched from **Hugging Face**, which needs no account and
no API key — unlike Kaggle, which requires a token. Total download ≈ 105 MB.

Fetch just one with `--only classic` or `--only kaggle4`.

---

## DATASET 1 — "classic" (253 images)

**Brain MRI Images for Brain Tumor Detection**, Navoneel Chakrabarty.

| | |
|---|---|
| images | 253 (155 tumor / 98 healthy) |
| task | binary — tumor present or absent |
| layout | `data/raw/yes/`, `data/raw/no/` |
| split | none provided; we use a stratified 75/25 random split |

This is the standard small benchmark that most published brain-tumor-detection projects
report on, which makes it the right dataset for comparing against prior work.

**What it is actually like.** Far messier than a tidy benchmark suggests, and this drove
several pipeline design decisions:

- **129 distinct image shapes** among the 155 tumor images alone
- sizes from 150×168 up to 1920×1080
- mixed channel counts: 1 (grayscale), 3 (BGR), and 4 (BGRA)
- mixed MRI sequences — T1, T1 with contrast, T2 and FLAIR are all present, and a tumor
  that is bright on one sequence can be dark on another
- several images are **photographs of printed figures**, complete with a white page border
- some carry burned-in scanner annotations (patient text, orientation letters, rulers)

---

## DATASET 2 — "kaggle4" (3264 images) — the main dataset

**Brain Tumor Classification (MRI)**, Sartaj Bhuvaji et al.

| | |
|---|---|
| images | 3264 |
| classes | glioma 926 · meningioma 937 · pituitary 901 · no tumor 500 |
| tasks | binary (tumor / no tumor) **and** 4-class tumor typing |
| layout | `data/kaggle4/train/<class>/`, `data/kaggle4/test/<class>/` |
| split | **provided by the curators** — we use it as given |

Also mixed: axial, sagittal **and** coronal slices all appear, so the pipeline cannot
assume a viewing plane.

### Why we use the provided split instead of a random one

This is the single most important methodological decision in the project.

Medical imaging datasets contain **several consecutive slices from the same patient**. Under
a random split, slice 14 of a patient lands in training and slice 15 of the same patient
lands in test. The two images are nearly identical, so the model is graded on images it has
effectively already seen. The reported score measures memorisation, not generalisation.

We measured the size of this effect on our own pipeline, using identical features and the
same model, changing only how the data was split:

| splitting strategy | accuracy | sensitivity | ROC-AUC |
|---|---|---|---|
| **provided split** (sources kept apart) | **93.65%** | **91.35%** | 0.9937 |
| random split of the pooled data | 97.21% | 98.50% | 0.9841 |

A random split looks **+3.6 accuracy points and +7.2 sensitivity points better** — and every
one of those points is an illusion. We report the lower, honest number.

---

## A MEASURED PROPERTY OF THE kaggle4 SPLIT

While investigating why glioma classification performs poorly, we found something worth
reporting: **the train and test halves are not drawn from the same source.**

We trained a deliberately trivial classifier — 32 histogram bins plus mean, standard
deviation and image height — to predict only *which folder an image came from*, train or
test. If the two halves were sampled from one population this should be impossible, scoring
around 0.5 AUC. Instead:

| class | train-vs-test separability (AUC) | distinct image sizes in train | in test |
|---|---|---|---|
| glioma | **0.858** | 1 | 64 |
| meningioma | 0.836 | 4 | 98 |
| pituitary | 0.798 | 15 | 28 |
| no tumor | 0.678 | 116 | 56 |

The glioma row tells the story plainly: **every** training glioma image is 512×512 — a single
uniform clinical source — while the test gliomas come in 64 different sizes, consistent with
being gathered from mixed sources. A classifier can tell the two apart from the histogram
alone.

**Three consequences, all of which we observed:**

1. **Glioma recall collapses to 24%.** The model learned what a 512×512 clinical glioma looks
   like and is then tested on something meaningfully different.
2. **Probability calibration does not transfer.** On training folds the model reaches 99.1%
   sensitivity at a 0.5 threshold; on the test set the same threshold gives 87.9%. This is
   why our first attempt at threshold tuning made results *worse* — see
   `docs/04_TECHNICAL_DEEP_DIVE.md`.
3. **It is a realistic dressed rehearsal for deployment.** A hospital installing this tool
   would face exactly this problem: a model trained at one site meeting scans from another.
   Our headline numbers are measured *under* that shift rather than in spite of it, which
   makes them conservative rather than flattering.

---

## THE SYNTHETIC SIMULATOR IS STILL THERE

`./run.sh train --dataset synthetic` still works and needs no download. Its role changed
once real data arrived: it is no longer the source of headline results, but it remains
useful as a **controlled test-bed**, because it is the only dataset where we know the exact
ground-truth tumor location and can therefore check the segmentation stage directly rather
than inferring its quality from classification accuracy.

The generator models skull, CSF, grey and white matter, ventricles and cortical folding,
injects tumors with irregular Fourier-perturbed boundaries, edema haloes and necrotic cores,
and applies B1 bias-field inhomogeneity plus **Rician** noise — the correct model for an MRI
magnitude image, since the magnitude of complex Gaussian noise is Rician, not Gaussian.

---

## USING YOUR OWN DATA

Drop it in and retrain — no code changes:

```
data/raw/yes/     scans WITH a tumor
data/raw/no/      scans WITHOUT a tumor
```

```bash
./run.sh train --dataset classic
```

The loader also accepts the folder names `tumor`, `notumor`, `no_tumor`, `positive`,
`negative`, `healthy`, `glioma`, `meningioma`, `pituitary`, `1` and `0`. For a dataset with
its own split, mirror the `data/kaggle4/{train,test}/<class>/` layout and use
`--dataset kaggle4`.
