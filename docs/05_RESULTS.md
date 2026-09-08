# 05 — RESULTS

**All numbers below are measured on REAL clinical brain MRI scans**, on held-out test sets
the model never saw during training or model selection.
Regenerate with `./run.sh train --dataset kaggle4`.

---

## HEADLINE RESULT — the shipped model

Binary tumor detection on the **3264-image** dataset, evaluated on the curators'
own held-out test split (394 images from separate sources).

| metric | value |
|---|---|
| **Accuracy** | **97.72%** |
| **Sensitivity (recall)** | **96.89%** |
| **Specificity** | **100.00%** |
| Balanced accuracy | 98.44% |
| Precision | 100.00% |
| F1 score | 98.42% |
| ROC-AUC | 0.9938 |
| Missed tumors | **9** out of 289 |
| False alarms | **0** out of 105 |
| Model | svm_rbf |
| Features | 141 |
| Training time | 164.41s on CPU |

### Confusion matrix

|  | predicted normal | predicted tumor |
|---|---|---|
| **actual normal** | 105 (TN) | 0 (FP) |
| **actual tumor** | 9 (FN) | 280 (TP) |

---

## THE DECISION THRESHOLD — where most of the clinical value came from

The model's raw output is a probability. Turning it into a decision needs a threshold, and
the default of 0.5 silently asserts that a missed tumor and a false alarm cost the same.
They do not. We chose the threshold deliberately, **using only out-of-fold predictions on
the training set** — the test set was never consulted.

| | default 0.500 | tuned 0.140 |
|---|---|---|
| Sensitivity (recall) | 87.89% | **96.89%** |
| Specificity | 100.00% | 100.00% |
| Accuracy | 91.12% | **97.72%** |
| Balanced accuracy | 93.94% | 98.44% |
| F1 | 93.55% | 98.42% |
| **Missed tumors** | **35** | **9** |
| False alarms | 0 | 0 |

**26 fewer missed tumors, at a cost of
0 extra false alarms.** For a screening tool
that is unambiguously the right trade.

Objective used: *max sensitivity subject to specificity >= 80%*. The specificity floor is a **clinical policy
parameter**, not something fitted to the data — it encodes how many extra reviews the
radiology department can absorb.

### The full trade-off curve on the test set

Published for transparency. The operating point we ship was chosen from training data alone;
this table exists so that choice is auditable, not so it can be cherry-picked after the fact.

| threshold | sensitivity | specificity | accuracy | missed tumors | false alarms |
|---|---|---|---|---|---|
| 0.05 | 99.3% | 100.0% | 99.5% | 2 | 0 |
| 0.10 | 97.2% | 100.0% | 98.0% | 8 | 0 |
| 0.15 | 96.2% | 100.0% | 97.2% | 11 | 0 |
| 0.20 | 94.8% | 100.0% | 96.2% | 15 | 0 |
| 0.25 | 91.7% | 100.0% | 93.9% | 24 | 0 |
| 0.30 | 91.7% | 100.0% | 93.9% | 24 | 0 |
| 0.35 | 91.0% | 100.0% | 93.4% | 26 | 0 |
| 0.40 | 88.9% | 100.0% | 91.9% | 32 | 0 |
| 0.45 | 88.2% | 100.0% | 91.4% | 34 | 0 |
| 0.50 | 87.5% | 100.0% | 90.9% | 36 | 0 |
| 0.55 | 87.5% | 100.0% | 90.9% | 36 | 0 |

---

## MODEL COMPARISON (5-fold stratified CV, training set only)

| model | CV ROC-AUC | CV balanced accuracy | |
|---|---|---|---|
| `logistic_regression` | 0.9837 ± 0.003 | 94.49% ± 1.32 |  |
| `svm_rbf` | 0.9940 ± 0.004 | 95.53% ± 1.22 | **← selected** |
| `random_forest` | 0.9874 ± 0.005 | 89.81% ± 1.53 |  |
| `mlp_neural_net` | 0.9895 ± 0.005 | 93.92% ± 1.74 |  |

Selection is automatic, by CV ROC-AUC with balanced accuracy as a tiebreak.

---

## MOST INFLUENTIAL FEATURES

| rank | feature | importance |
|---|---|---|
| — | *SVM-RBF exposes no per-feature importances* | — |

---

## SECOND DATASET — the classic 253-image benchmark

| metric | value |
|---|---|
| Accuracy | **95.31%** |
| Sensitivity (recall) | 94.87% |
| Specificity | 96.00% |
| F1 | 96.10% |
| ROC-AUC | 0.9723 |
| Model | mlp_neural_net |
| Missed tumors | 2 of 39 |

Threshold tuning **did nothing here**, and that is the correct behaviour rather than a
failure: our tuner adopts a new threshold only when it actually buys sensitivity on the
training folds, and on 189 training images it did not. The guard exists because lowering a
threshold always costs specificity, so a move that gains no recall is a pure loss.

Caveat worth stating out loud: with only 64 test images, **one image is worth 1.6
percentage points**. Differences of two or three points on this dataset are noise. That is
precisely why the 3264-image dataset is the one we report as the headline.

---

## THIRD TASK — 4-class tumor typing (the stretch goal)

Same pipeline, asked to name the tumor type rather than just detect one.

| metric | value |
|---|---|
| Accuracy | 72.08% |
| Balanced accuracy | 70.00% |
| Macro F1 | 67.91% |
| ROC-AUC (one-vs-rest, macro) | 0.9073 |

### Per-class recall

| class | recall |
|---|---|
| No Tumor | 100.00% |
| Glioma | 24.00% |
| Meningioma | 96.52% |
| Pituitary | 59.46% |

### Confusion matrix (rows = actual, columns = predicted)

| | No Tumor | Glioma | Meningioma | Pituitary |
|---|---|---|---|---|
| **No Tumor** | 105 | 0 | 0 | 0 |
| **Glioma** | 27 | 24 | 49 | 0 |
| **Meningioma** | 4 | 0 | 111 | 0 |
| **Pituitary** | 20 | 0 | 10 | 44 |

**We report this as a partial success, because that is what it is.** Detecting *that* a
tumor exists works well; naming *which* type does not, and the failure is concentrated
almost entirely in the glioma class.

Two honest reasons:

1. **The dataset's glioma split is inconsistent.** We measured this rather than assuming it.
   A deliberately trivial classifier — 32 histogram bins plus mean, standard deviation and
   image height — can tell a *training* glioma from a *test* glioma at **0.858 AUC**. Every
   training glioma is 512×512; the test gliomas come in 64 different sizes. The model is
   being tested on a different population from the one it learned. Full analysis in
   [06_DATASETS.md](06_DATASETS.md).
2. **Tumor type is not primarily a texture property.** Radiologists distinguish these types
   largely by anatomical *location* (pituitary tumors sit at the skull base, meningiomas
   attach to the meninges) and by contrast-enhancement behaviour across sequences. Our
   global texture and intensity features do not encode location, and a single slice does not
   show enhancement dynamics. This is the part of the problem that genuinely needs a CNN.

Collapsed to the clinical question, the multi-class model still answers "is there a tumor
at all" at 87.06% accuracy — but the dedicated binary model
above beats it, so **the binary model is the one we ship.**

---

## SEGMENTATION STAGE, MEASURED SEPARATELY

Evaluated on all 253 images of the classic dataset, independently of the classifier:

| | |
|---|---|
| tumors localised | **129 / 155 (83.2%)** |
| false regions on healthy scans | 56 / 98 (57.1%) |

The false-region rate is high, and that is a deliberate choice rather than an oversight.
Segmentation is tuned toward sensitivity: a false region is not a false diagnosis, because
the classifier still weighs all 141 features and overrules it — final
specificity is 100%. A tumor *missed* at the segmentation stage,
by contrast, would be unrecoverable downstream.

Adding the contralateral-asymmetry criterion (comparing each pixel with its mirror
counterpart, the way a radiologist scans for a lesion) raised localisation from **72.3% to
83.2%** and was the fix for tumors on T2 scans, where the whole brain is bright and an
absolute intensity threshold finds nothing.

---

## HOW MUCH EACH IMPROVEMENT WAS WORTH

Measured on the 3264-image dataset, in the order we made them:

| change | accuracy | sensitivity |
|---|---|---|
| baseline: 111 features, threshold 0.5 | 88.58% | 84.43% |
| \+ asymmetry and regional features (141 total) | 93.65% | 91.35% |
| \+ tuned clinical threshold | **97.72%** | **96.89%** |

The two feature families added +5.1 accuracy points; the threshold added a further +4.1 and
cut missed tumors from 35 to 9.

---

## HONEST LIMITATIONS

- **Single-dataset validation.** Strong results on one dataset's test split are not clinical
  validity. Proper validation needs multiple institutions, multiple scanner vendors and
  expert annotation.
- **The test set is 105 healthy vs 289 tumor scans.** Real screening populations are
  overwhelmingly healthy, so specificity matters far more in deployment than this test set
  can demonstrate. Our 100% specificity here would certainly degrade.
- **Tumor typing does not work well enough to be useful** (glioma recall
  24%).
- **2D single-slice analysis**, not 3D volumetric.
- **No external validation.** Every number here comes from one curated public dataset.
- The system is **assistive**: it is not a medical device, has no regulatory approval, and
  must never be the sole basis for a clinical decision.

---

## REPRODUCIBILITY

Fixed seed (42) throughout: split, CV folds and every model. `./run.sh check`
runs 27 tests covering every stage in about 40 seconds, including regression tests for the
four real bugs described in [04_TECHNICAL_DEEP_DIVE.md](04_TECHNICAL_DEEP_DIVE.md).
