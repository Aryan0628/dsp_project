# 05 — RESULTS

**All numbers below are measured on REAL clinical brain MRI scans**, on held-out test sets
the model never saw during training or model selection.
Regenerate everything with `./run.sh train-all`.

Two models ship, answering two different questions:

| model | question | headline |
|---|---|---|
| **Detection** (binary) | is there a tumor at all? | **95.43% accuracy · 93.77% sensitivity · 100% specificity** |
| **Typing** (4-class) | which kind of tumor? | **80.20% accuracy · 79.48% balanced accuracy** |

---

## MODEL 1 — TUMOR DETECTION

Binary tumor detection on the **3264-image** dataset, evaluated on the curators' own
held-out test split (394 images from separate sources).

| metric | value |
|---|---|
| **Accuracy** | **95.43%** |
| **Sensitivity (recall)** | **93.77%** |
| **Specificity** | **100.00%** |
| Balanced accuracy | 96.89% |
| Precision | 100.00% |
| F1 score | 96.79% |
| ROC-AUC | 0.9968 |
| Missed tumors | **18** out of 289 |
| False alarms | **0** out of 105 |
| Model | svm_rbf, 171 features |
| Training time | 81 s on CPU |

### Confusion matrix

|  | predicted normal | predicted tumor |
|---|---|---|
| **actual normal** | 105 (TN) | 0 (FP) |
| **actual tumor** | 18 (FN) | 271 (TP) |

### The decision threshold

The shipped threshold is **0.153**, not 0.5. It is chosen on out-of-fold training
predictions by *max sensitivity subject to specificity ≥ 80%*; the test set is never
consulted. Against the default 0.5 it converts 33 missed tumors into 18, at a cost of zero
extra false alarms.

> **Note on a change since the previous run.** Adding the location feature family (141 →
> 171 features) raised ROC-AUC from 0.9938 to **0.9968** — the model's *ranking* of scans
> genuinely improved. Accuracy nonetheless moved from 97.72% to 95.43%, because the tuner
> landed on threshold 0.153 rather than 0.140 and 15 borderline scans sit between them.
> Both runs used the identical, unchanged procedure. On a 394-image test set one threshold
> step is worth several points, which is exactly why the threshold is reported alongside
> the accuracy rather than the accuracy alone.

---

## MODEL 2 — TUMOR TYPING (the hard problem, and what fixed it)

Same pipeline, asked to name the tumor type rather than only detect one.

| metric | value |
|---|---|
| **Accuracy** | **80.20%** |
| **Balanced accuracy** | **79.48%** |
| Macro F1 | 78.10% |
| ROC-AUC (one-vs-rest, macro) | 0.9610 |
| Shipped model | ensemble: svm_rbf × 0.05 + ResNet-18 × 0.95 |

### Per-class recall

| class | recall |
|---|---|
| No Tumor | 100.00% |
| Glioma | 35.00% |
| Meningioma | 99.13% |
| Pituitary | 83.78% |

### Confusion matrix (rows = actual, columns = predicted)

| | No Tumor | Glioma | Meningioma | Pituitary |
|---|---|---|---|---|
| **No Tumor** | 105 | 0 | 0 | 0 |
| **Glioma** | 24 | 35 | 40 | 1 |
| **Meningioma** | 0 | 1 | 114 | 0 |
| **Pituitary** | 6 | 0 | 6 | 62 |

---

## WHAT WE CHANGED, AND WHAT EACH CHANGE WAS WORTH

The previous version of this document reported tumor typing at **72.08% accuracy with 24%
glioma recall**, and called it a partial success. Two changes were made, both aimed at the
specific diagnosis in that write-up: *texture features cannot express where a mass sits,
and tumor type is largely a question of where a mass sits.*

| version | accuracy | balanced | macro F1 | glioma | pituitary |
|---|---|---|---|---|---|
| 141 texture/shape features (previous) | 72.08% | 70.00% | 67.91% | 24.0% | 59.5% |
| \+ location feature family (171 features) | 69.29% | 66.37% | 63.92% | 23.0% | 45.9% |
| \+ fine-tuned CNN, ensembled | **80.20%** | **79.48%** | **78.10%** | **35.0%** | **83.8%** |

**+8.1 accuracy points, +9.5 balanced accuracy, +11 points of glioma recall, +24 points of
pituitary recall.**

### Change 1 — anatomical location features (Family 7)

30 new features measuring *where* the detected region sits, all normalised by the brain's
own radius so they cannot encode image resolution:

- **distance from the midline** — a pituitary adenoma grows from a midline gland, so this
  is near zero for pituitary and large for a hemispheric glioma
- **depth inside the parenchyma** (distance transform of the brain mask) — separates
  intra-axial glioma from extra-axial meningioma
- **border contact** — the share of the lesion hugging the brain edge, the meningioma cue
- radial position, polar angle as sin/cos, superior/middle/inferior mass bands,
  left/centre/right bands, midline crossing
- brain-shape descriptors as a proxy for the acquisition plane, since this dataset mixes
  axial, sagittal and coronal slices and a location means different things in each

**Measured honestly: on their own these features made the classical model slightly worse**
(72.08% → 69.29%). They are real signal — the API surfaces them per-scan and they behave
exactly as designed on individual cases — but an SVM on 171 mostly-texture features could
not exploit them, and the extra dimensions cost more than the signal gained. They earn
their place because they are interpretable and because a clinician can check them; they did
not, by themselves, solve the problem.

### Change 2 — the fine-tuned CNN (what actually worked)

A **ResNet-18** pretrained on ImageNet, fine-tuned for 14 epochs on Apple Metal (MPS) in
**396 seconds**. It consumes the same CLAHE-enhanced image the classical branch measures,
so both halves see an identical picture.

| branch | accuracy | balanced accuracy | macro F1 |
|---|---|---|---|
| Engineered features only | 69.29% | 66.37% | 63.92% |
| Fine-tuned CNN only | 80.20% | 79.48% | 78.10% |
| **Ensemble (shipped)** | **80.20%** | **79.48%** | **78.10%** |

Per-class, the CNN is where every gain came from:

| class | classical | shipped | change |
|---|---|---|---|
| No Tumor | 100.00% | 100.00% | +0.00 |
| Glioma | 23.00% | 35.00% | **+12.00** |
| Meningioma | 96.52% | 99.13% | +2.61 |
| Pituitary | 45.95% | 83.78% | **+37.84** |

**How the ensemble weight was chosen.** 431 images were carved out of the *training* set as
a validation split. Both branches were fit on the remaining 2439, and the weight was picked
by a grid search maximising balanced accuracy on that validation split. The grid includes
0.0 (pure classical) and 1.0 (pure CNN), so the weight search *is* the branch choice — we
never compare the three test scores and pick the winner, because the maximum of three draws
on the test set is not an estimate of anything.

The search chose **0.95**, and at that weight the ensemble is numerically identical to the
CNN alone on this test set. We report that plainly rather than claiming the ensemble as a
separate achievement. The classical branch is retained because it contributes the
segmentation, the measurements and the audit trail the CNN cannot provide.

### Why the CNN is not "just cheating"

The engineered features remain the reason this system is inspectable: the segmented region,
its geometry and its location are all shown per-scan and can be checked by a human. What
the CNN adds is the one judgement those measurements provably could not make. The honest
summary is that **detection is solved by interpretable features, and typing needs a
network** — which is what the failure of Change 1 measured rather than assumed.

---

## THE REMAINING FAILURE — GLIOMA, AND WHY

Glioma recall is 35%. It is better than 24%, and it is still the weakest part of the
system. The 65 missed gliomas go to meningioma (40) and no-tumor (24). Two reasons, both
measured rather than assumed:

1. **The dataset's glioma split is inconsistent.** A deliberately trivial classifier — 32
   histogram bins plus mean, standard deviation and image height — can tell a *training*
   glioma from a *test* glioma at **0.858 AUC**. Every training glioma is 512×512; the test
   gliomas come in 64 different sizes. The model is tested on a different population from
   the one it learned. Full analysis in [06_DATASETS.md](06_DATASETS.md).
2. **The gap between validation and test proves it.** The CNN reached **98.57% balanced
   accuracy on the validation split** (drawn from the training distribution) and **79.48%
   on the test split**. A 19-point drop with no change to the model is the domain shift
   made visible. Our headline number is measured *under* that shift, which makes it
   conservative rather than flattering.

Test-time augmentation (4 label-preserving views: mirror × mild zoom) and matching training
augmentation were added specifically to blunt this, and they are why the drop is 19 points
rather than more.

### Collapsed to the clinical question

The typing model still answers "is there a tumor at all" at **92.39% accuracy, 89.62%
sensitivity, 100% specificity** — but the dedicated binary model beats it on sensitivity
(93.77%), so **for detection, ship the binary model.** The two are offered as separate
modes in the web interface for exactly this reason.

---

## MODEL COMPARISON (5-fold stratified CV, training set only)

Binary task:

| model | CV ROC-AUC | CV balanced accuracy | |
|---|---|---|---|
| `logistic_regression` | 0.9837 ± 0.003 | 94.49% ± 1.32 | |
| `svm_rbf` | **0.9940 ± 0.004** | 95.53% ± 1.22 | **← selected** |
| `random_forest` | 0.9874 ± 0.005 | 89.81% ± 1.53 | |
| `mlp_neural_net` | 0.9895 ± 0.005 | 93.92% ± 1.74 | |

Typing task:

| model | CV ROC-AUC (ovr) | CV balanced accuracy | |
|---|---|---|---|
| `logistic_regression` | 0.9621 ± 0.004 | 84.86% ± 0.7 | |
| `svm_rbf` | **0.9834 ± 0.002** | 90.02% ± 0.8 | **← selected as the classical branch** |
| `random_forest` | 0.9673 ± 0.001 | 84.82% ± 0.9 | |
| `mlp_neural_net` | 0.9791 ± 0.003 | 88.43% ± 0.9 | |

Selection is automatic, by CV ROC-AUC with balanced accuracy as a tiebreak.

---

## SEGMENTATION STAGE, MEASURED SEPARATELY

Evaluated on all 253 images of the classic dataset, independently of the classifier:

| | |
|---|---|
| tumors localised | **129 / 155 (83.2%)** |
| false regions on healthy scans | 56 / 98 (57.1%) |

The false-region rate is high, and that is a deliberate choice. Segmentation is tuned
toward sensitivity: a false region is not a false diagnosis, because the classifier still
weighs all 171 features and overrules it — final specificity is 100%. A tumor *missed* at
the segmentation stage would be unrecoverable downstream.

Adding the contralateral-asymmetry criterion raised localisation from **72.3% to 83.2%**
and was the fix for tumors on T2 scans, where the whole brain is bright and an absolute
intensity threshold finds nothing.

---

## HONEST LIMITATIONS

- **Glioma typing does not work well enough to be useful** (35% recall). Detection does.
- **Single-dataset validation.** Strong results on one dataset's test split are not clinical
  validity. Proper validation needs multiple institutions, multiple scanner vendors and
  expert annotation.
- **The test set is 105 healthy vs 289 tumor scans.** Real screening populations are
  overwhelmingly healthy, so specificity matters far more in deployment than this test set
  can demonstrate. Our 100% specificity would certainly degrade.
- **The CNN is not interpretable.** The classical branch carries the audit trail; the branch
  that produces most of the typing accuracy does not. That is a real trade, not a solved
  problem, and Grad-CAM is the obvious next step.
- **2D single-slice analysis**, not 3D volumetric.
- The system is **assistive**: not a medical device, no regulatory approval, and never the
  sole basis for a clinical decision.

---

## REPRODUCIBILITY

Fixed seed (42) throughout: split, CV folds, every classical model, and the CNN's
initialisation and data order. `./run.sh check` runs **66 tests** covering every stage,
the REST API and the deep branch. Deep-branch tests skip cleanly when torch is absent.
