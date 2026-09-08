# SYNOPSIS

## Title

**Medical Image Enhancement and Diagnostic Assistance System for Brain Tumor Detection
using MRI and OpenCV**

---

## 1. Introduction

Medical imaging plays an important role in the detection and analysis of disease. Brain
Magnetic Resonance Imaging (MRI) provides detailed information about brain tissues and is
widely used for identifying abnormalities such as brain tumors. However, MRI images contain
noise, low contrast, and unclear boundaries, which make the identification of abnormal
regions difficult and time-consuming even for trained radiologists.

This project implements a **Medical Image Enhancement and Diagnostic Assistance System** for
brain tumor detection from MRI. The system uses **OpenCV-based image processing** to denoise
and enhance MRI images, automatically segments the suspicious region, extracts quantitative
image features, and applies a **machine learning classifier** to classify the scan as normal
or tumor-affected — reporting a confidence score alongside a full visual explanation of how
the conclusion was reached.

The system is a **diagnostic assistance tool**, not a replacement for professional medical
diagnosis.

---

## 2. Problem Statement

Manual examination of brain MRI requires expertise and is time-consuming, particularly when
abnormalities have unclear boundaries or low contrast. Poor image quality makes tumor
regions harder to identify, and radiologist fatigue and shortage compound the problem.

A system is therefore needed that can:

* Improve the quality and visibility of brain MRI images
* Reduce noise and imaging artefacts
* Automatically highlight suspicious regions
* Assist in detecting the presence of a brain tumor
* Present its analysis in a form a clinician can verify

---

## 3. Objectives

1. Develop a complete processing system for brain MRI images.
2. Enhance MRI images using OpenCV image processing techniques.
3. Reduce noise and improve contrast, with objectively measured improvement.
4. Identify and extract the region of interest (ROI) corresponding to a possible tumor.
5. Extract quantitative image features from the processed MRI.
6. Integrate a machine learning model for tumor classification.
7. Classify MRI images into **Tumor** and **No Tumor** categories with a confidence score.
8. Provide visual diagnostic assistance showing every processing stage.
9. Develop a simple user interface for uploading and analysing MRI images.

---

## 4. Methodology

### Stage 1 — MRI Image Acquisition
The system accepts brain MRI images from any publicly available medical imaging dataset.
It additionally ships a physically-motivated MRI phantom simulator (modelling skull, CSF,
grey and white matter, ventricles, tumor with edema and necrosis, B1 bias field and Rician
noise), so the system is fully reproducible without a dataset download.

### Stage 2 — Preprocessing
Grayscale conversion · resize to 256×256 (INTER_AREA) · median filter for salt-and-pepper
noise · bilateral filter for edge-preserving smoothing · min-max intensity normalisation.

### Stage 3 — Image Enhancement
**CLAHE (Contrast Limited Adaptive Histogram Equalisation)** with an 8×8 tile grid and clip
limit 2.0, followed by **unsharp masking**. Global histogram equalisation is implemented for
comparison and is demonstrably worse here, because the bright skull dominates the global
histogram. Improvement is measured objectively as contrast (standard deviation), entropy
(bits) and sharpness (Laplacian variance).

### Stage 4 — Tumor Region Segmentation
Skull stripping via Otsu thresholding, largest-connected-component selection, flood-fill
hole filling and distance-transform erosion · a tissue-relative statistical threshold
(mean + 1.5σ over pixels above the brain median) · morphological opening and closing ·
contour detection with area and aspect-ratio plausibility filtering · selection of the
best candidate by area weighted by hyperintensity.

### Stage 5 — Feature Extraction
171 features in seven interpretable families:
* **First-order intensity statistics** — mean, variance, skewness, kurtosis, entropy,
  percentiles, bright-pixel fractions, 16-bin histogram
* **GLCM (Haralick) texture** — contrast, dissimilarity, homogeneity, energy, correlation,
  ASM and entropy at 3 distances × 4 angles
* **Local Binary Patterns** — uniform LBP histogram (P=24, R=3)
* **Shape / region descriptors** — area, perimeter, circularity, solidity, extent, aspect
  ratio, hyperintensity ratio
* **Bilateral asymmetry** — comparison of each pixel with its mirror counterpart across the
  mid-sagittal plane, the way a radiologist scans for a lesion
* **Regional statistics** — a coarse 3×3 spatial map of the brain
* **Anatomical location** — distance from the midline, depth inside the parenchyma, contact
  with the brain border, radial position and mass bands, all normalised by the brain's own
  radius so they encode anatomy rather than image resolution. These are the geometric cues
  that distinguish tumor *types*: a pituitary adenoma sits on the midline at the skull base,
  a meningioma hugs the brain's outer edge, a glioma is embedded deep in white matter.

### Stage 6 — Classification
Four candidate models (Logistic Regression, SVM with RBF kernel, Random Forest, and an MLP
neural network), each as a `StandardScaler → classifier` pipeline to prevent data leakage.
All are compared by 5-fold stratified cross-validated ROC-AUC and the best is selected
automatically, then evaluated once on a held-out test set. For the binary detection task the
decision threshold is additionally tuned on out-of-fold training predictions to maximise
sensitivity within a specificity budget, because a missed tumor costs far more than a second
review.

### Stage 6b — Deep Branch (tumor typing only)
The engineered features were measured to be insufficient for naming a tumor type — 69.3%
accuracy with 23% glioma recall. A **ResNet-18 pretrained on ImageNet** is therefore
fine-tuned on the same CLAHE-enhanced images, with augmentation targeted at the measured
scale and framing shift between this dataset's train and test halves. Its probabilities are
blended with the classical model's using a weight chosen on a validation split carved out of
the training data, never on the test set. This lifted typing accuracy to 80.2% and pituitary
recall from 45.9% to 83.8%.

### Stage 7 — Result Presentation
An 8-panel figure from the CLI, and a web interface showing the verdict, the full per-class
probability distribution, all six pipeline stages, and every measurement behind the
decision — so a clinician can check the reasoning rather than only the conclusion.

### Stage 7b — Web Application
A **FastAPI** REST backend exposing health, model metadata, per-scan analysis and the
complete feature vector, consumed by a **Next.js** frontend. Uploaded scans are analysed in
memory and never written to disk.

---

## 5. System Architecture

```text
                    Brain MRI Image
                          │
                          ▼
                   Image Acquisition
                          │
                          ▼
                    Preprocessing
             (resize + median + bilateral)
                          │
                          ▼
                  Image Enhancement
                  (CLAHE + unsharp mask)
                          │
              ┌───────────┴────────────┐
              │                        │
              ▼                        │
      ROI / Segmentation               │  the same enhanced image
  (skull strip + threshold +           │  feeds both branches
   morphology + contours)              │
              │                        │
              ▼                        ▼
     Feature Extraction         Fine-tuned ResNet-18
  (intensity + GLCM + LBP +      (typing task only)
   shape + asymmetry +                 │
   regional + LOCATION)                │
       = 171 features                  │
              │                        │
              ▼                        │
       ML Classifier                   │
  (best of 4, by CV ROC-AUC)           │
              │                        │
              └───────────┬────────────┘
                          ▼
                  Weighted soft vote
            (weight chosen on validation)
                          │
                          ▼
              Tumor / No Tumor      or
        Glioma / Meningioma / Pituitary / None
              + confidence score
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
      8-panel figure            FastAPI  ->  Next.js
         (CLI)                    (web interface)
```

---

## 6. Technologies Used

**Backend:** Python 3.9+ · OpenCV · NumPy · scikit-image · scikit-learn · PyTorch /
torchvision · Matplotlib · Pandas · Joblib · FastAPI · Uvicorn · pytest

**Frontend:** Next.js 16 (App Router) · TypeScript · Tailwind CSS v4

**Hardware:** any standard laptop; 8 GB RAM recommended. **No GPU required** — PyTorch is an
optional dependency and the entire detection pipeline, CLI and API run without it. The
classical training run takes about 80 seconds on CPU; on Apple Silicon the CNN fine-tunes on
Metal (MPS) in about 6 minutes.

---

## 7. Results Achieved

Both models are evaluated on the dataset's **own** held-out test split (394 images), so
near-duplicate slices of one patient cannot appear on both sides.

### Detection — is there a tumor?

| metric | value |
|---|---|
| Accuracy | **95.43%** |
| Sensitivity (recall) | **93.77%** |
| Specificity | 100.00% |
| Precision | 100.00% |
| F1 score | 96.79% |
| ROC-AUC | 0.9968 |
| Missed tumors | 18 of 289 |
| False alarms | 0 of 105 |

### Typing — which kind of tumor?

| metric | value |
|---|---|
| Accuracy | **80.20%** |
| Balanced accuracy | **79.48%** |
| Macro F1 | 78.10% |
| ROC-AUC (one-vs-rest) | 0.9610 |

Per-class recall: No Tumor 100% · Meningioma 99.1% · Pituitary 83.8% · Glioma 35.0%.

Adding the location feature family and the fine-tuned CNN raised typing accuracy from 72.1%
to 80.2%, pituitary recall from 59.5% to 83.8%, and glioma recall from 24.0% to 35.0%.
**Glioma remains the weak class**, and the measured reason is a defect in the dataset rather
than the model: a trivial classifier using only image histogram and height can tell a
training glioma from a test glioma at 0.858 AUC, because every training glioma is 512×512
while the test gliomas come in 64 different sizes.

Segmentation measured independently on the 253-image benchmark: 129/155 tumors localised
(83.2%). Enhancement measured per scan: contrast 45.5 → 59.1, entropy 6.82 → 7.71 bits,
sharpness 77.7 → 683.1. Verified by **66 automated tests** covering every stage, the deep
branch and the REST API.

---

## 8. Scope of the Project

The system covers brain MRI for **both** tumor detection and tumor typing, delivered as a
REST API with a web interface. The architecture separates the image-processing stages from
the classifier — Stages 2–5 are model-agnostic — which is what allowed a CNN to be added at
Stage 6 for the typing task without touching anything upstream. It can be extended further
to pixel-level U-Net segmentation, Grad-CAM explanations of the deep branch, 3D volumetric
analysis, automated reporting, and other imaging modalities.

---

## 9. Limitations

* **Glioma typing is weak (35% recall).** Detection is strong; naming the tumor type is only
  partially solved, and the failure is concentrated almost entirely in this one class.
* The CNN that supplies most of the typing accuracy is **not interpretable**. The audit
  trail — the segmented region and its measurements — comes from the classical branch only.
* Performance depends on the quality and diversity of the dataset; models degrade across
  scanners (domain shift). We measured this directly: the CNN scores 98.6% balanced accuracy
  on a validation split drawn from the training distribution and 79.5% on the test split.
* The test set is 105 healthy against 289 tumor scans. Real screening populations are
  overwhelmingly healthy, so the reported 100% specificity would degrade in deployment.
* 2D slice analysis rather than 3D volumetric analysis; splits are by slice, not by patient.
* Intensity-based segmentation can be misled by calcifications or imaging artefacts.
* Results come from one curated public dataset and demonstrate pipeline correctness rather
  than clinical validity.
* The system is for **diagnostic assistance only** and cannot replace examination and
  diagnosis by a qualified medical professional.

---

## 10. Conclusion

The implemented system combines traditional image processing with machine learning for brain
MRI analysis. OpenCV performs preprocessing, enhancement and segmentation; a cross-validated
machine learning model performs classification; and for the harder tumor-typing task a
fine-tuned CNN is blended in as a second opinion.

The hybrid design is deliberate, and the project measured where each half earns its place
rather than assuming it. The classical stages produce a visible, measurable region that a
clinician can audit, and they are sufficient to answer *is there a tumor* at 93.8%
sensitivity with zero false alarms. They were **not** sufficient to answer *which tumor* —
69.3% accuracy — so the deep branch was added for that task alone, lifting it to 80.2%.
Reporting that the interpretable half fell short, rather than quietly replacing it
everywhere, is the honest version of the result.

The project demonstrates an effective application of computer vision and machine learning to
medical image analysis, and provides a foundation for more advanced automated diagnostic
assistance systems.
