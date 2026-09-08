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
111 features in four interpretable families:
* **First-order intensity statistics** — mean, variance, skewness, kurtosis, entropy,
  percentiles, bright-pixel fractions, 16-bin histogram
* **GLCM (Haralick) texture** — contrast, dissimilarity, homogeneity, energy, correlation,
  ASM and entropy at 3 distances × 4 angles
* **Local Binary Patterns** — uniform LBP histogram (P=24, R=3)
* **Shape / region descriptors** — area, perimeter, circularity, solidity, extent, aspect
  ratio, hyperintensity ratio

### Stage 6 — Classification
Four candidate models (Logistic Regression, SVM with RBF kernel, Random Forest, and an MLP
neural network), each as a `StandardScaler → classifier` pipeline to prevent data leakage.
All are compared by 5-fold stratified cross-validated ROC-AUC and the best is selected
automatically, then evaluated once on a held-out test set.

### Stage 7 — Result Visualisation
An 8-panel figure showing the original MRI, the preprocessed image, the enhanced image, the
before/after intensity histogram, the skull-stripped brain, the segmentation mask, the
detected region with contour and bounding box, and a verdict card carrying the label,
confidence, probability bar and measured region properties.

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
              ▼
      ROI / Segmentation
  (skull strip + threshold +
   morphology + contours)
              │
              ▼
     Feature Extraction
 (intensity + GLCM + LBP + shape)
              │
              ▼
      ML Classifier
  (best of 4, by CV ROC-AUC)
              │
              ▼
      Tumor / No Tumor
      + confidence score
              │
              ▼
    Result Visualisation
```

---

## 6. Technologies Used

**Software:** Python 3.9+ · OpenCV · NumPy · scikit-image · scikit-learn · Matplotlib ·
Pandas · Joblib · Streamlit (GUI) · pytest

**Hardware:** any standard laptop; 8 GB RAM recommended. **No GPU required** — the full
training run completes in about 15 seconds on CPU.

---

## 7. Results Achieved

| metric | value |
|---|---|
| Accuracy | 98.00% |
| Sensitivity (recall) | 96.00% |
| Specificity | 100.00% |
| Precision | 100.00% |
| F1 score | 97.96% |
| ROC-AUC | 1.0000 |
| Training time | 15 seconds (CPU) |

Segmentation stage measured independently: 40/40 tumors localised, 5/40 false regions on
healthy scans. Enhancement stage measured: contrast 73.7 → 81.4, entropy 6.16 → 7.08 bits,
sharpness 272 → 1100. Verified by 19 automated tests.

---

## 8. Scope of the Project

Version 1 covers brain MRI and binary tumor classification. The architecture separates the
image-processing stages from the classifier, so the system can be extended to classify
tumor types, estimate tumor size and location, perform pixel-level U-Net segmentation, use
deep-learning backbones, generate automated reports, and support other imaging modalities.

---

## 9. Limitations

* Performance depends on the quality and diversity of the dataset; models degrade across
  scanners (domain shift).
* Binary classification only — tumor present or absent, not tumor type.
* 2D slice analysis rather than 3D volumetric analysis.
* Intensity-based segmentation can be misled by calcifications or imaging artefacts.
* Reported results are on synthetic data and demonstrate pipeline correctness rather than
  clinical validity.
* The system is for **diagnostic assistance only** and cannot replace examination and
  diagnosis by a qualified medical professional.

---

## 10. Conclusion

The implemented system combines traditional image processing with machine learning for brain
MRI analysis. OpenCV performs preprocessing, enhancement and segmentation, while a
cross-validated machine learning model performs classification. The hybrid design is
deliberate: the classical stages produce a visible, measurable region that a clinician can
audit, so the system is **explainable as well as accurate** — which is what makes such a tool
usable in practice. The project demonstrates an effective application of computer vision and
machine learning to medical image analysis and provides a foundation for more advanced
automated diagnostic assistance systems.
