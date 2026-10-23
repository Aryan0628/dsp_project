# Lesson 10 — Revision Sheet, Glossary & Final Quiz

## What you'll learn
- The whole project on one page, in the order you would explain it.
- The numbers you must know by heart.
- Plain definitions of every term an examiner might use.

---

## 1. The project on one page

1. **Input:** a brain MRI slice (JPG). It is turned into a 2-D grid of grey values (0–255).
2. **Preprocess** (`preprocess.py`): grayscale → crop → resize to 256×256 → **median filter**
   (removes salt-and-pepper noise) → **bilateral filter** (smooths but keeps edges) → normalise.
3. **Enhance** (`enhance.py`): **CLAHE** (local contrast, clip limit 2.0, 8×8 tiles) →
   **unsharp masking** (sharpen: original + 1.2 × (original − blurred)).
4. **Segment** (`segment.py`): **Otsu** separates the head from the black background; skull
   stripping keeps the brain. Suspicious pixels are those **brighter than mean + 1.5σ** of
   normal tissue, or brighter than their **mirror-image side** by 2σ. Then **morphology**
   (opening removes specks, closing fills holes) → **contours** → reject anything too small
   (< 120 px) or too thin (aspect ratio > 5).
5. **Features** (`features.py`): **171 numbers** per scan: intensity statistics, histogram,
   GLCM texture, LBP texture, shape, location, and more (lesson 05).
6. **Classify — detection** (`model.py`, `train.py`): four candidates (Logistic Regression,
   SVM-RBF, Random Forest, MLP), each inside a `StandardScaler` pipeline, compared with
   5-fold stratified cross-validation. **SVM-RBF wins.** Decision threshold tuned to **0.153**
   on out-of-fold training predictions (never on the test set).
7. **Classify — typing** (`deep.py`): an ImageNet-pretrained **ResNet-18** fine-tuned for
   4 classes (glioma / meningioma / pituitary / no tumor). It is blended with the classical
   model, with weight 0.95 on the CNN.
8. **Output** (`predict.py`, `api.py`, `frontend/`): verdict + probabilities + measurements +
   a picture of every stage. The output comes from a FastAPI server (port 3000) and is shown
   in a Next.js app (port 5713).

**Why classical + ML?** The image-processing half is deterministic and **explainable**:
you can see the region it found and its measurements. The ML half learns patterns that
are too subtle to hand-code.

---

## 2. Numbers to memorise (measured on the held-out test set)

| | value |
|---|---|
| Dataset | 3264 images; held-out test set of 394 (289 tumor, 105 normal) |
| Detection accuracy | **95.43%** |
| Sensitivity (recall) | **93.77%**: 18 tumors missed out of 289 |
| Specificity | **100%**: 0 false alarms out of 105 |
| ROC-AUC | **0.9968** |
| Detection model | SVM-RBF on 171 features, threshold 0.153 |
| Typing accuracy (4-class) | **80.20%** (balanced accuracy 79.48%) |
| Weakest class | glioma (recall ≈ 35%), so be ready to discuss this honestly |

---

## 3. Glossary

| Term | Meaning |
|---|---|
| **Pixel / intensity** | one cell of the image grid; here a grey value 0 (black) to 255 (white) |
| **DSP / image processing** | fixed mathematical operations on the signal (the image); nothing is learned |
| **Kernel / filter** | a small grid of numbers slid over the image |
| **Convolution** | at each position, multiply the kernel by the pixels under it and sum (a *linear* filter) |
| **Median filter** | replace each pixel by the median of its neighbours; non-linear; removes salt-and-pepper noise |
| **Bilateral filter** | averages neighbours that are close in *both* position and brightness, so edges survive |
| **Histogram equalisation** | spreads grey levels across the full range to increase contrast (whole image at once) |
| **CLAHE** | equalisation done per tile, with a clip limit so noise is not over-amplified |
| **Unsharp masking** | sharpening: add back the difference between the image and a blurred copy |
| **Thresholding** | turn a grey image into black/white by a cut-off value |
| **Otsu's method** | picks the threshold automatically by minimising within-class variance |
| **Morphology** | shape operations on binary masks: erosion, dilation, opening, closing |
| **Contour** | the outline of a connected white region |
| **Segmentation** | separating the region of interest (the tumor candidate) from everything else |
| **Feature / feature vector** | a number that describes the image; the list of all of them (171 here) |
| **GLCM** | Grey-Level Co-occurrence Matrix: counts how often grey level *i* sits next to level *j*; gives texture measures |
| **LBP** | Local Binary Pattern: encodes whether each neighbour is brighter than the centre; a texture code |
| **Supervised learning** | learning from examples that already have the correct label |
| **Classifier** | a model that outputs a class (tumor / no tumor) |
| **Binary vs multiclass** | 2 possible answers vs more than 2 (here 4) |
| **Training / test set** | data the model learns from vs unseen data used only to measure it |
| **Overfitting** | memorising the training data instead of learning general patterns |
| **Cross-validation (k-fold)** | split training data into k parts and train/validate k times; gives a fairer estimate |
| **Stratified** | each fold keeps the same class proportions as the whole set |
| **StandardScaler** | rescales each feature to mean 0, standard deviation 1 |
| **SVM (RBF kernel)** | finds the widest-margin boundary between classes; the RBF kernel lets it be curved |
| **Random Forest** | many decision trees voting together |
| **Logistic Regression** | a linear model that outputs a probability |
| **MLP** | a small fully-connected neural network |
| **CNN** | neural network whose layers are *learned* convolution filters |
| **ResNet-18** | an 18-layer CNN with skip connections; ours starts from ImageNet weights |
| **Transfer learning** | reuse a network trained on one task (ImageNet) and fine-tune it for ours |
| **Data augmentation** | random flips/rotations etc. during training so the model generalises |
| **Epoch / batch** | one full pass over the training data / the group of images per update step |
| **Softmax** | turns raw scores into probabilities that sum to 1 |
| **Cross-entropy loss** | penalty that grows when the model is confidently wrong |
| **Decision threshold** | probability above which we say "tumor" (ours is 0.153, not 0.5) |
| **Confusion matrix** | table of actual vs predicted classes (TP, TN, FP, FN) |
| **Accuracy** | (TP+TN) / all |
| **Sensitivity / recall** | TP / (TP+FN): how many real tumors we caught |
| **Specificity** | TN / (TN+FP): how many healthy scans we correctly cleared |
| **Precision** | TP / (TP+FP): when we say tumor, how often we're right |
| **F1** | harmonic mean of precision and recall |
| **ROC-AUC** | how well the model *ranks* tumors above non-tumors across all thresholds (1.0 = perfect) |
| **Explainable AI** | the system shows *why* it decided, not just the decision |
| **API** | the web endpoint (`POST /analyze`) the frontend sends the image to |

---

## 4. Final quiz

1. Why do we use a median filter *before* the bilateral filter?
2. Why CLAHE rather than ordinary histogram equalisation?
3. Otsu is used for the head mask but not for finding the tumor. Why?
4. What is the difference between opening and closing?
5. Name four feature families and say what each one captures.
6. Why put `StandardScaler` inside the pipeline instead of scaling all data first?
7. Why is the threshold 0.153 and not 0.5? On which data was it chosen?
8. In medicine, which error is worse, FN or FP? Which metric measures it?
9. Our specificity is 100%. Does that mean the model is perfect? Why not?
10. Why use a pretrained ResNet-18 instead of training a CNN from scratch?
11. What does ROC-AUC = 0.997 tell you that accuracy does not?
12. Trace one uploaded image from the browser to the verdict on screen.

<details>
<summary>Answers</summary>

1. The median filter removes isolated extreme pixels (salt-and-pepper noise). If the bilateral
   filter ran first, it would treat those specks as "edges" and preserve them.
2. Global equalisation uses one mapping for the whole image, so it can wash out local detail
   and amplify noise. CLAHE equalises each tile separately and clips the histogram, which
   limits noise amplification.
3. Otsu *always* splits the image into two classes. Head vs background really has two groups.
   A healthy brain has no lesion group, so Otsu would still "find" one. On 60 scans it flagged
   100% of healthy brains. The mean + kσ rule flagged 3%.
4. Opening = erosion then dilation: removes small bright specks. Closing = dilation then
   erosion: fills small holes and gaps inside a blob.
5. Examples: intensity statistics (brightness, spread), GLCM (texture: contrast, homogeneity),
   LBP (micro-texture patterns), shape (area, solidity, circularity of the region), location
   (where the region is).
6. So that during cross-validation the scaler learns its mean and standard deviation only from
   the training folds. Otherwise information leaks from the validation data.
7. Missing a tumor is worse than a false alarm, so the threshold was tuned for maximum
   sensitivity while keeping specificity ≥ 80%. It was chosen on **out-of-fold training
   predictions**; the test set was never used.
8. The false negative (a missed tumor). Sensitivity / recall measures it.
9. No. 18 of 289 tumors were still missed (sensitivity 93.77%), and the test set is only
   394 images, so one more error moves the numbers noticeably.
10. We have only a few thousand images. ImageNet weights already detect edges, textures and
    shapes, so fine-tuning needs far less data and generalises better.
11. AUC is threshold-independent. It measures how well the model ranks tumor scans above
    healthy ones overall. Accuracy depends on the one threshold chosen.
12. Browser upload → `POST /analyze` (FastAPI) → `predict.py`: preprocess → enhance → segment →
    features → SVM (+ ResNet for typing) → JSON (verdict, probabilities, measurements, stage
    images) → React components (VerdictCard, ProbabilityBars, StageGallery, MeasurementPanel).

</details>

---

**You've finished the course.** Go back to [00 — Course Map](00_COURSE_MAP.md) and re-read any lesson whose quiz you found hard.