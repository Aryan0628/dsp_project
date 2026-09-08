# 04 — TECHNICAL DEEP DIVE
### The maths and the reasoning behind every stage. Read this if you want to answer *any* question.

---

## STAGE 2 — PREPROCESSING

### Grayscale conversion
MRI intensity encodes tissue proton density and relaxation times, not colour. A stored
3-channel file has three identical channels, so we drop to one and cut memory and compute
by 3× with no information loss. OpenCV's `BGR2GRAY` uses the luminance weighting
`Y = 0.299R + 0.587G + 0.114B`.

### Resizing to 256×256
Every image must be the same size or the feature vector length changes between samples.
We use `INTER_AREA` when shrinking — it computes a weighted average over the source pixel
area, which acts as a low-pass filter and prevents aliasing. Using `INTER_NEAREST` or
`INTER_LINEAR` when downscaling folds high-frequency noise into false low-frequency
structure, which would corrupt the texture features.

### Median filter
Replaces each pixel with the median of its 3×3 neighbourhood. The median is a **rank
statistic**, so an extreme outlier occupies one slot in the sorted list and is discarded.
A mean filter would instead *average the outlier in*, smearing it across the neighbourhood.
This is why median is the standard choice for salt-and-pepper noise.

### Bilateral filter
A Gaussian blur computes

```
    I'(p) = (1/W) * SUM_q  G_s(||p - q||) * I(q)
```

which weights neighbours purely by distance — so it blurs across edges. The bilateral
filter adds a second, intensity-based weight:

```
    I'(p) = (1/W) * SUM_q  G_s(||p - q||) * G_r(|I(p) - I(q)|) * I(q)
```

`G_r` is small when the neighbour's intensity differs a lot from the centre, so pixels on
the far side of an edge contribute almost nothing. Result: noise inside a homogeneous
region is averaged away, while the tumor boundary stays crisp. We use
`sigmaColor = sigmaSpace = 50`, `d = 7`.

### Min-max normalisation
`I' = 255 * (I - min) / (max - min)`. Puts every scan on the same brightness scale so a
dim scan and a bright scan of the same anatomy produce comparable features.

---

## STAGE 3 — ENHANCEMENT

### Why global histogram equalisation fails here
Histogram equalisation maps intensities through the cumulative distribution function:
`s = T(r) = (L-1) * CDF(r)`. This spreads intensities to occupy the full range and
maximises global entropy. But it uses **one mapping for the entire image**. In a brain MRI
the skull and scalp fat are very bright and occupy a large fraction of the histogram mass,
so the transform allocates most of the output range to them. The subtle grey-on-grey
contrast between tumor and healthy parenchyma — the thing we actually care about — is
compressed rather than expanded. It also amplifies background noise into visible texture.

### CLAHE
Three ideas stacked:

1. **Adaptive (AHE).** Split the image into a grid of tiles (we use 8×8, so 32×32 px tiles
   at our resolution) and compute a separate equalisation mapping for each tile from that
   tile's own histogram. Local contrast now improves everywhere, independent of the global
   distribution.
2. **Contrast limiting.** Plain AHE over-amplifies noise: in a near-uniform tile the
   histogram is a tall narrow spike, so the CDF is very steep and tiny intensity differences
   (i.e. noise) get stretched enormously. CLAHE clips each tile histogram at
   `clipLimit` (we use 2.0) *before* computing the CDF, and redistributes the clipped mass
   uniformly across all bins. This bounds the maximum slope of the transform, and therefore
   bounds noise amplification.
3. **Bilinear interpolation.** Applying each tile's mapping only within its own tile would
   produce visible blocky seams. Instead each pixel's output is bilinearly interpolated
   between the mappings of the four nearest tile centres, giving a smooth result.

### Unsharp masking
Counter-intuitive name: you sharpen by *subtracting a blur*.

```
    blurred = GaussianBlur(I, sigma)          # low-frequency content
    detail  = I - blurred                     # high-frequency content: edges, texture
    result  = I + amount * detail
            = (1 + amount) * I - amount * blurred
```

The second form is what we implement with `cv2.addWeighted`, which is a single pass.
`sigma = 1.5` selects the spatial scale of the detail to boost; `amount = 1.2` sets the
strength. Too high and you get halo artefacts around strong edges plus amplified noise.

### Measured evidence that CLAHE beats global HE

This is not just a visual claim — `outputs/demo_enhancement.png` measures it on the same scan:

| method | contrast (std) | what the image looks like |
|---|---|---|
| Preprocessed input | 73.2 | baseline |
| **Global histogram equalisation** | **68.2** | background noise amplified into visible chaos; brain interior washed out toward white |
| CLAHE | 76.9 | local contrast improved, background left alone |
| CLAHE + unsharp mask | **81.6** | final output |

The striking result is that **global HE actually *reduces* contrast here (73.2 → 68.2)**
while making the image far worse to look at. That is the skull-domination effect made
quantitative: HE spends the output range on the bright skull and the background noise, and
compresses the mid-grey tissue range where the diagnostic information lives.

### Measuring the improvement objectively
We do not just claim "it looks better". `enhancement_report()` computes:
- **contrast** = standard deviation of intensities (wider tonal spread)
- **entropy** = `-SUM p_i log2(p_i)` over the normalised histogram, in bits (more
  information content)
- **sharpness** = variance of the Laplacian, the standard no-reference focus measure

Typical measured result on our data: contrast 73.7 → 81.4, entropy 6.16 → 7.08 bits,
sharpness 272 → 1100.

---

## STAGE 4 — SEGMENTATION

### The core difficulty
The skull is the brightest thing in the image — brighter than any tumor. Any
"find the bright region" algorithm returns the skull. So the skull must go first.

### Skull stripping, step by step
1. **Otsu threshold** the whole image to separate head from background.
2. **Morphological closing** with a 7×7 ellipse, 3 iterations, to seal small gaps.
3. **Largest connected component** — `connectedComponentsWithStats` with 8-connectivity;
   label 0 is background, so we take the largest label from 1 onward. This discards
   scanner artefacts and isolated noise blobs.
4. **Hole filling by flood fill.** Flood the background from a corner, invert to obtain the
   interior holes, OR them back into the mask.
5. **Distance-transform erosion** to pull the mask inside the skull rim (see below).

### Bug #1 we hit and fixed: the flood-fill split-background bug
The head ellipse often touches the top and bottom image borders. When it does, the
background is **split into two disconnected regions** (left and right of the head). A
flood-fill from corner (0,0) reaches only the left one, so the entire right background is
misclassified as an "interior hole" and gets filled in. The mask became a huge crescent
covering half the frame.

**Fix:** pad a 1-pixel black frame around the image with `copyMakeBorder` before flood
filling. The frame connects all background regions into one, so a single flood-fill is
always sufficient. Then crop the frame off. `tests/test_pipeline.py::test_brain_mask_is_one_solid_region`
is the regression test.

### Why a distance transform instead of a fixed erosion kernel
Our first version eroded with a fixed 9×9 kernel. That is wrong because heads vary in size:
a kernel tuned for a large head removes too little skull from it and destroys a small one.

The **distance transform** labels every foreground pixel with its Euclidean distance to the
nearest background pixel. Keeping only pixels with
`distance > 0.22 * max_distance` scales automatically with head size. The skull plus the CSF
gap is roughly 15% of the head radius, so 0.22 clears it with margin. Measured effect: brain
mask area went from wildly varying (23k–40k px) to consistent (21k–26k px), and the retained
maximum intensity dropped from 255 (skull included) to ~230 (brain only).

### Bug #2 we hit and fixed: Otsu flags every healthy brain
Otsu's method finds the threshold `t` minimising the intra-class variance

```
    sigma_w^2(t) = w0(t) * sigma0^2(t) + w1(t) * sigma1^2(t)
```

equivalently maximising the between-class variance. It is elegant and parameter-free, and
it is the textbook answer — but it has a fatal property for this task: **it always returns a
split, even when the image contains nothing abnormal.** On a healthy brain it simply
separates grey matter from white matter and reports a large "lesion".

Measured over 60 scans: Otsu flagged **100% of healthy brains**.

**Fix — a tissue-relative statistical threshold.** A pixel is suspicious when it is
statistically anomalous *for that patient's own scan*:

```
    tissue    = brain_pixels[brain_pixels > median(brain_pixels)]   # excludes dark CSF
    threshold = mean(tissue) + 1.5 * std(tissue)
```

The CSF exclusion matters enormously: the lateral ventricles are very dark, and including
them inflates `std` so much that `mean + k*sigma` exceeded 255 — meaning **nothing was ever
detected**. That was the first symptom we saw. Restricting to pixels above the brain median
drops CSF while keeping grey and white matter.

Measured comparison (60 scans, area > 200 px counted as a detection):

| threshold rule | tumors detected | healthy falsely flagged |
|---|---|---|
| `mean + 2σ` (all brain pixels) | 10% | 0% |
| `median + 2.5·MAD` | 17% | 0% |
| Otsu within tissue | 100% | **100%** |
| 97th percentile | 83% | 0% |
| `tissue mean + 2σ` | 90% | 0% |
| **`tissue mean + 1.5σ`  ← chosen** | **100%** | **3%** |

Because the reference is the patient's own tissue distribution, the rule is
self-calibrating and transfers across scanners without retuning.

### Morphological cleanup
Both operations use a 5×5 elliptical structuring element.

- **Opening** (erode → dilate, 2 iterations) removes bright specks smaller than the kernel.
  Erosion deletes them; the following dilation restores the size of surviving large objects.
- **Closing** (dilate → erode, 3 iterations) fills small dark holes inside the blob so the
  contour traces one clean outline instead of a ring.

Order matters: opening first (remove noise), then closing (consolidate the signal). Doing it
the other way round would first grow the noise specks into solid blobs.

### Contour detection and candidate selection
`findContours` with `RETR_EXTERNAL` (outermost boundaries only — we do not care about holes
as separate objects) and `CHAIN_APPROX_SIMPLE` (compresses straight runs into endpoints).

Candidates are filtered by two plausibility rules:
- area ≥ 120 px — smaller is noise
- bounding-box aspect ratio ≤ 5.0 — a mass is a blob, not a streak. This kills the thin
  crescent slivers produced by partial-volume effects at the brain-mask boundary.

We also apply a **boundary guard**: candidates are detected inside a mask eroded by a
further 7 px relative to the mask used for statistics, because right at the boundary bright
skull-adjacent tissue blends into the brain and generates false detections.

The winning candidate maximises `area × (region mean intensity / brain mean intensity)` —
a large *and* bright region outranks a large dim one.

### Region properties
| property | formula | why it discriminates |
|---|---|---|
| area | pixel count | tumor burden |
| perimeter | `arcLength` | boundary length |
| circularity | `4πA / P²` | 1.0 for a circle; tumors are irregular so this drops |
| solidity | `A / A_convexhull` | measures concavity / raggedness |
| extent | `A / (w·h)` | how well the shape fills its bounding box |
| aspect ratio | `w / h` | elongation |
| brightness ratio | region mean / brain mean | degree of hyperintensity |

---

## STAGE 5 — FEATURE EXTRACTION (111 features)

### Why engineered features rather than raw pixels
A 256×256 image is 65,536 raw values. With a few hundred training samples that is a
catastrophic curse-of-dimensionality problem — the model would memorise rather than
generalise. Engineered features compress the image into ~111 *meaningful* numbers, which is
a tractable ratio, and each one is interpretable so we can explain the decision.

### Family 1 — first-order intensity statistics (34 features)
Computed over brain pixels only (including the black background would swamp everything with
zeros). Mean, variance, std, min, max, median, range, RMS, energy.

- **Skewness** `E[(x-µ)³]/σ³` — asymmetry. A bright lesion pulls the right tail out, so
  skewness rises.
- **Kurtosis** `E[(x-µ)⁴]/σ⁴` — tail weight / peakedness.
- **Entropy** — histogram information content.
- **p90 / p95 / p99** — how bright the brightest tissue is. `int_p99` turned out to be the
  single most influential feature in the trained model, which makes sense: tumors are
  hyperintense by definition.
- **bright_frac / verybright_frac** — fraction of brain above µ+2σ and µ+3σ.
- **16-bin normalised histogram** — a compact shape descriptor of the whole distribution.

### Family 2 — GLCM / Haralick texture (38 features)
The Grey Level Co-occurrence Matrix `P(i,j | d, θ)` counts how often a pixel of grey level
`i` occurs at offset `(d, θ)` from a pixel of grey level `j`. It is a **second-order**
statistic: it describes spatial relationships that mean and variance cannot see.

We quantise to 32 grey levels rather than 256. Reason: a 256×256 GLCM from a 256×256 image
has 65,536 cells for 65,536 pixel pairs, so it is almost all zeros and the statistics are
noise. A 32×32 matrix is dense and stable, and it is 64× cheaper.

Distances 1, 2, 3 and angles 0°, 45°, 90°, 135°, averaged across angles for rotation
robustness. Properties computed:

| property | formula | meaning |
|---|---|---|
| contrast | `Σ P(i,j)·(i-j)²` | local intensity variation — high for rough texture |
| dissimilarity | `Σ P(i,j)·|i-j|` | like contrast but linear, less outlier-sensitive |
| homogeneity | `Σ P(i,j)/(1+(i-j)²)` | closeness to the diagonal — high for smooth texture |
| energy / ASM | `Σ P(i,j)²` | uniformity — high when few pairings dominate |
| correlation | linear dependency of `i` and `j` | how predictable a neighbour is |
| entropy | `-Σ P·log₂P` | texture randomness |

Clinically: tumor tissue is heterogeneous and chaotic (high contrast, high entropy, low
homogeneity); healthy white matter is smooth and regular (the opposite).

### Family 3 — Local Binary Patterns (28 features)
For each pixel, compare it to `P = 24` neighbours on a circle of radius `R = 3`. Each
comparison yields one bit (1 if the neighbour is brighter), producing an integer code. The
normalised histogram of codes is the descriptor.

We use the **uniform** variant: patterns with at most two 0↔1 transitions around the circle
get their own bin, and all others collapse into a single bin. This reduces `2²⁴` possible
codes to 26 bins, which is statistically stable, and uniform patterns correspond to
meaningful primitives — flat regions, edges, corners, spots.

Key property: LBP compares pixels to *each other*, never to an absolute value, so it is
invariant to any monotonic grey-level transformation. That means it survives brightness and
gain differences between scanners — exactly the domain-shift problem that hurts raw-pixel
models.

### Family 4 — shape / region descriptors (12 features)
The Stage-4 measurements, plus `shape_found` (whether any region was detected at all) and
`shape_n_candidates`. When nothing is detected every value is 0, which is itself a strong
signal of a normal scan.

---

## STAGE 6 — CLASSIFICATION

### The pipeline pattern and data leakage
Every candidate is `StandardScaler → classifier` wrapped in a `sklearn.pipeline.Pipeline`.
`StandardScaler` computes `z = (x - µ)/σ` per feature, which matters because our features
span wildly different scales — `shape_area` is in the thousands, `glcm_homogeneity` is in
[0,1]. Distance-based and gradient-based methods would otherwise be dominated by the
large-magnitude features.

Wrapping it in a Pipeline is not cosmetic. During cross-validation the scaler is re-fitted
on **each training fold only**, so the validation fold's statistics never influence
training. Scaling the whole dataset up front — the common shortcut — leaks test information
into training and inflates the reported score. It is one of the most frequent and most
invisible mistakes in student ML projects.

### The four candidates
| model | what it does | why it is in the comparison |
|---|---|---|
| Logistic Regression | linear decision boundary on the scaled features | the **baseline** — if a linear model already wins, complexity is unjustified. Also gives signed coefficients for interpretation. |
| SVM (RBF kernel) | maximum-margin hyperplane in an implicit high-dimensional space via `K(x,x') = exp(-γ‖x-x'‖²)` | strong on moderate-dimensional, medium-sized datasets; `C` trades margin width against training error |
| Random Forest | 300 trees on bootstrap samples with random feature subsets, majority vote | captures non-linear interactions, resists overfitting by averaging, and yields **feature importances** |
| MLP (128, 64) | a small feed-forward neural network with ReLU | shows the pipeline is not tied to one algorithm family; this is the "AI model" in the classical sense |

### Model selection
5-fold **stratified** cross-validation (stratified = each fold preserves the class ratio),
scored by ROC-AUC with accuracy as a tiebreak. ROC-AUC is threshold-independent: it measures
the probability that a randomly chosen tumor case is ranked above a randomly chosen healthy
one, so it evaluates the model's *ranking quality* rather than its behaviour at one
arbitrary cut-off. The winner is refitted on the full training set and evaluated once on the
held-out test set.

### Evaluation metrics and why accuracy is not enough
From the confusion matrix (TN, FP, FN, TP):

```
    accuracy    = (TP + TN) / total
    sensitivity = TP / (TP + FN)      "of all real tumors, how many did we catch"
    specificity = TN / (TN + FP)      "of all healthy patients, how many did we clear"
    precision   = TP / (TP + FP)      "when we say tumor, how often are we right"
    F1          = 2·precision·recall / (precision + recall)
```

**Sensitivity is the headline metric.** A false negative sends a tumor patient home
untreated; a false positive costs one extra review. Those errors are not remotely equal in
cost, and accuracy averages over both as though they were. This is also why a 90%-accurate
model on a 90%-healthy dataset can be completely worthless — it may simply be predicting
"healthy" every time.

---

## STAGE 7 — VISUALISATION

The 8-panel figure shows every intermediate stage, so the system's reasoning is auditable.
The histogram panel gives objective before/after evidence for the enhancement claim. The
verdict card carries the label, the confidence, a probability bar, the measured region
properties, and — always — the disclaimer that this is assistive, not diagnostic.

---

## REPRODUCIBILITY

Fixed seed (42) throughout: dataset generation, train/test split, CV folds, and every model.
Running `python main.py train` twice gives byte-identical results. The trained model,
feature names and metrics are all persisted, and `predict.py` verifies at inference time
that the feature names match what the model was trained on — so a change to the feature code
fails loudly instead of silently predicting from misaligned inputs.
