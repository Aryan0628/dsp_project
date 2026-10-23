# Lesson 5 - Feature Extraction

> **In one minute**
> A classical model such as an SVM cannot look at a picture. It needs a list of numbers.
> Feature extraction turns each cleaned-up MRI into one list of 171 named measurements.
> These measurements come from seven "families": brightness, texture (two kinds), shape, left-right difference, regional brightness and location.
> The numbers are left raw here. Scaling them happens later, inside the model pipeline (Lesson 6).

## What you'll learn
- What a "feature" and a "feature vector" are, and why we do not feed raw pixels to the classical model.
- The seven feature families this project computes, and what each one measures.
- How a GLCM works, with a small example you can verify by hand.
- How LBP, asymmetry and the other families work, one idea at a time.
- Why the project has exactly 171 features, and where each one comes from.
- Why features must be scaled (the link to Lesson 6).

---

## 1. What is a feature vector?

### Why do we need it?
A classifier such as an SVM cannot look at a picture. It only understands a **fixed-length list of numbers** for every example.

Think of a doctor describing a scan over the phone. They do not read out every pixel. They say "bright blob on the left, uneven texture, not symmetric". Feature extraction is the same job, but the description is written as numbers.

### The words
- A **feature** is one measurement of the image. Example: "average brightness of the brain".
- A **feature vector** is the list of all features for one image.
- The **feature matrix** `X` stacks the vectors of all images. One row per image, one column per feature.
- The **labels** (tumor / no tumor) form a separate vector `y`.

```
  image 1  ──► [ f1, f2, f3, ... f171 ] ─┐
  image 2  ──► [ f1, f2, f3, ... f171 ] ─┤
  image 3  ──► [ f1, f2, f3, ... f171 ] ─┼──► feature matrix X
   ...                                   │    (rows = images,
  image N  ──► [ f1, f2, f3, ... f171 ] ─┘     columns = features)

  labels:  tumor / no tumor, one per image  ──► vector y
```

### Why not just use raw pixels?
- A 256x256 image has 65,536 pixels.
- We only have a few thousand images.
- Many columns and few rows makes it easy for a model to memorise noise instead of learning real patterns.
- Hand-designed features squeeze the image into a few hundred meaningful numbers.
- Every feature has a **name**, so later we can ask "which measurements does the model rely on?"

The deep CNN branch (Lesson 7) does use pixels. This lesson is only about the classical branch.

### What goes in
Feature extraction takes two inputs:

```
  ┌───────────────────────────────┐   ┌─────────────────────────────────┐
  │ Enhanced image                │   │ Segmentation result "seg"       │
  │ (after preprocessing + CLAHE, │   │ (brain mask, candidate tumor    │
  │  Lessons 3-4)                 │   │  region and its properties)     │
  └───────────────┬───────────────┘   └────────────────┬────────────────┘
                  └──────────────┬─────────────────────┘
                                 ▼
                      [ Feature extraction ]
                                 ▼
                      171 named numbers
```

---

## 2. The seven families (171 features in total)

The counts below were checked against the `feature_names` list stored in `backend/outputs/metrics_binary.json`. The prefix counts add up to 171.

| Family | Prefix in name | Count | Question it answers |
|---|---|---|---|
| 1. Intensity statistics + coarse histogram | `int_` (17), `hist_` (16) | 33 | How bright / spread out is the brain tissue? |
| 2. GLCM texture | `glcm_` | 38 | How do neighbouring pixel brightnesses relate? |
| 3. LBP micro-texture | `lbp_` | 28 | What small local patterns appear? |
| 4. Shape / region | `shape_` | 12 | What does the detected suspicious region look like geometrically? |
| 5. Bilateral asymmetry | `asym_` | 10 | Do the left and right halves of the brain disagree? |
| 6. Regional statistics | `reg_` | 20 | Is one part of the brain unusually bright? |
| 7. Location | `loc_` (23), `plane_` (7) | 30 | Where is the region, and which slice orientation is this? |
| **Total** | | **171** | |

Check the sum: 33 + 38 + 28 + 12 + 10 + 20 + 30 = 171.

A quick way to remember the families: **brightness, texture, texture, shape, symmetry, regions, location.**

---

### Family 1 - Intensity statistics (first-order)

**The idea.** Look at the brain tissue pixels as a pile of brightness values. Describe the pile with simple summary numbers.

**Why only brain pixels?** The brain mask is used so that the black background does not swamp everything. If we counted the background, almost every image would have a mean near zero.

**What is measured:**
- **Basic summary:** mean, standard deviation, variance, median, min, max, range.
- **Skewness:** is the pile lopsided? A bright lesion stretches the bright tail, so skewness goes up.
- **Kurtosis:** how heavy are the tails compared with a normal bell curve? More extreme pixels means a larger value.
- **Entropy:** how unpredictable the brightness is. It is Shannon entropy in bits: add up `-p x log2(p)` over all brightness bins, where `p` is the share of pixels in a bin.
- **Upper percentiles:** the 90th, 95th and 99th percentile brightness.
- **Outlier fractions:** the share of pixels more than 2 or 3 standard deviations above the mean. Tumors tend to be hyperintense (brighter than tissue), so these go up when a tumor is present.

**Plus a coarse histogram.** The 16-bin normalised histogram (`HIST_BINS = 16`) splits brightness into 16 bands. Each feature is the share of brain pixels that fall in one band. Together the bands add up to 1.

```
  brain pixels (mask applied)
          │
          ├──► summary numbers (mean, std, median, ...)      ┐
          ├──► shape of the pile (skewness, kurtosis)        │  17 "int_" features
          ├──► unpredictability (entropy)                    │
          ├──► bright tail (90/95/99th percentile,           │
          │     fraction above mean + 2 std and + 3 std)     ┘
          │
          └──► split brightness into 16 bands ──► 16 "hist_" features
```

**What "first-order" means.** Each pixel is looked at **alone**. The arrangement of pixels in space is ignored. If you shuffled all the pixels randomly, these numbers would not change at all. That is exactly why we need the next family.

---

### Family 2 - GLCM texture (second-order)

**The idea.** Texture is about **how pixels sit next to each other**.
- In a smooth region, neighbours have similar brightness.
- In a chaotic region, neighbours are very different.

First-order statistics cannot tell these apart. The **Grey Level Co-occurrence Matrix (GLCM)** can.

**What a GLCM is.** It is a table that counts how often grey level `i` appears right next to grey level `j`. You choose the neighbour with an **offset**: a distance and a direction.

#### Worked example, step by step

Take a tiny 4x4 image with 4 grey levels (0 to 3):

```
0 0 1 1
0 0 1 1
0 2 2 2
2 2 3 3
```

**Step 1 - choose the offset.** Distance 1, direction 0 degrees. This means "look at the neighbour immediately to the right".

**Step 2 - list every horizontal pair.** Each row of 4 pixels gives 3 pairs, so there are 12 pairs in total:

```
  row 1:  (0,0) (0,1) (1,1)
  row 2:  (0,0) (0,1) (1,1)
  row 3:  (0,2) (2,2) (2,2)
  row 4:  (2,2) (2,3) (3,3)
```

**Step 3 - make it symmetric.** The project uses `symmetric=True`. This means every pair is counted in both orders. A pair (0,1) also adds a count for (1,0). So 12 pairs become 24 counts.

**Step 4 - fill the table.** Row `i` is the first grey level, column `j` is the neighbour:

```
        j=0 j=1 j=2 j=3
 i=0     4   2   1   0
 i=1     2   4   0   0
 i=2     1   0   6   1
 i=3     0   0   1   2          (total = 24 counts)
```

Check one cell. The pair (2,2) appears 3 times in the list. Because of symmetry each one counts twice, so the cell (2,2) holds 6.

**Step 5 - turn counts into probabilities.** Divide every cell by 24. This is `normed=True`. The result is `P(i,j)`: the chance that a random neighbour pair is (i, j).

**Step 6 - read off properties.** These values were checked numerically:

| Property | Intuition and formula | Value here |
|---|---|---|
| **Contrast** | Large when neighbours differ a lot. `sum P(i,j) x (i-j)^2`. Big jumps are penalised heavily because the gap is squared. | 0.583 |
| **Homogeneity** | Large when neighbours are similar (mass near the diagonal). `sum P(i,j) / (1+(i-j)^2)`. | 0.808 |
| **Energy** | Large when a few pairings dominate, i.e. a regular, repetitive texture. `sqrt(sum P^2)`. | 0.382 |
| **Correlation** | How linearly related neighbouring grey levels are, from -1 to 1. | 0.720 |

**How to read the result.** Most of the mass sits on the diagonal (pairs of equal or nearly equal levels). So homogeneity is high and contrast is low. This is a fairly smooth image. A noisy checkerboard would push mass far from the diagonal. Contrast would go up, and homogeneity and energy would go down.

```
  smooth image                      noisy checkerboard
  ┌───────────────┐                 ┌───────────────┐
  │ mass near the │                 │ mass far from │
  │ diagonal      │                 │ the diagonal  │
  └───────┬───────┘                 └───────┬───────┘
          ▼                                 ▼
  low contrast, high homogeneity    high contrast, low homogeneity
```

#### What our pipeline does with GLCM (done in `glcm_features`)

```
  Enhanced image (0-255)
          │
          ▼
  [ Quantise 256 grey levels down to 32 ]   (GLCM_LEVELS = 32)
          │       so the table is 32x32 and dense enough to be stable
          ▼
  [ Build GLCMs for 3 distances x 4 angles ]
          │       distances 1, 2, 3
          │       angles 0°, 45°, 90°, 135°
          ▼
  [ Measure 6 properties on each GLCM ]
          │       contrast, dissimilarity, homogeneity,
          │       energy, correlation, ASM
          │       (ASM = angular second moment = energy squared)
          ▼
  [ For each property and each distance, squash the 4 angles into 2 numbers ]
          │       • mean over the 4 angles
          │       • range over the 4 angles (max minus min)
          │       6 properties x 3 distances x 2 = 36 features
          ▼
  [ Add GLCM entropy: mean and std = 2 features ]
          ▼
  38 "glcm_" features
```

**Why average over angles?** A texture can be rotated. Averaging over the four angles makes the description less sensitive to orientation. The "range" feature still records how much the angles disagreed, which hints at directional texture.

---

### Family 3 - LBP (Local Binary Patterns)

**The idea.** Describe the tiny pattern around every pixel, then count which patterns are common.

**Step by step for one pixel:**
1. Pick a centre pixel.
2. Look at `P` neighbours on a circle of radius `R` around it.
3. For each neighbour, write a 1 if it is at least as bright as the centre. Otherwise write 0.
4. The string of bits is a **code** for the local pattern (spot, edge, flat area, corner and so on).

Then repeat for every pixel in the brain and make a **histogram** of the codes. That histogram is the feature.

**Our settings:** `LBP_RADIUS = 3`, `LBP_POINTS = 24`, method `"uniform"`.

**What "uniform" means.**
- Walk once around the circle of bits. Count how many times the bit flips between 0 and 1.
- A pattern with **at most two flips** is called uniform. Example: 000011110000 has two flips. Smooth edges and flat areas look like this.
- Every uniform pattern gets its own bin.
- All the remaining messy patterns share **one** extra bin.

**Counting the bins.** With 24 points, the uniform patterns are counted by how many 1s they contain: 0 ones, 1 one, up to 24 ones. That gives 25 bins. Add the one shared bin for all non-uniform patterns, and you get 26 bins. In the library's usual form this is written `P + 2 = 24 + 2 = 26`.

**The final count.** 26 bin features, plus LBP entropy and LBP uniformity, gives **28** features.

```
  for every brain pixel:
     centre pixel
        │
        ▼
  [ Compare with 24 neighbours on a circle, radius 3 ]
        │   neighbour >= centre → 1,  otherwise → 0
        ▼
  [ Bit string = pattern code ]
        │
        ▼
  [ "Uniform" rule: at most 2 flips between 0 and 1? ]
        ├── yes ──► its own bin    ┐
        └── no  ──► the shared bin ┴─► 26 bins
        ▼
  [ Histogram over all brain pixels ] ──► 26 numbers
        ▼
  [ Add entropy + uniformity ] ──► 28 "lbp_" features
```

**Why it is useful.** LBP only asks "brighter or darker?". So it is unaffected if a whole image is brightened by a monotonic change. That helps when scans come from different scanners.

---

### Family 4 - Shape / region features

**The idea.** Segmentation (Stage 4) finds a candidate bright region. Shape features describe that region's geometry.

**What is recorded (12 features):**
- **Size:** area, and area ratio (the region's area relative to the brain's area).
- **Outline:** perimeter, circularity (how round), solidity (how filled-in compared with its convex outline), extent (how much of its bounding box it fills), aspect ratio (long versus wide).
- **Brightness:** mean intensity, max intensity, and brightness ratio.
- **Counting:** the number of candidate regions, and a flag `shape_found`.

**The `shape_found` flag.** If segmentation found nothing, every shape value is 0 and `shape_found = 0`. This flag matters. It lets the model tell "no region found" apart from "a region of area zero". By itself it is a strong hint that the scan is normal.

```
  seg (from Stage 4)
        │
        ▼
  [ Region found? ]
     ├── no  ──► all shape values = 0,  shape_found = 0
     └── yes ──► [ measure size, outline, brightness, count ]
                  shape_found = 1
```

---

### Family 5 - Bilateral asymmetry

**The idea.** A healthy brain is roughly a mirror image of itself, left to right. A mass usually appears on one side only. So if we flip the image and compare it with the original, a tumor shows up as a difference.

**Everyday analogy.** Hold a photo of a face next to its mirror image. If both halves match, the face is symmetric. A mark on only one cheek stands out.

**Steps (10 features):**

```
  Enhanced image + brain mask
          │
          ▼
  [ Shift the image so the brain's own centroid is in the middle ]
          │     (avoids fake asymmetry from an off-centre scan)
          ▼
  [ Flip it left-to-right ]
          │
          ▼
  [ Absolute difference: |original - flipped| ]
          │
          ▼
  [ Summarise the difference image ]
          │   • mean, std, max
          │   • percentiles
          │   • fraction of pixels differing by more than 30 grey levels
          │   • fraction differing by more than 60 grey levels
          │   • a normalised ratio
          │   • area of the largest connected "disagreeing" blob
          ▼
  10 "asym_" features
```

**Why the brain's centroid?** If we used the image centre instead, a scan that is simply shifted sideways would look asymmetric even for a healthy brain.

**Honest caveat (also stated in the code).** On sagittal slices (side view) a left-right flip is not a true symmetry. So this family is uninformative there.

---

### Family 6 - Regional statistics

**The problem.** Global statistics (Family 1) cannot see "one corner is bright". A small bright patch barely changes the overall mean.

**The fix.** Split the brain into a grid of cells and measure each cell separately.

```
  Brain bounding box
          │
          ▼
  [ Divide into a 3 x 3 grid = 9 cells ]
          │
          ▼
  [ In each cell: mean intensity and std ]   9 cells x 2 = 18 features
          │
          ▼
  [ reg_spread = spread of the 9 cell means ]            1 feature
  [ reg_max_over_mean = brightest cell / average cell ]  1 feature
          ▼
  20 "reg_" features
```

If one cell is much brighter than the others, `reg_spread` and `reg_max_over_mean` both rise.

---

### Family 7 - Location features

**Why they were added.** Texture alone could detect that a tumor exists, but could not name its **type**. Before this family, the project's multiclass accuracy was 72.1% and glioma recall was 24% (measured in the project's results, `backend/outputs/metrics*.json`).

Radiologists use position as a clue:
- **Pituitary tumors** sit near the midline at the skull base.
- **Meningiomas** hug the brain's outer border.
- **Gliomas** sit inside the brain tissue.

**Two groups of features:**

**`loc_*` (23 features): where the region is.** Examples:
- `loc_midline_abs`: distance from the midline.
- `loc_radial`: 0 means the centre, about 1 means at the rim.
- `loc_depth_*`: how deep inside the brain it is (via a distance transform).
- `loc_border_contact`: does it touch the brain's edge?
- Band fractions: how much lies in the superior (top), middle and inferior (bottom) bands.
- Left / centre / right fractions.
- `loc_crosses_midline`: does it straddle the middle?
- `loc_found`: was any region found at all?

The angle of the region is stored as **sine and cosine**, not as a raw number. Reason: +179 and -179 degrees are neighbours on a circle but far apart as numbers. Sine and cosine make them close.

**`plane_*` (7 features): what kind of slice this is.** These describe the shape of the head itself (aspect, fill, coverage, circularity, solidity, offsets). They act as a proxy for slice orientation (axial, sagittal or coronal). This matters because a "central" position means different things in different planes.

**Everything is scale-free.** Positions are divided by the brain's own size (its bounding box or equivalent radius), never by image pixels. So a 512x512 and a 256x256 copy of the same scan give nearly the same values. This is tested in `test_location_is_scale_free`.

```
  Enhanced image + seg
         │
         ▼
  [ Find the brain's size and midline ]
         │
         ▼
  [ Measure the region's position relative to the brain ]
         │     divide by brain size, not by pixels
         ▼
  ┌─────────────────────────┬──────────────────────────┐
  │ loc_*  (23)             │ plane_* (7)              │
  │ midline distance,       │ head shape:              │
  │ radial position, depth, │ aspect, fill, coverage,  │
  │ border contact, bands,  │ circularity, solidity,   │
  │ sin/cos of angle        │ offsets                  │
  └─────────────────────────┴──────────────────────────┘
```

---

## 3. Putting it together, and scaling

### Combining the families
```
  [ Family 1 ] [ Family 2 ] [ Family 3 ] [ Family 4 ] [ Family 5 ] [ Family 6 ] [ Family 7 ]
      (dict)       (dict)       (dict)       (dict)       (dict)       (dict)       (dict)
         └───────────┴───────────┴─────┬──────┴───────────┴───────────┴───────────┘
                                       ▼
                    [ Merge into one named dictionary ]      extract_features
                                       ▼
                    [ Sort the names alphabetically ]        feature_vector
                                       ▼
                    [ Return (values, names) ]
```

**Why sorting matters.** The saved model expects the columns in one fixed order. If training and prediction disagreed on the order, the model would read the wrong number as the wrong feature. The test `test_feature_names_are_stable_and_sorted` guards this.

### Why scaling is needed
Features live on wildly different scales. `shape_area` can be thousands. `hist_03` is a fraction below 1.

Models that use distances or gradients (SVM-RBF, logistic regression, the MLP) would be dominated by the big-number columns. It is like comparing people by "height in millimetres" and "age in years" - the millimetres would drown out everything.

**Standardisation** rescales each column to mean 0 and standard deviation 1: `z = (x - mean) / std`.

In this project scaling is done by `StandardScaler` **inside the model pipeline** (Lesson 6), not here. That way it is fitted on training data only. Feature extraction itself returns raw, unscaled values.

---

## Where this lives in our code

All the feature code is in `backend/src/braintumor/features.py`.

| Function | Family |
|---|---|
| `intensity_features` (image, mask) | 1 - intensity + histogram |
| `glcm_features` (image) | 2 - GLCM texture |
| `lbp_features` (image, mask) | 3 - LBP |
| `shape_features` (seg) | 4 - shape / region |
| `asymmetry_features` (image, mask) | 5 - asymmetry |
| `regional_features` (image, mask, 3x3 grid) | 6 - regional |
| `location_features` (image, seg) | 7 - location + plane |
| `extract_features`, `feature_vector` | merge, then sort |

Settings live in `backend/src/braintumor/config.py`:

| Setting | Value |
|---|---|
| `GLCM_DISTANCES` | 1, 2, 3 |
| `GLCM_ANGLES_DEG` | 0, 45, 90, 135 |
| `GLCM_LEVELS` | 32 |
| `LBP_RADIUS` | 3 |
| `LBP_POINTS` | 24 |
| `HIST_BINS` | 16 |

The whole path for one image:

```
  enhanced image + seg
          │
          ▼
  ┌───────────────────────────────────────────────┐
  │ features.py: seven family functions           │
  └──────────────────────┬────────────────────────┘
                         ▼
  [ extract_features: merge into one named dictionary ]
                         ▼
  [ feature_vector: sort names, make a number array ]
                         ▼
  one row of 171 numbers
```

The loop over a whole dataset is `build_feature_matrix` in `backend/src/braintumor/dataset.py`. It runs the path above for every image and stacks the rows into `X`.

Tests for the location family are in `backend/tests/test_location_features.py`. For example, moving a synthetic lesion sideways must increase `loc_midline_abs`, and moving it toward the rim must increase `loc_radial`.

---

## Common viva questions

**Q1. What is a feature vector and why do you need one?**
A fixed-length list of numbers describing one image. Classical ML models need tabular numeric input. An image is converted into measurements such as brightness statistics, texture and shape.

**Q2. Why use first-order statistics and GLCM? Is one not enough?**
First-order statistics ignore pixel positions. Two images with the same pixels shuffled give identical values. GLCM is second-order: it captures how neighbouring pixels relate, so it sees texture (smooth versus heterogeneous). They are complementary.

**Q3. Explain contrast and homogeneity in a GLCM.**
Both are weighted sums over the normalised matrix. Contrast weights each pair by `(i-j)^2`, so it is high when neighbours differ strongly. Homogeneity weights by `1/(1+(i-j)^2)`, so it is high when neighbours are similar. They move in opposite directions.

**Q4. Why quantise to 32 grey levels before the GLCM?**
A 256-level GLCM is 256x256 = 65,536 cells, mostly empty and noisy for a small image. With 32 levels it is 1,024 cells. That is denser, more stable, and much cheaper (64 times fewer cells).

**Q5. What is LBP and why is it useful?**
Each pixel is compared with neighbours on a circle, producing a binary code. The histogram of codes describes micro-texture. It depends only on brighter/darker comparisons, so it is robust to brightness shifts between scanners. We use the "uniform" variant to reduce the number of bins to 26.

**Q6. Why add location features, and how do you know they work?**
Texture features detect tumors but could not distinguish types (72.1% multiclass accuracy and 24% glioma recall before). Location is how radiologists distinguish them. Unit tests verify the direction of response (for example, midline distance grows when a lesion moves sideways). Be honest in the viva: on their own, they did not improve the classical typing accuracy. The CNN branch gave the big typing gain. They did raise binary ROC-AUC from 0.9938 to 0.9968 (measured in the project's results, `backend/outputs/metrics*.json`).

**Q7. How many features, and why do names matter?**
171. Named features let us report which measurements drive the model. They also keep the column order consistent between training and prediction.

---

## Check yourself

1. Why are intensity statistics computed only inside the brain mask?
<details><summary>Answer</summary>The black background would dominate the pixels and make mean, variance and the histogram meaningless. The mask restricts them to tissue.</details>

2. A region of an MRI is very smooth. Roughly how would its GLCM contrast and homogeneity compare with a noisy region?
<details><summary>Answer</summary>Lower contrast and higher homogeneity (and typically higher energy), because neighbouring pixels have similar grey levels, so probability mass sits near the diagonal.</details>

3. How is the `glcm_` count of 38 obtained?
<details><summary>Answer</summary>6 properties x 3 distances x 2 statistics (mean and range over the 4 angles) = 36, plus 2 for GLCM entropy (mean and std) = 38.</details>

4. Why is the angle stored as sine and cosine?
<details><summary>Answer</summary>Raw angles wrap around: +179 degrees and -179 degrees are neighbours but numerically far apart. Sine and cosine make them close.</details>

5. Why do we need `shape_found` / `loc_found` flags?
<details><summary>Answer</summary>When no region is detected the other values are set to 0. The flag lets the model distinguish "nothing found" from a real measurement that happens to be 0 (such as a lesion at the exact centre).</details>

6. In the worked GLCM example, why does the table total 24 and not 12?
<details><summary>Answer</summary>There are 12 horizontal neighbour pairs. Because the matrix is symmetric, each pair is counted in both orders, which doubles the total to 24.</details>

**Next:** [Lesson 6 - Classical ML Training](06_CLASSICAL_ML_TRAINING.md)
