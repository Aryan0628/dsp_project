# Lesson 3 - Preprocessing and Enhancement

> **In one minute**
> Brain MRI images arrive in different sizes, brightness levels and noise levels, so we first clean and standardise them.
> Stage 2 (preprocessing) converts to grey, trims borders, resizes to 256 x 256, removes noise with a median filter and a bilateral filter, and stretches brightness to 0-255.
> Stage 3 (enhancement) then makes the tumor easier to see: CLAHE boosts local contrast and unsharp masking sharpens edges.
> The order matters: we always clean noise first, because contrast boosting would also boost the noise.

## What you'll learn

- Why every MRI is "standardised" before analysis, and the exact order our code does it in.
- What an image is to a computer, and what **noise** looks like (Gaussian, salt-and-pepper, Rician).
- What a **kernel** and **convolution** are, with a worked 3x3 example.
- How the **median filter** and **bilateral filter** remove noise (and why neither is a plain blur).
- **Histogram equalisation vs CLAHE** (clip limit, tiles) and **unsharp masking**.
- Which files do each job, and why the parameters in `config.py` have the values they do.

---

## 1. The big picture: two stages, fixed order

**Why do we need this at all?** Imagine grading exam papers where some are handwritten in pencil on tiny paper and others are typed on huge sheets. You would first photocopy them all to the same size and darkness. Preprocessing is that photocopy step. Without it, the classifier could learn "which scanner made this picture" instead of "is there a tumor".

Our pipeline runs two stages one after the other, then hands the result to segmentation (the stage that outlines the tumor).

```
 original MRI
      │
      ▼
┌───────────────────────────────┐
│ STAGE 2: preprocess()         │
│ (file: preprocess.py)         │
│                               │
│ 1. make it one-channel grey   │
│ 2. make values 0-255 (uint8)  │
│ 3. trim uniform borders       │
│ 4. resize to 256 x 256        │
│ 5. denoise (median, then      │
│    bilateral)                 │
│ 6. stretch brightness 0-255   │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ STAGE 3: enhance()            │
│ (file: enhance.py)            │
│                               │
│ 1. CLAHE (local contrast)     │
│ 2. unsharp mask (sharpen)     │
│ 3. gamma (optional; default   │
│    1.0 = do nothing)          │
└───────────────┬───────────────┘
                ▼
        segment_tumor()  (Lesson 4)
```

The three calls are chained in `predict.py`: preprocess first, then enhance, then segment.

**Why this order?** Contrast enhancement amplifies whatever is in the image, including noise. If you boosted contrast first, you would make the noise louder and then ask a filter to clean up a bigger mess. So: clean first, enhance second.

---

## 2. What is a digital image?

A greyscale image is a 2D grid of numbers (a NumPy array). Each number is one **pixel**. In our code each pixel is an integer from 0 (black) to 255 (white). This type is called `uint8`, meaning "unsigned 8-bit integer". A 256 x 256 image is 65,536 numbers. Everything in this lesson is arithmetic on that grid.

### Step: grayscale

MRI measures tissue properties, not colour. So one number per pixel holds all the information. But files such as JPEGs are often saved with 3 colour channels. OpenCV (the image library we use) loads colour as **B, G, R** order, not RGB. The function `to_grayscale` handles three cases:

```
        input image
             │
             ▼
   ┌─────────────────────┐
   │ Already 2D (1 chan)?│── yes ──► return unchanged
   └──────────┬──────────┘
              no
              ▼
   ┌─────────────────────┐
   │ Has 4 channels?     │── yes ──► convert BGRA ──► grey
   └──────────┬──────────┘
              no
              ▼
        convert BGR ──► grey
```

### Step: resize to 256 x 256

Different scanners give different sizes. A classifier needs every input to be the same size, so every image becomes `IMAGE_SIZE = (256, 256)` (width, height).

How should we invent or discard pixels? The code (`resize`) picks the method by checking whether we are shrinking or enlarging:

```
        image height > 256 ?
           │            │
          yes           no
           ▼            ▼
   shrinking:        enlarging:
   INTER_AREA        INTER_CUBIC
```

- `INTER_AREA` (shrinking) averages the source pixels that fall inside each output pixel. This avoids **aliasing**, which is when fine texture gets misrepresented as false patterns.
- `INTER_CUBIC` (enlarging) fits smooth curves between neighbouring pixels to invent plausible in-between values.

A precise note: the code only compares heights. That is a simplification, and it is fine for the roughly square images in our datasets.

### Step: autocrop the border

Real-world images often come with a solid white page frame (from a scanned textbook figure) or black letterbox bars. These frames confuse later steps. Before this step existed, a white frame broke skull stripping on about 3% of the real dataset.

The idea: real anatomy varies from pixel to pixel, but padding is flat. So a row or column whose **standard deviation** (a measure of how much the values vary) is below `tol = 6.0` is treated as padding.

```
   start with the image
          │
          ▼
┌──────────────────────────────────────┐
│ repeat up to max_passes = 4 times:   │
│                                      │
│  image smaller than 32 px a side? ───┼─► stop
│          │ no                        │
│          ▼                           │
│  mark rows/columns with std > 6.0    │
│  as "real content"                   │
│          │                           │
│  everything flat? ───────────────────┼─► stop, leave image alone
│          │ no                        │
│  nothing to trim? ───────────────────┼─► stop
│          │ no                        │
│          ▼                           │
│  crop to the content rows/columns    │
│          │                           │
│  crop would keep < 10% of image? ────┼─► stop, refuse to crop
│          │ no                        │
│  accept crop, go round again         │
└──────────────────────────────────────┘
```

Why several passes? Removing a flat top and bottom edge can reveal that the left and right edges are flat too. So rows and columns are trimmed in turn.

### Step: normalise

Min-max normalisation stretches the darkest pixel to 0 and the brightest to 255, linearly:

```
 new value = (old - min) / (max - min) × 255
```

Example: if a scan only uses values 50 to 150, then 50 becomes 0, 100 becomes 127.5, and 150 becomes 255. This makes brightness comparable across scans.

---

## 3. Noise: what are we removing?

**Noise** is random variation that is not part of the anatomy. Think of the crackle on an old radio. Three kinds are worth knowing:

| Type | What it looks like | Typical cause |
|---|---|---|
| **Gaussian** | Every pixel nudged by a small random amount drawn from a bell curve | Electronic/thermal noise |
| **Salt-and-pepper** | A few pixels forced to pure white (255) or pure black (0) | Transmission/bit errors, dead pixels |
| **Rician** | Looks like grainy speckle; adds a positive bias, especially in dark regions | How MRI magnitude images are formed (Gaussian noise in the complex signal becomes Rician in the magnitude image) |

Rician noise is the one specific to MRI. Our synthetic data generator uses it: `SYNTH_NOISE_SIGMA = 8.0` ("Rician noise level") in `config.py`. Our filters do not model Rician noise explicitly. They are general-purpose denoisers that cope with the speckle it produces.

---

## 4. Kernels and convolution (the core DSP idea)

Many image operations are **neighbourhood operations**. The new value of a pixel depends on the pixel and its neighbours. The recipe for combining the neighbours is a small grid of numbers called a **kernel** (also called a mask or filter).

**Convolution** slides the kernel across the image:

```
   place kernel over a 3x3 patch
              │
              ▼
   multiply each kernel number by the pixel beneath it
              │
              ▼
   add all 9 products together
              │
              ▼
   that sum is the NEW value of the centre pixel
              │
              ▼
   slide one pixel along and repeat
```

**Worked example.** A 3x3 patch with one very bright pixel, and a 3x3 averaging ("box") kernel where every weight is 1/9:

```
Patch            Kernel (box)
10  10  10       1/9 1/9 1/9
10 200  10       1/9 1/9 1/9
10  10  10       1/9 1/9 1/9
```

New centre value = (8 x 10 + 200) / 9 = 280 / 9 = **31.1**

The spike of 200 has been dragged down, but not removed. It is smeared into a mild glow of about 31, and over the whole image it would also bleed into every neighbour. Averaging **blurs**.

Two honest technical notes for the viva:

- Strictly, convolution flips the kernel before sliding it, whereas OpenCV's `filter2D` does correlation (no flip). For **symmetric** kernels such as a box or Gaussian the results are identical, so the distinction does not matter for us.
- A kernel-based filter is **linear**: the output is a weighted sum. The **median** and **bilateral** filters below are **not** plain convolutions. Median is non-linear (it sorts), and the bilateral filter has weights that depend on the image content itself.

A **Gaussian blur** is a convolution whose weights follow a bell curve: the centre pixel counts most, distant neighbours count less. Its width is set by a number called **sigma**. We use it inside unsharp masking (Section 7).

---

## 5. Denoising (the `denoise()` step)

Denoising is two filters in a row.

```
 resized image
      │
      ▼
┌──────────────────────────┐
│ Median filter, 3 x 3     │  kills isolated white/black dots
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ Bilateral filter         │  smooths flat areas, keeps edges
│  diameter d = 7          │
│  sigmaColor = 50         │
│  sigmaSpace = 50         │
└────────────┬─────────────┘
             ▼
      denoised image
```

### 5a. Median filter - for salt-and-pepper speckle

`MEDIAN_KERNEL = 3`, so a 3x3 neighbourhood. For each pixel, it lists the 9 values, sorts them, and takes the middle one.

Same patch as before:

```
 values:   10 10 10 10 10 10 10 10 200
 sorted:   10 10 10 10 [10] 10 10 10 200
                        ▲
                 5th of 9 = the median = 10
```

The median is **10**. The spike vanishes completely instead of being smeared. One extreme value cannot drag the median, because it only ever lands at the end of the sorted list. Everyday analogy: the "middle-income person" in a room does not change when a billionaire walks in, but the average does.

The median is the standard tool for salt-and-pepper noise, and it tends to keep step edges reasonably sharp.

### 5b. Bilateral filter - smooth flat areas, keep edges

A plain Gaussian blur weights neighbours only by **distance**. Near a tumor boundary it would average bright tumor pixels with dark tissue pixels and blur the edge. But the edge is exactly what we need to keep.

The bilateral filter gives each neighbour **two** weights and multiplies them:

```
 For each neighbour of the pixel being fixed:

   ┌────────────────────────┐   ┌─────────────────────────────┐
   │ Spatial weight         │   │ Range (intensity) weight    │
   │ closer pixel = bigger  │   │ similar brightness = bigger │
   └───────────┬────────────┘   └───────────────┬─────────────┘
               └──────────┬─────────────────────┘
                          ▼
              final weight = spatial × range
                          │
                          ▼
         new pixel = weighted average of neighbours
```

Across a strong edge the brightness difference is large, so the range weight is nearly zero. Those pixels barely contribute. So flat regions get smoothed, and edges survive.

The cost: it is slower than a Gaussian blur, and it is not a perfect edge-keeper (very weak edges can still be softened).

---

## 6. Contrast: histogram equalisation vs CLAHE

### The histogram

A **histogram** counts how many pixels have each brightness value, 0 to 255. A low-contrast image has its histogram bunched into a narrow range, like a crowd squeezed into one corner of a room.

### Global histogram equalisation (HE)

The function `histogram_equalization` uses `cv2.equalizeHist`. The steps:

```
 histogram of the whole image
          │
          ▼
 running total (CDF: cumulative distribution)
          │
          ▼
 use that curve as a brightness remapping
 (crowded brightness levels get spread apart)
          │
          ▼
 ONE mapping applied to the WHOLE image
```

Why that fails on a brain MRI (as the module docstring says): the very bright skull and the large dark background dominate the histogram. The mapping is tuned for them. The subtle brightness difference between a tumor and healthy tissue gets little extra separation.

The function is kept in the code only for a side-by-side comparison. `enhance()` does not use it.

### CLAHE

**C**ontrast **L**imited **A**daptive **H**istogram **E**qualisation fixes two problems.

**Adaptive**: split the image into a grid of tiles and give each tile its own equalisation mapping. A dim region is brightened relative to *its own* surroundings. With `CLAHE_TILE_GRID = (8, 8)` on a 256 x 256 image, each tile is 32 x 32 pixels.

**Contrast limited**: in a nearly uniform tile, plain equalisation would stretch tiny noise differences into large ones. CLAHE clips each tile's histogram at the **clip limit** and spreads the clipped excess evenly over all bins before equalising. This caps how steep the mapping can be, and so caps noise amplification.

**No seams**: to avoid visible borders between tiles, the result is bilinearly interpolated between neighbouring tile mappings (stated in the `clahe` docstring).

```
 denoised image
       │
       ▼
 split into 8 x 8 grid of tiles (each 32 x 32 px)
       │
       ▼
 for each tile:
   ┌─ build its histogram
   ├─ clip tall bins at the clip limit (2.0)
   ├─ spread the clipped excess evenly over all bins
   └─ equalise using this tile's own mapping
       │
       ▼
 blend neighbouring tile mappings smoothly (bilinear)
       │
       ▼
 locally contrast-boosted image
```

In OpenCV the clip limit is relative to a perfectly flat histogram. `1.0` means essentially no contrast boost, and larger values allow more.

---

## 7. Unsharp masking

The name is historical (from darkroom photography) and confusing: it **sharpens**. The old method used a blurred ("unsharp") copy of the photo to find the edges.

The idea in three steps:

```
 original image
     │
     ├──────────────┐
     ▼              ▼
 Gaussian blur   (keep original)
 (sigma = 1.5)
     │              │
     ▼              │
 blurred copy       │
     └──────┬───────┘
            ▼
   detail = original - blurred
   (what the blur removed: edges and texture)
            │
            ▼
   result = original + amount × detail
            (amount = 1.2)
            │
            ▼
   clip to the range 0-255
```

**Tiny worked example** with `amount = 1.2`. Take a bright pixel next to a dark-to-bright edge. Its value is 100, and its blurred value is 80 (the blur pulled it down towards the dark side).

```
 detail = 100 - 80 = 20
 result = 100 + 1.2 × 20 = 124        (brighter: the edge pops)
```

On the dark side: pixel 10, blurred 30. Detail = -20, result = 10 - 24 = -14, which is clipped to 0 (darker). The edge gets a steeper step.

The code uses an algebraically identical form: `(1 + amount) × image - amount × blurred`. This is a single weighted-sum call (`cv2.addWeighted`) and is then clipped to 0-255. Passing a kernel size of `(0, 0)` to the blur tells OpenCV to work out the kernel size from sigma.

In frequency terms, unsharp masking is a **high-boost**: it keeps everything and adds extra high-frequency content (fine edges and texture).

**Gamma correction** (`gamma_correct`) applies the brightness curve `out = 255 × (in/255)^(1/gamma)` using a 256-entry lookup table. The default is `gamma = 1.0`. At exactly 1.0 the function returns the image unchanged, so in the normal pipeline this step does nothing.

---

## Where this lives in our code

| Job | File and function |
|---|---|
| Whole Stage 2 chain | `preprocess.py`, `preprocess()` |
| Grayscale | `preprocess.py`, `to_grayscale()` |
| Trim borders | `preprocess.py`, `autocrop_border()` |
| Resize | `preprocess.py`, `resize()` |
| Median + bilateral | `preprocess.py`, `denoise()` |
| Min-max normalise | `preprocess.py`, `normalize()` |
| Whole Stage 3 chain | `enhance.py`, `enhance()` = CLAHE, unsharp, gamma |
| Global HE (comparison only) | `enhance.py`, `histogram_equalization()` |
| Local contrast | `enhance.py`, `clahe()` |
| Sharpen | `enhance.py`, `unsharp_mask()` |
| Numbers proving it helped | `enhance.py`, `enhancement_report()` |

The `enhancement_report` step measures the improvement instead of just claiming it:

```
 original image ─┐
                 ├─► for each image, compute:
 enhanced image ─┘     • standard deviation  (how spread out the tones are)
                       • entropy, in bits    (Shannon entropy of the histogram:
                                              information content)
                       • sharpness           (variance of the Laplacian, a
                                              second-derivative edge detector;
                                              blurry images score low)
                          │
                          ▼
                 compare before vs after
```

---

## Why these parameter values (`config.py`)

| Constant | Value | Reasoning |
|---|---|---|
| `IMAGE_SIZE` | (256, 256) | Big enough to keep detail, small enough to be fast; fixed size is needed for consistent features. |
| `MEDIAN_KERNEL` | 3 | Smallest useful window. A larger window would also erase small real structures. |
| `BILATERAL_D` | 7 | Neighbourhood diameter in pixels. |
| `BILATERAL_SIGMA_COLOR` | 50 | On a 0-255 scale: brightness differences much larger than ~50 get little weight, so strong edges are preserved while small noise fluctuations are averaged. |
| `BILATERAL_SIGMA_SPACE` | 50 | Spatial falloff. Because `d=7` caps the window, the window size is effectively set by `d`. |
| `CLAHE_CLIP_LIMIT` | 2.0 | Config comment: "higher = more contrast but more amplified noise". 2.0 is a common moderate default. |
| `CLAHE_TILE_GRID` | (8, 8) | 32 x 32 pixel tiles: small enough to be local, large enough to contain enough pixels for a meaningful histogram. |
| `UNSHARP_SIGMA` | 1.5 | Sets the scale of detail that is boosted: fine edges, not broad shading. |
| `UNSHARP_AMOUNT` | 1.2 | How strongly detail is added back; above 1 gives a clear but not extreme sharpening. |

Be honest in the viva: these are sensible, standard-range values chosen for this project. They were not derived by exhaustive optimisation (except where a comment in the code states a measurement).

---

## Common viva questions

**Q1. Why median filter and then bilateral? Why not just a Gaussian blur?**
Median removes isolated extreme pixels (salt-and-pepper) without smearing them. Bilateral then smooths the remaining small fluctuations in flat regions while keeping strong edges, because pixels across an edge have very different intensity and get almost no weight. A Gaussian blur weights only by distance, so it would blur the tumor boundary, which our segmentation later depends on.

**Q2. Why is the median better than the mean for a single bright noise pixel?**
The mean includes the outlier's full value (our example: 31.1 instead of 10). The median ignores how extreme an outlier is. It only cares about rank, so it returns 10.

**Q3. Why does global histogram equalisation not work well here, and what does CLAHE change?**
Global HE uses one mapping dominated by the bright skull and dark background. CLAHE equalises per tile (8x8 grid), so local contrast is enhanced. It also clips each histogram at the clip limit (2.0), so noise in flat areas is not blown up.

**Q4. What does the clip limit do?**
It caps how tall any histogram bin may be before equalisation, and the excess is spread evenly over all bins. A tall bin means a steep mapping slope, i.e. a strong contrast stretch (and noise amplification) in that brightness range. Clipping limits the slope. Higher means more contrast and more noise.

**Q5. Explain unsharp masking. Why is it called that?**
Blur the image, subtract to get the detail layer (edges), and add a multiple of that back. It is named after the analogue technique that used a blurred ("unsharp") copy. It sharpens by boosting high frequencies.

**Q6. Why do you enhance after denoising, not before?**
Enhancement amplifies contrast, including noise. Clean first, then enhance.

**Q7. Is the bilateral filter a convolution?**
Not a plain one. Its weights depend on pixel values as well as position, so it is non-linear and not a single fixed kernel. The Gaussian blur is a true (linear) convolution.

---

## Check yourself

**1.** Compute the 3x3 box-filter and median results for the centre of this patch: `[5 5 5; 5 95 5; 5 5 5]`.
<details><summary>Answer</summary>
Box: (8 x 5 + 95) / 9 = 135 / 9 = 15. Median: the sorted values are eight 5s and one 95, so the 5th value is 5.
</details>

**2.** In what order does `preprocess()` run its steps, and which comes first, denoise or resize?
<details><summary>Answer</summary>
Grayscale, convert to uint8 if needed, autocrop, resize, denoise (median then bilateral), normalise. Resize comes before denoise.
</details>

**3.** With a 256 x 256 image and `CLAHE_TILE_GRID = (8, 8)`, how large is each tile?
<details><summary>Answer</summary>
256 / 8 = 32, so 32 x 32 pixels.
</details>

**4.** Using `UNSHARP_AMOUNT = 1.2`, a pixel has value 100 and blurred value 80. What is the result?
<details><summary>Answer</summary>
100 + 1.2 x (100 - 80) = 124. Equivalently 2.2 x 100 - 1.2 x 80 = 124.
</details>

**5.** Is `histogram_equalization` part of the normal pipeline?
<details><summary>Answer</summary>
No. `enhance()` runs CLAHE, unsharp mask and optional gamma. Global HE is kept only for the comparison demonstration.
</details>

---

Next: [Lesson 4 - Segmentation](04_SEGMENTATION.md)
