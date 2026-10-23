# Lesson 4 - Segmentation

> **In one minute**
> Segmentation means colouring in the part of the scan that looks suspicious, so we can show it to a doctor and measure it.
> The big trap is that the skull is brighter than the tumor, so a "find the brightest thing" rule would always pick the skull.
> We therefore remove the skull first, then look only inside the brain for pixels that are unusually bright for that patient.
> We also compare the left half of the brain with the right half, the way a radiologist does.
> Finally we clean up the result, throw away blobs that are too small or too thin, and keep the single best candidate.

## What you'll learn

- What segmentation is, and why the **skull** is the central difficulty.
- **Thresholding**, **Otsu's method**, and why we use Otsu only for the head and *not* for the tumor.
- The **mean + k x sigma** rule, and why statistics are taken from "tissue" pixels only.
- The **asymmetry (mirror) check**, the way radiologists compare left and right.
- **Morphology**: erosion, dilation, opening, closing, and the distance transform.
- **Contours** and the **area / aspect-ratio** filters, how the final candidate is picked, and every **fallback** in `segment.py`.

---

## 1. The problem and the order of operations

**Segmentation** means labelling which pixels belong to a region of interest.

Think of it as colouring in a map. You take a black pen and mark "this part is the region I care about".

Here the output is a **mask**. A mask is a second image the same size as the scan. White pixels mean "suspicious region". Black pixels mean "everything else". We use the mask to outline the region for a doctor and to measure its shape for the classifier.

The obstacle is stated in the module's own notes: *in an MRI the skull is brighter than the tumor*. Imagine searching a dark room for the brightest lamp. If a floodlight is hanging on the wall, you will always find the floodlight, never the small lamp. The skull is our floodlight.

So the order of the steps matters. We must remove the skull before we search for bright spots.

```
 Enhanced image (from Lesson 3)
        │
        ▼
 ┌──────────────────────────┐
 │ 4a. Skull strip          │  keep only brain tissue
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 4b. Threshold INSIDE the │  "brighter than normal for this brain?"
 │     brain only           │
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ Mirror check (left vs    │  "brighter than the other side?"
 │ right), then OR the two  │
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 4c. Morphology           │  remove specks, fill holes
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 4d. Contours -> filter   │  drop tiny / thin blobs
 │     -> pick best         │
 └──────────────────────────┘
```

The function `segment_tumor()` in `segment.py` runs all of this. Its input is the **enhanced** image from Lesson 3.

---

## 2. Step 4a - skull stripping (brain_mask)

The goal is a mask that contains only brain tissue. We build it in stages, and each stage fixes a problem the previous one leaves behind.

### 2.1 Otsu thresholding to find the head

A **threshold** is a cut-off brightness. Pixels at or above the cut-off become white. All other pixels become black. This turns a greyscale picture into a black-and-white one.

The hard question is where to put the cut-off. **Otsu's method** answers it automatically.

Here is the idea. Otsu tries every possible cut-off t from 0 to 255. For each t it splits the pixels into two groups: darker than t, and brighter than t. It then asks how "spread out" each group is. It picks the t that makes both groups as tight as possible. (The technical name is minimising the **within-class variance**. It is equivalent to maximising the between-class variance.)

Otsu works best when the histogram has two clear humps. "Dark empty background" and "bright head" give exactly that shape.

```
 Greyscale scan
      │
      ▼
 [Otsu: try t = 0..255, pick the t that gives
  the two tightest groups]
      │
      ▼
 [Pixels >= t become white, the rest black]
      │
      ▼
 "Head" mask (white head, black background)
```

OpenCV computes t by itself here. The 0 you pass in is ignored when the Otsu option is switched on.

One important caveat: Otsu **always** splits the pixels into two groups. It does this even when there are not really two groups in the data. Section 4 shows why this matters.

### 2.2 Close gaps, keep the largest component

The raw head mask can have small dark gaps. We fix them with a **closing** operation (explained fully in Section 6). Closing fills small dark holes and cracks.

After that, the mask may still contain stray white specks far from the head. A **connected component** is one separate white blob. The code labels every blob, measures its area, and keeps only the biggest one. Label 0 is the background, so the search starts at label 1. Everything else is speckle or scanner artefact.

```
 Head mask
    │
    ▼
 [Closing, ellipse 7×7, 3 iterations]   ← fills small dark gaps
    │
    ▼
 [Label every separate white blob]
    │
    ▼
 [Keep only the blob with the largest area]
    │
    ▼
 One solid head blob
```

### 2.3 Fill interior holes (safely)

The ventricles and CSF (the fluid-filled spaces) are dark. They can appear as holes inside the head. We want the head to be solid, so we fill those holes.

The trick is to work from the outside. Flood the *background* with colour, starting from the top-left corner (0,0), then invert. Anything the flood could not reach must be inside the head, so it counts as an interior hole and becomes white.

Think of pouring water on a sandcastle with a moat. Water that stays outside the walls is "background". Any dry pocket the water cannot reach is "interior".

There are two safety details.

**Safety 1: the 1-pixel frame.** If the head touches both the top and bottom edges of the image, the background is cut into a left part and a right part. A flood from one corner would only reach one of them. The other would be wrongly treated as an interior hole. So the code first adds a 1-pixel black frame around the image. The frame reconnects the two background parts.

**Safety 2: the hollow-ring guard.** Sometimes the chosen blob is a thin closed ring. Examples are a scanner bezel, a scanned page frame, or a skull whose inside was thresholded away. Filling a ring floods the whole frame, which is a disaster. So the code compares areas.

- Real hole filling adds only a modest amount of area.
- If the filled mask is more than `MAX_HOLE_FILL_GROWTH = 3.0` times bigger than the original, we assume a ring was flooded.
- In that case we reject the fill and keep the unfilled blob.

```
 Solid head blob
    │
    ▼
 [Add 1-pixel black frame around the image]
    │
    ▼
 [Flood the background from corner (0,0), then invert]
    │          ──► "filled" version (interior holes become white)
    ▼
 ┌──────────────────────────────────────────┐
 │ Is filled area  >  3.0 × original area ? │
 └───────┬───────────────────────┬──────────┘
       YES                       NO
   (a ring was flooded)    (normal hole filling)
         │                       │
         ▼                       ▼
 keep UNFILLED blob       use FILLED blob
```

### 2.4 Peel off the skull with the distance transform

Now we have a solid head. It still includes the skull and the thin CSF layer under it. We must peel that off.

The **distance transform** gives every white pixel a number: how far it is from the nearest black pixel. Pixels at the head's centre get the biggest numbers. Pixels on the rim get the smallest.

We keep only pixels whose distance is greater than a **fraction of the maximum distance**. The fraction is `SKULL_STRIP_FRACTION = 0.22`.

*Tiny worked example.* Suppose the centre of a head is 100 pixels from the background. The maximum distance is 100. The cut-off is 0.22 x 100 = 22. Every pixel less than 22 pixels from the edge is peeled away.

Why a fraction and not a fixed number of pixels? Heads differ in size. A fixed peel that clears the skull on a large head could eat a small head completely. A fraction scales with the head. The notes say the skull plus CSF layer is about 15% of the head radius, so 0.22 clears it with some margin.

One more detail. A 1-pixel black frame is added again before the distance transform. This makes "outside the image" count as background. Without it, skull touching the image edge would look far from any background and would survive the peel.

```
 Solid head mask
    │
    ▼
 [Add 1-pixel black frame]
    │
    ▼
 [Distance transform: each white pixel gets its
  distance to the nearest black pixel]
    │
    ▼
 [Find the maximum distance D_max]
    │   (if D_max is 0, skip this step)
    ▼
 [Keep pixels with distance  >  0.22 × D_max]
    │
    ▼
 Brain-only mask (skull and CSF rim removed)
```

### 2.5 Plausibility check and ellipse fallback

Last, we check that the answer makes sense. We measure **coverage**, the fraction of the image the mask covers.

- Below 5% (`MIN_BRAIN_COVERAGE = 0.05`): thresholding collapsed and the mask is almost empty.
- Above 88% (`MAX_BRAIN_COVERAGE = 0.88`): no background was found. This happens when the picture was already cropped tightly to the brain.

In both cases the code throws the mask away. It draws a **centred filled ellipse** instead. The ellipse has semi-axes of 42% of the width and 46% of the height. This is a crude guess. Still, a sensible guess is better than a broken mask that poisons the statistics in later steps.

```
 Brain-only mask
    │
    ▼
 [coverage = fraction of image that is white]
    │
    ▼
 ┌──────────────────────────────────────────┐
 │ Is  5%  <=  coverage  <=  88% ?          │
 └───────┬───────────────────────┬──────────┘
       YES                       NO
         │                       │
         ▼                       ▼
   keep the mask     replace with a centred filled ellipse
                     (semi-axes: 42% of width, 46% of height)
```

After this, `apply_brain_mask` blanks everything outside the mask, so only brain pixels remain in the picture.

---

## 3. Step 4b - the mean + k x sigma rule (_threshold_within_brain)

Now we search for pixels that are unusually bright **for this particular brain**.

First, two terms.

- The **mean** is the average brightness of normal tissue.
- **Sigma** (the standard deviation) measures how spread out normal brightness values are. A small sigma means most pixels are close to the mean.

The rule is: call a pixel suspicious if its brightness is above `mean + k x sigma`. Here `k = 1.5` (`TUMOR_SIGMA_K`).

*Worked example.* Normal tissue has mean 140 and sigma 20. The cut-off is 140 + 1.5 x 20 = 170. Any pixel at 170 or brighter is a candidate.

For bell-shaped data, about 7% of values lie above mean + 1.5 sigma. So this rule only creates *candidates*. It is not a verdict. Later filters decide.

Three ideas make the rule work.

1. **Mean and sigma describe normal tissue.** "Unusually bright" only means something relative to what is normal.
2. **Tissue only.** The ventricles are very dark. If we include them, sigma becomes huge, and mean + k x sigma becomes too high. The code notes that in the first tests the cut-off exceeded 255, so nothing was ever detected. The fix is to use only pixels above the brain's median brightness. This drops the dark CSF and keeps grey and white matter.
3. **Self-calibrating.** The reference is that patient's own scan. So the rule adapts to different scanners and brightness levels.

```
 Enhanced image  +  detect mask (see Section 5)
        │
        ▼
 [Collect all brain pixels]
        │
        ▼
 [Median brightness of brain pixels]
        │
        ▼
 [Keep only pixels ABOVE the median = "tissue"]
        │         (fewer than 50 tissue pixels?
        │          use all brain pixels instead)
        ▼
 [mean and sigma of tissue pixels]
        │
        ▼
 [threshold = mean + 1.5 × sigma, clipped to 0..254]
        │
        ▼
 [Pixel is white if  brightness >= threshold  AND  inside the brain]
        │
        ▼
 Binary candidate mask
```

If there are no brain pixels at all, the function returns an empty mask with threshold 255.0. The chosen threshold is stored in the result under the name "threshold".

---

## 4. Why not Otsu for the tumor? (a classic viva question)

Remember that Otsu **always** splits pixels into two groups, even when nothing is abnormal.

Now picture a healthy brain. It has grey matter and white matter, which are two slightly different brightness levels. Otsu happily splits these into two groups and calls one of them "lesion". That is a false alarm.

This was measured, not guessed. Over 60 scans:

| Method | Healthy brains wrongly flagged | Tumors caught |
|---|---|---|
| Otsu on the brain | **100%** | - |
| Tissue-relative mean + k x sigma rule | **3%** | **100%** |

(The full comparison is in the project's measured results, `backend/outputs/metrics*.json`.)

So the rule of thumb is:

- Otsu is the right tool for "head vs empty background". That question truly has two groups.
- Otsu is the wrong tool for "is there a lesion?". That question might have no second group at all.

---

## 5. The boundary guard and the asymmetry (mirror) check

### Boundary guard

Near the edge of the brain mask, bright skull-adjacent tissue blends with brain tissue. This is called a partial-volume effect. It creates thin bright crescents that look like lesions but are not.

To avoid them, the code searches in a slightly smaller mask called `detect_mask`. It shrinks the brain mask using an elliptical 7 x 7 element (`BOUNDARY_GUARD = 7`). That removes roughly 3 pixels from the border. Real tumors sit well inside the brain, so they are not affected.

```
 Brain mask
    │
    ▼
 [Erode once with an ellipse 7×7]   (shrinks ~3 pixels inward)
    │
    ▼
 detect_mask  (the search region used from now on)
```

### Mirror check (_contralateral_excess)

A brain is roughly left-right symmetric. A radiologist often asks: "is this spot brighter than the same spot on the other side?" The code asks the same question for every pixel.

Why is this useful? On some MRI types (T2) the whole brain is bright, and a tumor may be only slightly brighter than normal tissue. The mean + k x sigma rule can miss it. But the tumor still looks different from its mirror image. The mirror check compares the scan with itself, so it works across scan types.

Here is how it works, step by step.

1. Find the centre of the brain mask (its centroid, using image moments).
2. Slide the whole image sideways so the brain's centre lies on the image's centre column.
3. Make a left-right flipped copy.
4. Subtract the flipped copy from the original. The result is the **excess**: how much brighter each pixel is than its mirror.
5. Do the subtraction using signed numbers, so negative differences stay negative.
6. Trust only pixels where both the pixel and its mirror are inside the brain mask.
7. Slide the result back to the original position.

A tumor on one side gives a positive excess on that side. The mirror location gets a negative excess. So only the tumor side passes a positive threshold.

```
 Enhanced image + detect_mask
        │
        ▼
 [Find brain centre (centroid of the mask)]
        │
        ▼
 [Shift image sideways so the brain centre = image centre column]
        │
        ├──────────────► [Flip a copy left-right]
        ▼                         │
 [excess = centred image  MINUS  flipped image]
        │         (signed numbers, so negatives are kept)
        ▼
 [Keep only pixels where pixel AND its mirror are in the brain]
        │
        ▼
 [Shift back to the original position]
        │
        ▼
 excess map  +  validity map
```

Then the pipeline turns the excess map into a second candidate mask, and combines it with the first.

```
 excess map + validity map
        │
        ▼
 [Take excess values at valid pixels]
        │       (50 or fewer valid pixels?  SKIP the mirror check)
        ▼
 [asym_threshold = mean + 2.0 × sigma of those values]
        │
        ▼
 [Pixel is white if excess >= asym_threshold AND valid]
        │
        ▼
 ┌──────────────────────────────────────────────┐
 │ Combine with the intensity candidate mask    │
 │ using OR  (a pixel counts if EITHER rule     │
 │ flags it)                                    │
 └──────────────────────────────────────────────┘
```

Key points:

- It is the same mean + k x sigma idea, with `ASYM_SIGMA_K = 2.0`. This is stricter than 1.5 because difference values are noisier.
- It is combined by **union (OR)**, not intersection (AND). The reason is that the two rules catch different failures. A large midline tumor is bright but symmetric, so only the intensity rule catches it. A subtle T2 lesion is asymmetric but not globally bright, so only the mirror rule catches it.
- Adding the mirror rule raised tumor localisation from 72.3% to 82.6% on the 253-image real dataset.
- Limitation to admit in a viva: it assumes the head is roughly symmetric and upright. A perfectly symmetric midline tumor will not look asymmetric.

---

## 6. Morphology (morphological_cleanup)

After thresholding, the mask is messy. It has isolated white specks, and the real blob has small holes in it. **Morphological operations** clean this up.

They work by sliding a small shape over the binary image. That shape is called the **structuring element**. Ours is an ellipse of size 5 x 5 (`MORPH_KERNEL_SIZE = 5`).

Think of the structuring element as a small stamp.

- **Erosion**: a pixel stays white only if the whole stamp fits inside the white area around it. White regions shrink. Specks smaller than the stamp vanish.
- **Dilation**: a pixel becomes white if the stamp touches any white pixel. White regions grow. Small holes fill in.
- **Opening** = erosion, then dilation. First the specks disappear. Then the survivors grow back to their original size. Used here with `MORPH_OPEN_ITER = 2` (two rounds).
- **Closing** = dilation, then erosion. First holes and gaps fill in. Then the object shrinks back to its original size. Used here with `MORPH_CLOSE_ITER = 3` (three rounds).

*Tiny one-dimensional example of opening with a 3-wide element.*

```
 before:   0 1 0 0 1 1 1 1 0
 after:    0 0 0 0 1 1 1 1 0
```

The lone 1 on the left (a speck) disappears. The run of four 1s survives intact.

The order matters. We open first, then close.

```
 Raw binary mask
    │
    ▼
 [Opening: ellipse 5×5, 2 iterations]   ← delete small bright specks
    │
    ▼
 [Closing: ellipse 5×5, 3 iterations]   ← fill small holes in the blob
    │
    ▼
 Clean binary mask
```

Why this order? Deleting noise first is safe. If we closed first, we might glue noise specks onto the real region and make them permanent.

Note: `brain_mask` also uses closing (7 x 7 ellipse, 3 iterations). Those numbers are written directly inside `brain_mask`, not in `config.py`.

---

## 7. Contours, filters and picking the winner

### Contours

A **contour** is the outline curve around a white region. Think of tracing the edge of a puddle with a pen.

The code collects only the **outer** outline of each blob and ignores holes inside blobs. It also compresses straight stretches of the outline down to their end points, to save memory.

### Plausibility filters (_plausible)

Not every blob is a tumor candidate. Each outline goes through two tests.

- **Area filter**: contours smaller than `MIN_TUMOR_AREA = 120` pixels are treated as noise.
- **Aspect-ratio filter**: draw the smallest upright rectangle around the blob (the bounding box). Divide the longer side by the shorter side. The result must be at most `MAX_ASPECT_RATIO = 5.0`. A real mass is a blob, not a long streak. Taking "longer divided by shorter" means the test works for both tall and wide shapes.

*Worked example.* A blob is 60 wide and 10 tall, with area 400. The area passes (400 is at least 120). But 60 / 10 = 6, which is more than 5. The blob is rejected as a sliver.

```
 Each contour
    │
    ▼
 ┌─────────────────────────┐
 │ Area  >=  120 pixels ?  │──NO──► reject (noise)
 └───────────┬─────────────┘
            YES
             ▼
 ┌─────────────────────────────────────────┐
 │ Bounding box has a zero side ?          │──YES──► reject
 └───────────┬─────────────────────────────┘
            NO
             ▼
 ┌─────────────────────────────────────────┐
 │ longer side ÷ shorter side  <=  5.0 ?   │──NO──► reject (sliver)
 └───────────┬─────────────────────────────┘
            YES
             ▼
        keep as a candidate
```

### Choosing the best candidate

Several candidates may survive. We want only one. Each gets a score:

score = area x brightness ratio

The **brightness ratio** is the candidate's mean brightness divided by the mean brightness of the whole brain. A big, bright region beats a big, dim one. The highest score wins and is reported as the ROI (region of interest).

```
 Surviving candidates
    │
    ▼
 [For each: measure area and mean brightness]
    │
    ▼
 [brightness ratio = candidate mean ÷ whole-brain mean]
    │
    ▼
 [score = area × brightness ratio]
    │
    ▼
 [Pick the candidate with the highest score]
    │
    ▼
 Chosen ROI  ──► measure its shape (below)
```

### Measuring the winner (_region_properties)

The chosen region is measured. These numbers later become shape features for the classifier (Lesson 5).

| Measurement | Meaning |
|---|---|
| **circularity** = 4 pi A / P^2 | 1.0 for a perfect circle. Lower for ragged shapes. |
| **solidity** = area / convex-hull area | 1.0 for a shape with no dents. (The convex hull is a rubber band stretched around the shape.) |
| **extent** = area / bounding-box area | How much of its box the shape fills. |
| **aspect ratio** = width / height | Tall or wide. |
| Also measured | Area, perimeter, bounding box, centroid, mean and max intensity. |

Then `segment_tumor` adds two more numbers: `area_ratio` (region area divided by brain area) and `brightness_ratio`.

### The picture for the doctor (draw_overlay)

`draw_overlay` draws the result on the scan.

```
 Segmentation result
    │
    ▼
 ┌────────────────────────┐
 │ Was a region found ?   │
 └───────┬─────────┬──────┘
       NO          YES
        │           │
        ▼           ▼
  Write the text    [Translucent red fill (30%)]
  "No suspicious            │
   region"                  ▼
                    [Red outline of the contour]
                            │
                            ▼
                    [Yellow bounding box]
                            │
                            ▼
                    [Yellow label "ROI ...px"]
```

---

## 8. Every fallback and guard, in one list

Real data is messy, so the code has a safety net at nearly every step.

| Situation | What the code does |
|---|---|
| Hole filling inflates the mask more than 3.0x | Keep the unfilled component (`MAX_HOLE_FILL_GROWTH`). |
| Brain mask covers under 5% or over 88% of the image | Replace with a centred ellipse (`MIN/MAX_BRAIN_COVERAGE`). |
| Distance transform maximum is 0 | Skip the peeling step. |
| No brain pixels at all in the threshold step | Return an empty mask and threshold 255.0. |
| Fewer than 50 tissue pixels | Compute statistics over all brain pixels. |
| Threshold would exceed 254 | Clip it to 254. |
| 50 or fewer valid mirror pixels | Skip the asymmetry criterion. |
| Centroid undefined (empty mask) | Use the image centre (width / 2). |
| No contour passes the area / aspect filters | `tumor_found = False` and an empty `tumor_mask`. |

---

## Where this lives in our code

All of it is in `backend/src/braintumor/segment.py`.

| Job | Function |
|---|---|
| Skull strip (Otsu, largest component, fill, distance transform, ellipse fallback) | `brain_mask()` |
| Blank outside the brain | `apply_brain_mask()` |
| mean + k x sigma rule | `_threshold_within_brain()` |
| Left-right comparison | `_contralateral_excess()` |
| Open then close | `morphological_cleanup()` |
| Shape measurements | `_region_properties()` |
| Whole pipeline | `segment_tumor()` |
| Picture for the doctor | `draw_overlay()` |

How `segment_tumor()` strings these together:

```
 enhanced image
      │
      ▼
 [brain_mask]  ──► brain mask ──► [apply_brain_mask] ──► masked picture
      │
      ▼
 [Erode mask (7×7 ellipse)] ──► detect_mask
      │
      ├──► [_threshold_within_brain]  ──► intensity candidates
      │
      ├──► [_contralateral_excess + mean + 2.0×sigma] ──► mirror candidates
      ▼
 [OR the two candidate masks]
      │
      ▼
 [morphological_cleanup]
      │
      ▼
 [Find outer contours] ──► [area / aspect filters]
      │
      ▼
 [Score = area × brightness ratio, pick best]
      │
      ▼
 [_region_properties + area_ratio + brightness_ratio]
      │
      ▼
 result: brain mask, raw and clean masks, threshold,
         contour, properties, tumor_found, tumor_mask
```

---

## Why these parameter values (config.py)

| Constant | Value | Reasoning |
|---|---|---|
| `SKULL_STRIP_FRACTION` | 0.22 | Skull + CSF is about 15% of head radius; 0.22 clears it with margin. |
| `MAX_HOLE_FILL_GROWTH` | 3.0 | Real hole filling adds a modest fraction; tripling the area means a ring was flooded. |
| `MIN_BRAIN_COVERAGE` / `MAX_BRAIN_COVERAGE` | 0.05 / 0.88 | A brain fills a middling fraction of the frame; outside this range the mask is degenerate. |
| `MORPH_KERNEL_SIZE` | 5 | Large enough to delete specks, small enough not to erase a real mass. |
| `MORPH_OPEN_ITER` / `MORPH_CLOSE_ITER` | 2 / 3 | Remove specks, then fill holes in the blob (config comments). |
| `MIN_TUMOR_AREA` | 120 px | Smaller contours are rejected as noise. |
| `BOUNDARY_GUARD` | 7 | Removes crescent false positives near the mask boundary. |
| `MAX_ASPECT_RATIO` | 5.0 | Rejects long thin slivers. |
| `TUMOR_SIGMA_K` | 1.5 | Brighter than mean + 1.5 sigma of normal tissue. |
| `ASYM_SIGMA_K` | 2.0 | Brighter than its mirror counterpart by 2 sigma. |

As in Lesson 3, some values are justified by measurement in code comments (the sigma rule vs Otsu, the 72.3% to 82.6% gain). Others are sensible engineering choices tuned for this dataset.

---

## Common viva questions

**Q1. Why remove the skull first?**
The skull is brighter than the tumor, so any brightness-based search would select the skull. We restrict the search to brain tissue.

**Q2. How does skull stripping work here?**
Otsu threshold to separate head from background, close gaps, keep the largest connected component, fill interior holes, then keep only pixels farther from the edge than 22% of the maximum distance-to-edge (distance transform). If the result covers under 5% or over 88% of the image, a centred ellipse replaces it.

**Q3. You used Otsu for the head; why not for the tumor?**
Otsu always forces a two-class split. On healthy brains it split grey from white matter and flagged 100% of them as lesions in our 60-scan test. The mean + 1.5 sigma rule on normal tissue flagged 3% while catching 100% of tumors.

**Q4. Why compute mean and sigma from pixels above the median only?**
Dark CSF/ventricles inflate sigma, pushing the threshold above 255 so nothing is detected. Using above-median tissue gives a tight "normal brightness" distribution.

**Q5. What is the asymmetry check and why is it unioned with the intensity rule?**
Each pixel is compared with its mirror position across the brain's centre. Unusually large positive differences (mean + 2 sigma) are flagged. Union (OR) catches both globally bright symmetric tumors and subtle asymmetric ones.

**Q6. Difference between opening and closing?**
Opening = erosion then dilation; it removes small bright specks. Closing = dilation then erosion; it fills small holes. We open (2 iterations) then close (3) with a 5 x 5 ellipse.

**Q7. Which regions are rejected and why?**
Area under 120 px (noise) and bounding-box aspect ratio above 5 (slivers). Of the survivors, the one with the highest area x brightness-ratio is reported.

**Q8. What happens when no tumor is found?**
`tumor_found` is False, `tumor_mask` is empty, and the overlay says "No suspicious region". Segmentation is not itself the diagnosis; classification is a separate stage.

---

## Check yourself

**1.** In skull stripping, why use a fraction of the maximum distance instead of a fixed erosion?
<details><summary>Answer</summary>
Head sizes vary. A fixed erosion that removes the skull from a large head could erase a small head. A fraction scales with the head (`SKULL_STRIP_FRACTION = 0.22`).
</details>

**2.** A brain's tissue pixels have mean 140 and standard deviation 20. What is the tumor threshold?
<details><summary>Answer</summary>
140 + 1.5 x 20 = 170 (`TUMOR_SIGMA_K = 1.5`), clipped to the range 0-254, so 170.
</details>

**3.** A candidate contour has a bounding box 60 wide by 10 tall and area 400 px. Is it kept?
<details><summary>Answer</summary>
No. The area passes (400 is at least 120), but the aspect ratio is 60/10 = 6, which exceeds 5.0.
</details>

**4.** After erosion with a large element, what does dilation with the same element do to a tiny isolated speck?
<details><summary>Answer</summary>
Nothing, because erosion already deleted it and dilation cannot bring back something that is gone. That is why opening removes specks permanently while preserving larger objects.
</details>

**5.** Why is the excess map stored as signed numbers (int16) rather than unsigned 8-bit?
<details><summary>Answer</summary>
Differences can be negative and can fall outside the 0-255 range. Unsigned 8-bit arithmetic would wrap around and produce nonsense.
</details>

---

Next: Lesson 5 - Feature extraction (`05_FEATURE_EXTRACTION.md`)
