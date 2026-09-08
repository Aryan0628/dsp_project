# 03 — 30 MINUTE PRESENTATION & DEMO SCRIPT
### Follow this in order. Timings are a guide, not a rule.

---

## BEFORE YOU START (do this 5 minutes early)

Open a terminal in the project folder and run these two commands so everything is warm:

```bash
cd /Users/aryangupta/Documents/dsp_project
./run.sh check
```

If it prints `ALL CHECKS PASSED`, you are ready. Also open these two images in an image
viewer so you can flip to them instantly:

- `outputs/demo_tumor_pipeline.png`   <- your single best slide
- `outputs/training_report.png`       <- your results slide

---

## MINUTE 0–3 — THE HOOK

> "Brain tumors are one of the most fatal cancers, and survival depends almost entirely on
> how early they are found. MRI is how we find them — but reading an MRI is slow, needs an
> expert, and the images are noisy and low in contrast. We built a system that enhances the
> MRI, automatically locates the suspicious region, and gives a first-pass tumor/no-tumor
> opinion with a confidence score — so the radiologist spends their time deciding, not
> squinting."

Then state the framing clearly, because it prevents half the hard questions:

> "It is a **diagnostic assistance** tool. It does not replace the doctor."

---

## MINUTE 3–8 — THE PROBLEM AND THE ARCHITECTURE

Draw or show the 7-stage block diagram (it is in `docs/01`, and in `README.md`).

Say the key architectural sentence:

> "Our project has two halves. **OpenCV** does the deterministic image work — denoise,
> enhance, segment. **Machine learning** does the pattern recognition — classify. We
> deliberately used both, because a pure deep-learning model is a black box that a doctor
> cannot audit, while our OpenCV half produces a visible, measurable region. So the system
> is accurate *and* explainable."

Walk the seven stages in one sentence each:
1. Acquisition — take the MRI in
2. Preprocessing — grayscale, resize to 256×256, median + bilateral denoise
3. Enhancement — CLAHE + unsharp masking
4. Segmentation — skull strip, threshold, morphology, contours
5. Feature extraction — 111 numbers: intensity, GLCM texture, LBP, shape
6. Classification — compare 4 ML models, keep the best
7. Visualisation — 8-panel figure with the verdict

---

## MINUTE 8–13 — LIVE DEMO PART 1: TRAIN THE WHOLE THING FROM SCRATCH

This is the strongest moment of the demo, because it takes about 15 seconds and proves the
project is real, not a saved screenshot.

```bash
./run.sh train
```

While it runs, narrate:

> "It is generating 400 brain MRIs, running all four processing stages on each one,
> extracting 111 features per image, then cross-validating four different classifiers and
> picking the winner. Watch the model comparison table."

When it finishes, point at the results block:

- **Accuracy 98%**
- **Sensitivity 96%** — "this is the number that matters, it is the fraction of real tumors
  we caught"
- **Specificity 100%**
- **ROC-AUC 1.0**
- Confusion matrix: 50 TN, 0 FP, 2 FN, 48 TP

Then say the line that shows you understand evaluation:

> "Notice we report sensitivity separately from accuracy. In medicine a false negative —
> telling a tumor patient they are fine — is far worse than a false positive. Accuracy alone
> would hide those two missed cases. That is why we show the full confusion matrix."

---

## MINUTE 13–20 — LIVE DEMO PART 2: ANALYSE INDIVIDUAL SCANS

```bash
./run.sh demo
```

This produces every figure. Then open `outputs/demo_tumor_pipeline.png` and walk the panels
left to right, top to bottom:

1. **Original MRI** — "grainy, low contrast, this is what comes off the scanner"
2. **Preprocessed** — "median filter killed the speckle, bilateral filter smoothed the
   tissue *without* blurring the edges"
3. **Enhanced** — "CLAHE. Look at the difference — the lesion has separated from the
   surrounding tissue" *(this is the visual wow moment, pause here)*
4. **Histogram** — "objective proof: the intensity distribution is wider after enhancement.
   Contrast went from 73.7 to 81.4 and entropy from 6.16 to 7.08 bits"
5. **Skull stripped** — "the skull is brighter than the tumor, so we had to remove it first
   or the system would always find the skull"
6. **Segmentation mask** — "threshold plus morphological opening and closing"
7. **Detected region** — "the contour, the bounding box, the exact ROI"
8. **Verdict card** — "TUMOR, 99.8% confidence, region area 1326 px = 5.7% of the brain,
   circularity 0.78, 1.46× brighter than surrounding tissue"

Then show the healthy case (`outputs/demo_healthy_pipeline.png`):

> "Same pipeline, healthy brain. No region passes the threshold, prediction is No Tumor at
> 99.9% confidence. The system does not just say yes to everything."

---

## MINUTE 20–25 — LIVE DEMO PART 3: THE WEB INTERFACE

```bash
./run.sh app
```

A browser opens. Demonstrate three things:

1. **Generate a tumor case** in the sidebar → point at the verdict metrics at the top.
2. Go to the **"Enhancement comparison" tab** → this shows Input vs Global HE vs CLAHE vs
   CLAHE+unsharp side by side. Say:
   > "This is why we chose CLAHE. Global histogram equalisation blows out the skull and
   > amplifies background noise. CLAHE works tile by tile with a clip limit, so it improves
   > local contrast without manufacturing fake structure."

   Then land the measured point — this one always impresses:
   > "And we measured it. Global HE actually *lowers* contrast, from 73.2 down to 68.2,
   > while CLAHE raises it to 76.9 and CLAHE plus unsharp masking to 81.6. The textbook
   > method is measurably worse than doing nothing on this kind of image."
3. **Drag the CLAHE clip-limit slider** live and let them watch the image change. Then go to
   the **"Region measurements" tab** and show the measured properties table.

Stop the app with `Ctrl+C` in the terminal.

---

## MINUTE 25–28 — LIMITATIONS AND FUTURE SCOPE

State the limitations yourself before they ask — it reads as confidence, not weakness:

- Dataset dependence and domain shift between scanners
- Binary only in v1 (not tumor *type*)
- 2D slice, not 3D volume
- Intensity-based segmentation can be fooled by calcifications or artefacts
- Synthetic data proves the pipeline is correct; clinical validity needs real multi-centre
  data and a formal trial

Then future scope: multi-class tumor typing, U-Net segmentation, transfer learning with a
CNN backbone, Grad-CAM explanations, 3D volume estimation, automated PDF reporting.

---

## MINUTE 28–30 — CLOSE AND QUESTIONS

> "To summarise: OpenCV makes the image readable, machine learning makes the first call, and
> the visualisation makes both of those verifiable by a human. We get 98% accuracy with 96%
> sensitivity, the whole thing trains in 15 seconds, and every decision it makes can be
> inspected. That combination — accurate *and* explainable — is what makes it usable in a
> real clinic."

Then take questions. Answers are in `docs/02_VIVA_QUESTIONS_ANSWERS.md`.

---

## EMERGENCY FALLBACKS

**If the live demo fails**, all figures are already saved in `outputs/`. Open them and
present from the images. Nothing in your talk depends on the code running live.

**If they ask to see real data**, say:
> "The pipeline is dataset-agnostic. Drop any Kaggle brain MRI dataset into `data/raw/yes/`
> and `data/raw/no/` and rerun the exact same command — the code auto-detects it and
> retrains. Nothing changes."

**If asked "did you write this or copy it?"** — walk them through
`src/braintumor/segment.py`. Explain the two real bugs you had to solve, which are
documented in the code comments:
1. The brain mask broke when the head touched the image border, because the flood-fill for
   hole-filling could only reach one side of a split background. Fixed by padding a 1-pixel
   frame first.
2. Otsu thresholding flagged 100% of healthy brains as having a lesion, because Otsu always
   splits a histogram in two. Fixed by using a tissue-relative statistical threshold instead.

Those two stories prove authorship better than any explanation.

---

## THE FOUR COMMANDS YOU NEED (write these on your hand)

```bash
./run.sh check      # verify everything works
./run.sh train      # full training run, ~15 seconds
./run.sh demo       # generate all presentation figures
./run.sh app        # launch the web GUI
```
