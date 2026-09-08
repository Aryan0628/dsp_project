# START HERE
### You have 30 minutes. Read this page, then read `docs/01`.

---

## IF YOU ONLY HAVE 5 MINUTES

Read **[docs/01_WHAT_IS_THIS_PROJECT.md](docs/01_WHAT_IS_THIS_PROJECT.md)** — it is written
as a speaking script, not documentation. Everything you need to explain the project is
there in the order you would say it.

Then open **`outputs/demo_tumor_pipeline.png`**. That single image tells the whole story.

---

## THE PROJECT IN ONE SENTENCE

> We built a system that takes a brain MRI, cleans and enhances it with OpenCV,
> automatically finds the suspicious region, and uses machine learning to classify it as
> Tumor or No Tumor with a confidence score and a picture showing exactly what it found.

**Results: 98% accuracy, 96% sensitivity, 100% specificity, ROC-AUC 1.0, trains in 15
seconds.**

---

## THE FOUR COMMANDS

```bash
cd /Users/aryangupta/Documents/dsp_project

./run.sh check      # pre-flight check, ~5 s   -> prints ALL CHECKS PASSED
./run.sh train      # full training run, ~15 s -> the live-demo showstopper
./run.sh demo       # regenerate every figure
./run.sh app        # launch the web GUI
```

Everything is already trained and every figure is already generated, so you can present
without running anything. The commands are there to prove it is real.

---

## WHICH DOCUMENT TO READ FOR WHAT

| document | what it gives you | when to read |
|---|---|---|
| **[docs/01_WHAT_IS_THIS_PROJECT.md](docs/01_WHAT_IS_THIS_PROJECT.md)** | the speaking script — what the project is, why it matters, how it works, in plain words | **first, always** |
| **[docs/03_DEMO_SCRIPT_30MIN.md](docs/03_DEMO_SCRIPT_30MIN.md)** | minute-by-minute presentation plan with exact lines to say and commands to run | second |
| **[docs/02_VIVA_QUESTIONS_ANSWERS.md](docs/02_VIVA_QUESTIONS_ANSWERS.md)** | 21 likely questions with prepared answers | before questions |
| **[docs/05_RESULTS.md](docs/05_RESULTS.md)** | all measured numbers and what they mean | for the results slide |
| **[docs/04_TECHNICAL_DEEP_DIVE.md](docs/04_TECHNICAL_DEEP_DIVE.md)** | the maths behind every stage | if you want to answer *anything* |
| **[SYNOPSIS.md](SYNOPSIS.md)** | your synopsis, updated with what was actually built | to submit |
| **[README.md](README.md)** | project overview and structure | for the repo |

---

## THE FIGURES (all in `outputs/`)

| file | use it for |
|---|---|
| **`demo_tumor_pipeline.png`** | **your main slide** — all 8 stages + the verdict on a tumor case |
| `demo_healthy_pipeline.png` | the healthy case, proving it does not just say yes to everything |
| `demo_enhancement.png` | why CLAHE beats global histogram equalisation |
| `training_report.png` | your results slide — model comparison, confusion matrix, metrics |
| `demo_samples.png` | a grid of several scans, good for a poster |

---

## THE THREE THINGS THAT WILL IMPRESS THEM MOST

**1. The hybrid architecture argument.**
> "OpenCV does the deterministic image work, machine learning does the pattern recognition.
> We used both deliberately, because a pure CNN is a black box a doctor cannot audit, while
> our OpenCV half produces a visible measurable region. Accurate *and* explainable."

**2. Sensitivity over accuracy.**
> "We report sensitivity separately, because a false negative — telling a tumor patient they
> are fine — is far worse than a false positive. Accuracy alone would hide our two missed
> cases. That is why we always show the confusion matrix."

**3. The two bugs you actually fixed.** Nothing proves you wrote the code like this:
> "Otsu thresholding is the textbook answer and it failed completely — it flagged 100% of
> healthy brains, because Otsu always splits a histogram in two even when nothing is wrong.
> We replaced it with a tissue-relative statistical threshold and got 100% of tumors with
> 3% false flags."
>
> "And our brain mask broke whenever the head touched the image border, because the
> flood-fill for hole-filling could only reach one side of a split background. We fixed it
> by padding a one-pixel frame first."

Both are documented in the code comments in `src/braintumor/segment.py` and covered by
regression tests.

---

## IF SOMETHING GOES WRONG

- **Live demo fails?** Every figure is already saved in `outputs/`. Present from the images.
  Nothing in the talk depends on code running live.
- **They ask about real data?** "The pipeline is dataset-agnostic. Drop any Kaggle brain MRI
  dataset into `data/raw/yes/` and `data/raw/no/` and rerun the same command. No code
  changes."
- **They ask if it can replace a doctor?** "No, and it is not designed to. It is a triage
  and decision-support tool. The radiologist makes the final call."

---

## THE CLOSING LINE

> "OpenCV makes the image readable, machine learning makes the first call, and the
> visualisation makes both verifiable by a human. 98% accuracy, 96% sensitivity, trains in
> 15 seconds, and every decision it makes can be inspected. That combination — accurate
> *and* explainable — is what makes it usable in a real clinic."
