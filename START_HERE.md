# START HERE
### You have 30 minutes. Read this page, then start the course in `docs/learn/`.

---

## IF YOU ONLY HAVE 5 MINUTES

Read **[docs/learn/10_REVISION_AND_GLOSSARY.md](docs/learn/10_REVISION_AND_GLOSSARY.md)**: the
whole project on one page, the numbers to memorise, and a final quiz.

Then open **`backend/outputs/demo_tumor_pipeline.png`**. That single image tells the whole story.

---

## THE PROJECT IN ONE SENTENCE

> We built a system that takes a brain MRI, cleans and enhances it with OpenCV,
> automatically finds the suspicious region, and uses machine learning to say whether there
> is a tumor **and what kind it is** — with a confidence score and a picture showing exactly
> what it found. It runs as a web app: Python API, Next.js frontend.

**Detection: 95.4% accuracy, 93.8% sensitivity, 100% specificity, ROC-AUC 0.997.**
**Typing: 80.2% accuracy across glioma / meningioma / pituitary / none.**

---

## RUNNING THE DEMO — TWO TERMINALS

```bash
cd /Users/aryangupta/Documents/dsp_project

./run.sh serve      # terminal 1: the API      -> http://localhost:3000
./run.sh client     # terminal 2: the web app  -> http://localhost:5713
```

Then open **http://localhost:5713** and drag in a scan from `backend/data/samples/` (2 of each class).

### The other commands

```bash
./run.sh check                       # pre-flight check, ~12 s -> 66 tests pass
./run.sh predict <image> --multiclass # analyse one scan from the terminal
./run.sh train-all                   # retrain both models (~10 min, the showstopper)
./run.sh demo                        # regenerate every figure
```

Everything is already trained and every figure is already generated, so you can present
without running anything. The commands are there to prove it is real.

> **If the web app misbehaves on the day**, `./run.sh predict <image> --multiclass` prints
> the same verdict, the same class probabilities and the same measurements to the terminal,
> and every figure is already in `backend/outputs/`. Nothing in the talk depends on the browser.

---

## WHICH DOCUMENT TO READ FOR WHAT

| document | what it gives you | when to read |
|---|---|---|
| **[docs/learn/00_COURSE_MAP.md](docs/learn/00_COURSE_MAP.md)** | step-by-step course (10 lessons) if you are new to ML / image processing | **first, in order** |
| **[docs/learn/10_REVISION_AND_GLOSSARY.md](docs/learn/10_REVISION_AND_GLOSSARY.md)** | one-page recap, key numbers, glossary, quiz | the night before |
| **[SYNOPSIS.md](SYNOPSIS.md)** | your synopsis, updated with what was actually built | to submit |
| **[README.md](README.md)** | project overview and structure | for the repo |

---

## THE FIGURES (all in `backend/outputs/`)

| file | use it for |
|---|---|
| **`demo_tumor_pipeline.png`** | **your main slide** — all 8 stages + the verdict on a tumor case |
| `demo_healthy_pipeline.png` | the healthy case, proving it does not just say yes to everything |
| `demo_enhancement.png` | why CLAHE beats global histogram equalisation |
| `training_report.png` | your results slide — model comparison, confusion matrix, metrics |
| `demo_samples.png` | a grid of several scans, good for a poster |

---

## THE THREE THINGS THAT WILL IMPRESS THEM MOST

**1. The hybrid architecture argument — and knowing where each half stops.**
> "OpenCV does the deterministic image work, machine learning does the pattern recognition.
> Our engineered features produce a visible measurable region a doctor can audit, and they
> answer *is there a tumor* at 93.8% sensitivity. But we measured that they could NOT answer
> *which tumor* — only 69% — because tumor type depends on spatial structure a texture
> histogram averages away. So the typing model adds a fine-tuned ResNet-18 and blends the
> two. We didn't guess where the classical approach ran out; we measured it."

**2. Sensitivity over accuracy.**
> "We report sensitivity separately, because a false negative — telling a tumor patient they
> are fine — is far worse than a false positive. Accuracy alone would hide our 18 missed
> cases. That is why we always show the confusion matrix, and why the decision threshold is
> 0.153 rather than the default 0.5."

**3. The two bugs you actually fixed.** Nothing proves you wrote the code like this:
> "Otsu thresholding is the textbook answer and it failed completely — it flagged 100% of
> healthy brains, because Otsu always splits a histogram in two even when nothing is wrong.
> We replaced it with a tissue-relative statistical threshold and got 100% of tumors with
> 3% false flags."
>
> "And our brain mask broke whenever the head touched the image border, because the
> flood-fill for hole-filling could only reach one side of a split background. We fixed it
> by padding a one-pixel frame first."

Both are documented in the code comments in `backend/src/braintumor/segment.py` and covered by
regression tests.

---

## IF SOMETHING GOES WRONG

- **Live demo fails?** Every figure is already saved in `backend/outputs/`. Present from the images.
  Nothing in the talk depends on code running live.
- **They ask about real data?** "The pipeline is dataset-agnostic. Drop any Kaggle brain MRI
  dataset into `backend/data/raw/yes/` and `backend/data/raw/no/` and rerun the same command. No code
  changes."
- **They ask if it can replace a doctor?** "No, and it is not designed to. It is a triage
  and decision-support tool. The radiologist makes the final call."

---

## THE CLOSING LINE

> "OpenCV makes the image readable, machine learning makes the first call, and the
> visualisation makes both verifiable by a human. 95.4% accuracy and 93.8% sensitivity for
> detection with zero false alarms, 80.2% for naming the tumor type, and every stage of
> every decision can be inspected in the browser. We also know exactly where it stops
> working — glioma typing, at 35% recall — and we measured why. That combination —
> accurate, explainable, and honest about its limits — is what makes it usable in a real
> clinic."
