# Learn This Project — Step by Step

A short course for someone who knows basic Python but **no machine learning**.
Read the lessons in order. Each one builds on the previous one and ends with
viva-style questions and a self-check.

| # | Lesson | You will be able to explain… | Time |
|---|---|---|---|
| 01 | [ML & Image Basics](01_ML_AND_IMAGE_BASICS.md) | what an image is to a computer, what "training a model" means | 30 min |
| 02 | [The Data](02_THE_DATA.md) | which scans we used, how they're labelled and split | 20 min |
| 03 | [Preprocessing & Enhancement](03_PREPROCESSING_AND_ENHANCEMENT.md) | denoising filters, CLAHE, sharpening | 40 min |
| 04 | [Segmentation](04_SEGMENTATION.md) | skull stripping, thresholding, morphology, contours | 40 min |
| 05 | [Feature Extraction](05_FEATURE_EXTRACTION.md) | turning an image into a list of numbers (GLCM, LBP, shape…) | 40 min |
| 06 | [Classical ML Training](06_CLASSICAL_ML_TRAINING.md) | the classifiers, cross-validation, model selection | 40 min |
| 07 | [Deep Learning Model](07_DEEP_LEARNING_MODEL.md) | CNNs and how we classify tumor *type* | 40 min |
| 08 | [Evaluation Metrics](08_EVALUATION_METRICS.md) | accuracy, sensitivity, specificity, ROC-AUC — and our results | 30 min |
| 09 | [Prediction, API & App](09_PREDICTION_API_AND_APP.md) | the full journey of one uploaded scan | 30 min |
| 10 | [Revision & Glossary](10_REVISION_AND_GLOSSARY.md) | one-page recap, glossary, final quiz | 20 min |

## The big picture (keep this in your head the whole time)

```
 MRI image ──► 1. Preprocess ──► 2. Enhance ──► 3. Segment ──► 4. Features ──► 5. Classify ──► Verdict
 (pixels)       (denoise,         (CLAHE,        (find the       (numbers that   (ML model)     + confidence
                 resize)           sharpen)       suspicious      describe it)                  + picture
                                                  region)
 └──────────── classical image processing / DSP ──────────┘ └──────────── machine learning ──────────┘
```

- Stages 1–3 are **fixed maths** (no learning): the same input always gives the same output.
- Stages 4–5 are where the computer **learns from labelled examples**.
- A separate deep-learning model (lesson 07) predicts the **tumor type**.

## How to study

1. Read a lesson, then open the code files it points to and find the functions it names.
2. Answer the "Check yourself" questions **before** opening the answers.
3. Run things: `./run.sh check` (tests), `./run.sh predict <image> --multiclass`.
4. Finish with lesson 10's final quiz. If you can answer it without peeking, you are ready.
