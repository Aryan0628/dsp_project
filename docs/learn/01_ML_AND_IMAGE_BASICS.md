# Lesson 1 — Images, Image Processing and Machine Learning Basics

> **In one minute**
> A computer sees a brain scan as a grid of numbers, where small numbers are dark and large numbers are bright.
> We first clean and measure that grid with fixed recipes (image processing, also called DSP). Nothing is learned in this step.
> Then a machine-learning model reads the measurements and decides "tumor" or "no tumor". It learned its rule from many labelled examples.
> To know whether the model really works, we test it on scans it never saw during training. Otherwise it could just have memorised the answers.

## What you'll learn

- What a digital image really is: a grid of numbers
- What pixels, grayscale, the 0–255 range and "shape" mean
- The difference between DSP / image processing and machine learning (and why this project uses both)
- Supervised learning, classification (binary vs multiclass), features and labels
- Training vs testing, overfitting, and why we split the data
- What a "model" is, concretely

---

## 1. An image is a table of numbers

**Why do we need to know this?** Every later lesson (filters, thresholds, features) is just arithmetic on a table of numbers. If you see the image as a table, the rest becomes simple.

A computer does not "see" a brain scan. It stores a rectangular grid of numbers. Each cell of the grid is a **pixel** (picture element). Its number says how bright that spot is.

A tiny 3x4 grayscale image might be:

```
  0   0  40  90
  0  60 200 220
 30 180 255 240
```

Dark spots are small numbers. Bright spots are large numbers. That is the entire idea.

**Grayscale and 0–255.** A grayscale image has one number per pixel. In the common 8-bit format (called `uint8`, "unsigned 8-bit integer") each number is a whole number from 0 (black) to 255 (white). This is because 8 bits can hold 2^8 = 256 different values.

**Colour images** have three numbers per pixel: blue, green and red in OpenCV's order (often written BGR). Some files also have a fourth "alpha" (transparency) channel (BGRA).

**Shape.** In Python the grid is held in a NumPy array. Its "shape" tells you its size:

| array shape | meaning |
|---|---|
| (256, 256) | grayscale, 256 rows (height) by 256 columns (width) = 65,536 pixels |
| (256, 256, 3) | colour, 3 channels per pixel |

Watch the order. OpenCV and NumPy report shape as (height, width). But OpenCV's resize function wants the size as (width, height). The project's `config.py` marks this clearly: the image size is (256, 256) written as (w, h). Our target is square, so it makes no difference here. It is still a classic source of bugs.

**Why MRI is grayscale.** An MRI scan does not record colour. Brightness encodes tissue properties (roughly, how tissue responds to the magnetic signal). So one channel carries all the information. The code comment in `preprocess.py` (the grayscale step) says exactly this: MRI encodes tissue density, not colour.

---

## 2. DSP / image processing vs machine learning

**Why do we need to know this?** The viva will almost certainly ask "which part of your project is machine learning and which is not?" You need a clean answer.

These are two different ways of getting value from numbers.

**Digital signal processing (DSP) / image processing** applies a *fixed, hand-designed recipe* to the numbers. A programmer decides the maths in advance. Nothing is learned. Nothing is trained. The same input always gives the same output.

Examples in our project:

- A **median filter** replaces each pixel with the median of its neighbours. This removes speckle noise.
- **CLAHE** (a local contrast-boosting method) makes subtle grey-on-grey differences easier to see.
- **Thresholding** turns "bright enough" pixels white and the rest black.

**Machine learning (ML)** does not get a recipe. It is shown many *examples* with known answers. It then **learns its own rule** from them. The rule is stored as numbers (called parameters) inside the model.

A useful analogy: DSP is a cooking recipe that you follow exactly. ML is a chef who has tasted thousands of dishes labelled "good" or "bad" and has developed a feel for it.

**Why our project uses both.** The classical pipeline cleans and enhances each scan (DSP) and measures things about it. ML then decides *tumor or not* from those measurements.

- The DSP part is auditable. A person can look at the mask and the boxed region.
- The ML part is flexible. Nobody can hand-write an exact rule that separates tumors from healthy brains.

```
 scan file ──► [DSP: clean, enhance, outline region, measure] ──► row of numbers
                         fixed recipes, nothing learned                 │
                                                                        ▼
                                                     [ML model: learned rule]
                                                                        │
                                                                        ▼
                                                          "Tumor" / "No Tumor"
```

---

## 3. Supervised learning, labels and classification

**Why do we need to know this?** These three words describe the exact kind of problem we are solving.

**Supervised learning** means learning from examples where the correct answer is supplied. Each example is a pair:

- the **input** (the scan), and
- the **label** (the correct answer, written by humans).

The model learns a mapping from input to label. Then it can guess the label for a *new* input it has never seen.

**Classification** is supervised learning where the label is a category. The other main kind is *regression*, where the label is a number such as a price.

- **Binary classification**: exactly two classes. In our project these are `No Tumor` (label 0) and `Tumor` (label 1). They are defined as the class names in `config.py`.
- **Multiclass classification**: more than two classes. Ours are `No Tumor`, `Glioma`, `Meningioma`, `Pituitary`, with labels 0, 1, 2, 3.

One detail is worth knowing. The order of the multiclass names is deliberate. A comment in `config.py` says index 0 must stay "No Tumor". Then "label greater than 0 means tumor" is always true. So the code can get the binary answer from a multiclass prediction with no lookup table.

The project therefore trains two models, one for each question:

| model | question | classes |
|---|---|---|
| Detection | Is there a tumor? | binary |
| Typing | Which kind? | 4-class |

---

## 4. Features: turning a picture into a row of numbers

**Why do we need features?** A classical ML algorithm such as an SVM cannot take a 2D picture. It expects each example as a **row of numbers of fixed length**. Those numbers are called **features**.

You could feed in raw pixels. A 256x256 image gives 65,536 numbers. But raw pixels are noisy, and a small dataset is not enough to learn from them. Instead we compute meaningful summaries, for example:

- how bright the image is on average, and how spread out the brightness is (intensity statistics),
- how "rough" or "smooth" the texture is (GLCM and LBP texture measures, explained in a later lesson),
- the area and shape of the region the segmentation stage found,
- left/right asymmetry (the `asym_*` features). A brain is roughly mirror-symmetric, and a tumor often breaks that.

The current trained model uses **171 features** (recorded as `n_features` in `backend/outputs/metrics.json`). Feature names such as `glcm_contrast_d1_mean` and `asym_ratio` are listed there too.

Put together, the data for the classical model is a table:

```
            feature_1  feature_2  ...  feature_171 |  label
scan A        0.31       12.4     ...     0.07      |   1
scan B        0.28       10.9     ...     0.02      |   0
...
```

`X` is the feature table (one row per image). `y` is the label column. This `X` and `y` vocabulary appears all over the code.

**Hand-crafted vs learned features.** The features above are *hand-crafted*. Humans decided what to measure. The project also has a deep-learning branch (a ResNet-18 convolutional neural network, in `deep.py`). It looks at the pixels directly and *learns* its own features. The project's measured results show that hand-crafted features were enough for detection but not for tumor typing. That is why the deep branch exists (see the comment above the deep-epochs setting in `config.py`).

---

## 5. A "model"

**Why do we need to know this?** "Model" is the most used and least explained word in ML. Here is the concrete meaning.

A **model** is a function with adjustable numbers (parameters) inside. **Training** (also called *fitting*) is the process of adjusting those numbers. The goal is that the model's outputs match the known labels on the training examples.

After training, the model is just a saved bundle of numbers plus the code to apply them. Predicting is cheap: features go in, an answer comes out.

In scikit-learn the pattern is always the same two steps:

```
  TRAINING                                   USING IT
  ┌─────────────────────────────┐            ┌─────────────────────────────┐
  │ feature table X_train       │            │ feature rows of new scans   │
  │ + known labels y_train      │            └──────────────┬──────────────┘
  └──────────────┬──────────────┘                           ▼
                 ▼                            ┌─────────────────────────────┐
  ┌─────────────────────────────┐             │ model asks: how likely is   │
  │ "fit": adjust the model's   │──saved──►   │ "tumor"? (a probability)    │
  │ internal numbers to match   │             └──────────────┬──────────────┘
  │ the labels                  │                            ▼
  └─────────────────────────────┘                  probability per scan
```

In our code, each candidate model is a scikit-learn *pipeline* with two steps (built in `model.py`):

```
 feature row ──► [Scaler: rescale every feature to a similar range] ──► [Classifier] ──► probability
```

The scaler matters because a feature measured in thousands would otherwise drown out a feature measured in fractions.

Four classifier families are compared: logistic regression, SVM with an RBF kernel (called `svm_rbf`), random forest, and an MLP (a small neural network). The winner on this data is recorded as `best_model` in `backend/outputs/metrics.json`. It is `svm_rbf`. The fitted model is saved to `backend/models/brain_tumor_model.joblib`.

The model outputs a **probability** of "tumor". A **threshold** then turns that probability into a decision. The project does not use the default 0.5. The shipped threshold is 0.153, chosen to hit the clinical goals set in `config.py` (a target sensitivity and a minimum specificity). The measured value is in the project's results (`backend/outputs/metrics*.json`).

---

## 6. Training vs testing, and overfitting

**Why do we need to know this?** This is the single most important idea for trusting any ML result.

**The core problem.** What we actually want is a model that works on *new* scans. Suppose we only measured performance on the scans it learned from. We would learn nothing about new scans, because a model can simply memorise its training examples.

**Overfitting** is when a model fits the training data so closely (quirks and noise included) that it does worse on unseen data. Analogy: a student memorises last year's exam answers. They score 100% on last year's paper and fail a fresh one.

**The remedy: hold data back.**

- **Training set**: the examples the model learns from.
- **Test set**: examples locked away and used *only once at the end* to estimate real-world performance.

If test performance is far below training performance, that gap is the signature of overfitting.

**Validation / cross-validation.** We also need to *choose* between models and settings. We must do this without peeking at the test set. The project uses **5-fold stratified cross-validation** (5 folds, set in `config.py`). Here is how it works:

```
 Training set, cut into 5 equal parts (keeping class proportions)

 Round 1:  [TEST][ train ][ train ][ train ][ train ]
 Round 2:  [ train ][TEST][ train ][ train ][ train ]
 Round 3:  [ train ][ train ][TEST][ train ][ train ]
 Round 4:  [ train ][ train ][ train ][TEST][ train ]
 Round 5:  [ train ][ train ][ train ][ train ][TEST]
                         │
                         ▼
   average the 5 scores ──► pick best model, tune threshold
   (the real test set is never touched here)
```

Each part takes a turn as a temporary "mini test" while the model trains on the other four. The best model is picked from these scores. The decision threshold is tuned on the out-of-fold predictions (the predictions made on the part that was held out each round). The real test set is not consulted for any of these choices.

**Why this matters for medicine.** A tool that looks 99% accurate on its own training scans but fails at a different hospital is dangerous. That is why the next lesson spends so long on how the data is split.

---

## 7. Why accuracy alone is not enough

**Why do we need to know this?** A missed tumor is worse than a false alarm. Plain accuracy treats both mistakes the same, so we need better measures.

- **Sensitivity (recall)**: of the scans that truly have a tumor, what fraction did we catch?
- **Specificity**: of the truly healthy scans, what fraction did we correctly clear?

A tiny worked example: 10 scans have tumors and the model catches 9. Sensitivity is 9/10 = 90%. Another 10 scans are healthy and the model clears all 10. Specificity is 10/10 = 100%.

The measured detection numbers on the held-out test set (from the project's results, `backend/outputs/metrics*.json`) are:

- accuracy 95.43%
- sensitivity 93.77%
- specificity 100.00%
- ROC-AUC 0.9968
- 18 missed tumors out of 289, and 0 false alarms out of 105

The 4-class typing model reaches 80.20% accuracy (`test_metrics.accuracy` in `backend/outputs/metrics.json`).

---

## Where this lives in our code

| concept | where |
|---|---|
| Read an image as a grayscale array | `backend/src/braintumor/preprocess.py` (load step, reads the file directly as grayscale) |
| Standard size and type | `preprocess.py` outputs an 8-bit grayscale image resized to 256 x 256 (`config.py`) |
| Normalise brightness to 0–255 | `preprocess.py` (min-max normalisation step) |
| DSP stages (fixed recipes) | `preprocess.py`, `enhance.py`, `segment.py` |
| Turn an image into a feature row | `features.py` (feature vector), called from `dataset.py` (single-image processing) |
| Build `X` and `y` for all images | `dataset.py` (feature-matrix builder) |
| The model (scaler + classifier) | `model.py` (candidate builder) |
| Training, CV, evaluation | `train.py` (train and evaluate functions) |
| Class names | `config.py` (binary and multiclass name lists) |

The data path for one image, as done in `dataset.py`:

```
 ┌──────────────────────────┐
 │ Read the file from disk  │   numbers from disk
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ Preprocess               │   grayscale, crop, resize, denoise, normalise
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ Enhance                  │   contrast + sharpening
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ Segment                  │   find the candidate tumor region
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ Extract features         │   one row of 171 numbers
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ Model (fit / predict)    │   ML starts here
 └──────────────────────────┘
```

Everything above the last box is DSP. The model step is ML.

---

## Common viva questions

**Q1. What is the difference between image processing and machine learning?**
Image processing applies fixed, hand-designed operations (filters, thresholds, morphology) with no learning. The same input always gives the same output. Machine learning learns its decision rule from labelled examples. Our project uses image processing to clean, enhance and localise. It uses ML to make the tumor/no-tumor decision from measured features.

**Q2. Why do you convert the images to grayscale?**
MRI intensity encodes tissue characteristics, not colour. So one channel holds all the information. Some source files are 3- or 4-channel, so the grayscale step in the code standardises them to one channel. (The image loader already reads files as grayscale.)

**Q3. What is supervised learning, and what is the label here?**
It is learning from input/answer pairs. The input is a scan (represented by its features). The label comes from the dataset's folder name. For example, a scan in the `no_tumor` folder has label 0. The folder-to-label mapping is in `dataset.py`.

**Q4. What is a feature, and why not feed raw pixels to the SVM?**
A feature is a number that summarises something meaningful about the image (texture, intensity spread, region shape). Raw pixels would mean tens of thousands of noisy, highly redundant inputs for only a few thousand examples. That encourages overfitting. Features compress the image into 171 informative numbers.

**Q5. What is overfitting and how do you guard against it?**
Overfitting is when a model learns training-set quirks and fails on new data. We guard against it in three ways. We keep a test set the model never trains on. We use 5-fold cross-validation, on training data only, for model selection and threshold tuning. And we use a provided train/test split that separates sources (next lesson).

**Q6. Binary vs multiclass: why have both?**
Detection (is there a tumor) is clinically the first question, and the classical features answer it well. Typing (glioma / meningioma / pituitary) is a harder 4-class problem. We train a separate model for each. The multiclass index order lets us derive a binary answer from it.

---

## Check yourself

1. A grayscale image has shape (256, 256) and type `uint8`. How many pixels is that, and what is the largest value a pixel can hold?
2. Is a median filter machine learning? Why or why not?
3. What is the difference between a feature and a label?
4. Why can't we just report accuracy on the data the model was trained on?
5. In the multiclass names, why must "No Tumor" stay at index 0?

<details>
<summary>Answers</summary>

1. 256 x 256 = 65,536 pixels; the maximum is 255.
2. No. It is a fixed formula (replace each pixel with the median of its neighbourhood) with nothing learned from data, so it is image processing / DSP.
3. A feature is an input measurement describing a scan. The label is the correct answer we want the model to predict (e.g. tumor / no tumor).
4. The model may have memorised those examples (overfitting), so the score would not tell us how it behaves on new scans.
5. So that "label greater than 0 means tumor" holds. This lets binary results be derived from multiclass predictions without a lookup table (comment in `config.py`).

</details>

---

**Next:** [Lesson 2 — The Data](02_THE_DATA.md)
