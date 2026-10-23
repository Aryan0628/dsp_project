# Lesson 2 — The Data: Datasets, Splits and Reproducibility

> **In one minute**
> The project uses real brain MRI scans. The main dataset has 3264 images in four folders: glioma, meningioma, pituitary and no tumor.
> The label of an image is simply the name of the folder it sits in.
> We keep a separate test set that the model never trains on. We use the dataset's own train/test split, because a random split would let near-identical slices from one patient leak into both sides and give a fake-good score.
> The data is imbalanced (about 85% tumors), so we use class weights and report sensitivity and specificity, not accuracy alone.
> A fixed random seed (42) makes the results repeatable.

## What you'll learn

- Which datasets the project uses and what is in them
- How folder names become class labels
- How an image gets from disk into the pipeline
- How train/test splits are made, and why the provided split matters
- Class imbalance and how the project handles it
- What the synthetic dataset is for
- What data leakage is, and the honest caveats that remain
- What `RANDOM_SEED = 42` does and why reproducibility matters

---

## 1. The datasets

**Why do we need to know this?** A model is only as good as its data. In the viva you must be able to say what data you used, how big it is, and where it came from.

The project uses real clinical MRI scans. The loader registry is in `train.py` (a table called `DATASETS`):

| key | what | size | split |
|---|---|---|---|
| `kaggle4` | Brain Tumor Classification (MRI), Sartaj Bhuvaji et al. — the main dataset | 3264 images | **provided** by curators |
| `classic` | Brain MRI Images for Brain Tumor Detection, Navoneel Chakrabarty | 253 images (155 tumor / 98 healthy) | none; random stratified split |
| `synthetic` | generated phantoms from our own simulator | default 400 (`SYNTH_DEFAULT_COUNT` in `config.py`) | none; random stratified split |

Both real datasets are downloaded from Hugging Face by `backend/scripts/download_datasets.py` (about 105 MB, no account needed).

### The main dataset: kaggle4

Classes and counts (totals are the `class_counts` in `backend/outputs/metrics.json`):

| class | total | train folder | test folder |
|---|---|---|---|
| No Tumor | 500 | 395 | 105 |
| Glioma | 926 | 826 | 100 |
| Meningioma | 937 | 822 | 115 |
| Pituitary | 901 | 827 | 74 |
| **all** | **3264** | **2870** | **394** |

The train and test columns are the actual file counts in `backend/data/kaggle4/`. They agree with `n_train = 2870` and `n_test = 394` in `metrics.json`.

Folder layout:

```
 data/
 ├── kaggle4/                      (main dataset, split already provided)
 │    ├── train/
 │    │     ├── glioma_tumor/
 │    │     ├── meningioma_tumor/
 │    │     ├── no_tumor/
 │    │     └── pituitary_tumor/
 │    └── test/
 │          └── (the same four folders)
 ├── raw/                          (the "classic" dataset)
 │     ├── yes/   scans with a tumor
 │     └── no/    scans without
 └── synthetic/                    (generated; same yes/ and no/ layout)
```

The images are messy in realistic ways. Sizes differ. The number of colour channels differs. Different MRI sequences are mixed. In kaggle4, axial, sagittal and coronal slices all appear. This is why `preprocess.py` standardises everything: grayscale, border crop, resize to 256x256, denoise, normalise.

---

## 2. From folder names to labels

**Why do we need to know this?** There is no label spreadsheet. If you understand the folder rule, you understand where every label comes from.

The label of an image is **the name of the folder it sits in**. No separate label file is used. The mapping lives in `dataset.py`, which keeps three lists of folder names:

| list | folder names it recognises | meaning |
|---|---|---|
| positive names | yes, tumor, tumour, positive, 1, brain_tumor | "has a tumor", type unknown |
| negative names | no, notumor, no_tumor, healthy, negative, 0, none | "no tumor" |
| tumor types | glioma / glioma_tumor = 1, meningioma = 2, pituitary / pituitary_tumor = 3 | a specific tumor type |

The helper that decides the label (the `_class_of` function) follows this flow:

```
 folder name
     │
     ▼
 [Normalise: lower-case, "-" and spaces become "_"]
     │
     ▼
 Is it a negative name? ──── yes ──► label 0
     │ no
     ▼
 Is it a tumor-type name? ── yes ──► multiclass mode?  ── yes ──► its type (1, 2 or 3)
     │ no                                              └─ no ───► 1 (just "tumor")
     ▼
 Is it a generic positive name (e.g. "yes")? ── yes ──► multiclass mode? ── yes ──► skipped
     │ no                                                               └─ no ───► 1
     ▼
 Anything else ──► skipped (folder ignored)
```

Why is a generic "yes" folder skipped in multiclass mode? Because "yes" only says a tumor exists. It says nothing about the tumor type.

So the same folders serve both tasks. In binary mode, a scan in `glioma_tumor/` becomes label 1 (tumor). In multiclass mode, the same scan becomes label 1 = Glioma. The index order matches the multiclass names in `config.py`.

### How images are loaded

```
 ┌──────────────────────────────────────────────────────────────┐
 │ 1. List image files under a folder (searched recursively,    │
 │    sorted so the order is stable). Accepted extensions:      │
 │    .png .jpg .jpeg .bmp .tif .tiff                           │
 └───────────────────────────┬──────────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ 2. Walk train/ and test/ (or one flat folder when there is   │
 │    no split) and collect  (file paths, labels)  for each     │
 └───────────────────────────┬──────────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ 3. For every path: read ► preprocess ► enhance ► segment ►   │
 │    extract features. Stack all rows into X, labels into y.   │
 └───────────────────────────┬──────────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ 4. Safety rules:                                             │
 │    - unreadable file: print "skipping <name>", drop it from  │
 │      X and y together so they stay aligned                   │
 │    - nothing processable at all: stop with an error          │
 │    - NaN or infinite feature values: replaced with 0         │
 └──────────────────────────────────────────────────────────────┘
```

The functions are in `dataset.py`: one that lists files, one for split datasets, one for flat folders, and the feature-matrix builder.

---

## 3. Train/test splits

**Why do we need to know this?** The split decides whether our final score is honest. We handle two situations (in `train.py`).

**A. The dataset ships its own split (kaggle4).** The project uses the train and test lists exactly as given. `metrics.json` records `"split_kind": "provided"`.

**B. No split provided (classic, synthetic).** The project does a **random stratified split**:

```
 all images (indices)  +  their labels
          │
          ▼
 ┌─────────────────────────────────────────────┐
 │ Random split                                │
 │   75% ──► training      25% ──► test        │
 │   (test size 0.25, set in config.py)        │
 │   "stratified": both parts keep the same    │
 │   class proportions                         │
 │   seeded with RANDOM_SEED = 42              │
 └─────────────────────────────────────────────┘
```

- **75% / 25%**: the test fraction is 0.25 (`TEST_SIZE` in `config.py`).
- **Stratified** means the class proportions stay the same in both parts. Without it, a small dataset could by bad luck put almost all healthy scans on one side.
- The split is done on *indices* (row numbers), so that features and images stay row-aligned. This is noted in a comment in `train.py`.

### Within training: validation

The test set is never used to make choices. Inside the training data:

- Model selection and threshold tuning use **5-fold stratified cross-validation** (5 folds, shuffled, seeded with the random seed). Lesson 1 shows how the folds rotate.
- For the deep (CNN) model, a fraction of the *training* set is held back (`DEEP_VAL_SIZE = 0.15` in `config.py`). It is used to pick the best epoch and the ensemble weight. The comment in `config.py` says explicitly that this slice is carved out of train, never out of test.

```
 kaggle4 train folder (2870 images)             kaggle4 test folder (394 images)
 ┌─────────────────────────────────┐            ┌──────────────────────────┐
 │ classical model: 5-fold CV      │            │ LOCKED AWAY              │
 │ deep model: 85% fit / 15% val   │            │ used only for the final  │
 │ (all choices are made here)     │            │ reported numbers         │
 └─────────────────────────────────┘            └──────────────────────────┘
```

---

## 4. Why the provided split matters: data leakage

**Why do we need to know this?** This is the most likely "trap" question in the viva, and it shows you understand evaluation.

**Data leakage** means information from the test set sneaks into training. The test score then looks better than real-world performance.

Medical datasets typically contain **several consecutive slices from the same patient**. Neighbouring slices look almost identical. Imagine a random split:

```
 Patient P, slices 13 14 15 16 ...  (nearly identical pictures)

 random split:   slice 14 ──► TRAIN        slice 15 ──► TEST
                      │                          │
                      └── model has effectively already seen slice 15
                          so it "passes" the test by memorising
```

The score then measures memorisation, not generalisation.

The curators of kaggle4 kept sources apart between train and test. So the docstring of the split loader in `dataset.py` calls using their split "the only way to get an honest number."

The project measured the effect. The features and model were identical. Only the split changed:

| splitting strategy | accuracy | sensitivity | ROC-AUC |
|---|---|---|---|
| provided split | 93.65% | 91.35% | 0.9937 |
| random split of the pooled data | 97.21% | 98.50% | 0.9841 |

The random split looks better by about 3.6 accuracy points and 7.2 sensitivity points. That gain is an illusion. We report the lower number.

### A second, related finding: train and test are not from the same source

The project's dataset analysis found that a trivial classifier could tell *which folder* a kaggle4 image came from. It used only 32 histogram bins, the mean, the standard deviation and the image height. Its AUC was as high as 0.858 for glioma. For example, every training glioma image is 512x512, while test gliomas appear in 64 different sizes.

This is a form of **domain shift**: the training and deployment data differ. It is linked to the poor glioma recall. The measured glioma recall in the current typing model is 35% (`per_class_recall` in `metrics.json`).

### Honest caveats you should know before a viva

- **Cross-validation inside the training set is not patient-aware.** The fold splitter has no patient or group information, because the dataset does not provide patient IDs. So near-duplicate slices from one patient can sit in different folds. CV scores may therefore be somewhat optimistic. This only affects model selection and threshold tuning on training data. The final test numbers come from the provided, source-separated test set.
- **The claim "patients are separated" comes from the dataset curators**, not from something the code can check. The code simply trusts the provided split (and prints that statement during training).
- The `classic` dataset has no provided split. Its random split has exactly the leakage risk described above. That is why kaggle4 is the main dataset and `classic` is used for comparison with prior work.

---

## 5. Class balance

**Why do we need to know this?** If one class is much bigger, a lazy model can look good without learning anything.

kaggle4 is **imbalanced**: 500 healthy vs 2764 tumor scans. About 85% of the images are tumors (2764 / 3264 = 84.7%, noted in the comment above `CLASS_WEIGHT` in `config.py`). The test set has 289 tumor and 105 healthy scans.

Here is the problem in a tiny example. A lazy model that always answers "tumor" scores about 85% accuracy. Yet it is useless for a healthy patient. Accuracy hides this.

What the project does about it:

```
 Imbalanced data (85% tumor)
          │
          ├─► 1. Class weights "balanced": mistakes on the rarer class
          │      cost more during training (weight is inverse to class frequency)
          │
          ├─► 2. Stratified split and stratified CV: same proportions everywhere
          │
          ├─► 3. Report balanced accuracy, per-class recall, sensitivity and
          │      specificity, not just accuracy
          │
          └─► 4. Tune the decision threshold for the clinical goal
                 (target sensitivity 0.95, minimum specificity 0.80)
```

For the 4-class task the tumor classes are closer to even (926 / 937 / 901). But "No Tumor" has fewer examples (500), so weighting still applies.

---

## 6. The synthetic dataset

**Why do we need to know this?** You will be asked why fake data is in a project about real scans.

The generator in `dataset.py` draws a fake axial brain slice from scratch:

```
 [Skull ring] ► [CSF] ► [grey and white matter] ► [ventricles] ► [cortical folds]
                                    │
                                    ▼
        optional: [irregular tumor + edema halo (swelling around it)]
                                    │
                                    ▼
        [bias field: smooth brightness gradient across the image]
                                    │
                                    ▼
        [Rician noise: the correct noise model for MRI magnitude images]
                                    │
                                    ▼
        save as PNG in backend/data/synthetic/yes/ or backend/data/synthetic/no/
        (set is balanced: half tumor, half healthy)
```

Its role today is not to produce headline results. It is a **controlled test-bed**. We know the exact tumor location, so segmentation can be checked directly. Training on it (`./run.sh train --dataset synthetic`) works without any download.

Do not quote synthetic results as clinical performance. Simulated data validates the pipeline's mechanics, not real-world accuracy.

---

## 7. Reproducibility: `RANDOM_SEED = 42`

**Why do we need to know this?** If you run the project twice and get different numbers, nobody can trust it.

Many steps involve randomness: shuffling the CV folds, choosing the train/test split, initialising neural-network weights, generating synthetic images, and random augmentation. A **random seed** fixes the starting point of the pseudo-random number generator. The same seed gives the same "random" choices every run. Think of it as shuffling a deck of cards in a way that you can repeat exactly.

Our single source is `RANDOM_SEED = 42` in `config.py`:

```
                       config.py: RANDOM_SEED = 42
                                   │
        ┌──────────────┬───────────┴───────────┬────────────────────┐
        ▼              ▼                       ▼                    ▼
  train.py         dataset.py              deep.py            backend/outputs/metrics.json
  - train/test     - synthetic image       - network weight   - records
    split            generation              initialisation     "random_seed": 42
  - CV fold                                - random numbers
    shuffling                              - augmentation
```

The number 42 has no special meaning. What matters is that it is fixed and recorded.

One caveat: bit-for-bit identical results can still vary across machines, library versions or GPU back-ends for the deep model. So say "reproducible" rather than "guaranteed identical".

---

## Where this lives in our code

| topic | location |
|---|---|
| Paths, seed, split sizes | `backend/src/braintumor/config.py`: dataset folders, `RANDOM_SEED`, `TEST_SIZE`, `CV_FOLDS`, `CLASS_WEIGHT` |
| Folder name to label | `dataset.py`: the label helper and the three name lists |
| Loading with a provided split | `dataset.py`: split loader |
| Loading a flat folder | `dataset.py`: flat-folder loader |
| Choosing a dataset and split kind | `train.py`: the `DATASETS` table and its loader |
| Random stratified split | `train.py`: the train function |
| Feature matrix | `dataset.py`: feature-matrix builder |
| Synthetic generator | `dataset.py`: brain-image generator and synthetic-set writer |
| Download script | `backend/scripts/download_datasets.py` |
| Recorded split info and class counts | `backend/outputs/metrics.json` |

The overall flow, from disk to a trained model:

```
 [download_datasets.py] ──► data/ folders
                                 │
                                 ▼
        [train.py picks a dataset] ──► provided split?  ── yes ──► use train/ and test/ as given
                                              │ no
                                              ▼
                                  random stratified split (75/25, seed 42)
                                 │
                                 ▼
        [dataset.py] folder name ► label, image ► features   ──►  X and y
                                 │
                                 ▼
        [train.py] 5-fold CV on the training part ► best model + threshold
                                 │
                                 ▼
        final score on the locked test part ──► backend/outputs/metrics.json
```

---

## Common viva questions

**Q1. Which dataset do you use, and why that one?**
Mainly the 3264-image four-class Brain Tumor Classification dataset (glioma 926, meningioma 937, pituitary 901, no tumor 500). It supports both binary detection and 4-class typing. It also ships with a curator-provided train/test split. The smaller 253-image dataset is used for comparison with published work.

**Q2. Why not just split randomly?**
Medical datasets have many near-identical slices per patient. A random split can put one slice in training and its neighbour in test, which inflates the score. We measured the inflation. A random split gave 97.21% accuracy and 98.50% sensitivity. The provided split gave 93.65% and 91.35% with the same features and model.

**Q3. Your dataset is about 85% tumor. How do you stop the model from always saying "tumor"?**
We use balanced class weights and stratified splits. We evaluate by sensitivity, specificity and balanced accuracy rather than accuracy alone. We also use a clinically tuned decision threshold.

**Q4. What is data leakage? Is there any left in your pipeline?**
Leakage is information from the test set influencing training. The final test split is kept separate. One honest remaining caveat: our internal 5-fold CV is not grouped by patient (no patient IDs are available), so it may be slightly optimistic. It only guides model and threshold selection, not the reported test numbers. Also, the kaggle4 train and test halves differ in source. That is a distribution shift, not leakage.

**Q5. What does the random seed do?**
It fixes the pseudo-random generator. So the split, CV folds, weight initialisation and synthetic data are the same each run. Ours is `RANDOM_SEED = 42` in `config.py`, recorded in `metrics.json`.

**Q6. Why do you have synthetic data at all?**
It runs with zero downloads. It also gives exact ground-truth tumor locations, which real data lacks. So it is a controlled test-bed for the segmentation stage. It is not used for headline results.

---

## Check yourself

1. In kaggle4, how many images are in train and test, and what folder name holds the healthy scans?
2. In binary mode, what label does a scan in `meningioma_tumor/` get? In multiclass mode?
3. Why does the train/test split use stratification?
4. Why is a random split potentially dishonest on this kind of data?
5. If you change `RANDOM_SEED`, which parts of the pipeline can change?

<details>
<summary>Answers</summary>

1. 2870 train and 394 test (3264 total). Healthy scans are in `no_tumor/`.
2. Binary: 1 (tumor). Multiclass: 2 (Meningioma), by the tumor-type list and the multiclass name order.
3. To keep the class proportions the same in both parts. This matters most when classes are imbalanced or the dataset is small.
4. Neighbouring slices of the same patient can fall on both sides. The test set then contains near-copies of training images, and the score reflects memorisation.
5. The random train/test split (for datasets without a provided one), the CV fold assignment, synthetic data generation, neural-network initialisation and augmentation. So potentially the chosen model, the threshold and the reported metrics too.

</details>

---

**Next:** [Lesson 3 — Preprocessing and Enhancement](03_PREPROCESSING_AND_ENHANCEMENT.md)
