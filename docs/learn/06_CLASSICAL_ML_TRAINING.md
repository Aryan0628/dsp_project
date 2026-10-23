# Lesson 6 - Classical ML Training

> **In one minute**
> Training means letting a classifier adjust itself until its answers match known labels. To be honest about how good it is, we keep some images locked away as a test set. We compare four classifiers (logistic regression, SVM, random forest, MLP) using 5-fold cross-validation on the training images only. The SVM with an RBF kernel wins. Its settings are fixed by hand (no grid search); cross-validation only picks the winner. The winner is refitted on all training data and saved, together with its scaler and tuned threshold, in one `.joblib` file.

## What you'll learn
- What "training" and "overfitting" mean, and why we split data.
- The four classifiers this project compares, in plain terms.
- What a scikit-learn Pipeline and StandardScaler do, and why the scaler lives inside the pipeline.
- Stratified k-fold cross-validation (5 folds) and how the winner is selected.
- What is, and is not, "hyperparameter search" in this project.
- Saving with joblib; binary vs multiclass models; class weighting and the decision threshold.

---

## 1. The basic idea

From Lesson 5 we have a table `X`. Each row is one image, described by 171 numbers (its features). We also have a list `y` of labels, one per image.

A **classifier** is a program that learns a rule: "given these 171 numbers, answer with a label". **Training** means adjusting the classifier's internal settings until its answers match the known labels.

Think of a student revising with past exam papers. If the student only memorises the answers to those exact papers, they will fail a new paper. That is **overfitting**: the model memorises the training images, noise included, and fails on new ones.

To catch overfitting we must test on images the model has never seen. So we split the data:

- **Training set**: used to fit the model and to choose between models.
- **Test set**: locked away. Used once, at the very end, for the honest report.

How the split is made:

- For the 4-class Kaggle dataset we use the split **provided by the dataset**: 2870 train and 394 test images. This keeps slices from the same patient from landing on both sides.
- For datasets with no provided split, the project splits the data itself with a test fraction of 0.25, keeping class proportions equal on both sides (called stratifying), and a fixed random seed of 42 so the split is repeatable.

```
 All labelled images
        │
        ▼
 ┌───────────────────┐
 │  Split (once)     │
 └───────────────────┘
     │            │
     ▼            ▼
 ┌──────────┐  ┌──────────────┐
 │ TRAINING │  │ TEST (locked)│
 │ fit and  │  │ used once at │
 │ choose   │  │ the very end │
 └──────────┘  └──────────────┘
```

---

## 2. The four candidate classifiers

The project defines exactly four candidates (in model.py). Each one is wrapped in the same two-step Pipeline: first a StandardScaler, then the classifier.

### Logistic regression (named `logistic_regression`)
Despite the name, it is a classifier. It takes a weighted sum of the features and squashes it through an S-shaped curve to get a probability between 0 and 1. Its decision boundary is a flat plane.

It is simple, fast and easy to interpret, so it is the **baseline**. If a plain linear model already does well, a fancier model must prove it is worth the extra complexity.

Settings:
- Regularisation strength 1.0. Regularisation is a penalty on complicated solutions. A smaller value means a stronger penalty and a simpler model.
- Up to 3000 optimisation iterations.
- Balanced class weights (explained in section 5).

### SVM with RBF kernel (named `svm_rbf`)
A Support Vector Machine looks for the boundary with the **widest margin**, meaning the biggest empty gap between the two classes.

The RBF kernel lets that boundary bend. Picture each training point as a small hill whose influence fades with distance. Adding up these hills lets the model draw curved boundaries.

Settings:
- Penalty parameter 10.0. This sets how harshly training mistakes are punished. Larger means a more flexible, tighter fit.
- Influence width set automatically from the data (the "scale" option).
- Probability output switched on. This is needed for ROC-AUC and for thresholds. scikit-learn adds an extra internal calibration step to produce the probabilities.

This model **wins** in our runs.

### Random forest (named `random_forest`)
A decision tree is a flowchart of yes/no questions about the features. A random forest builds many trees. Each tree sees a random resample of the images and a random subset of the features. Then all trees vote.

One tree overfits easily. Averaging many different trees cancels out much of that.

Settings:
- 400 trees.
- No limit on tree depth.
- At least 2 images in each leaf.
- Use all CPU cores.

It can also report which features mattered most.

### MLP neural network (named `mlp_neural_net`)
MLP stands for multi-layer perceptron: a small neural network that works on the 171 features. It has two hidden layers of 128 and 64 units, ReLU activation, a small L2 penalty of 0.001, and up to 1200 iterations. It can learn non-linear combinations of features.

Two facts to remember:
- kNN is not used in this project.
- scikit-learn's MLP has no class-weight option, so it is the one candidate without class weighting.

The ResNet-18 CNN (in deep.py) is a separate branch. It is covered in Lesson 7 and is not one of these four.

---

## 3. Scaling and the Pipeline

The 171 features live on very different scales (Lesson 5). An area might be in the thousands while a ratio is between 0 and 1.

**StandardScaler** fixes this. For every column it computes `(value - mean) / standard deviation`. The mean and standard deviation are **learned from the data it is fitted on**. After scaling, every column is centred on 0 with a spread of about 1.

SVM, logistic regression and the MLP need this. Random forests do not strictly need it, but the pipeline applies it to all four for simplicity and fairness.

A **Pipeline** chains steps so they behave like one model:

```
 fit(training data)                 predict(new image)
 ┌────────────────────┐             ┌────────────────────┐
 │ Scaler: learn mean │             │ Scaler: apply the  │
 │ and std, then scale│             │ SAME mean and std  │
 └────────────────────┘             └────────────────────┘
          │                                   │
          ▼                                   ▼
 ┌────────────────────┐             ┌────────────────────┐
 │ Classifier: learn  │             │ Classifier: answer │
 └────────────────────┘             └────────────────────┘
```

**Why must the scaler be inside the pipeline?** Because of cross-validation (next section). The pipeline is re-fitted for each fold, so the scaler only ever sees that fold's training part. If you scaled the whole dataset first, the mean and standard deviation would secretly include the validation images. That is **data leakage**, and it makes scores look better than they really are. It is like a student who glimpsed the exam questions while preparing.

---

## 4. Cross-validation and model selection

**Why not just one split?** One validation split can be lucky or unlucky. **K-fold cross-validation** repeats the experiment several times.

How it works with k = 5 (the project's `CV_FOLDS`):

1. Cut the training set into 5 equal parts, called folds.
2. Train on 4 folds. Validate on the 5th.
3. Repeat 5 times, so each fold is the validation fold exactly once.
4. Average the 5 scores. The mean is a stable estimate. The standard deviation shows how much it wobbles.

The project uses a **stratified** version, with shuffling and seed 42. Stratified means every fold keeps the same class proportions as the whole set. This matters because the data is imbalanced (about 85% tumor in the 4-class set).

```
 Training set split into 5 folds:   [1] [2] [3] [4] [5]

 Round 1:  validate on 1, train on 2 3 4 5  ──► score 1
 Round 2:  validate on 2, train on 1 3 4 5  ──► score 2
 Round 3:  validate on 3, train on 1 2 4 5  ──► score 3
 Round 4:  validate on 4, train on 1 2 3 5  ──► score 4
 Round 5:  validate on 5, train on 1 2 3 4  ──► score 5
                                                │
                                                ▼
                                  mean score  and  spread
```

Each candidate is scored twice:

- **ROC-AUC** (one-vs-rest version for multiclass). It measures how well the model ranks tumors above normals, whatever the threshold. Lesson 8 explains it.
- **Balanced accuracy**: the average recall over classes. It stays honest on imbalanced data.

The winner is the model with the highest mean CV ROC-AUC. Ties are broken by balanced accuracy. This selection is done in train.py.

Real numbers from `backend/outputs/metrics_binary.json` (the stored accuracy value there is balanced accuracy):

| Model | Binary CV ROC-AUC | Binary CV bal. acc. | Multiclass CV ROC-AUC (ovr) |
|---|---|---|---|
| logistic_regression | 0.9841 | 0.9422 | 0.9621 |
| **svm_rbf** | **0.9934** | **0.9501** | **0.9834** |
| random_forest | 0.9874 | 0.8937 | 0.9673 |
| mlp_neural_net | 0.9904 | 0.9443 | 0.9791 |

`svm_rbf` is selected for both the binary and the multiclass task.

After selection, the whole flow is:

```
 ┌────────────────────────────────────────────┐
 │ For each of the 4 candidate pipelines:     │
 │   5-fold stratified CV on TRAINING data    │
 │   record mean ROC-AUC + balanced accuracy  │
 └────────────────────────────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────────┐
 │ Pick highest mean ROC-AUC                  │
 │ (tie-break: balanced accuracy)             │
 └────────────────────────────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────────┐
 │ Refit the winner on ALL training data      │
 └────────────────────────────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────────┐
 │ Evaluate ONCE on the locked test set       │
 └────────────────────────────────────────────┘
```

Headline binary result on the test set (measured in the project's results, `backend/outputs/metrics*.json`): accuracy 95.43%, sensitivity 93.77%, specificity 100%, ROC-AUC 0.9968, at a tuned threshold of 0.153.

### Is there hyperparameter search? Be precise.

**Hyperparameters** are settings chosen by us rather than learned from data. Examples: the SVM penalty, the number of trees, the layer sizes.

In this project the classical models' hyperparameters are **fixed by hand** in `build_candidates`. There is **no grid search** over them. Cross-validation only chooses **which of the four models** is best. Never tell the examiner you grid-searched the SVM.

Two things are tuned, always on training data only:

1. The **decision threshold** (see section 5).
2. The **CNN/classical blend weight**, found by a grid search on a validation split carved out of the training set. This only applies when the deep branch is used.

---

## 5. Class imbalance and the decision threshold

### Class weights
Our data has far more tumor images than normal ones. A lazy model could score well by always saying "tumor".

Setting the class weight to "balanced" fixes this. Each class's contribution to the training penalty is made inversely proportional to how common it is. Rare classes get a louder voice.

Worked example: with 90 tumor images and 10 normal images, each normal image counts about nine times as much as a tumor image. This is applied to logistic regression, SVM and random forest.

### The decision threshold
A classifier that gives probabilities outputs something like "P(tumor) = 0.30". We turn that into a yes/no by comparing with a cut-off.

The default cut-off of 0.5 treats a missed tumor and a false alarm as equally bad. In screening they are not: a missed tumor is far worse.

The project picks a better cut-off like this:

```
 ┌─────────────────────────────────────────────┐
 │ Get out-of-fold probabilities on the        │
 │ TRAINING set (each image predicted by a     │
 │ model that never saw it)                    │
 └─────────────────────────────────────────────┘
                     │
                     ▼
 ┌─────────────────────────────────────────────┐
 │ Try cut-offs; keep only those with          │
 │ specificity >= 0.80 (MIN_SPECIFICITY)       │
 └─────────────────────────────────────────────┘
                     │
                     ▼
 ┌─────────────────────────────────────────────┐
 │ Choose the LOWEST cut-off (capped at 0.5)   │
 │ = highest sensitivity that is still allowed │
 └─────────────────────────────────────────────┘
```

The test set is never used for this. In our run it chose 0.153. On the test set this cut missed tumors from 33 to 18 with no extra false alarms (measured in the project's results).

---

## 6. Binary vs multiclass

- **Binary** task: labels 0 = No Tumor, 1 = Tumor. Metrics: sensitivity, specificity, precision, F1, ROC-AUC and the confusion matrix. Threshold tuning applies.
- **Multiclass** task: labels 0 to 3 = No Tumor, Glioma, Meningioma, Pituitary. It uses the same four candidates and the same features. Metrics: per-class recall and precision, macro F1, one-vs-rest ROC-AUC, plus a "binary view" that collapses to tumor vs no tumor. Threshold tuning is skipped, because it is a binary-only step.

Index 0 is always "No Tumor", so "label greater than 0" means tumor in both tasks.

Honest results to know:

- Detection (is there a tumor?) works very well with these features.
- Typing (which tumor?) is harder. The classical typing model alone was weak: about 69% accuracy on the test set, with glioma recall of 23%. That is why a fine-tuned CNN branch exists for typing.
- The shipped multiclass model (a CNN-dominated ensemble) reaches 80.2% accuracy and 78.1% macro F1 in `backend/outputs/metrics_multiclass.json`.

---

## 7. Saving with joblib

After training, the model must be stored so the app can load it later. **joblib** is a Python library that saves objects to disk efficiently (it is good with NumPy arrays).

The project does not save just the model. It saves a small "bundle" holding everything prediction needs:

```
 ┌───────────────────────────── .joblib bundle ─────────────────────────────┐
 │  model          the fitted Pipeline (scaler + classifier together)       │
 │  feature_names  the 171 column names, so columns can be checked/ordered  │
 │  model_name     which candidate won (e.g. svm_rbf)                       │
 │  task / dataset_key / source   what it was trained for and on            │
 │  class_names    the label names                                          │
 │  threshold      the tuned cut-off (0.5 if none)                          │
 └──────────────────────────────────────────────────────────────────────────┘
```

Because the model is a Pipeline, the fitted scaler travels with it. A new image is scaled exactly as the training images were.

Files:
- `backend/models/brain_tumor_model.joblib` for binary.
- `backend/models/brain_tumor_multiclass.joblib` for multiclass.
- A dataset-tagged copy of each.
- Metrics go to `backend/outputs/metrics.json`.
- CNN weights live in a separate `.pt` file.

Security note: never load a joblib file from an untrusted source. It is built on pickle, which can run arbitrary code.

---

## Where this lives in our code

| File | What it does |
|---|---|
| `backend/src/braintumor/model.py` | Defines the four candidate pipelines. Also reports feature importances (random forest importances, absolute coefficients for logistic regression, nothing for SVM-RBF and MLP). |
| `backend/src/braintumor/train.py` | The whole training run: load, features, split, CV selection, final fit and test. Also threshold tuning, evaluation and model loading. |
| `backend/src/braintumor/config.py` | The constants: seed 42, test size 0.25, 5 CV folds, balanced class weight, minimum specificity 0.80. |

The selection step in train.py, in plain steps:

```
 ┌──────────────────────────────┐
 │ Make a stratified 5-fold     │
 │ splitter (shuffle, seed 42)  │
 └──────────────────────────────┘
               │
               ▼
 ┌──────────────────────────────┐
 │ For each candidate pipeline: │
 │  cross-validate on TRAINING  │
 │  data, all CPU cores         │
 └──────────────────────────────┘
               │
               ▼
 ┌──────────────────────────────┐
 │ Keep the highest mean score  │
 └──────────────────────────────┘
```

---

## Common viva questions

**Q1. Which classifiers did you compare and which won?**
Logistic regression (baseline), SVM with RBF kernel, random forest and an MLP. They were compared by 5-fold stratified CV ROC-AUC. SVM-RBF won (binary CV AUC 0.9934).

**Q2. Why put StandardScaler inside the Pipeline?**
To prevent data leakage. The scaler is re-fitted on each training fold only, so validation statistics never influence training. It also guarantees the same scaling is applied automatically at prediction time.

**Q3. What is cross-validation and why not a single split?**
It means training and validating on several different splits and averaging. One split can be lucky or unlucky, especially on a small dataset. CV gives a mean and a standard deviation. Stratified keeps class ratios in each fold.

**Q4. Did you tune hyperparameters?**
Not by search for the classical models. Their settings are fixed in the code, and CV picks among the four models. We did tune the decision threshold, using out-of-fold predictions on training data. Say this plainly; it is accurate.

**Q5. Why is test data never used to choose the model or threshold?**
Anything chosen using the test set makes the reported score optimistic. Model, threshold and (for the deep branch) blend weight are all chosen using training data only.

**Q6. Why ROC-AUC and balanced accuracy rather than plain accuracy for selection?**
The data is imbalanced, and plain accuracy can be inflated by predicting the majority class. ROC-AUC measures ranking quality independent of the threshold. Balanced accuracy averages recall over classes.

**Q7. How does an SVM work, in one minute?**
It finds the separating boundary with the widest margin between classes. The RBF kernel lets the boundary curve by measuring similarity between points with a distance-based bump. The penalty parameter controls how strongly misclassified training points are punished.

---

## Check yourself

1. Which hyperparameter selection method does the project use for the SVM's penalty and width settings?
<details><summary>Answer</summary>None. They are fixed (penalty 10.0, width "scale"). Cross-validation only selects among the four candidate models.</details>

2. What would go wrong if you scaled all of `X` before cross-validation?
<details><summary>Answer</summary>Data leakage. The mean and standard deviation would be computed using validation rows too, so validation scores would be slightly optimistic. The pipeline avoids this by fitting the scaler per training fold.</details>

3. Why is the MLP the only candidate without class weighting?
<details><summary>Answer</summary>scikit-learn's MLP has no such option. It sees the raw class distribution, and if that hurts it, it loses in the CV comparison.</details>

4. What does stratified k-fold guarantee compared with plain k-fold?
<details><summary>Answer</summary>Each fold has approximately the same class proportions as the whole training set.</details>

5. What is stored inside the `.joblib` file, and why more than just the model?
<details><summary>Answer</summary>A bundle with the fitted Pipeline (scaler plus classifier), the feature names, model name, task, class names and tuned threshold. Prediction can then reproduce the exact same processing and decision rule.</details>

**Next:** [Lesson 7 - Deep Learning Model](07_DEEP_LEARNING_MODEL.md)
