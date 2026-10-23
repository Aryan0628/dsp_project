# Lesson 08 - Evaluation Metrics

> **In one minute**
> Accuracy alone can fool you: a program that always says "healthy" scores 95% when 95 of 100 scans are healthy, yet finds no tumors. So we count four kinds of outcome (true and false positives and negatives) and compute sensitivity, specificity, precision, F1 and AUC from them. In medical screening, sensitivity (catching real tumors) matters most, so we tune the decision threshold for it. Our detection model reaches 95.43% accuracy and 93.77% sensitivity; our 4-class typing reaches 80.20% accuracy but only 35% recall on glioma, which we report openly.

## What you'll learn

- How to read a confusion matrix, with a worked numeric example.
- Accuracy, precision, recall (sensitivity), specificity and F1, and when each misleads.
- ROC curves and AUC, and what a decision threshold does.
- Why sensitivity matters most in medical screening.
- How these extend to 4 classes (per-class, macro and balanced metrics).
- The project's real results and its honest limitations.

---

## 1. Why we need more than "accuracy"

Suppose 95 of 100 scans are healthy. A useless program that always says "healthy" scores 95% accuracy and finds zero tumors.

Accuracy hides *which* mistakes are made. So we split the results into types of right and wrong.

---

## 2. The confusion matrix

For a yes/no question ("is there a tumor?"), call **tumor = positive**. Every prediction falls into one of four boxes:

|  | predicted normal | predicted tumor |
|---|---|---|
| **actually normal** | TN (true negative: correctly cleared) | FP (false positive: false alarm) |
| **actually tumor** | FN (false negative: missed tumor) | TP (true positive: tumor caught) |

A good way to remember: the second letter (N or P) is what the model *said*. The first letter (T or F) says whether the model was *right*.

### Worked example: the project's detection model

On the 394-image test set: 105 normal and 289 tumor scans.

|  | predicted normal | predicted tumor |
|---|---|---|
| **actually normal** | 105 (TN) | 0 (FP) |
| **actually tumor** | 18 (FN) | 271 (TP) |

Now every metric is arithmetic on these four numbers.

| Metric | Question it answers | Formula | Value here |
|---|---|---|---|
| **Accuracy** | What share of all answers were right? | (TP + TN) / total | (271 + 105) / 394 = **95.43%** |
| **Recall / Sensitivity** | Of the real tumors, how many did we catch? | TP / (TP + FN) | 271 / 289 = **93.77%** |
| **Specificity** | Of the healthy scans, how many did we correctly clear? | TN / (TN + FP) | 105 / 105 = **100%** |
| **Precision** | When we say tumor, how often are we right? | TP / (TP + FP) | 271 / 271 = **100%** |
| **F1** | One balanced score of precision and recall | 2PR / (P + R) | 2(1.0)(0.9377) / 1.9377 = **96.79%** |

The F1 uses the harmonic mean. It is dragged toward the lower of the two numbers, so a model cannot hide poor recall behind great precision (or the other way round).

**Balanced accuracy** is the average of sensitivity and specificity. More generally, it is the mean recall over all classes. Here: (93.77 + 100) / 2 = **96.89%**. It is the honest headline when classes are imbalanced, as here (289 tumor vs 105 normal).

```
 Test set: 394 scans
     │
     ├── 105 normal ──► 105 cleared (TN)   0 false alarms (FP)
     │
     └── 289 tumor  ──► 271 caught (TP)   18 missed (FN)
```

---

## 3. Why sensitivity matters in medicine

The two errors do not cost the same.

- A **false negative** means a patient with a tumor is told they are fine. The delay can be harmful or fatal.
- A **false positive** means a healthy patient is sent for a follow-up. It is stressful and costly, but usually recoverable.

So for screening we aim for high sensitivity while keeping specificity acceptable. The project does this deliberately. In train.py, the threshold-tuning step chooses the probability cut-off like this:

```
 ┌──────────────────────────────────────────┐
 │ Use out-of-fold predictions on the       │
 │ TRAINING data (never the test set)       │
 └──────────────────────────────────────────┘
                   │
                   ▼
 ┌──────────────────────────────────────────┐
 │ Keep only cut-offs where specificity     │
 │ is at least 80%                          │
 └──────────────────────────────────────────┘
                   │
                   ▼
 ┌──────────────────────────────────────────┐
 │ Choose the one with the highest          │
 │ sensitivity (lowest cut-off, max 0.5)    │
 └──────────────────────────────────────────┘
```

The resulting threshold is **0.153** instead of the default 0.5. On the test set it cut missed tumors from 33 to 18, with no extra false alarms.

---

## 4. Threshold, ROC curve and AUC

Most models output a probability. We turn it into a yes/no with a **threshold**: predict "tumor" if probability is at least the threshold.

Think of a smoke alarm's sensitivity dial. Turn it up and it catches every fire but also goes off when you make toast. Lowering the threshold catches more tumors (sensitivity up) but raises false alarms (specificity down). It is a trade-off dial, not a fact about the model.

The **ROC curve** shows this whole trade-off at once. Each point on it is one threshold:

- x-axis: false positive rate (1 minus specificity).
- y-axis: true positive rate (sensitivity).

A curve hugging the top-left corner is excellent. The diagonal line is random guessing.

**AUC** (area under the ROC curve) squeezes the curve into one number from 0.5 (random) to 1.0 (perfect). A useful reading: it is the probability that the model gives a randomly chosen tumor scan a higher score than a randomly chosen normal scan. AUC measures **ranking quality, independent of any threshold**.

Our detection model has ROC-AUC **0.9968**.

AUC and accuracy are separate things. In the project's results, adding location features raised AUC (0.9938 to 0.9968), yet accuracy fell (97.72% to 95.43%). The tuned threshold moved from 0.140 to 0.153, and 15 borderline scans fell on the other side. Better ranking, different operating point.

---

## 5. Multiclass: per-class metrics

With 4 classes (No Tumor, Glioma, Meningioma, Pituitary) there is no single "positive". The confusion matrix becomes 4x4. Rows are the actual class and columns are the predicted class.

Recall and precision are computed **one class at a time**, treating it as "this class vs all the others":

- **Recall of class c** = correct predictions of c / all real c images (the row total).
- **Precision of class c** = correct predictions of c / all images predicted as c (the column total).

The project's 4-class test matrix:

| | No Tumor | Glioma | Meningioma | Pituitary | row total |
|---|---|---|---|---|---|
| **No Tumor** | 105 | 0 | 0 | 0 | 105 |
| **Glioma** | 24 | 35 | 40 | 1 | 100 |
| **Meningioma** | 0 | 1 | 114 | 0 | 115 |
| **Pituitary** | 6 | 0 | 6 | 62 | 74 |

Worked examples:

- Overall accuracy = diagonal / total = (105 + 35 + 114 + 62) / 394 = 316 / 394 = **80.20%**.
- Glioma recall = 35 / 100 = **35%**.
- Glioma precision = 35 / (0 + 35 + 1 + 0) = 35 / 36 = 97.2%. So when the model says glioma it is nearly always right, but it says it far too rarely.
- Meningioma recall = 114 / 115 = 99.13%. But its precision is 114 / (0 + 40 + 114 + 6) = 114 / 160 = 71.3%, because 40 gliomas and 6 pituitary tumors were labelled meningioma.

Aggregating over classes:

- **Macro average**: compute the metric for each class, then take the plain mean. Every class counts equally, whatever its size. **Macro F1 = 78.10%**.
- **Balanced accuracy** = mean of per-class recalls = (100 + 35 + 99.13 + 83.78) / 4 = **79.48%**.
- **One-vs-rest ROC-AUC (macro)**: for each class, treat it as "positive" against all others, compute an AUC, and average: **0.9610**.

Plain accuracy (80.20%) and balanced accuracy (79.48%) are close here. But the per-class view reveals what both averages hide: glioma at 35%.

---

## 6. Where honest evaluation can go wrong

- **Held-out test set.** Every number must come from images the model never trained or tuned on. The project selects models by cross-validation on the training set, and picks the threshold and ensemble weight without touching the test set.
- **Cross-validation** (5-fold, stratified) repeatedly trains on part of the training data and scores on the rest. It gives a mean and a standard deviation instead of one lucky number.
- **Validation vs test gap.** The CNN reached 98.57% balanced accuracy on its validation split but 79.48% on the test split. That gap is evidence of dataset shift, not a bug. The validation images came from the same distribution as the training images, and the test images did not.

---

## Where this lives in our code

| File | What it does |
|---|---|
| `backend/src/braintumor/train.py` | The evaluation function computes all metrics. The threshold-tuning function picks the cut-off. |
| `backend/outputs/metrics.json`, `backend/outputs/metrics_*.json` | Where the results are saved. |
| `backend/src/braintumor/deep.py` | Picks the CNN's best epoch and the ensemble weight using balanced accuracy. |

What the evaluation function does for the binary task:

```
 true labels + predictions + probabilities
                   │
                   ▼
 ┌───────────────────────────────────────┐
 │ Build the 2x2 confusion matrix        │
 │ and read off TN, FP, FN, TP           │
 └───────────────────────────────────────┘
                   │
                   ▼
 ┌───────────────────────────────────────┐
 │ Compute accuracy, sensitivity,        │
 │ specificity, precision, F1            │
 │ (a zero denominator gives 0)          │
 └───────────────────────────────────────┘
                   │
                   ▼
 ┌───────────────────────────────────────┐
 │ Compute ROC-AUC from the probabilities│
 └───────────────────────────────────────┘
```

For multiclass it additionally stores: the full 4x4 matrix, macro F1, per-class recall and precision, one-vs-rest macro AUC, and a "binary view" that collapses the 4 classes to tumor vs no tumor.

Selection of the deep model's best epoch and the ensemble weight uses balanced accuracy, deliberately. Plain accuracy would allow the network to give up on the glioma class and still look better.

---

## The project's actual results

All on held-out test sets (measured in the project's results, `backend/outputs/metrics*.json`).

**Detection (tumor vs none, 394 test images):**
- Accuracy 95.43%, sensitivity 93.77%, specificity 100%, precision 100%, F1 96.79%, ROC-AUC 0.9968.
- Missed tumors: 18 of 289. False alarms: 0 of 105.
- Model: SVM (RBF) on 171 features, threshold 0.153.

**Typing (4 classes):**
- Accuracy 80.20%, balanced accuracy 79.48%, macro F1 78.10%, one-vs-rest AUC 0.9610.
- Per-class recall: No Tumor 100%, Glioma 35%, Meningioma 99.13%, Pituitary 83.78%.

## Honest limitations

- **Glioma typing is weak** (35% recall; 40 gliomas called meningioma, 24 called No Tumor). The 24 gliomas called "No Tumor" would be dangerous in practice. So detection should use the dedicated binary model, which has higher sensitivity than the typing model's 89.62%.
- **Dataset shift.** The training and test halves differ. For example, all training gliomas are 512x512, while test gliomas come in 64 sizes. Results therefore measure generalisation under shift, which is why they are lower than validation scores.
- **Small, imbalanced test set** (105 normal vs 289 tumor). On 394 images a handful of scans moves accuracy by points. Also, 100% specificity on 105 normals would very likely fall in a real screening population, which is overwhelmingly healthy. Precision would then fall too, since false positives would be counted against far more normal scans.
- **Single dataset, 2D slices.** Not multi-hospital validation, not 3D.
- **The CNN is not interpretable**; the classical branch is.
- The system is assistive only, not a medical device.

---

## Common viva questions

**Q1. Why not just report accuracy?**
Accuracy can be high while missing the cases that matter, especially with imbalanced classes. We report sensitivity, specificity, precision, F1, balanced accuracy and AUC, plus the confusion matrix, so each kind of error is visible.

**Q2. Define sensitivity and specificity, and say which matters more here.**
Sensitivity = TP / (TP + FN), the share of real tumors found. Specificity = TN / (TN + FP), the share of healthy scans correctly cleared. For screening, sensitivity matters more because a missed tumor costs more than a false alarm. We keep a specificity floor (80%) so the system stays useful.

**Q3. Why is the threshold 0.153 and not 0.5?**
0.5 treats both errors as equal. We choose the cut-off that maximises sensitivity subject to specificity of at least 80%, using out-of-fold predictions on training data only. On the test set this reduced missed tumors from 33 to 18 with no added false alarms.

**Q4. What does AUC of 0.9968 mean?**
There is about a 99.7% chance that a random tumor scan gets a higher tumor-probability than a random normal scan. It measures ranking quality across all thresholds, independent of the chosen threshold.

**Q5. Accuracy went down but AUC went up. Is that a contradiction?**
No. AUC judges the ranking. Accuracy judges one specific threshold. The better ranking was cut at a slightly different threshold (0.153 vs 0.140), and 15 borderline scans changed sides.

**Q6. How do you evaluate a 4-class problem?**
With per-class recall and precision from the 4x4 confusion matrix, then macro F1, balanced accuracy and one-vs-rest AUC. Each class counts equally, so a weak class like glioma is not hidden by the strong ones.

**Q7. What is your biggest weakness and how do you report it?**
Glioma recall of 35%, partly caused by the dataset's inconsistent glioma split. We report it openly, show the confusion matrix, and recommend the binary model for detection.

---

## Check yourself

1. A test set has 50 tumor and 50 normal scans. The model finds 40 tumors and wrongly flags 5 normals. Compute sensitivity, specificity and precision.
<details><summary>Answer</summary>TP = 40, FN = 10, FP = 5, TN = 45. Sensitivity = 40/50 = 80%. Specificity = 45/50 = 90%. Precision = 40/45 = 88.9%.</details>

2. A model predicts "healthy" for everything on a set with 95 healthy and 5 tumor scans. What is its accuracy, sensitivity and balanced accuracy?
<details><summary>Answer</summary>Accuracy 95%, sensitivity 0%, balanced accuracy (0 + 100)/2 = 50%. This is why accuracy alone misleads.</details>

3. Lowering the decision threshold: what happens to sensitivity and specificity?
<details><summary>Answer</summary>Sensitivity rises or stays the same (more scans are called tumor). Specificity falls or stays the same (more healthy scans are flagged).</details>

4. In the 4-class matrix, how is glioma recall 35% while glioma precision is about 97%?
<details><summary>Answer</summary>Recall divides by the 100 real gliomas (only 35 found). Precision divides by the 36 scans predicted as glioma (35 correct). The model rarely says glioma, but is usually right when it does.</details>

5. Why do we use balanced accuracy to select the deep model's best epoch?
<details><summary>Answer</summary>It averages recall over classes, so the model cannot improve its score by abandoning the hard class (glioma) while getting the easy classes slightly better, which plain accuracy would allow.</details>

Next: [09 - Prediction, API & App](09_PREDICTION_API_AND_APP.md).
