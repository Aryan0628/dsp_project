# 02 — VIVA / Q&A PREPARATION
### Every question they are likely to ask, with a ready answer.

---

## A. CONCEPT QUESTIONS

**Q1. Why did you choose OpenCV instead of doing everything with deep learning?**
Two reasons. First, explainability — a CNN gives a label but a doctor cannot audit it.
Our OpenCV stages produce a visible, measurable region, so the reasoning is inspectable.
Second, data efficiency — deep learning needs thousands of labelled scans; classical
feature engineering plus a classical classifier works well on a few hundred. We also
designed the architecture so a CNN can be dropped into Stage 6 later without touching
Stages 1-5.

**Q2. What exactly is CLAHE and why not ordinary histogram equalisation?**
Ordinary histogram equalisation computes one intensity mapping for the whole image. On a
brain MRI, the skull is very bright and dominates the histogram, so global equalisation
saturates the bright areas and leaves the subtle grey-on-grey tumor contrast unimproved.
CLAHE divides the image into an 8x8 grid of tiles and equalises each tile independently,
so local contrast improves everywhere. "Contrast Limited" means the histogram is clipped at
a limit (we use 2.0) before equalising, and the clipped excess is redistributed uniformly.
Without that clipping, equalisation would massively amplify noise in near-uniform regions.
Tile boundaries are removed by bilinear interpolation.

**Q3. What is Otsu's thresholding and why is it better than a fixed threshold?**
A fixed threshold like 127 fails as soon as the image brightness changes. Otsu's method
examines the image histogram and searches every possible threshold for the one that
**minimises the intra-class variance** — equivalently maximises the between-class variance —
of the two resulting pixel groups. It is fully automatic and adapts per image. Its
assumption is a bimodal histogram, which holds reasonably well after skull stripping.

**Q4. Explain erosion, dilation, opening and closing.**
- **Erosion** shrinks bright regions — a pixel stays white only if all pixels under the
  structuring element are white. Removes small white noise.
- **Dilation** grows bright regions — a pixel becomes white if any pixel under the element
  is white. Fills small gaps.
- **Opening** = erosion then dilation. Removes small bright specks while keeping the size
  of large objects roughly unchanged.
- **Closing** = dilation then erosion. Fills small dark holes inside bright objects.
We open first to delete noise, then close to make the tumor blob solid.

**Q5. What is GLCM and why is it good for tumors?**
Grey Level Co-occurrence Matrix. It is a second-order statistical texture measure: for a
given offset (distance d and angle theta) it counts how often a pixel of intensity i occurs
next to a pixel of intensity j. From that matrix we derive Haralick features —
**contrast** (local intensity variation), **homogeneity** (how close values are to the
diagonal), **energy** (uniformity), **correlation** (linear dependency), **dissimilarity**
and **ASM**. It matters because tumor tissue is texturally heterogeneous and chaotic while
healthy white matter is smooth and regular — a difference in *texture*, which first-order
statistics like plain mean and variance completely miss. We compute it at distances 1,2,3
and angles 0, 45, 90, 135 degrees so the descriptor is rotation-robust.

**Q6. What is LBP?**
Local Binary Pattern. For each pixel, compare it to its P neighbours on a circle of radius
R. Each neighbour contributes a bit — 1 if brighter than the centre, 0 otherwise — producing
an integer code per pixel. The histogram of those codes describes micro-texture. We use the
"uniform" variant, which groups codes with at most two bit-transitions and is
grey-scale-monotonic invariant — meaning it is robust to overall brightness changes between
scanners.

**Q7. Why SVM? / Why did you compare several models?**
We did not assume. We train Logistic Regression, SVM with an RBF kernel, Random Forest and
an MLP, evaluate all of them with stratified 5-fold cross-validation, and automatically
select the best by ROC-AUC. SVM with RBF tends to do well here because our feature space is
moderate-dimensional and the classes are not linearly separable; the RBF kernel maps into a
higher-dimensional space where a maximum-margin hyperplane exists. Random Forest is a strong
alternative and gives us feature importances for free.

**Q8. Why is accuracy not enough as a metric?**
Because of class imbalance and asymmetric cost. If 90% of scans are healthy, a model that
always says "no tumor" scores 90% accuracy and is clinically worthless. More importantly, a
**false negative — telling a tumor patient they are fine — can be fatal**, while a false
positive only causes an extra review. So we report **sensitivity/recall** (of all real
tumors, how many did we catch), **specificity**, **precision**, **F1**, **ROC-AUC**, and the
full confusion matrix. We prioritise sensitivity.

**Q9. What is the difference between segmentation and classification here?**
Classification is image-level: one label for the whole scan. Segmentation is pixel-level:
which pixels belong to the tumor. We do classical segmentation in Stage 4 both to localise
the tumor for the doctor and to extract shape features that feed the classifier. So
segmentation serves two purposes: explanation and feature generation.

**Q10. Why a bilateral filter instead of a plain Gaussian blur?**
A Gaussian blur averages over a neighbourhood regardless of content, so it blurs edges —
and the tumor boundary is exactly the edge we need to keep. A bilateral filter weights
neighbours by **both** spatial distance **and** intensity similarity, so pixels across a
strong edge barely contribute. Result: noise is smoothed inside regions, edges stay sharp.

**Q11. Why Rician noise in your simulator and not Gaussian?**
MRI raw data is complex-valued with Gaussian noise in the real and imaginary channels.
Scanners display the **magnitude** image, and the magnitude of a complex Gaussian is
**Rician-distributed**. In low-signal (dark) regions this makes noise non-zero-mean and
biased upward, which is why MRI backgrounds look grainy rather than black. Simulating it
correctly means our denoising stage is tested against realistic conditions.

**Q12. What is the vanishing/overfitting risk and how did you handle it?**
We use stratified k-fold cross-validation rather than a single split, a held-out test set
never seen during model selection, a fixed random seed for reproducibility, regularisation
in the SVM (C parameter) and depth limits in the Random Forest, and StandardScaler fitted
**inside** the pipeline so scaling statistics never leak from test to train.

---

## B. CODE / IMPLEMENTATION QUESTIONS

**Q13. Walk me through the code structure.**
`src/braintumor/` holds one module per pipeline stage — `preprocess.py`, `enhance.py`,
`segment.py`, `features.py` — so each stage of the block diagram maps to exactly one file.
`dataset.py` handles both the synthetic generator and real-folder loading. `model.py` builds
the scikit-learn pipelines, `train.py` runs training and evaluation, `predict.py` does
single-image inference, `visualize.py` renders the 6-panel figure, `cli.py` is the command
line entry point, and `app/streamlit_app.py` is the web GUI.

**Q14. Why a scikit-learn Pipeline instead of scaling manually?**
To prevent **data leakage**. If you scale the whole dataset before splitting, the test set's
mean and standard deviation influence the training data. Wrapping the scaler and classifier
in a Pipeline means that during cross-validation the scaler is re-fitted on each training
fold only. It also makes the saved model self-contained — one joblib file that does scaling
and prediction together.

**Q15. How long does training take?**
Seconds to a couple of minutes on CPU for a few hundred images, because we use classical
features rather than training a deep network. That is a deliberate design advantage — the
whole project is reproducible live during a demo.

**Q16. What happens if a non-brain image is uploaded?**
The pipeline still runs, but the segmentation and feature values fall outside the learned
distribution and the confidence score drops. A production version would add an
out-of-distribution check — for example, verify a single large elliptical connected
component consistent with a head — and refuse to classify otherwise. We list this as
future work.

---

## C. TRICKY / PRESSURE QUESTIONS

**Q17. Can this replace a radiologist?**
No, and it is not designed to. It is a triage and decision-support tool. It reduces reading
time and prioritises urgent scans. Regulatory approval, prospective clinical trials and
multi-centre validation would be prerequisites for anything more, and even approved tools
worldwide operate in an assistive role with the clinician making the final call.

**Q18. Your accuracy looks high — is the problem too easy or are you overfitting?**
On synthetic data the accuracy is high because the generator produces a clean, controlled
separation between classes — its purpose is to verify the pipeline is correct end to end,
not to claim clinical performance. On real clinical data we would expect lower numbers and
more variance. We report results from a held-out test set the model never saw, and we are
explicit about which dataset each number comes from.

**Q19. What was the hardest part?**
Segmentation robustness. The skull is brighter than the tumor, so naive thresholding always
selected the skull. We solved it by skull-stripping first — threshold, take the largest
connected component as the brain mask, erode it slightly to drop the bright rim, and only
then apply Otsu inside the brain region.

**Q20. What would you do differently with more time?**
Train a U-Net for true pixel-level segmentation with a Dice-score metric, add transfer
learning with a pretrained CNN backbone and compare it head-to-head against our feature-based
model, validate on a real multi-institution dataset, and add Grad-CAM so the deep model
becomes as explainable as the classical one.

**Q21. Who does what in your team?** *(agree on this before you walk in)*
Split it by pipeline stage: preprocessing + enhancement / segmentation + features /
model training + evaluation / GUI + documentation. Everyone should still be able to explain
the whole pipeline end to end.
