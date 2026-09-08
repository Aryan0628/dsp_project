"""
STAGE 6 - CLASSIFICATION MODELS
===============================
We do NOT assume a single algorithm is best. We define four candidate models, evaluate all
of them with stratified cross-validation, and let the data pick the winner.

Every candidate is a scikit-learn Pipeline of  StandardScaler -> classifier.
Wrapping the scaler inside the pipeline is not cosmetic: during cross-validation the scaler
is re-fitted on each training fold only, so the test fold's mean and variance never leak
into training. Scaling the full dataset up front is one of the most common and most
invisible sources of over-optimistic results in student ML projects.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config


def build_candidates(seed: int = config.RANDOM_SEED) -> dict[str, Pipeline]:
    """The four models we compare, with a note on why each is a reasonable choice here.

    All of them are given `class_weight="balanced"` where the algorithm supports it. Real
    clinical datasets are lopsided - the 4-class set we train on is 85% tumor - and without
    weighting, a model can score 85% accuracy by predicting "tumor" for everything while
    being clinically worthless. Balanced weighting scales each class's contribution to the
    loss by the inverse of its frequency, so the rare class cannot be ignored for free.
    """
    weight = config.CLASS_WEIGHT
    return {
        # Linear baseline. If a linear model already does well, a complex one is not needed.
        # Always include a baseline - it is what makes a "good" score meaningful.
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=1.0, class_weight=weight,
                                       random_state=seed)),
        ]),

        # SVM with an RBF kernel. Maps features into a higher-dimensional space where a
        # maximum-margin separating hyperplane exists. Strong on moderate-dimensional,
        # medium-sized datasets like ours, and C controls the margin/error trade-off.
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                        class_weight=weight, random_state=seed)),
        ]),

        # Random Forest: an ensemble of decision trees on bootstrapped samples and random
        # feature subsets. Handles non-linear interactions, resists overfitting through
        # averaging, and gives feature importances for free - which is how we explain
        # WHICH measurements drive the decision.
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=400, max_depth=None,
                                           min_samples_leaf=2, n_jobs=-1,
                                           class_weight=weight, random_state=seed)),
        ]),

        # A small neural network on the engineered features. This is our "AI model" in the
        # classical sense and shows the pipeline is not tied to one algorithm family.
        # Note: MLPClassifier has no class_weight parameter, so it is the one candidate that
        # sees the raw class distribution. That is a fair part of the comparison - if the
        # imbalance hurts it, it loses on the balanced CV score and another model is picked.
        "mlp_neural_net": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                                  alpha=1e-3, max_iter=1200, random_state=seed)),
        ]),
    }


def feature_importances(model: Pipeline, names: list[str], top_k: int = 15) -> list[tuple[str, float]]:
    """Return the most influential features, whatever the underlying model type.

    Random Forest exposes impurity-based importances; linear models expose coefficients
    (we take the absolute value since sign only indicates direction). SVM-RBF and MLP have
    no direct equivalent, so we return an empty list rather than inventing a number.
    """
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        scores = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        scores = np.abs(np.asarray(clf.coef_, dtype=float)).ravel()
    else:
        return []

    if scores.size != len(names):
        return []
    order = np.argsort(scores)[::-1][:top_k]
    return [(names[i], float(scores[i])) for i in order]
