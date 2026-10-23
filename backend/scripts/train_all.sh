#!/usr/bin/env bash
# Retrain every shipped model, in the order the project depends on them.
#
# Both runs are needed after any change to Stage 5: the saved bundles record the exact
# feature-name list they were fitted on, and predict.py refuses to run when that list does
# not match the code. Adding the location family (141 -> 171 features) is exactly such a
# change.
set -e
cd "$(dirname "$0")/.."
PY="../.venv/bin/python"   # the venv lives at the repo root

echo "############ 1/2  BINARY DETECTION (classical, kaggle4) ############"
$PY main.py train --dataset kaggle4

echo
echo "############ 2/2  TUMOR TYPING (classical + deep CNN, kaggle4) ############"
$PY main.py train --dataset kaggle4 --multiclass --deep

echo
echo "ALL TRAINING RUNS COMPLETE"
