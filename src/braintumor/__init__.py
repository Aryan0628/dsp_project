"""
Medical Image Enhancement and Diagnostic Assistance System
for Brain Tumor Detection using MRI and OpenCV.

Pipeline stages (see docs/01_WHAT_IS_THIS_PROJECT.md):
    1. Acquisition      -> dataset.py
    2. Preprocessing    -> preprocess.py
    3. Enhancement      -> enhance.py
    4. Segmentation     -> segment.py
    5. Feature extract  -> features.py
    6. Classification   -> model.py / train.py / predict.py
    7. Visualisation    -> visualize.py
"""

__version__ = "1.0.0"
__all__ = [
    "config",
    "preprocess",
    "enhance",
    "segment",
    "features",
    "dataset",
    "model",
    "visualize",
]
