#!/usr/bin/env python3
"""
Brain Tumor Detection from MRI - project entry point.

    python main.py --help
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from braintumor.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
