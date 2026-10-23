"""Entry point for a Hugging Face Space (Gradio SDK).

Spaces runs `python app.py` and serves whatever listens on port 7860. We start the same
FastAPI app that `main.py serve` runs, so the deployed API is identical to the local one.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn  # noqa: E402

from braintumor.api import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
