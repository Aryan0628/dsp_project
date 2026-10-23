"""Entry point for a Hugging Face Space (Gradio SDK, ZeroGPU hardware).

Spaces runs `python app.py` and serves whatever listens on port 7860. We start the same
FastAPI app that `main.py serve` runs, so the deployed API is identical to the local one.
"""
import os
import sys
from pathlib import Path

# ZeroGPU refuses to start unless at least one @spaces.GPU function exists, and `spaces`
# must be imported before torch. Inference here runs on CPU, so the function is only a
# placeholder. `spaces` is preinstalled on Spaces and absent locally, hence the guard.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        return None
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn  # noqa: E402

from braintumor.api import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
