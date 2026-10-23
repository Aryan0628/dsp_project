"""Entry point for a Hugging Face Space (Gradio SDK, ZeroGPU hardware).

Spaces runs `python app.py` and serves whatever listens on port 7860. We serve the same
FastAPI app that `main.py serve` runs, so the deployed API is identical to the local one.

ZeroGPU kills a Space unless it reports at least one @spaces.GPU function at startup. The
`spaces` package only sends that report from inside gradio's Blocks.launch(), which we never
call, so on Spaces we define a placeholder GPU function (inference here runs on CPU) and send
the report ourselves via spaces.zero.startup(). Locally none of this applies.
"""
import os
import sys
from pathlib import Path

ON_SPACES = bool(os.environ.get("SPACE_ID"))

if ON_SPACES:
    import spaces  # must come before torch

    @spaces.GPU
    def gpu_probe() -> str:
        return "ok"

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn  # noqa: E402

from braintumor.api import app  # noqa: E402

if __name__ == "__main__":
    if ON_SPACES:
        from spaces import zero

        if hasattr(zero, "startup"):     # only defined on ZeroGPU hardware
            zero.startup()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
