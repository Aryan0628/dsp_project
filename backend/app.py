"""Entry point for a Hugging Face Space (Gradio SDK, ZeroGPU hardware).

Spaces runs `python app.py` and serves whatever listens on port 7860. We serve the same
FastAPI app that `main.py serve` runs, so the deployed API is identical to the local one.

ZeroGPU only supports Gradio apps and refuses to start unless at least one @spaces.GPU
function exists, so on Spaces we (1) import `spaces` before torch, (2) define a placeholder
GPU function (inference here runs on CPU) and (3) mount a minimal Gradio page at /ui
inside our FastAPI app. Locally none of this applies and the plain API is served.
"""
import os
import sys
from pathlib import Path

ON_SPACES = bool(os.environ.get("SPACE_ID"))

if ON_SPACES:
    import spaces  # must come before torch; on Spaces a failure here should be loud

    @spaces.GPU
    def gpu_probe() -> str:
        return "ok"

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn  # noqa: E402

from braintumor.api import app  # noqa: E402

if ON_SPACES:
    import gradio as gr

    with gr.Blocks(title="Brain Tumor API") as demo:
        gr.Markdown("Brain tumor detection API. Endpoints: `/health`, `/analyze`, `/docs`.")
        gr.Button("GPU probe").click(gpu_probe, outputs=gr.Textbox(label="result"))

    app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
