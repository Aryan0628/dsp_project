# ML API image: FastAPI + the trained models. Build from this folder:
#   docker build -t braintumor-api backend/
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=3000
WORKDIR /app

# OpenCV (even the headless build) links against glib at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY main.py .
# The trained bundles must exist locally before building (./run.sh train-all):
# models/*.joblib are git-ignored, so a build from a fresh clone has no models.
COPY models ./models
COPY outputs ./outputs

EXPOSE 3000
# Shell form so $PORT (injected by most hosts) is expanded.
CMD python main.py serve --host 0.0.0.0 --port $PORT
