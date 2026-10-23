# 09 - Prediction, the API, and the Web App

## What you'll learn

- The full journey of **one uploaded scan**, from the browser to the verdict on screen.
- What `predict.py` does at inference time (using an already-trained model).
- How the **FastAPI** backend receives the file, checks it, and builds a JSON answer.
- How the **Next.js** frontend turns that JSON into the verdict card, probability bars, stage gallery and measurements.
- The command-line shortcuts in `run.sh`, what our tests check, and how to run the demo.

You should already know the pipeline stages (preprocess, enhance, segment, features) and the two models (classical features + classifier, and the ResNet-18 CNN) from earlier chapters. This chapter shows how they are *wired together* for a real user.

---

## 1. Key idea: training is done, now we only "predict"

During training (chapter on training) the model learned from thousands of labelled scans and was saved to disk as a **bundle** (a Python dictionary saved with `joblib`). The bundle holds more than the model:

| Bundle key | Meaning |
|---|---|
| `model` | the trained classical classifier |
| `feature_names` | the exact list (and order) of the 171 measurements it was trained on |
| `threshold` | the tuned decision threshold (binary task) |
| `class_names` | e.g. `No Tumor`, `Glioma`, ... |
| `deep_model`, `blend_weight_deep`, `shipped_branch` | the optional fine-tuned CNN and how much to trust it |

At prediction time we load the bundle once and push each new image through the same pipeline used in training. **Same steps, same order** - otherwise the numbers fed to the model would mean something different from what it learned.

---

## 2. The journey of one scan (ASCII diagram)

```
 BROWSER (Next.js, http://localhost:5713)
 ┌────────────────────────────────────────────────────────────┐
 │ TaskToggle: "Detection" (binary) or "Typing" (multiclass)  │
 │ UploadZone: user drops / picks a PNG/JPEG/BMP/TIFF/WebP    │
 │   page.tsx  -> analyze(file, task)   (lib/api.ts)          │
 └───────────────┬────────────────────────────────────────────┘
                 │  POST /analyze?multiclass=true|false&include_images=true
                 │  body: multipart form, field name "file"
                 ▼
 FASTAPI BACKEND (backend/src/braintumor/api.py, http://localhost:3000)
 ┌────────────────────────────────────────────────────────────┐
 │ _read_upload():  type allowed? (else 415)                  │
 │                  empty? (400)   > 20 MB? (413)             │
 │                  decodes as image? (else 400) -> grayscale │
 │ get_bundle():    load model bundle once, cache it          │
 │                  (not trained yet -> 503)                  │
 │ analyse_image()  <-- predict.py                            │
 └───────────────┬────────────────────────────────────────────┘
                 ▼
 PREDICT.PY  analyse_image(image, bundle)
 ┌────────────────────────────────────────────────────────────┐
 │ preprocess  -> enhance -> segment_tumor -> draw_overlay    │
 │        (denoise)  (CLAHE+sharpen) (skull strip+threshold)  │
 │ feature_vector(enh, seg)  -> 171 numbers                   │
 │ check feature names match training                         │
 │ classical model.predict_proba(features) ─┐                 │
 │ CNN deep_model.predict_proba(enhanced) ──┼─> blend         │
 │ apply threshold (binary) / argmax (multiclass)             │
 │ -> dict: label, probabilities, confidence, all stage images│
 └───────────────┬────────────────────────────────────────────┘
                 ▼
 api.py _serialise():  numbers -> JSON, images -> base64 data URIs
                 │
                 │  200 OK  JSON { prediction, model, segmentation,
                 │                 enhancement, images, location,
                 │                 meta, disclaimer }
                 ▼
 BROWSER renders:
   VerdictCard  ProbabilityBars (typing only)  StageGallery  MeasurementPanel
```

---

## 3. Step by step

### Step 1 - The browser sends the file

In `frontend/src/lib/api.ts`, `analyze(file, task)` builds a `FormData` object (the browser's way to send a file), appends the file under the field name `"file"`, and sends a `POST` to `/analyze` with two query parameters: `multiclass` (true when the "Typing" task is chosen) and `include_images` (true by default).

Two details worth knowing:

- The request goes **directly from the browser to the FastAPI server** (default `http://localhost:3000`), not through a Next.js middle layer. This avoids copying the scan (and the large image replies) through an extra process.
- The code deliberately does **not** set a `Content-Type` header. The browser must set it itself because a file upload needs a "multipart boundary" in that header; setting it by hand breaks the upload.

Because the page (port 5713) and the API (port 3000) are different *origins*, the browser would normally block the call. The API therefore enables **CORS** (Cross-Origin Resource Sharing) and lists the allowed client addresses in `config.CORS_ORIGINS`.

In `page.tsx`, choosing a file immediately runs the analysis, and flipping the task toggle **re-runs the same file** with the other task. The old result is cleared first so a stale verdict is never mistaken for the new scan.

### Step 2 - The API checks the upload

The backend uses **FastAPI** (served by **uvicorn**). The `/analyze` endpoint calls `_read_upload()`, which rejects bad input *before* any heavy work, each with a clear error code:

| Problem | HTTP status | `error` field |
|---|---|---|
| Not an allowed image type | 415 | `unsupported_type` |
| Empty file | 400 | `empty_file` |
| Larger than 20 MB | 413 | `file_too_large` |
| Cannot be decoded as an image | 400 | `decode_failed` |
| Model not trained yet | 503 | `model_not_trained` (with a `fix` command) |
| Saved model's feature list differs from the code | 500 | `feature_mismatch` |

The file is read into memory and decoded with OpenCV as a **grayscale** image. Size is checked *before* decoding because decoding a malicious file can allocate a lot of memory.

Then `get_bundle(multiclass)` fetches the right model: the **binary** bundle (tumor / no tumor) or the **multiclass** bundle (No Tumor, Glioma, Meningioma, Pituitary). Bundles are loaded lazily on first use and kept in a dictionary cache, so only the first request pays the loading cost (a couple of seconds, since a ResNet-18 may be inside).

### Step 3 - `predict.py` runs the pipeline

`analyse_image(image, bundle)` in `backend/src/braintumor/predict.py` does the work:

1. `preprocess` - resize to 256x256 and denoise.
2. `enhance` - improve contrast (CLAHE) and sharpen.
3. `segment_tumor` - remove the skull, threshold bright regions, clean with morphology; `draw_overlay` draws the contour and box.
4. `feature_vector` - measure 171 numbers (intensity, texture, shape, asymmetry, location, ...).
5. **Safety check** - the feature *names* must match the ones saved at training. If the code changed, it either re-orders the values to match or raises an error, rather than silently predicting from mismatched numbers.
6. **Classical probabilities** - `model.predict_proba(x)`.
7. **CNN probabilities (if present)** - the enhanced image goes to the fine-tuned CNN. If a CNN exists and its blend weight is above 0, the two probability vectors are mixed:
   `final = (1 - w) * classical + w * deep` (a *weighted soft vote*, `blend_probabilities` in `deep.py`). If the CNN fails to run, the code falls back to classical only. The result records which branch decided (`classical`, `deep` or `ensemble`).
8. **Decision**
   - *Binary:* `tumor_probability = proba[1]`, and the label is "tumor" if it is at least the **tuned threshold** (not 0.5). The threshold was chosen in training to catch most tumors (high sensitivity), because a missed tumor is worse than an extra review.
   - *Multiclass:* label = the class with the highest probability (`argmax`); `tumor_probability` = 1 minus P(No Tumor).
9. **Confidence** - for binary, rescaled around the threshold: 0.5 means "right on the boundary", 1.0 means "maximally sure" in whichever direction. Without this, a probability of 0.23 with a 0.14 threshold would show as "23% confident" while really being a clear tumor call. For multiclass, confidence is simply the winning class's probability.

The function returns one dictionary containing the verdict **and** every intermediate image, so the CLI, the API and report scripts can all reuse it without recomputing anything.

### Step 4 - The API builds the JSON reply

`_serialise()` converts the result into JSON-safe data. NumPy arrays cannot go into JSON, so numbers become plain floats and each stage image is PNG-encoded and turned into a **base64 data URI** (a text string like `data:image/png;base64,...` that an `<img>` tag can display directly). Nothing is saved on the server's disk, so an uploaded scan does not outlive the request.

The response has these top-level fields (mirrored by `frontend/src/lib/types.ts`):

| Field | Contains |
|---|---|
| `prediction` | `label`, `label_name`, `is_tumor`, `confidence`, `tumor_probability`, `class_names`, `probabilities`, `threshold` |
| `model` | `name`, `branch`, `blend_weight_deep`, `deep_probabilities` |
| `segmentation` | `tumor_found`, `n_candidates`, `threshold`, `properties` (centroid, bbox, area, area_ratio, perimeter, circularity, solidity, extent, aspect_ratio, intensities, brightness_ratio) |
| `enhancement` | contrast, entropy, sharpness - before and after |
| `images` | `original`, `preprocessed`, `enhanced`, `brain_mask`, `segmentation_mask`, `overlay` (omitted if `include_images=false`) |
| `location` | a few location features (midline distance, depth, edge contact, ...) |
| `meta` | `filename`, `elapsed_ms`, `task` |
| `disclaimer` | "assistive only, not a medical device" |

The full 171-value feature vector is left out (too big for any screen); it is available from a separate endpoint, `POST /analyze/features`.

### Step 5 - The UI shows the result

`page.tsx` stores the reply in state and renders four components (in `frontend/src/components/`):

- **VerdictCard** - the big label (colour-coded: red for tumor, green for none), the confidence, and (binary only) tumor probability and threshold. Chips show the model name, **"Decided by"** (engineered features / fine-tuned CNN / ensemble with CNN weight) and elapsed milliseconds.
- **ProbabilityBars** - shown only for the **Typing** task. One bar per class, so a close second choice is visible (52% vs 41% is very different from 95% vs 2%). If the CNN contributed, its own opinion is marked on each bar.
- **StageGallery** - the six images: Input, Preprocessed, Enhanced, Skull stripped, Candidate mask, Detected region. Clicking enlarges one. This is the "explainability": you can see how the system reached its answer.
- **MeasurementPanel** - the region's numbers (area, share of brain, brightness vs brain, circularity, solidity, aspect ratio, centroid, perimeter, candidate count), the location features, and enhancement before -> after (contrast, entropy, sharpness).

If something goes wrong, `ErrorPanel` shows the backend's message and its suggested `fix` (for example "Run: ./run.sh serve" when the API is unreachable).

### Other API endpoints

| Method and path | Purpose |
|---|---|
| `GET /` | name, version, list of endpoints |
| `GET /health` | status, which models exist on disk, whether PyTorch is available |
| `GET /models` | description of each loaded model |
| `GET /classes` | label names and plain-language descriptions |
| `POST /analyze` | the main call described above |
| `POST /analyze/features` | all 171 named features, grouped by family |
| `GET /metrics` | the last training report (`?multiclass=true` for typing) |

FastAPI also auto-generates interactive documentation at `/docs` (e.g. `http://localhost:3000/docs`) where you can try endpoints from the browser.

---

## 4. The command line

`main.py` is the entry point and `braintumor/cli.py` defines the sub-commands. `run.sh` is a convenience wrapper that creates the virtual environment on first use and then calls `main.py`.

| Command | What it does |
|---|---|
| `./run.sh serve` | start the REST API on port **3000** |
| `./run.sh client` | start the Next.js frontend on port **5713** (runs `npm install` the first time) |
| `./run.sh check` | run the whole test suite (`pytest tests -q`) and print pass/fail |
| `./run.sh predict <image>` | analyse one image in the terminal and save the 8-panel figure (add `--multiclass` to name the tumor type; `-o` sets the output file) |
| `./run.sh demo` | generate every presentation figure into `backend/outputs/` |
| `./run.sh train`, `./run.sh train-all` | train models (see the training chapter) |
| `./run.sh metrics [--multiclass]` | reprint the last training report |
| `./run.sh generate` | regenerate the synthetic dataset |
| `./run.sh setup` | install Python and client dependencies |
| `./run.sh build-client` | install and build the frontend for production |

The terminal output of `predict` comes from `verdict_text()`: prediction, confidence, tumor probability, threshold, model, class probabilities (for typing), region measurements and the enhancement gain. The saved figure (`pipeline_figure` in `visualize.py`) has 8 panels: original, preprocessed, enhanced, intensity histogram, skull stripped, segmentation mask, detected region, and a verdict card.

---

## 5. Tests

Run with `./run.sh check` (or `python -m pytest tests -q` from inside `backend/`). Tests that need a trained model **skip** instead of failing on a fresh clone.

- **`backend/tests/test_api.py`** uses FastAPI's `TestClient`, which runs the app in-process (no server, no port). It checks: root lists endpoints; `/health` answers even with no model; `/classes` describes every label; CORS allows the client origin; bad uploads are rejected (undecodable, empty, wrong type, oversized, missing file field); `/analyze` returns a complete payload with all six images as `data:image/png;base64,...`; multiclass probabilities sum to 1 and the label is the argmax; `include_images=false` shrinks the response; `/analyze/features` groups every family; `/metrics` and `/models` respond sensibly.
- **`backend/tests/test_pipeline.py`** checks the image-processing chain and model: synthetic image generator, preprocessing (output size, noise reduction, colour input), enhancement (more contrast/entropy/sharpness, values stay in range), brain mask (excludes skull, one solid region), segmentation (localises tumors, correct structure), feature vector (fixed length, finite, all families, separates classes), auto-crop, no degenerate masks on real data, a trained model exists and predicts, the bundle carries a threshold, **inference actually uses the stored threshold**, reported metrics are sane, and the figures render.
- **`backend/tests/test_deep.py`** covers the CNN: device selection, classifier head size, valid probabilities, test-time augmentation, blending being a proper weighted average, blend-weight search, save/load round trip, and graceful handling of missing weights.
- **`backend/tests/test_location_features.py`** checks the location features (finite values, zeroed when no region, midline distance grows with lateral position, edge contact higher at the edge, scale-free, stable names).

---

## 6. How to run the demo

Two terminals, from the project folder:

```
# Terminal 1 - backend
./run.sh serve          # API at http://localhost:3000  (docs at /docs)

# Terminal 2 - frontend
./run.sh client         # UI at http://localhost:5713
```

Open `http://localhost:5713`, pick **Detection** or **Typing**, upload a slice (sample images are in `backend/data/samples/`), and walk through verdict, probabilities, stage gallery and measurements. Before presenting, run `./run.sh check` to confirm everything passes. If the models are missing, `./run.sh train-all` retrains them.

---

## Where this lives in our code

| Piece | File |
|---|---|
| Inference pipeline, threshold, blending, `verdict_text` | `backend/src/braintumor/predict.py` |
| REST API, upload checks, model cache, JSON building | `backend/src/braintumor/api.py` |
| CLI sub-commands (`predict`, `demo`, `serve`, ...) | `backend/src/braintumor/cli.py`, `main.py` |
| 8-panel figure | `backend/src/braintumor/visualize.py` (`pipeline_figure`) |
| Ports, CORS origins, upload limits | `backend/src/braintumor/config.py` |
| Command shortcuts | `run.sh` |
| Browser-side API calls and response types | `frontend/src/lib/api.ts`, `frontend/src/lib/types.ts` |
| Main page | `frontend/src/app/page.tsx` |
| UI pieces | `frontend/src/components/` (`UploadZone`, `TaskToggle`, `VerdictCard`, `ProbabilityBars`, `StageGallery`, `MeasurementPanel`, `ErrorPanel`) |
| Tests | `backend/tests/test_api.py`, `backend/tests/test_pipeline.py`, `backend/tests/test_deep.py`, `backend/tests/test_location_features.py` |

---

## Common viva questions

**Q1. What happens, in order, when a user uploads a scan?**
The browser sends a `POST /analyze` multipart request to the FastAPI server. The API validates the file (type, empty, size 20 MB limit, decodable), decodes it to grayscale, loads the cached model bundle, and calls `analyse_image`. That preprocesses, enhances, segments, extracts 171 features, gets probabilities from the classical model (blended with the CNN when present), applies the threshold or argmax, and returns everything. The API converts it to JSON with base64 images, and the Next.js page displays it.

**Q2. Which web framework and ports do you use, and why is CORS needed?**
Backend: FastAPI run by uvicorn, on port 3000. Frontend: Next.js on port 5713. They are different origins, and browsers block cross-origin requests unless the server allows them, so the API adds CORS middleware with the client's address in its allowed list.

**Q3. Why return images as base64 inside the JSON instead of saving files?**
It keeps the API stateless: nothing about a patient's scan is left on the server after the response, and there is no file naming or cleanup to manage. The trade-off is a larger response, so clients can pass `include_images=false` to skip them.

**Q4. Why does the binary task not use 0.5 as the cut-off, and what does "confidence" mean then?**
A tuned threshold was chosen during training to reach high sensitivity, because missing a tumor is costlier than a false alarm. Confidence is rescaled around that threshold (0.5 = on the boundary, 1.0 = very sure), so the number is not misleading.

**Q5. What does "Decided by: ensemble" mean?**
The final probabilities are a weighted average of the classical (feature-based) model and the fine-tuned CNN, with the weight picked on validation data. `branch` records which opinion produced the answer so the UI is honest about it.

**Q6. How do you know the system works and keeps working?**
`./run.sh check` runs pytest: API tests (using `TestClient`, no server needed), pipeline tests, CNN tests and location-feature tests. Model-dependent tests skip when no model is trained. One important test confirms inference really uses the stored tuned threshold.

---

## Check yourself

1. Which HTTP method, path and form-field name does the browser use to upload a scan?
<details><summary>Answer</summary>`POST /analyze`, with the file in a multipart form field named `file`. Query parameters `multiclass` and `include_images` choose the task and whether to return images.</details>

2. A user uploads a 30 MB file. What does the API return?
<details><summary>Answer</summary>HTTP 413 with `error: "file_too_large"`, because the limit is 20 MB. The check happens before decoding.</details>

3. In the multiclass task, how is the label chosen and how is `tumor_probability` computed?
<details><summary>Answer</summary>The label is the class with the highest probability (argmax). `tumor_probability` is 1 minus the probability of "No Tumor".</details>

4. Why are the models loaded lazily and cached?
<details><summary>Answer</summary>Loading takes a couple of seconds, so it is done once on first use and reused. Loading lazily also means the server can start (and `/health` can report what is missing) even if no model has been trained yet.</details>

5. Which UI component appears only for the Typing task, and why?
<details><summary>Answer</summary>`ProbabilityBars`. It shows the full probability for each of the four classes, so a close runner-up is visible; the Detection task has just two outcomes, which the verdict card already covers.</details>

6. What command starts the backend, and what starts the frontend?
<details><summary>Answer</summary>`./run.sh serve` (API on port 3000) and `./run.sh client` (UI on port 5713).</details>

---

**Next:** [10_REVISION_AND_GLOSSARY.md](10_REVISION_AND_GLOSSARY.md)
