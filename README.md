## Abstract Classification Service

This project trains a small, fast Transformer-based classifier for academic paper abstracts
and exposes a lightweight REST API for inference.

### 1. Environment setup

- **Python**: 3.9 or 3.10 recommended.
- **Install dependencies**:

```bash
pip install -r requirements.txt
```

If you have a GPU and CUDA installed, `torch` will use it automatically. On CPU-only
machines, training will automatically limit the subset size to keep things fast.

### 2. Training the model

1. Place your dataset (CSV/TSV) in the project root (already provided as `arxiv_data.csv`).
2. Run the training script:

```bash
python train_model.py
```

The script will:

- Auto-detect the abstract and label columns.
- Clean and filter the data (remove missing and very short abstracts).
- Build a balanced subset (up to ~1,200 samples, or 800 on CPU-only).
- Split into **train / val / test** (80/10/10, stratified by label).
- Train a **DistilBERT** classifier for 1 epoch.
- Save artifacts:
  - `final_model/` (model + tokenizer + `labels.json`)
  - `train.csv`, `val.csv`, `test.csv`
  - `metrics.json` (includes accuracy, macro precision/recall/F1, per-class metrics)
  - `confusion_matrix.csv`
  - `misclassified_samples.csv`

### 3. Running the API locally

After training has completed and `final_model/` exists:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

#### Health check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "ok"}
```

#### Get labels

```bash
curl http://localhost:8000/labels
```

Returns a JSON array of label names used during training.

#### Predict

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"abstract":"This paper proposes a transformer-based method for efficient document classification..."}'
```

Example response:

```json
{
  "label": "cs.LG",
  "score": 0.9234,
  "all_scores": [
    {"label": "cs.LG", "score": 0.9234},
    {"label": "cs.AI", "score": 0.0345}
  ]
}
```

### 4. Docker usage

Build the image:

```bash
docker build -t abstract-classifier .
```

Train inside the container (optional, you can also train on the host and just copy `final_model/`):

```bash
docker run --rm -v ${PWD}:/app abstract-classifier python train_model.py
```

Serve the API:

```bash
docker run --rm -p 8000:8000 -v ${PWD}:/app abstract-classifier \
  uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. Deployment options

- **Render / Railway**:
  - Use this repo.
  - Set the start command to:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

- **Hugging Face Space**:
  - Create a Space with **Docker** or **FastAPI** backend.
  - Use this `Dockerfile` or `app.py` + `requirements.txt`.

Once deployed, Cursor (or any client) can:

- Call `GET /health` for liveness.
- Call `GET /labels` to show possible categories.
- Call `POST /predict` with an abstract string to get the predicted label and scores.


