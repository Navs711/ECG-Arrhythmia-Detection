# ECG Arrhythmia Detection — Project

## 📁 Project Structure

```
ecg_arrhythmia/
├── data/
│   ├── __init__.py
│   └── loader.py          # MIT-BIH data loading & beat segmentation
├── model/
│   ├── __init__.py
│   ├── cnn_model.py       # 1D CNN architecture
│   ├── trainer.py         # Training, evaluation, model save/load
│   └── predictor.py       # Inference & patient-level decision
├── api/
│   ├── __init__.py
│   └── main.py            # FastAPI application
├── ui/
│   └── app.py             # Streamlit frontend
├── train.py               # Training entry-point
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
# From the ecg_arrhythmia/ directory
python train.py
```
This downloads record 106 from PhysioNet, trains the CNN for 10 epochs,
and saves `ecg_cnn_model.h5` in the project root.

---

## 🚀 Running the API

```bash
# From the ecg_arrhythmia/ directory
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs available at:
- Swagger UI → http://localhost:8000/docs
- ReDoc      → http://localhost:8000/redoc

### API Endpoints

| Method | Endpoint             | Description                               |
|--------|----------------------|-------------------------------------------|
| GET    | /health              | Liveness check + model status             |
| POST   | /predict             | Classify custom beat segments (JSON)      |
| POST   | /predict/record      | Fetch a MIT-BIH record & classify beats   |

#### Example — /predict/record
```bash
curl -X POST http://localhost:8000/predict/record \
  -H "Content-Type: application/json" \
  -d '{"record_id": "106", "window_size": 100}'
```

#### Example — /predict (custom segments)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"segments": [[0.1, 0.2, ...200 values...]]}'
```

---

## 🖥️ Running the Streamlit UI

Make sure the FastAPI server is running first, then:

```bash
# From the ecg_arrhythmia/ directory
streamlit run ui/app.py
```

Opens at → http://localhost:8501

### UI Features
- **Tab 1**: Enter any MIT-BIH record ID → fetch from PhysioNet → instant classification
- **Tab 2**: Upload a CSV of custom beat segments (each row = 200 samples)
- Live API health indicator in the sidebar
- Pie chart + beat-sequence scatter plot
- Full beat-level predictions table

---

## 🔬 Model Details

| Parameter       | Value              |
|-----------------|--------------------|
| Architecture    | 1D CNN             |
| Input           | 200-sample window  |
| Conv layers     | 2 × (Conv1D + MaxPool1D) |
| Dense layers    | 64 → 1 (sigmoid)   |
| Loss            | Binary crossentropy |
| Optimizer       | Adam               |
| Epochs          | 10                 |
| Batch size      | 32                 |
| Arrhythmia threshold | >10 % abnormal beats |

---

## 📝 Notes

- Requires internet connection for first run (downloads MIT-BIH data from PhysioNet).
- `ecg_cnn_model.h5` must exist before starting the API — run `train.py` first.
- The Streamlit UI communicates with the FastAPI backend via HTTP — both must run simultaneously.
