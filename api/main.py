"""
api/main.py
FastAPI REST API for ECG arrhythmia detection.

Endpoints:
    POST /predict          — classify a list of beat segments
    POST /predict/record   — load a MIT-BIH record and classify all beats
    GET  /health           — liveness check
"""

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# ── adjust path when running from project root ──────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.trainer import load_saved_model, MODEL_PATH
from model.predictor import predict_beats, patient_decision
from data.loader import load_ecg_record, segment_beats

# ── app setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ECG Arrhythmia Detection API",
    description="1D CNN model trained on the MIT-BIH arrhythmia database.",
    version="1.0.0",
)

# Load model once at startup
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at '{MODEL_PATH}'. Run train.py first.",
            )
        _model = load_saved_model(MODEL_PATH)
    return _model


# ── schemas ──────────────────────────────────────────────────────────────────
class BeatSegmentsRequest(BaseModel):
    segments: List[List[float]] = Field(
        ...,
        description="List of beat segments. Each segment must be 200 samples (window_size=100).",
        example=[[0.1] * 200],
    )


class RecordRequest(BaseModel):
    record_id: str = Field("106", description="MIT-BIH record ID (e.g. '100', '106')")
    window_size: int = Field(100, ge=50, le=300, description="Samples before/after each R-peak")


class BeatPrediction(BaseModel):
    beat_index: int
    label: str          # "Normal" or "Abnormal"
    label_code: int     # 0 or 1


class PredictionResponse(BaseModel):
    beat_predictions: List[BeatPrediction]
    summary: dict


# ── endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Utility"])
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_segments(body: BeatSegmentsRequest):
    """
    Classify a list of raw beat segments provided by the caller.
    Each segment must contain exactly 200 float values.
    """
    model = get_model()
    arr = np.array(body.segments, dtype=np.float32)

    if arr.ndim != 2:
        raise HTTPException(400, "Segments must be a 2-D list of floats.")
    if arr.shape[1] != 200:
        raise HTTPException(400, f"Each segment must have 200 samples, got {arr.shape[1]}.")

    preds = predict_beats(model, arr)
    summary = patient_decision(preds)

    beat_preds = [
        BeatPrediction(beat_index=i, label="Abnormal" if p else "Normal", label_code=int(p))
        for i, p in enumerate(preds)
    ]

    return PredictionResponse(beat_predictions=beat_preds, summary=summary)


@app.post("/predict/record", response_model=PredictionResponse, tags=["Prediction"])
def predict_record(body: RecordRequest):
    """
    Download a MIT-BIH record from PhysioNet, segment all beats, and classify them.
    Requires an internet connection.
    """
    model = get_model()

    try:
        signal, labels, positions = load_ecg_record(body.record_id)
    except Exception as e:
        raise HTTPException(400, f"Could not load record '{body.record_id}': {e}")

    X, _ = segment_beats(signal, labels, positions, window_size=body.window_size)

    if len(X) == 0:
        raise HTTPException(400, "No valid segments found in this record.")

    preds = predict_beats(model, X)
    summary = patient_decision(preds)

    beat_preds = [
        BeatPrediction(beat_index=i, label="Abnormal" if p else "Normal", label_code=int(p))
        for i, p in enumerate(preds)
    ]

    return PredictionResponse(beat_predictions=beat_preds, summary=summary)
