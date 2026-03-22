"""
model/predictor.py
Inference utilities: predict single beats and patient-level arrhythmia decision.
"""

import numpy as np


ARRHYTHMIA_THRESHOLD = 0.1   # >10 % abnormal beats → arrhythmia detected


def predict_beats(model, segments: np.ndarray) -> np.ndarray:
    """
    Run inference on an array of beat segments.

    Args:
        model: Trained Keras model
        segments: Shape (N, segment_length) or (N, segment_length, 1)

    Returns:
        Binary predictions array of shape (N,)
    """
    if segments.ndim == 2:
        segments = segments.reshape(segments.shape[0], segments.shape[1], 1)

    probs = model.predict(segments, verbose=0)
    return (probs.flatten() > 0.5).astype(int)


def patient_decision(predictions: np.ndarray, threshold: float = ARRHYTHMIA_THRESHOLD) -> dict:
    """
    Aggregate beat-level predictions into a patient-level arrhythmia decision.

    Args:
        predictions: Binary beat predictions
        threshold: Fraction of abnormal beats that triggers a positive diagnosis

    Returns:
        dict with total_beats, abnormal_beats, abnormal_pct, arrhythmia_detected
    """
    total = len(predictions)
    abnormal = int(predictions.sum())
    pct = abnormal / total if total > 0 else 0.0

    return {
        "total_beats": total,
        "abnormal_beats": abnormal,
        "abnormal_pct": round(pct * 100, 2),
        "arrhythmia_detected": pct > threshold,
    }
