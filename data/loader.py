"""
data/loader.py
Handles loading and preprocessing ECG data from MIT-BIH database.
"""

import wfdb
import numpy as np


def load_ecg_record(record_id: str = '106', pn_dir: str = 'mitdb'):
    """
    Load ECG signal and annotations from PhysioNet MIT-BIH database.

    Args:
        record_id: Record identifier (e.g., '106')
        pn_dir: PhysioNet directory name

    Returns:
        signal (np.ndarray): Raw ECG signal (lead 0)
        labels (list): Beat annotation symbols
        positions (np.ndarray): Sample positions of each beat
    """
    record = wfdb.rdrecord(record_id, pn_dir=pn_dir)
    annotation = wfdb.rdann(record_id, 'atr', pn_dir=pn_dir)

    signal = record.p_signal[:, 0]
    labels = annotation.symbol
    positions = annotation.sample

    return signal, labels, positions


def segment_beats(signal: np.ndarray, labels: list, positions: np.ndarray, window_size: int = 100):
    """
    Slice heartbeat segments centered on each annotated R-peak.

    Args:
        signal: Raw ECG signal
        labels: Beat annotation symbols
        positions: Sample positions of annotations
        window_size: Samples before and after each R-peak

    Returns:
        X (np.ndarray): Beat segments of shape (N, 2*window_size)
        y (np.ndarray): Binary labels — 0=Normal, 1=Abnormal
    """
    X, y = [], []

    for i, pos in enumerate(positions):
        if pos - window_size < 0 or pos + window_size > len(signal):
            continue

        segment = signal[pos - window_size: pos + window_size]
        label = 0 if labels[i] == 'N' else 1

        X.append(segment)
        y.append(label)

    return np.array(X), np.array(y)
