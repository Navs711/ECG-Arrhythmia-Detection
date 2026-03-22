"""
model/cnn_model.py
Defines and compiles the 1D CNN model for ECG arrhythmia detection.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input


def build_model(input_length: int) -> Sequential:
    """
    Build and compile a 1D CNN for binary ECG classification.

    Architecture:
        Conv1D(32) → MaxPool → Conv1D(64) → MaxPool → Flatten → Dense(64) → Sigmoid

    Args:
        input_length: Length of each beat segment (e.g., 200 for window_size=100)

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        Input(shape=(input_length, 1)),

        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),

        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),

        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model
