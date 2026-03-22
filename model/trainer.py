"""
model/trainer.py
Handles training, evaluation, and persistence of the CNN model.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model as keras_load_model


from model.cnn_model import build_model

MODEL_PATH = "ecg_cnn_model.h5"


def prepare_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Split and reshape data for CNN input.

    Returns:
        X_train, X_test, y_train, y_test — reshaped to (N, L, 1)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Add channel dimension for Conv1D
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, X_test, y_train, y_test


def train(X_train: np.ndarray, y_train: np.ndarray,
          X_test: np.ndarray, y_test: np.ndarray,
          epochs: int = 10, batch_size: int = 32):
    """
    Train the CNN model and save weights.

    Returns:
        model: Trained Keras model
        history: Training history object
    """
    model = build_model(input_length=X_train.shape[1])

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )

    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    return model, history


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Print accuracy and full classification report.

    Returns:
        dict with loss, accuracy, and report string
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    report = classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"])

    print(f"\nLoss: {loss:.4f} | Accuracy: {accuracy:.4f}")
    print(report)

    return {"loss": loss, "accuracy": accuracy, "report": report}


def load_saved_model(path: str = MODEL_PATH):
    """Load a previously saved Keras model."""
    return keras_load_model(path)
