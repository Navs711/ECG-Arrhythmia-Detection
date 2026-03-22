"""
train.py
Entry-point: load data → train CNN → evaluate → save model.
Run with: python train.py
"""

from data.loader import load_ecg_record, segment_beats
from model.trainer import prepare_data, train, evaluate


def main():
    print("📥 Loading ECG data from MIT-BIH database …")
    signal, labels, positions = load_ecg_record(record_id='106')

    print("✂️  Segmenting heartbeats …")
    X, y = segment_beats(signal, labels, positions, window_size=100)
    print(f"   Segments: {X.shape}  |  Labels distribution: {dict(zip(*[['Normal','Abnormal'],[int((y==0).sum()),int((y==1).sum())]])) }")

    print("🔀 Splitting data …")
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    print("🚀 Training CNN model …")
    model, history = train(X_train, y_train, X_test, y_test, epochs=10, batch_size=32)

    print("📈 Evaluating model …")
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
