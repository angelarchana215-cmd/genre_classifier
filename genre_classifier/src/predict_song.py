"""
predict_song.py
------------------
Give this script the path to ANY audio file, and it will:
  1. Extract the exact same 55 features used during training
  2. Scale them using the SAME scaler fitted on the training data
  3. Ask both the saved KNN and DNN models what genre they think it is

Usage:
    python src/predict_song.py "path/to/some_song.wav"

Requirements:
    You must have already run train_evaluate.py at least once - it saves
    the trained models and scaler into output/models/, which this script
    loads back.

Note on file formats: librosa can read most common audio formats (wav,
mp3, ogg, flac, m4a) as long as ffmpeg is available on your system. If a
file fails to load, converting it to .wav first (e.g. with an online
converter or ffmpeg) is the simplest fix.
"""

import os
import sys
import json
import joblib
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features  # reuse the exact same function used in training

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "..", "output", "models")


def load_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    with open(os.path.join(MODELS_DIR, "label_classes.json")) as f:
        label_classes = json.load(f)
    knn_model = joblib.load(os.path.join(MODELS_DIR, "knn_model.joblib"))
    dnn_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "dnn_model.keras"))
    return scaler, feature_columns, label_classes, knn_model, dnn_model


def predict(file_path):
    scaler, feature_columns, label_classes, knn_model, dnn_model = load_artifacts()

    print(f"Extracting features from: {file_path}")
    feats = extract_features(file_path)

    # Build the feature vector in EXACTLY the same column order used during training
    x = np.array([[feats[col] for col in feature_columns]])
    x_scaled = scaler.transform(x)

    # --- KNN prediction ---
    knn_pred_idx = knn_model.predict(x_scaled)[0]
    knn_genre = label_classes[knn_pred_idx]
    knn_proba = knn_model.predict_proba(x_scaled)[0]
    knn_confidence = knn_proba[knn_pred_idx]

    # --- DNN prediction ---
    dnn_proba = dnn_model.predict(x_scaled, verbose=0)[0]
    dnn_pred_idx = int(np.argmax(dnn_proba))
    dnn_genre = label_classes[dnn_pred_idx]
    dnn_confidence = dnn_proba[dnn_pred_idx]

    print("\n" + "=" * 50)
    print(f"KNN prediction:  {knn_genre}   (confidence: {knn_confidence:.1%})")
    print(f"DNN prediction:  {dnn_genre}   (confidence: {dnn_confidence:.1%})")
    print("=" * 50)

    # Show the DNN's top-3 guesses too - often more informative than just the top-1
    top3_idx = np.argsort(dnn_proba)[::-1][:3]
    print("\nDNN's top 3 guesses:")
    for i in top3_idx:
        print(f"  {label_classes[i]:<12} {dnn_proba[i]:.1%}")

    return knn_genre, dnn_genre


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict_song.py <path_to_audio_file>")
        sys.exit(1)
    predict(sys.argv[1])
