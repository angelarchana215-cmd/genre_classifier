"""
train_evaluate.py
-------------------
Loads the feature CSV produced by feature_extraction.py, trains:
  1. An instance-based learner - K-Nearest Neighbors (KNN), with a
     sweep over k to find the best value.
  2. A deep neural network (feed-forward / MLP) built with Keras.

Then compares them on accuracy, precision/recall/F1, confusion
matrices, and training/inference time, and saves all plots + a
text summary report to output/.

WHY these two are a fair, interesting comparison:
  - KNN is "instance-based" / "lazy learning": it stores the entire
    training set and classifies a new point by looking at its nearest
    neighbours at prediction time. It learns NO internal parameters.
  - A DNN is "eager, parametric learning": it compresses everything
    it learns from the training data into a fixed set of weights,
    then throws the training data away at prediction time.
  This is exactly the conceptual contrast the assignment is asking
  you to demonstrate empirically.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
HERE = os.path.dirname(__file__)
FEATURES_CSV = os.path.join(HERE, "..", "output", "features_30_sec.csv")
PLOTS_DIR = os.path.join(HERE, "..", "output", "plots")
MODELS_DIR = os.path.join(HERE, "..", "output", "models")
REPORT_PATH = os.path.join(HERE, "..", "output", "comparison_report.txt")
RANDOM_STATE = 42

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(csv_path=FEATURES_CSV):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["filename", "label"]).values
    y_raw = df["label"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le, df


def preprocess(X_train, X_test):
    """Feature scaling is essential for KNN (distance-based) and helps
    the DNN converge faster."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ----------------------------------------------------------------------
# 1. KNN - instance-based learning
# ----------------------------------------------------------------------
def train_knn(X_train, y_train, X_test, y_test, k_range=range(1, 26)):
    """Sweep k to find the best number of neighbours, then fit the best model."""
    # k can't exceed the number of training samples available
    max_k = len(X_train)
    k_range = range(1, min(max(k_range) + 1, max_k + 1))
    accuracies = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        accuracies.append(acc)

    best_k = list(k_range)[int(np.argmax(accuracies))]
    print(f"[KNN] Best k = {best_k} (test accuracy {max(accuracies):.4f})")

    # plot accuracy vs k
    plt.figure(figsize=(7, 4))
    plt.plot(list(k_range), accuracies, marker="o")
    plt.axvline(best_k, color="red", linestyle="--", label=f"best k = {best_k}")
    plt.xlabel("k (number of neighbours)")
    plt.ylabel("Test accuracy")
    plt.title("KNN: accuracy vs k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "knn_k_sweep.png"), dpi=150)
    plt.close()

    # final fit + timing
    knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
    t0 = time.time()
    knn.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = knn.predict(X_test)
    inference_time = time.time() - t0

    return {
        "model": knn,
        "name": f"KNN (k={best_k})",
        "y_pred": y_pred,
        "train_time": train_time,
        "inference_time": inference_time,
    }


# ----------------------------------------------------------------------
# 2. Deep Neural Network - eager, parametric learning
# ----------------------------------------------------------------------
def build_dnn(input_dim, n_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_dnn(X_train, y_train, X_test, y_test, n_classes, epochs=100):
    model = build_dnn(X_train.shape[1], n_classes)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True
    )

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    inference_time = time.time() - t0

    # training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("DNN loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("DNN accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dnn_training_curves.png"), dpi=150)
    plt.close()

    return {
        "model": model,
        "name": "Deep Neural Network",
        "y_pred": y_pred,
        "train_time": train_time,
        "inference_time": inference_time,
    }


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------
def plot_confusion(y_test, y_pred, labels, title, filename):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()


def evaluate(result, y_test, label_encoder):
    y_pred = result["y_pred"]
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    plot_confusion(
        y_test, y_pred, label_encoder.classes_,
        title=f"Confusion Matrix - {result['name']}",
        filename=f"confusion_{result['name'].split()[0].lower()}.png",
    )
    return acc, report


def plot_side_by_side(results, accuracies):
    names = [r["name"] for r in results]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, accuracies, color=["#2c3e6b", "#8a2be2"])
    plt.ylabel("Test accuracy")
    plt.title("KNN vs Deep Neural Network")
    plt.ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                  f"{acc:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("Loading features...")
    X, y, le, df = load_data()
    n_classes = len(le.classes_)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, {n_classes} genres.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test, scaler = preprocess(X_train, X_test)

    # Save the scaler and the exact feature column order now, while we have
    # both handy - predict_song.py needs both to process a brand-new audio
    # file exactly the same way this training data was processed.
    feature_columns = [c for c in df.columns if c not in ("filename", "label")]
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f)
    with open(os.path.join(MODELS_DIR, "label_classes.json"), "w") as f:
        json.dump(list(le.classes_), f)

    print("Training KNN (instance-based learner)...")
    knn_result = train_knn(X_train, y_train, X_test, y_test)
    joblib.dump(knn_result["model"], os.path.join(MODELS_DIR, "knn_model.joblib"))

    print("\nTraining DNN (deep neural network)...")
    dnn_result = train_dnn(X_train, y_train, X_test, y_test, n_classes)
    dnn_result["model"].save(os.path.join(MODELS_DIR, "dnn_model.keras"))

    results = [knn_result, dnn_result]
    lines = []
    accuracies = []

    for result in results:
        acc, report = evaluate(result, y_test, le)
        accuracies.append(acc)
        lines.append(f"\n{'='*60}\n{result['name']}\n{'='*60}")
        lines.append(f"Test accuracy      : {acc:.4f}")
        lines.append(f"Training time (s)  : {result['train_time']:.3f}")
        lines.append(f"Inference time (s) : {result['inference_time']:.4f}")
        lines.append("\nClassification report:\n" + report)

    plot_side_by_side(results, accuracies)

    summary = "\n".join(lines)
    print(summary)

    with open(REPORT_PATH, "w") as f:
        f.write(summary)

    print(f"\nAll plots saved to {PLOTS_DIR}")
    print(f"Full report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
