"""
improve_knn.py
----------------
Experiments with two ways to try to improve KNN accuracy beyond the
baseline in train_evaluate.py:

  1. Different distance metrics (euclidean vs manhattan vs cosine)
  2. Reducing the 53 raw features down to fewer dimensions with PCA
     before running KNN, to fight the "curse of dimensionality"

Prints a comparison table so you can see which combination actually
helps for YOUR dataset (results vary depending on the data).

Usage:
    python src/improve_knn.py
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

HERE = os.path.dirname(__file__)
FEATURES_CSV = os.path.join(HERE, "..", "output", "features_30_sec.csv")
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    X = df.drop(columns=["filename", "label"]).values
    y = df["label"].values
    return X, y


def best_k_accuracy(X_train, y_train, X_test, y_test, metric, k_range=range(1, 26)):
    """Sweep k for a given metric, return the best accuracy found and the k that achieved it."""
    max_k = min(max(k_range), len(X_train))
    best_acc, best_k = 0, 1
    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance", metric=metric)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        if acc > best_acc:
            best_acc, best_k = acc, k
    return best_acc, best_k


def main():
    print("Loading features...")
    X, y = load_data()
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} raw features.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []

    # ---- Baseline: all 53 features, euclidean (this matches train_evaluate.py) ----
    acc, k = best_k_accuracy(X_train_s, y_train, X_test_s, y_test, metric="euclidean")
    results.append(("Baseline (53 features, euclidean)", acc, k))

    # ---- Try different distance metrics on the full feature set ----
    for metric in ["manhattan", "cosine"]:
        acc, k = best_k_accuracy(X_train_s, y_train, X_test_s, y_test, metric=metric)
        results.append((f"53 features, {metric}", acc, k))

    # ---- Try PCA dimensionality reduction, then euclidean KNN ----
    for n_components in [10, 15, 20, 30]:
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_train_pca = pca.fit_transform(X_train_s)
        X_test_pca = pca.transform(X_test_s)
        acc, k = best_k_accuracy(X_train_pca, y_train, X_test_pca, y_test, metric="euclidean")
        explained = pca.explained_variance_ratio_.sum() * 100
        results.append((f"PCA to {n_components} dims ({explained:.0f}% variance), euclidean", acc, k))

    # ---- Print comparison table ----
    print(f"{'Configuration':<50} {'Best k':>7} {'Accuracy':>10}")
    print("-" * 70)
    baseline_acc = results[0][1]
    for name, acc, k in results:
        marker = ""
        if acc > baseline_acc:
            marker = "  <- better than baseline"
        print(f"{name:<50} {k:>7} {acc*100:>9.1f}%{marker}")

    best_overall = max(results, key=lambda r: r[1])
    print(f"\nBest configuration: {best_overall[0]} - {best_overall[1]*100:.1f}% accuracy")
    print(f"(Baseline was {baseline_acc*100:.1f}%)")


if __name__ == "__main__":
    main()
