"""
visualize_split.py
--------------------
Visualizes two things your teacher likely wants to see:

  1. How many samples (songs) exist per genre - confirming the
     dataset is roughly balanced (100 songs per genre in GTZAN).
  2. How the train/test split divides each genre's songs - showing
     that the 80/20 split is applied PROPORTIONALLY within each
     genre (this is what stratify=y in train_test_split guarantees),
     not just overall.

Run this AFTER feature_extraction.py.

Usage:
    python src/visualize_split.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(__file__)
FEATURES_CSV = os.path.join(HERE, "..", "output", "features_30_sec.csv")
PLOTS_DIR = os.path.join(HERE, "..", "output", "plots")
RANDOM_STATE = 42

os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} total songs across {df['label'].nunique()} genres.\n")

    # ---- Plot 1: how many songs per genre ----
    counts = df["label"].value_counts().sort_index()
    print("Songs per genre:")
    print(counts.to_string())

    plt.figure(figsize=(9, 5))
    bars = plt.bar(counts.index, counts.values, color="#2c3e6b")
    plt.title("Number of songs per genre (full dataset)")
    plt.xlabel("Genre")
    plt.ylabel("Number of songs")
    plt.xticks(rotation=45, ha="right")
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 1, str(val), ha="center", fontsize=9)
    plt.tight_layout()
    out1 = os.path.join(PLOTS_DIR, "samples_per_genre.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"\nSaved {out1}")

    # ---- Plot 2: how the train/test split divides EACH genre ----
    X = df.drop(columns=["filename", "label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    genres = sorted(df["label"].unique())

    x_pos = np.arange(len(genres))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x_pos - width/2, [train_counts.get(g, 0) for g in genres], width,
            label="Training set", color="#2c3e6b")
    plt.bar(x_pos + width/2, [test_counts.get(g, 0) for g in genres], width,
            label="Test set", color="#8a2be2")
    plt.title("Train/test split per genre (stratified - proportions preserved)")
    plt.xlabel("Genre")
    plt.ylabel("Number of songs")
    plt.xticks(x_pos, genres, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(PLOTS_DIR, "train_test_split_per_genre.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved {out2}")

    print("\nTrain/test counts per genre:")
    summary = pd.DataFrame({"train": train_counts, "test": test_counts}).fillna(0).astype(int)
    print(summary.to_string())


if __name__ == "__main__":
    main()
