"""
visualize_clusters.py
-----------------------
Your 55 audio features live in 55-dimensional space, which nobody can
actually "see". This script compresses them down to just 2 dimensions
(while trying to preserve how close/far songs are from each other) so
you can plot them on a normal 2D scatter plot and SEE whether songs
from the same genre naturally group together.

Two different compression techniques are used, since they show
slightly different things:

  - PCA (Principal Component Analysis): a simple, fast, linear method.
    Finds the 2 directions in feature space that capture the most
    overall variance. Good for a quick, honest look at the data.

  - t-SNE (t-distributed Stochastic Neighbor Embedding): a more
    powerful, non-linear method that specifically tries to keep
    NEARBY points nearby in the 2D plot. It usually produces much
    more visually distinct clusters - which is exactly why KNN
    (which relies on nearby points sharing a label) works as well as
    it does.

Run this AFTER feature_extraction.py, since it needs features_30_sec.csv.

Usage:
    python src/visualize_clusters.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

HERE = os.path.dirname(__file__)
FEATURES_CSV = os.path.join(HERE, "..", "output", "features_30_sec.csv")
PLOTS_DIR = os.path.join(HERE, "..", "output", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


def load_features():
    df = pd.read_csv(FEATURES_CSV)
    X = df.drop(columns=["filename", "label"]).values
    y = df["label"].values
    return X, y


def scatter_plot(coords, labels, title, filename, xlabel, ylabel):
    genres = sorted(set(labels))
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(9, 7))
    for i, genre in enumerate(genres):
        mask = labels == genre
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            label=genre, alpha=0.75, s=35,
            color=cmap(i % 10),
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    print("Loading features...")
    X, y = load_features()
    print(f"Loaded {X.shape[0]} songs, {X.shape[1]} features, {len(set(y))} genres.\n")

    # Features MUST be scaled before PCA/t-SNE, same reason as for KNN -
    # otherwise features with naturally bigger numbers (like spectral
    # centroid in Hz) would dominate over features with small numbers
    # (like zero-crossing rate, which is between 0 and 1).
    X_scaled = StandardScaler().fit_transform(X)

    # ---- PCA: fast, linear ----
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    scatter_plot(
        pca_coords, y,
        title=f"Genre clusters (PCA) - {explained.sum()*100:.1f}% of variance captured in 2D",
        filename="clusters_pca.png",
        xlabel=f"Principal Component 1 ({explained[0]*100:.1f}% variance)",
        ylabel=f"Principal Component 2 ({explained[1]*100:.1f}% variance)",
    )

    # ---- t-SNE: slower, non-linear, usually much cleaner clusters ----
    print("Running t-SNE (this can take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0] // 4))
    tsne_coords = tsne.fit_transform(X_scaled)
    scatter_plot(
        tsne_coords, y,
        title="Genre clusters (t-SNE)",
        filename="clusters_tsne.png",
        xlabel="t-SNE dimension 1",
        ylabel="t-SNE dimension 2",
    )

    print("\nDone. Open output/plots/clusters_pca.png and clusters_tsne.png to view them.")


if __name__ == "__main__":
    main()
