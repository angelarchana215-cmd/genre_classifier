# Music Genre Classification: KNN vs Deep Neural Network

Classifies music genres from extracted audio features, and compares an
instance-based learner (K-Nearest Neighbors) against a deep neural network.

---

## 1. Get the dataset (GTZAN)

This project is built for **GTZAN**, the standard dataset for this exact
assignment: 1000 audio clips, 30 seconds each, 10 genres, 100 clips per genre
(blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock).

Steps:

1. Go to Kaggle and search **"GTZAN Dataset - Music Genre Classification"**
   (uploaded by andradaolteanu). You'll need a free Kaggle account.
   Link: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
2. Download it and unzip.
3. Inside, find the `genres_original` folder — it already has one
   sub-folder per genre, each full of `.wav` files. That's exactly the
   structure this project expects.
4. Copy that `genres_original` folder into this project's `data/` folder,
   so you end up with:

```
genre_classifier/
  data/
    genres_original/
      blues/
        blues.00000.wav
        blues.00001.wav
        ...
      classical/
        ...
      rock/
        ...
```

(GTZAN also ships a ready-made `features_30_sec.csv` — you can ignore
it, since the point of this project is to do the feature extraction
yourself and show that step in your report.)

**Don't have Kaggle access / want a different dataset?** Any collection
of audio clips organized as `data/genres_original/<genre_name>/*.wav`
(or `.mp3`, `.au`) will work — the code doesn't hardcode genre names.

---

## 2. Set up your environment

```bash
cd genre_classifier
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This installs `librosa` (audio processing), `tensorflow` (deep learning),
`scikit-learn` (KNN + metrics), `pandas`/`numpy` (data handling), and
`matplotlib`/`seaborn` (plots).

---

## 3. Run it

**Step A — extract features from audio → CSV**

```bash
python src/feature_extraction.py
```

This reads every clip in `data/genres_original/`, extracts a feature
vector per clip (MFCCs, chroma, spectral features, tempo, etc.), and
writes `output/features_30_sec.csv`. On a laptop CPU, 1000 clips takes
roughly 5-15 minutes — librosa's beat/tempo tracking is the slow part.

**Step B — train both models and compare**

```bash
python src/train_evaluate.py
```

This trains the KNN (sweeping k from 1 up to 25 to find the best value)
and the DNN (with early stopping so it won't overfit), evaluates both
on a held-out test set, and writes everything to `output/`:

```
output/
  features_30_sec.csv
  comparison_report.txt         <- accuracy, precision/recall/F1, timings
  plots/
    knn_k_sweep.png             <- accuracy vs k
    dnn_training_curves.png     <- loss/accuracy over epochs
    confusion_knn.png
    confusion_deep.png
    model_comparison.png        <- head-to-head bar chart
```

---

## 4. What's actually being compared, and why it matters for the report

| | KNN (instance-based) | DNN (parametric) |
|---|---|---|
| **Learning** | Stores all training data; no training phase to speak of | Learns weights via backpropagation over many epochs |
| **Prediction** | Compares new point to every stored point (slow at large scale) | Single forward pass through fixed weights (fast) |
| **What it captures** | Local similarity in feature space | Non-linear combinations of features, can model more complex boundaries |
| **Data hunger** | Works OK with a few hundred samples per class | Usually wants more data to generalize well |
| **Curse of dimensionality** | Badly affected — with ~50+ features, "nearest" neighbors become less meaningful | Less affected, thanks to learned feature weighting |

With ~1000 clips and ~50 features, you'll typically see KNN and a small
DNN land in a similar ballpark (both around 60-75% accuracy on 10-way
GTZAN classification) — GTZAN is a famously noisy, hard dataset. That's
a legitimate and interesting result to discuss: it shows that having
*good features* (the MFCCs) matters more here than model sophistication,
and that KNN's simplicity isn't a strict disadvantage when the feature
space is well-behaved.

Things you can point to in your report/viva:
- The `knn_k_sweep.png` plot shows how accuracy changes with k — too
  small k overfits to noise, too large k oversmooths across genres.
- The `dnn_training_curves.png` plot shows whether the DNN over/underfit
  (watch for train accuracy climbing while val accuracy plateaus).
- The confusion matrices show *which* genres get confused for which —
  usually rock/metal or jazz/blues bleed into each other, which is a
  genuinely interesting musicological point, not just a model failure.

---

## 5. Visualizing genre clusters

A useful question is: *do songs from the same genre naturally group
together, before we even train a classifier?* This script compresses
your 55 features down to just 2 dimensions (so they can be plotted)
and colors each song by its genre:

```bash
python src/visualize_clusters.py
```

This produces two plots in `output/plots/`:

- **`clusters_pca.png`** - a fast, simple compression (PCA). Honest but
  the clusters may overlap somewhat, since PCA can only capture
  "straight-line" relationships between features.
- **`clusters_tsne.png`** - a more powerful compression (t-SNE) that
  usually produces much clearer, tighter clusters, since it specifically
  tries to keep similar songs close together in the 2D plot.

**Why this matters for your project:** if genres form visible, distinct
clusters in this plot, that's direct visual evidence for *why* KNN
works at all - it's a "nearest neighbour" method, so it only works
well when similar songs really do end up close together in feature
space. This is a great plot to show alongside your KNN results.

## 6. Predicting the genre of your own song

Once you've run `train_evaluate.py` (which trains AND saves the models
into `output/models/`), you can hand the project any audio file and
ask it to guess the genre:

```bash
python src/predict_song.py "path/to/any_song.mp3"
```

It prints both models' predictions with confidence percentages, e.g.:

```
==================================================
KNN prediction:  rock   (confidence: 65.0%)
DNN prediction:  rock   (confidence: 78.3%)
==================================================

DNN's top 3 guesses:
  rock         78.3%
  country      12.1%
  blues         5.4%
```

This works on ANY audio file - not just GTZAN clips. Try a song from
your own music library and see what it guesses. It won't always be
right (remember, the models only got 66-70% accuracy on the test set),
but seeing it work on a real song you know is the best way to actually
understand what the model learned.

If it fails to load a file, librosa needs `ffmpeg` installed on your
system for non-`.wav` formats (mp3, m4a, etc.) - install it via your OS
package manager, or convert the file to `.wav` first.

## 7. Extending it (optional, if you want to go further)

- Add a CNN on **mel-spectrograms** (images) instead of the tabular
  features — this usually beats both KNN and the tabular DNN, since it
  can learn spatial/temporal patterns the summary statistics throw away.
- Try other instance-based methods (e.g. weighted KNN with different
  distance metrics) for a broader "instance-based" comparison.
- Cross-validate instead of a single train/test split for more robust
  numbers.
