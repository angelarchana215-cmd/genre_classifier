"""
feature_extraction.py
----------------------
Walks through the dataset folder (one sub-folder per genre, containing
.wav / .au / .mp3 clips), extracts a set of audio features from every
clip using librosa, and saves everything into a single CSV file.

Why these features?
  - MFCCs (Mel-Frequency Cepstral Coefficients): capture the "timbre"
    or texture of a sound - the single most important feature family
    for genre/instrument/speaker classification.
  - Chroma STFT: captures pitch-class content (which of the 12 notes
    are active) - useful for tonal/harmonic style differences.
  - Spectral centroid / bandwidth / rolloff: describe the "brightness"
    and spread of the sound's frequency content.
  - Zero-crossing rate: how noisy/percussive a signal is.
  - RMS energy: loudness.
  - Tempo: beats per minute - genres often cluster around tempo ranges.

For each feature that varies over time (MFCCs, chroma, etc.) we take
the MEAN and VARIANCE across the clip, which is the standard way to
collapse a time-series into a fixed-length vector so classical ML
models (like KNN) can use it. This is exactly what the well-known
GTZAN "features_30_sec.csv" file does.
"""

import os
import numpy as np
import pandas as pd
import librosa
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIG - edit these paths for your machine
# ----------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "genres_original")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "output", "features_30_sec.csv")
N_MFCC = 20          # how many MFCC coefficients to extract
SAMPLE_RATE = 22050  # librosa's default; GTZAN clips are ~22kHz mono


def extract_features(file_path, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Load one audio file and return a dict of extracted features."""
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    feats = {}

    # --- Core time/frequency-domain features ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # tempo (beats per minute)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    feats["chroma_stft_mean"] = float(np.mean(chroma))
    feats["chroma_stft_var"] = float(np.var(chroma))
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_var"] = float(np.var(rms))
    feats["spectral_centroid_mean"] = float(np.mean(spec_centroid))
    feats["spectral_centroid_var"] = float(np.var(spec_centroid))
    feats["spectral_bandwidth_mean"] = float(np.mean(spec_bw))
    feats["spectral_bandwidth_var"] = float(np.var(spec_bw))
    feats["rolloff_mean"] = float(np.mean(rolloff))
    feats["rolloff_var"] = float(np.var(rolloff))
    feats["zero_crossing_rate_mean"] = float(np.mean(zcr))
    feats["zero_crossing_rate_var"] = float(np.var(zcr))
    feats["tempo"] = tempo

    # MFCCs 1..N, mean and variance each
    for i in range(n_mfcc):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_var"] = float(np.var(mfcc[i]))

    return feats


def build_dataset(data_dir=DATA_DIR, output_csv=OUTPUT_CSV):
    """
    Expects data_dir to contain one sub-folder per genre, e.g.:
        data/genres_original/blues/*.wav
        data/genres_original/classical/*.wav
        ...
    Writes a CSV with one row per audio file: filename, all features, genre label.
    """
    rows = []
    genres = sorted(
        g for g in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, g))
    )
    if not genres:
        raise RuntimeError(
            f"No genre sub-folders found in {data_dir}. "
            "Download GTZAN (or your dataset) and place each genre's clips "
            "in its own sub-folder under data/genres_original/."
        )

    print(f"Found {len(genres)} genres: {genres}")

    for genre in genres:
        genre_dir = os.path.join(data_dir, genre)
        files = [
            f for f in os.listdir(genre_dir)
            if f.lower().endswith((".wav", ".au", ".mp3"))
        ]
        print(f"  {genre}: {len(files)} files")
        for fname in files:
            fpath = os.path.join(genre_dir, fname)
            try:
                feats = extract_features(fpath)
                feats["filename"] = fname
                feats["label"] = genre
                rows.append(feats)
            except Exception as e:
                print(f"    [skipped] {fname}: {e}")

    df = pd.DataFrame(rows)
    # put filename/label first for readability
    cols = ["filename", "label"] + [c for c in df.columns if c not in ("filename", "label")]
    df = df[cols]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows x {df.shape[1]} columns to {output_csv}")
    return df
if __name__ == "__main__":
    build_dataset()