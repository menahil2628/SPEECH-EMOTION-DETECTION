"""
train_model.py
--------------
Extracts MFCC features from WAV files in the dataset/ directory,
trains a Random Forest classifier, and saves the model + label encoder
to the model/ directory.

Usage:
    python train_model.py
"""

import os
import glob
import joblib
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "emotion_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

N_MFCC       = 40      # Number of MFCC coefficients
SAMPLE_RATE  = 22050
MAX_DURATION = 3.0     # Clip or pad to this length (seconds)


# ── Feature Extraction ─────────────────────────────────────────────────────────
def extract_features(file_path: str) -> np.ndarray | None:
    """
    Load a WAV file and extract a fixed-length MFCC feature vector.
    Returns a 1-D numpy array of shape (N_MFCC * 3,) containing:
        mean, std, max of each MFCC coefficient across time.
    Returns None if the file cannot be read.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE,
                                  duration=MAX_DURATION, mono=True)

        # Pad short clips to ensure consistent shape
        target_len = int(SAMPLE_RATE * MAX_DURATION)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))

        # ── MFCC (primary feature) ──────────────────────────────────────────
        mfccs       = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean   = np.mean(mfccs, axis=1)
        mfcc_std    = np.std(mfccs, axis=1)
        mfcc_max    = np.max(mfccs, axis=1)

        # ── Chroma ──────────────────────────────────────────────────────────
        chroma      = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # ── Spectral contrast ────────────────────────────────────────────────
        contrast      = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        # ── ZCR & RMS Energy ────────────────────────────────────────────────
        zcr    = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        rms    = np.mean(librosa.feature.rms(y=audio))

        features = np.concatenate([
            mfcc_mean, mfcc_std, mfcc_max,
            chroma_mean,
            contrast_mean,
            [zcr, rms],
        ])
        return features

    except Exception as exc:
        print(f"   ⚠  Skipping {file_path}: {exc}")
        return None


# ── Dataset Loading ────────────────────────────────────────────────────────────
def load_dataset(dataset_dir: str):
    """
    Walk through dataset_dir/<EmotionLabel>/*.wav and collect
    feature vectors + labels.
    """
    X, y = [], []
    emotion_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not emotion_dirs:
        raise FileNotFoundError(
            f"No emotion sub-directories found in '{dataset_dir}'.\n"
            "Run  python prepare_dataset.py  first."
        )

    print(f"\n📂 Loading dataset from: {dataset_dir}")
    print(f"   Found {len(emotion_dirs)} emotion classes: {emotion_dirs}\n")

    for emotion in emotion_dirs:
        emotion_path = os.path.join(dataset_dir, emotion)
        wav_files    = glob.glob(os.path.join(emotion_path, "*.wav"))

        if not wav_files:
            print(f"   ⚠  No .wav files in {emotion_path}, skipping.")
            continue

        count = 0
        for wav in wav_files:
            features = extract_features(wav)
            if features is not None:
                X.append(features)
                y.append(emotion)
                count += 1

        print(f"   ✅ {emotion:<14}  {count}/{len(wav_files)} files loaded")

    return np.array(X), np.array(y)


# ── Training ───────────────────────────────────────────────────────────────────
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = load_dataset(DATASET_DIR)

    if len(X) == 0:
        print("\n❌ No features extracted. Check your dataset directory.")
        return

    # Encode string labels → integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train / test split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\n🤖 Training Random Forest …")
    print(f"   Train samples : {len(X_train)}")
    print(f"   Test  samples : {len(X_test)}")
    print(f"   Feature size  : {X_train.shape[1]}\n")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluation ─────────────────────────────────────────────────────────
    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print(f"  Test Accuracy : {accuracy * 100:.2f}%")
    print("=" * 60)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=le.classes_))

    # ── Save artefacts ─────────────────────────────────────────────────────
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le,  ENCODER_PATH)
    print(f"\n💾 Model saved   → {MODEL_PATH}")
    print(f"💾 Encoder saved → {ENCODER_PATH}")
    print("\n✔  Training complete! You can now run:  python gui_app.py\n")


if __name__ == "__main__":
    train()
