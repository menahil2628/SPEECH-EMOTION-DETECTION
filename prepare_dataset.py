"""
prepare_dataset.py
------------------
Downloads the Audio MNIST dataset from Kaggle or generates synthetic
sample data if Kaggle credentials are not available.

For full dataset: place your kaggle.json in ~/.kaggle/ then run:
    python prepare_dataset.py --kaggle

For quick demo with synthetic samples:
    python prepare_dataset.py
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf

# ── Emotion label mapping (Audio MNIST digits → mapped emotions for demo) ─────
# Since Audio MNIST is spoken digits (0-9), we creatively map them to
# emotion-like categories for the Speech Emotion Recognition demo.
# In a real SER project you would use RAVDESS or CREMA-D.

DIGIT_TO_EMOTION = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Fearful",
    5: "Disgusted",
    6: "Surprised",
    7: "Calm",
    8: "Excited",
    9: "Bored",
}

SAMPLE_RATE = 22050
DURATION    = 1.0   # seconds per synthetic clip


def generate_synthetic_audio(emotion_idx: int, seed: int = 0) -> np.ndarray:
    """
    Generate a synthetic audio signal that mimics basic spectral
    characteristics associated with each emotion class.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))

    # Base frequency varies by emotion to create distinct patterns
    base_freqs = [220, 330, 180, 440, 260, 200, 380, 165, 480, 140]
    freq = base_freqs[emotion_idx % len(base_freqs)]

    # Fundamental tone
    signal = 0.5 * np.sin(2 * np.pi * freq * t)

    # Add harmonics
    signal += 0.25 * np.sin(2 * np.pi * freq * 2 * t)
    signal += 0.12 * np.sin(2 * np.pi * freq * 3 * t)

    # Add emotion-specific modulation
    mod_rate = [1, 4, 0.5, 6, 2, 1.5, 5, 0.3, 7, 0.2][emotion_idx % 10]
    signal  *= (0.6 + 0.4 * np.sin(2 * np.pi * mod_rate * t))

    # Add slight noise
    signal += rng.normal(0, 0.02, len(t))

    # Normalise
    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    return signal.astype(np.float32)


def create_synthetic_dataset(dataset_dir: str, samples_per_class: int = 30):
    """Create a synthetic dataset with WAV files organised by emotion label."""
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"\n📁 Creating synthetic dataset in: {dataset_dir}")
    print(f"   {len(DIGIT_TO_EMOTION)} emotions × {samples_per_class} samples each\n")

    total = 0
    for digit, emotion in DIGIT_TO_EMOTION.items():
        emotion_dir = os.path.join(dataset_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)

        for i in range(samples_per_class):
            audio   = generate_synthetic_audio(digit, seed=digit * 1000 + i)
            fname   = os.path.join(emotion_dir, f"{emotion.lower()}_{i:03d}.wav")
            sf.write(fname, audio, SAMPLE_RATE)
            total  += 1

        print(f"   ✅ {emotion:<12}  {samples_per_class} files written")

    print(f"\n✔  Total files created: {total}")
    return dataset_dir


def download_kaggle_dataset(dataset_dir: str):
    """Download Audio MNIST from Kaggle (requires kaggle.json credentials)."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("❌ kaggle package not found. Install with: pip install kaggle")
        sys.exit(1)

    os.makedirs(dataset_dir, exist_ok=True)
    print("⬇  Downloading Audio MNIST from Kaggle …")
    os.system(
        f"kaggle datasets download -d sripaadsrinivasan/audio-mnist "
        f"--path {dataset_dir} --unzip"
    )
    print("✔  Download complete.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Speech Emotion Dataset")
    parser.add_argument(
        "--kaggle", action="store_true",
        help="Download real Audio MNIST from Kaggle (requires kaggle.json)"
    )
    parser.add_argument(
        "--samples", type=int, default=30,
        help="Synthetic samples per emotion class (default: 30)"
    )
    parser.add_argument(
        "--out", type=str, default="dataset",
        help="Output directory (default: dataset/)"
    )
    args = parser.parse_args()

    if args.kaggle:
        download_kaggle_dataset(args.out)
    else:
        create_synthetic_dataset(args.out, args.samples)
