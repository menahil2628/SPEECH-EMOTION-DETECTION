# 🎙 Speech Emotion Recognition
### Built with Python · Tkinter GUI · Random Forest · MFCC Features

---

## 📁 Project Structure

```
speech_emotion_project/
│
├── prepare_dataset.py     ← Step 1: Generate or download dataset
├── train_model.py         ← Step 2: Extract features & train model
├── gui_app.py             ← Step 3: Launch the GUI application
├── requirements.txt       ← All Python dependencies
├── run_project.bat        ← One-click setup + run (Windows)
│
├── dataset/               ← Created by prepare_dataset.py
│   ├── Neutral/
│   ├── Happy/
│   ├── Sad/
│   ├── Angry/
│   └── ...
│
└── model/                 ← Created by train_model.py
    ├── emotion_model.pkl  ← Trained Random Forest classifier
    └── label_encoder.pkl  ← Encodes emotion labels
```

---

## 🚀 Quick Start (Windows)

Double-click `run_project.bat` — it will:
1. Install all dependencies
2. Generate the dataset
3. Train the model
4. Launch the GUI

---

## 🛠 Manual Setup (VS Code / Terminal)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset
```bash
# Option A: Synthetic data (no Kaggle account needed — works immediately)
python prepare_dataset.py

# Option B: Real Audio MNIST from Kaggle
#   1. Download kaggle.json from https://www.kaggle.com/settings
#   2. Place it in C:\Users\<you>\.kaggle\kaggle.json
#   3. Run:
python prepare_dataset.py --kaggle
```

### 3. Train the model
```bash
python train_model.py
```
You'll see accuracy score and a classification report printed.

### 4. Launch the GUI
```bash
python gui_app.py
```

---

## 🖥 GUI Features

| Feature | Description |
|---|---|
| ⏺ Record | Records 3 seconds of audio from your microphone |
| 📂 Upload WAV | Load any WAV file from disk |
| 🔍 Analyse | Predicts emotion with confidence score |
| 📈 Waveform | Live waveform plot of the audio signal |
| 🔬 MFCC | Mel-Frequency Cepstral Coefficient spectrogram |
| 🧠 Top-3 | Shows top 3 predicted emotions with probabilities |

---

## 🔬 How It Works

```
Audio File / Microphone
        │
        ▼
  Preprocessing
  (resample, pad/trim to 3s)
        │
        ▼
  Feature Extraction
  ┌─────────────────────────┐
  │  MFCC (mean, std, max)  │  ← 120 features
  │  Chroma STFT (mean)     │  ←  12 features
  │  Spectral Contrast      │  ←   7 features
  │  ZCR + RMS Energy       │  ←   2 features
  └─────────────────────────┘
  Total: 141-dimensional vector
        │
        ▼
  Random Forest Classifier
  (200 trees, trained on dataset)
        │
        ▼
  Predicted Emotion + Probability
```

---

## 🏷 Emotion Classes

| Label | Colour |
|---|---|
| Neutral | Gray |
| Happy | Yellow |
| Sad | Blue |
| Angry | Red |
| Fearful | Purple |
| Disgusted | Green |
| Surprised | Orange |
| Calm | Teal |
| Excited | Pink |
| Bored | Lavender |

---

## ⚙ VS Code Setup Tips

1. Open the `speech_emotion_project/` folder in VS Code
2. Select your Python interpreter: `Ctrl+Shift+P` → `Python: Select Interpreter`
3. Open terminal: `` Ctrl+` ``
4. Run steps 1–4 from the **Manual Setup** section above
5. To run the GUI directly, press `F5` with `gui_app.py` open

### Recommended VS Code Extensions
- Python (Microsoft)
- Pylance
- Jupyter (optional, for exploring data)

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| librosa | 0.10.1 | Audio loading & MFCC extraction |
| numpy | 1.24.4 | Numerical operations |
| scikit-learn | 1.3.2 | Random Forest model |
| sounddevice | 0.4.6 | Microphone recording |
| soundfile | 0.12.1 | WAV file read/write |
| matplotlib | 3.7.2 | Waveform & MFCC plots |
| joblib | 1.3.2 | Model serialization |
| scipy | 1.11.3 | Signal processing |

---

## 🔄 Using a Real Dataset (Kaggle Audio MNIST)

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Settings → API → Create New Token → download `kaggle.json`
3. Place `kaggle.json` in `C:\Users\<YourName>\.kaggle\`
4. Run: `python prepare_dataset.py --kaggle`
5. Then re-train: `python train_model.py`

For even better emotion accuracy, consider the **RAVDESS** dataset:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `No module named 'sounddevice'` | Run `pip install sounddevice` |
| `No model found` warning in GUI | Run `python train_model.py` first |
| Microphone not working | Check Windows microphone permissions |
| `PortAudio` error on Windows | Install `pipwin` then `pipwin install pyaudio` |
| Low accuracy | Increase `--samples` in `prepare_dataset.py` or use real dataset |
