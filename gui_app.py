"""
gui_app.py
----------
Speech Emotion Recognition — Tkinter GUI
Run with:  python gui_app.py
"""

import os
import time
import threading
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import librosa
import joblib
import sounddevice as sd
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("model", "emotion_model.pkl")
ENCODER_PATH = os.path.join("model", "label_encoder.pkl")

# ── Audio config ───────────────────────────────────────────────────────────────
SAMPLE_RATE      = 22050
RECORD_DURATION  = 3        # seconds
N_MFCC           = 40

# ── Emotion colours (one per class) ───────────────────────────────────────────
EMOTION_COLORS = {
    "Neutral":   "#94A3B8",
    "Happy":     "#FBBF24",
    "Sad":       "#60A5FA",
    "Angry":     "#F87171",
    "Fearful":   "#C084FC",
    "Disgusted": "#4ADE80",
    "Surprised": "#FB923C",
    "Calm":      "#34D399",
    "Excited":   "#F472B6",
    "Bored":     "#A78BFA",
}
DEFAULT_COLOR = "#6366F1"

# ── Feature extraction (must match train_model.py) ─────────────────────────────
def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    target_len = int(sr * 3.0)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mfccs         = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean     = np.mean(mfccs, axis=1)
    mfcc_std      = np.std(mfccs, axis=1)
    mfcc_max      = np.max(mfccs, axis=1)
    chroma        = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean   = np.mean(chroma, axis=1)
    contrast      = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    zcr           = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    rms           = np.mean(librosa.feature.rms(y=audio))

    return np.concatenate([
        mfcc_mean, mfcc_std, mfcc_max,
        chroma_mean, contrast_mean,
        [zcr, rms],
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class SpeechEmotionApp(tk.Tk):

    # ── Colour palette ────────────────────────────────────────────────────────
    BG       = "#0F172A"
    PANEL    = "#1E293B"
    ACCENT   = "#6366F1"
    TEXT     = "#F1F5F9"
    SUBTEXT  = "#94A3B8"
    SUCCESS  = "#34D399"
    WARNING  = "#FBBF24"
    DANGER   = "#F87171"

    FONT_H1  = ("Segoe UI", 20, "bold")
    FONT_H2  = ("Segoe UI", 13, "bold")
    FONT_BODY= ("Segoe UI", 11)
    FONT_SM  = ("Segoe UI", 9)
    FONT_BIG = ("Segoe UI", 42, "bold")

    def __init__(self):
        super().__init__()

        self.title("🎙 Speech Emotion Recognition")
        self.geometry("980x760")
        self.minsize(900, 700)
        self.configure(bg=self.BG)
        self.resizable(True, True)

        # State
        self.model        = None
        self.label_enc    = None
        self.recording    = False
        self.audio_data   = None
        self.current_file = None
        self._record_buf  = []

        self._load_model()
        self._build_ui()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            self.model     = joblib.load(MODEL_PATH)
            self.label_enc = joblib.load(ENCODER_PATH)
        else:
            self.model     = None
            self.label_enc = None

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=self.PANEL, height=72)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(
            header, text="🎙  Speech Emotion Recognition",
            font=self.FONT_H1, bg=self.PANEL, fg=self.TEXT
        ).pack(side="left", padx=24, pady=16)

        self.model_badge = tk.Label(
            header,
            text="✅ Model Loaded" if self.model else "⚠  No Model — Run train_model.py",
            font=self.FONT_SM,
            bg=self.SUCCESS if self.model else self.WARNING,
            fg=self.BG, padx=10, pady=4
        )
        self.model_badge.pack(side="right", padx=24, pady=20)

        # ── Main body ─────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=self.BG)
        body.pack(fill="both", expand=True, padx=20, pady=16)

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=0)
        body.rowconfigure(1, weight=1)

        # ── LEFT column ───────────────────────────────────────────────────────
        left = tk.Frame(body, bg=self.BG)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))

        self._build_input_panel(left)
        self._build_result_panel(left)

        # ── RIGHT column ──────────────────────────────────────────────────────
        right = tk.Frame(body, bg=self.BG)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")

        self._build_waveform_panel(right)
        self._build_mfcc_panel(right)

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready. Record or upload an audio file.")
        status_bar = tk.Label(
            self, textvariable=self.status_var,
            font=self.FONT_SM, bg=self.PANEL, fg=self.SUBTEXT,
            anchor="w", padx=16, pady=6
        )
        status_bar.pack(fill="x", side="bottom")

    # ── Input panel ───────────────────────────────────────────────────────────
    def _build_input_panel(self, parent):
        card = self._card(parent, "🎤  Audio Input")
        card.pack(fill="x", pady=(0, 12))

        # Record button
        self.rec_btn = tk.Button(
            card, text="⏺  Record  (3 sec)",
            font=self.FONT_H2,
            bg=self.ACCENT, fg=self.TEXT, activebackground="#4F46E5",
            relief="flat", bd=0, padx=20, pady=12, cursor="hand2",
            command=self._toggle_record
        )
        self.rec_btn.pack(fill="x", padx=16, pady=(0, 10))

        # Upload button
        tk.Button(
            card, text="📂  Upload WAV File",
            font=self.FONT_BODY,
            bg=self.PANEL, fg=self.TEXT, activebackground="#334155",
            relief="flat", bd=0, padx=20, pady=10, cursor="hand2",
            command=self._upload_file,
            highlightbackground=self.SUBTEXT, highlightthickness=1
        ).pack(fill="x", padx=16, pady=(0, 10))

        # File label
        self.file_label = tk.Label(
            card, text="No file selected",
            font=self.FONT_SM, bg=self._card_bg(), fg=self.SUBTEXT,
            anchor="w", wraplength=340
        )
        self.file_label.pack(fill="x", padx=16, pady=(0, 12))

        # Analyse button
        self.analyse_btn = tk.Button(
            card, text="🔍  Analyse Emotion",
            font=self.FONT_H2,
            bg=self.SUCCESS, fg=self.BG, activebackground="#059669",
            relief="flat", bd=0, padx=20, pady=12, cursor="hand2",
            command=self._analyse,
            state="disabled"
        )
        self.analyse_btn.pack(fill="x", padx=16, pady=(0, 16))

    # ── Result panel ──────────────────────────────────────────────────────────
    def _build_result_panel(self, parent):
        card = self._card(parent, "🧠  Prediction Result")
        card.pack(fill="both", expand=True)

        # Big emotion label
        self.emotion_var = tk.StringVar(value="—")
        self.emotion_lbl = tk.Label(
            card, textvariable=self.emotion_var,
            font=self.FONT_BIG, bg=self._card_bg(), fg=self.ACCENT
        )
        self.emotion_lbl.pack(pady=(16, 4))

        # Confidence
        self.conf_var = tk.StringVar(value="Confidence: —")
        tk.Label(
            card, textvariable=self.conf_var,
            font=self.FONT_BODY, bg=self._card_bg(), fg=self.SUBTEXT
        ).pack(pady=(0, 12))

        # Confidence bar
        self.conf_bar = ttk.Progressbar(
            card, length=280, mode="determinate", maximum=100
        )
        self.conf_bar.pack(padx=24, pady=(0, 16))

        # Top-3 probabilities
        tk.Label(
            card, text="Top Predictions",
            font=("Segoe UI", 10, "bold"), bg=self._card_bg(), fg=self.SUBTEXT
        ).pack()

        self.top3_frame = tk.Frame(card, bg=self._card_bg())
        self.top3_frame.pack(fill="x", padx=16, pady=(4, 16))

        for i in range(3):
            row = tk.Frame(self.top3_frame, bg=self._card_bg())
            row.pack(fill="x", pady=3)
            lbl  = tk.Label(row, text=f"—", font=self.FONT_BODY,
                            bg=self._card_bg(), fg=self.TEXT, width=14, anchor="w")
            bar  = ttk.Progressbar(row, length=140, mode="determinate", maximum=100)
            pct  = tk.Label(row, text="—%", font=self.FONT_SM,
                            bg=self._card_bg(), fg=self.SUBTEXT, width=5)
            lbl.pack(side="left")
            bar.pack(side="left", padx=6)
            pct.pack(side="left")
            setattr(self, f"top3_lbl_{i}",  lbl)
            setattr(self, f"top3_bar_{i}",  bar)
            setattr(self, f"top3_pct_{i}",  pct)

    # ── Waveform panel ────────────────────────────────────────────────────────
    def _build_waveform_panel(self, parent):
        card = self._card(parent, "📈  Waveform")
        card.pack(fill="both", expand=True, pady=(0, 10))

        fig, self.ax_wave = plt.subplots(figsize=(4.8, 2.1),
                                          facecolor=self._card_bg())
        self.ax_wave.set_facecolor(self._card_bg())
        self.ax_wave.tick_params(colors=self.SUBTEXT, labelsize=7)
        for spine in self.ax_wave.spines.values():
            spine.set_edgecolor("#334155")
        self.ax_wave.set_xlabel("Time", color=self.SUBTEXT, fontsize=8)
        self.ax_wave.set_ylabel("Amplitude", color=self.SUBTEXT, fontsize=8)
        fig.tight_layout(pad=1.2)

        self.wave_canvas = FigureCanvasTkAgg(fig, master=card)
        self.wave_canvas.get_tk_widget().pack(fill="both", expand=True,
                                               padx=10, pady=(0, 10))
        self.wave_fig = fig

    # ── MFCC panel ────────────────────────────────────────────────────────────
    def _build_mfcc_panel(self, parent):
        card = self._card(parent, "🔬  MFCC Spectrogram")
        card.pack(fill="both", expand=True)

        fig, self.ax_mfcc = plt.subplots(figsize=(4.8, 2.2),
                                          facecolor=self._card_bg())
        self.ax_mfcc.set_facecolor(self._card_bg())
        self.ax_mfcc.tick_params(colors=self.SUBTEXT, labelsize=7)
        for spine in self.ax_mfcc.spines.values():
            spine.set_edgecolor("#334155")
        self.ax_mfcc.set_xlabel("Frames", color=self.SUBTEXT, fontsize=8)
        self.ax_mfcc.set_ylabel("MFCC", color=self.SUBTEXT, fontsize=8)
        fig.tight_layout(pad=1.2)

        self.mfcc_canvas = FigureCanvasTkAgg(fig, master=card)
        self.mfcc_canvas.get_tk_widget().pack(fill="both", expand=True,
                                               padx=10, pady=(0, 10))
        self.mfcc_fig = fig

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _card_bg(self):
        return self.PANEL

    def _card(self, parent, title: str):
        frame = tk.Frame(parent, bg=self.PANEL, bd=0, relief="flat")
        tk.Label(
            frame, text=title,
            font=self.FONT_H2, bg=self.PANEL, fg=self.TEXT,
            anchor="w", padx=16, pady=10
        ).pack(fill="x")
        ttk.Separator(frame, orient="horizontal").pack(fill="x", padx=16)
        return frame

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    # ── Recording ─────────────────────────────────────────────────────────────
    def _toggle_record(self):
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self.recording    = True
        self._record_buf  = []
        self.rec_btn.config(text="⏹  Stop Recording", bg=self.DANGER)
        self._set_status("🔴 Recording … speak now!")
        threading.Thread(target=self._record_thread, daemon=True).start()

    def _record_thread(self):
        frames = int(SAMPLE_RATE * RECORD_DURATION)
        audio  = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32")
        sd.wait()
        self.audio_data = audio.flatten()

        # Save to temp wav
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, self.audio_data, SAMPLE_RATE)
        self.current_file = tmp.name
        self.after(0, self._recording_done)

    def _recording_done(self):
        self.recording = False
        self.rec_btn.config(text="⏺  Record  (3 sec)", bg=self.ACCENT)
        self.file_label.config(text="🎙 Recorded audio (3 sec)", fg=self.SUCCESS)
        self.analyse_btn.config(state="normal")
        self._plot_waveform(self.audio_data)
        self._plot_mfcc(self.audio_data)
        self._set_status("✔ Recording complete. Click 'Analyse Emotion'.")

    # ── File upload ───────────────────────────────────────────────────────────
    def _upload_file(self):
        path = filedialog.askopenfilename(
            title="Select a WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load file:\n{e}")
            return

        self.audio_data   = audio
        self.current_file = path
        self.file_label.config(
            text=f"📄 {os.path.basename(path)}", fg=self.TEXT
        )
        self.analyse_btn.config(state="normal")
        self._plot_waveform(audio)
        self._plot_mfcc(audio)
        self._set_status(f"File loaded: {os.path.basename(path)}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    def _analyse(self):
        if self.model is None:
            messagebox.showwarning(
                "No Model",
                "No trained model found.\n\n"
                "Please run:\n    python train_model.py\n\nfirst."
            )
            return
        if self.audio_data is None:
            messagebox.showinfo("No Audio", "Please record or upload audio first.")
            return

        self._set_status("⚙  Extracting features & predicting …")
        self.after(50, self._run_predict)

    def _run_predict(self):
        try:
            feats = extract_features(self.audio_data, SAMPLE_RATE)
            proba = self.model.predict_proba([feats])[0]
            classes = self.label_enc.classes_

            top_idx  = np.argsort(proba)[::-1]
            top_emo  = classes[top_idx[0]]
            top_conf = proba[top_idx[0]] * 100

            # Update big result
            color = EMOTION_COLORS.get(top_emo, DEFAULT_COLOR)
            self.emotion_var.set(top_emo)
            self.emotion_lbl.config(fg=color)
            self.conf_var.set(f"Confidence: {top_conf:.1f}%")
            self.conf_bar["value"] = top_conf

            # Update top-3
            for i in range(3):
                idx  = top_idx[i] if i < len(top_idx) else 0
                emo  = classes[idx]
                pct  = proba[idx] * 100
                getattr(self, f"top3_lbl_{i}").config(text=emo)
                getattr(self, f"top3_bar_{i}")["value"] = pct
                getattr(self, f"top3_pct_{i}").config(text=f"{pct:.1f}%")

            self._set_status(
                f"✅ Predicted: {top_emo}  ({top_conf:.1f}% confidence)"
            )

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self._set_status("❌ Prediction failed.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    def _plot_waveform(self, audio: np.ndarray):
        self.ax_wave.clear()
        times = np.linspace(0, len(audio) / SAMPLE_RATE, len(audio))
        self.ax_wave.plot(times, audio, color=self.ACCENT, linewidth=0.6, alpha=0.85)
        self.ax_wave.set_facecolor(self._card_bg())
        self.ax_wave.tick_params(colors=self.SUBTEXT, labelsize=7)
        self.ax_wave.set_xlabel("Time (s)", color=self.SUBTEXT, fontsize=8)
        self.ax_wave.set_ylabel("Amplitude", color=self.SUBTEXT, fontsize=8)
        for spine in self.ax_wave.spines.values():
            spine.set_edgecolor("#334155")
        self.wave_fig.tight_layout(pad=1.2)
        self.wave_canvas.draw()

    def _plot_mfcc(self, audio: np.ndarray):
        self.ax_mfcc.clear()
        mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        img = self.ax_mfcc.imshow(
            mfccs, aspect="auto", origin="lower",
            cmap="magma", interpolation="nearest"
        )
        self.ax_mfcc.set_facecolor(self._card_bg())
        self.ax_mfcc.tick_params(colors=self.SUBTEXT, labelsize=7)
        self.ax_mfcc.set_xlabel("Frames", color=self.SUBTEXT, fontsize=8)
        self.ax_mfcc.set_ylabel("MFCC Coeff", color=self.SUBTEXT, fontsize=8)
        for spine in self.ax_mfcc.spines.values():
            spine.set_edgecolor("#334155")
        self.mfcc_fig.tight_layout(pad=1.2)
        self.mfcc_canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = SpeechEmotionApp()
    app.mainloop()
