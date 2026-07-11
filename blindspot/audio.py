"""
Audio scene module - YAMNet 521-class sound classification.

Continuously listens to the microphone and, every AUDIO_WINDOW_SECONDS, reports
the top few sound classes it hears (e.g. "vehicle", "speech", "car horn").

This is the SECOND modality. Its output is text facts that:
  - feed the fusion engine (Tier 4) to raise/lower danger severity, and
  - go into the VLM prompt so the brain reasons across sight + sound.

WALLED OFF ON PURPOSE:
  YAMNet needs TensorFlow, which is the heaviest / most install-fragile dep in
  the stack. This module imports TF lazily and is NOT imported by Tiers 1-2. If
  TF won't install on the laptop, everything else still runs. Install audio with
  the separate `requirements-audio.txt`.

Runs in a background thread; read the latest result with .latest().
"""

from __future__ import annotations
import csv
import threading
import time
from dataclasses import dataclass, field

import numpy as np

import config


@dataclass
class AudioResult:
    # List of (label, score) sorted by score desc, above AUDIO_CONFIDENCE.
    labels: list[tuple[str, float]] = field(default_factory=list)
    timestamp: float = 0.0

    def facts_line(self, max_items: int = 4) -> str:
        """Compact fact string for the VLM/fusion, e.g. 'speech(0.71); vehicle(0.40)'."""
        return "; ".join(f"{lbl}({sc:.2f})" for lbl, sc in self.labels[:max_items])

    def label_set(self) -> set[str]:
        """Lowercased label words, for loose matching in fusion."""
        words: set[str] = set()
        for lbl, _ in self.labels:
            for w in lbl.lower().replace(",", " ").replace("(", " ").replace(")", " ").split():
                words.add(w)
        return words


class AudioScene:
    """Background microphone listener + YAMNet classifier."""

    def __init__(self):
        self._latest = AudioResult()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._model = None
        self._class_names: list[str] = []

    # -- model loading (lazy, TF only touched here) --
    def _load(self):
        import tensorflow as tf          # noqa
        import tensorflow_hub as hub
        print("[audio] loading YAMNet from TF Hub (first time downloads it)...")
        self._model = hub.load(config.YAMNET_HANDLE)
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path) as f:
            reader = csv.DictReader(f)
            self._class_names = [row["display_name"] for row in reader]
        print(f"[audio] YAMNet ready ({len(self._class_names)} classes).")

    def _classify(self, waveform: np.ndarray) -> list[tuple[str, float]]:
        # waveform: mono float32 in [-1, 1] at 16 kHz.
        scores, _embeddings, _spectro = self._model(waveform)
        mean_scores = np.mean(scores.numpy(), axis=0)
        top = np.argsort(mean_scores)[::-1][:6]
        out = []
        for i in top:
            s = float(mean_scores[i])
            if s >= config.AUDIO_CONFIDENCE:
                out.append((self._class_names[i], s))
        return out

    # -- background loop --
    def _run(self):
        import sounddevice as sd
        self._load()
        window = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_WINDOW_SECONDS)
        print("[audio] listening...")
        while not self._stop.is_set():
            try:
                rec = sd.rec(window, samplerate=config.AUDIO_SAMPLE_RATE,
                             channels=1, dtype="float32")
                sd.wait()
                waveform = rec.reshape(-1)
                labels = self._classify(waveform)
                with self._lock:
                    self._latest = AudioResult(labels=labels, timestamp=time.time())
            except Exception as e:
                print(f"[audio] error: {e}")
                time.sleep(0.5)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def latest(self) -> AudioResult:
        with self._lock:
            return self._latest

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


class PhoneAudio:
    """YAMNet that classifies audio chunks PUSHED from the phone (no local mic).

    The phone streams short mono float32 @16kHz chunks; we buffer ~1s and classify
    on demand. Loads YAMNet lazily (TensorFlow), same walled-off dependency as
    AudioScene, so the web server only pays for it if phone audio is enabled.
    """

    def __init__(self):
        self._model = None
        self._class_names: list[str] = []
        self._buf = np.zeros(0, dtype="float32")
        self._latest = AudioResult()
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self):
        import tensorflow as tf          # noqa
        import tensorflow_hub as hub
        print("[phone-audio] loading YAMNet from TF Hub...")
        self._model = hub.load(config.YAMNET_HANDLE)
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path) as f:
            reader = csv.DictReader(f)
            self._class_names = [row["display_name"] for row in reader]
        self._loaded = True
        print(f"[phone-audio] YAMNet ready ({len(self._class_names)} classes).")

    def _classify(self, waveform: np.ndarray) -> list[tuple[str, float]]:
        scores, _e, _s = self._model(waveform)
        mean_scores = np.mean(scores.numpy(), axis=0)
        top = np.argsort(mean_scores)[::-1][:6]
        out = []
        for i in top:
            s = float(mean_scores[i])
            if s >= config.AUDIO_CONFIDENCE:
                out.append((self._class_names[i], s))
        return out

    def push(self, chunk: np.ndarray):
        """Add a chunk (mono float32 @16kHz); classify once we have ~1s."""
        if not self._loaded:
            self._load()
        self._buf = np.concatenate([self._buf, chunk.astype("float32")])
        window = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_WINDOW_SECONDS)
        if len(self._buf) >= window:
            waveform = self._buf[-window:]
            self._buf = self._buf[-window:]   # keep a rolling window
            try:
                labels = self._classify(waveform)
                with self._lock:
                    self._latest = AudioResult(labels=labels, timestamp=time.time())
            except Exception as e:
                print(f"[phone-audio] classify error: {e}")

    def latest(self) -> AudioResult:
        with self._lock:
            return self._latest


# --------------------------------------------------------------------------
# Standalone test:  python -m blindspot.audio
# Prints what it hears once a second. Ctrl+C to stop.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    scene = AudioScene()
    scene.start()
    print("[audio] Make some noise (talk, play a car horn on YouTube). Ctrl+C to stop.")
    try:
        while True:
            time.sleep(config.AUDIO_WINDOW_SECONDS)
            r = scene.latest()
            if r.labels:
                print("[heard]", r.facts_line())
    except KeyboardInterrupt:
        pass
    finally:
        scene.stop()
        print("\n[audio] done.")
