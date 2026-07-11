"""
Speech-to-text module - Whisper (base) push-to-talk.

Lets the user ASK a question ("what does this sign say?"). We use push-to-talk
(hold a key, speak, release) instead of always-on wake-word detection because
it's dead simple and rock solid for a hackathon demo.

Flow:
  - Hold config.PUSH_TO_TALK_KEY -> we record from the mic.
  - Release -> we transcribe with Whisper -> return the text.

This text becomes the `question` passed to the VLM. When a question arrives, the
trigger layer fires the VLM regardless of the scene.

Runs its own listener thread; poll .get_question() for a finished transcription.
"""

from __future__ import annotations
import queue
import threading
import time

import numpy as np

import config

SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono


class PushToTalk:
    """Hold-to-record, release-to-transcribe, using Whisper."""

    def __init__(self):
        self._model = None
        self._questions: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _load(self):
        import whisper
        print(f"[listen] loading Whisper '{config.WHISPER_MODEL}'...")
        self._model = whisper.load_model(config.WHISPER_MODEL)
        print("[listen] Whisper ready.")

    def _transcribe(self, waveform: np.ndarray) -> str:
        # waveform: mono float32 [-1,1] @ 16 kHz
        result = self._model.transcribe(waveform, fp16=False, language="en")
        return (result.get("text") or "").strip()

    def _run(self):
        import sounddevice as sd
        from pynput import keyboard

        self._load()

        recording = {"on": False}
        chunks: list[np.ndarray] = []

        def on_press(key):
            try:
                if key == getattr(keyboard.Key, config.PUSH_TO_TALK_KEY, None):
                    recording["on"] = True
            except Exception:
                pass

        def on_release(key):
            try:
                if key == getattr(keyboard.Key, config.PUSH_TO_TALK_KEY, None):
                    recording["on"] = False
            except Exception:
                pass

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        print(f"[listen] Hold [{config.PUSH_TO_TALK_KEY}] to ask a question.")

        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        stream.start()
        was_on = False
        try:
            while not self._stop.is_set():
                data, _ = stream.read(int(SAMPLE_RATE * 0.1))  # 100 ms blocks
                if recording["on"]:
                    chunks.append(data.reshape(-1).copy())
                    was_on = True
                elif was_on:
                    # Just released -> transcribe what we captured.
                    was_on = False
                    if chunks:
                        waveform = np.concatenate(chunks)
                        chunks = []
                        if len(waveform) > SAMPLE_RATE * 0.3:  # ignore < 0.3s taps
                            text = self._transcribe(waveform)
                            if text:
                                print(f"[question] {text}")
                                self._questions.put(text)
                else:
                    time.sleep(0.01)
        finally:
            stream.stop()
            stream.close()
            listener.stop()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def get_question(self) -> str | None:
        """Non-blocking: return a finished question, or None."""
        try:
            return self._questions.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


# --------------------------------------------------------------------------
# Standalone test:  python -m blindspot.listen
# Hold SPACE, speak a question, release. It prints the transcription. Ctrl+C quits.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ptt = PushToTalk()
    ptt.start()
    print("[listen] Ready. Hold SPACE and speak. Ctrl+C to quit.")
    try:
        while True:
            q = ptt.get_question()
            if q:
                print("  -> got question:", q)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        ptt.stop()
        print("\n[listen] done.")
