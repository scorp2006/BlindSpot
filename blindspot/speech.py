"""
Speech / TTS module - turns a sentence into spoken audio.

Design goals:
  - Best-sounding voice by default (Microsoft Edge neural voices via edge-tts).
  - Never block the main perception loop -> speaking happens on a background
    thread with a small queue.
  - Anti-spam -> the same sentence won't be repeated within a cooldown window.
  - Graceful fallback -> if edge-tts (needs internet) fails, drop to offline
    pyttsx3 automatically so a demo never goes silent.

Public API is tiny:
    voice = Voice()
    voice.say("A person is close ahead.")     # non-blocking, de-duplicated
    voice.say("...", urgent=True)             # jumps the queue, ignores cooldown
    voice.shutdown()
"""

from __future__ import annotations
import queue
import threading
import time

import config


class _EdgeBackend:
    """Microsoft Edge neural TTS. Great voice, needs internet."""

    def __init__(self, voice: str, rate: str):
        import edge_tts            # noqa: F401  (verify it imports)
        import asyncio             # noqa: F401
        self.voice = voice
        self.rate = rate
        self.name = "edge-tts"

    def speak(self, text: str) -> None:
        import asyncio
        import edge_tts
        import tempfile
        import os

        async def _run(path):
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            await communicate.save(path)

        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            asyncio.run(_run(path))
            _play_mp3(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


class _Pyttsx3Backend:
    """Offline robotic TTS. Always works, no internet, no VRAM."""

    def __init__(self):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.name = "pyttsx3"

    def speak(self, text: str) -> None:
        # pyttsx3 is not thread-friendly across calls on some platforms;
        # a fresh runAndWait per utterance is the most reliable.
        self.engine.say(text)
        self.engine.runAndWait()


def _play_mp3(path: str) -> None:
    """Play an mp3 file, trying a couple of backends so it works out of the box."""
    # playsound is simplest; fall back to the OS default player on Windows.
    try:
        from playsound import playsound
        playsound(path, block=True)
        return
    except Exception:
        pass
    try:
        # Windows built-in, no extra deps.
        import winsound  # noqa
        # winsound can't do mp3; convert path expectation -> use start via os.
        raise ImportError
    except Exception:
        import os
        import sys
        if sys.platform.startswith("win"):
            os.system(f'start /min "" "{path}"')
            time.sleep(2.0)  # rough wait so the file isn't deleted mid-play
        else:
            os.system(f'xdg-open "{path}" >/dev/null 2>&1')
            time.sleep(2.0)


def _make_backend():
    """Pick the configured backend, falling back to pyttsx3 on failure."""
    engine = config.TTS_ENGINE.lower()
    if engine == "edge":
        try:
            b = _EdgeBackend(config.EDGE_VOICE, config.EDGE_RATE)
            print(f"[speech] using {b.name} ({config.EDGE_VOICE})")
            return b
        except Exception as e:
            print(f"[speech] edge-tts unavailable ({e}); falling back to pyttsx3")
    b = _Pyttsx3Backend()
    print(f"[speech] using {b.name}")
    return b


class Voice:
    """Threaded, de-duplicated speaker."""

    def __init__(self):
        self._backend = _make_backend()
        # Priority queue items: (priority, seq, text). 'seq' is a monotonically
        # increasing tie-breaker so two items with the same priority never try to
        # compare their text strings (and it preserves insertion order).
        self._q: "queue.PriorityQueue[tuple[int, int, str]]" = queue.PriorityQueue()
        self._last_said: dict[str, float] = {}
        self._stop = threading.Event()
        self._seq = 0
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def say(self, text: str, urgent: bool = False) -> None:
        text = (text or "").strip()
        if not text:
            return
        now = time.time()
        # Cooldown: skip repeats of the same text (urgent bypasses it).
        if not urgent:
            last = self._last_said.get(text, 0.0)
            if now - last < config.SPEECH_COOLDOWN_SECONDS:
                return
        self._last_said[text] = now
        # Lower priority number => spoken first. Urgent jumps the queue.
        priority = 0 if urgent else 1
        with self._lock:
            self._seq += 1
            seq = self._seq
        self._q.put((priority, seq, text))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                _prio, _seq, text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._backend.speak(text)
            except Exception as e:
                print(f"[speech] backend error: {e}")

    def shutdown(self) -> None:
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)


# --------------------------------------------------------------------------
# Standalone test:  python -m blindspot.speech
# --------------------------------------------------------------------------
if __name__ == "__main__":
    v = Voice()
    print("[speech] Speaking three test lines...")
    v.say("BlindSpot voice test. This is the neural voice.")
    time.sleep(4)
    v.say("A person is close ahead.")
    time.sleep(3)
    v.say("Car approaching from your left. Wait.", urgent=True)
    time.sleep(4)
    v.shutdown()
    print("[speech] done.")
