"""
VLM client - the bridge from the laptop to the Qwen2.5-VL-7B "brain".

The brain is too big for a 6 GB laptop, so it lives on a rented GPU exposed as a
Hugging Face Gradio Space. This module is a *pluggable client* with three modes
(set config.VLM_MODE):

  "stub"   -> returns a canned answer instantly, NO network, NO GPU cost.
              Use this to build & test the whole pipeline for free. START HERE.
  "gradio" -> calls a Hugging Face Gradio Space (default; matches hf_space/app.py).
  "openai" -> calls any OpenAI-compatible /chat/completions vision endpoint.

Every mode takes the SAME inputs:
    frame_bgr : the raw camera frame (numpy BGR, straight from OpenCV)
    facts     : YOLO's text facts   (from vision.facts_line)
    sounds    : audio text facts     (from audio module; "" until Tier 3)
    question  : the user's spoken question, or "" for proactive mode

...and returns ONE short spoken-ready sentence (str).

Because the interface is identical across modes, the rest of the app never knows
or cares where the brain actually runs.
"""

from __future__ import annotations
import base64
import io

import config


# --------------------------------------------------------------------------
# Prompt construction (shared by all real backends)
# --------------------------------------------------------------------------
def build_prompt(facts: str, sounds: str, question: str) -> str:
    """Assemble the text half of the multimodal prompt.

    This is where cross-modal fusion becomes explicit: the VLM is told what was
    SEEN (YOLO facts) and what was HEARD (sound facts) alongside the actual
    image, and asked to reason across all of it.
    """
    lines = [
        "You are BlindSpot, a calm assistant for a blind user.",
        "You are given a camera image plus sensor hints.",
        f"Objects detected (vision): {facts or 'none'}.",
        f"Sounds detected (audio): {sounds or 'none'}.",
    ]
    if question:
        lines.append(f'The user asked: "{question}"')
        lines.append("Answer that question directly using what you see. "
                     "Read any relevant text aloud if asked.")
    else:
        lines.append("No question was asked. Only speak if something genuinely "
                     "matters for safety or navigation; otherwise say exactly "
                     "'NOTHING'.")
    lines.append("Reply with ONE short spoken sentence. No preamble, no lists.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Image encoding helpers
# --------------------------------------------------------------------------
def _bgr_to_jpeg_bytes(frame_bgr, quality: int = 85) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame.")
    return buf.tobytes()


def _bgr_to_data_url(frame_bgr) -> str:
    b = _bgr_to_jpeg_bytes(frame_bgr)
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")


def _bgr_to_temp_png(frame_bgr) -> str:
    """Gradio's client wants a file path for image inputs. Write a temp file."""
    import cv2
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(path, frame_bgr)
    return path


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class _StubBackend:
    name = "stub"

    def ask(self, frame_bgr, facts, sounds, question) -> str:
        if question:
            return f"(stub) You asked: '{question}'. I see {facts or 'nothing notable'}."
        # Proactive: pretend nothing important unless a hazard-ish word appears.
        if any(w in (facts + sounds).lower() for w in ("car", "truck", "engine", "horn")):
            return "(stub) A vehicle may be approaching. Please be careful."
        return "NOTHING"


def _patch_gradio_client_schema_bug():
    """Work around a gradio_client bug that crashes on API schemas containing a
    boolean node (e.g. additionalProperties: true):
        TypeError: argument of type 'bool' is not iterable
    We wrap the two offending helpers so a bool schema returns a safe type
    instead of crashing. Safe to call multiple times; must run before Client().
    """
    try:
        import gradio_client.utils as _gcu
    except Exception:
        return
    if getattr(_gcu, "_blindspot_patched", False):
        return
    _orig_get_type = _gcu.get_type
    _orig_json = _gcu._json_schema_to_python_type

    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "bool"
        return _orig_get_type(schema)

    def _safe_json(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_json(schema, defs)

    _gcu.get_type = _safe_get_type
    _gcu._json_schema_to_python_type = _safe_json
    _gcu._blindspot_patched = True


class _GradioBackend:
    name = "gradio"

    def __init__(self):
        _patch_gradio_client_schema_bug()  # must run before Client() connects
        from gradio_client import Client, handle_file  # noqa: F401
        self._Client = Client
        self._handle_file = handle_file
        kwargs = {}
        if config.VLM_API_KEY:
            # If the Space is private, an HF token authenticates the client.
            kwargs["hf_token"] = config.VLM_API_KEY
        self.client = Client(config.VLM_GRADIO_SPACE, **kwargs)
        print(f"[vlm] connected to Gradio Space: {config.VLM_GRADIO_SPACE}")

    def ask(self, frame_bgr, facts, sounds, question) -> str:
        import os
        path = _bgr_to_temp_png(frame_bgr)
        try:
            result = self.client.predict(
                self._handle_file(path),   # image
                facts or "",               # vision facts
                sounds or "",              # audio facts
                question or "",            # user question
                api_name=config.VLM_GRADIO_API_NAME,
            )
            return str(result).strip()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


class _OpenAIBackend:
    name = "openai"

    def __init__(self):
        import requests  # noqa: F401
        self._requests = requests

    def ask(self, frame_bgr, facts, sounds, question) -> str:
        prompt = build_prompt(facts, sounds, question)
        data_url = _bgr_to_data_url(frame_bgr)
        payload = {
            "model": config.VLM_OPENAI_MODEL,
            "max_tokens": 120,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }
        headers = {"Content-Type": "application/json"}
        if config.VLM_API_KEY:
            headers["Authorization"] = f"Bearer {config.VLM_API_KEY}"
        resp = self._requests.post(
            config.VLM_OPENAI_URL, json=payload, headers=headers,
            timeout=config.VLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


class _HttpBackend:
    """Primary path: POST the frame + hints to the Space's /api/describe route.

    Robust and version-proof (no gradio_client coupling). Matches the FastAPI
    endpoint in vlm_space/app.py exactly.
    """

    name = "http"

    def __init__(self):
        import requests  # noqa: F401
        self._requests = requests
        base = config.VLM_HTTP_URL.rstrip("/")
        # Allow either the base Space URL or the full endpoint.
        if base.endswith("/api/describe"):
            self.url = base
        else:
            self.url = base + "/api/describe"
        print(f"[vlm] HTTP endpoint: {self.url}")

    def ask(self, frame_bgr, facts, sounds, question) -> str:
        import base64
        b = _bgr_to_jpeg_bytes(frame_bgr)
        payload = {
            "image": base64.b64encode(b).decode("ascii"),
            "facts": facts or "",
            "sounds": sounds or "",
            "question": question or "",
        }
        headers = {"Content-Type": "application/json"}
        if config.VLM_API_KEY:
            headers["Authorization"] = f"Bearer {config.VLM_API_KEY}"
        resp = self._requests.post(
            self.url, json=payload, headers=headers, timeout=config.VLM_TIMEOUT
        )
        resp.raise_for_status()
        return str(resp.json().get("answer", "NOTHING")).strip()


def _make_backend():
    mode = config.VLM_MODE.lower()
    if mode == "http":
        return _HttpBackend()
    if mode == "gradio":
        return _GradioBackend()
    if mode == "openai":
        return _OpenAIBackend()
    return _StubBackend()


class VLMBrain:
    """The one object the rest of the app uses."""

    def __init__(self):
        self.backend = _make_backend()
        print(f"[vlm] mode: {self.backend.name}")

    def ask(self, frame_bgr, facts: str = "", sounds: str = "",
            question: str = "") -> str:
        """Return one spoken-ready sentence, or 'NOTHING' if nothing matters.

        Never raises on network trouble - a demo should degrade, not crash.
        """
        try:
            answer = self.backend.ask(frame_bgr, facts, sounds, question)
        except Exception as e:
            print(f"[vlm] error: {e}")
            # Proactive failures stay silent; a real question gets an apology.
            return "Sorry, I could not reach the vision brain." if question else "NOTHING"
        return (answer or "NOTHING").strip()

    @staticmethod
    def is_silent(answer: str) -> bool:
        """True if the VLM decided nothing should be spoken."""
        return answer.strip().upper().strip(".!") in ("", "NOTHING", "NONE")


# --------------------------------------------------------------------------
# Standalone test:  python -m blindspot.vlm
# Uses a single webcam frame (or a black frame if no camera) and asks a question.
# Works fully in "stub" mode with zero setup.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    brain = VLMBrain()
    try:
        from blindspot.camera import Camera
        with Camera() as cam:
            frame = cam.read()
        if frame is None:
            raise RuntimeError("no frame")
    except Exception:
        print("[vlm] no camera; using a blank test frame.")
        frame = np.zeros((480, 640, 3), dtype="uint8")

    print("\n-- proactive (should often be NOTHING) --")
    print(brain.ask(frame, facts="person(close, ahead, 0.9)", sounds=""))

    print("\n-- question --")
    print(brain.ask(frame, facts="person(close, ahead, 0.9)",
                    sounds="speech", question="What is in front of me?"))
