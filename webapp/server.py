"""
BlindSpot web server - phone is the camera/mic/speaker, laptop is the brain.

The phone opens /phone in its browser:
  - streams back-camera frames to us (base64 JPEG over HTTP POST)
  - listens for the wake word + questions in-browser, sends the question text
  - speaks our replies using the browser's own voice

We run the SAME pipeline you already built (YOLO -> flag/tracker -> VLM) on this
machine's GPU, and push everything live to /dashboard for the jury to watch.

Run:
    python -m webapp.server
Then (separate terminal) expose it:
    ngrok http 8000
Open the ngrok https URL + /phone on the phone, and /dashboard on the laptop.

NOTE: phone camera/mic need HTTPS -> that's why we use ngrok (it gives https).
"""

from __future__ import annotations
import base64
import time
import threading
from dataclasses import asdict

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import config
from blindspot.vision import VisionModel, facts_line
from blindspot.flag import FlagEngine
from blindspot.vlm import VLMBrain

import os
_HERE = os.path.dirname(__file__)
_STATIC = os.path.join(_HERE, "static")


# ---------------------------------------------------------------------------
# One shared brain (loaded once). For a hackathon demo, a single session is fine.
# ---------------------------------------------------------------------------
print("[server] loading models...")
vision = VisionModel()
flag = FlagEngine()
brain = VLMBrain()
print("[server] models ready.")

# Latest state, pushed to the dashboard.
_state_lock = threading.Lock()
_latest = {
    "tier": "silent",
    "reason": "waiting for phone...",
    "objects": [],
    "sounds": "",
    "spoken": "",       # last thing spoken to the user
    "vlm_log": [],      # recent VLM answers
    "ts": 0.0,
}
_vlm_busy = threading.Lock()
_question_busy = {"on": False}
_recent_said: list[str] = []       # memory the VLM sees (no-repeat)
_instruction = {"text": ""}        # user's standing "behave like this" wish


def _remember_said(text: str):
    _recent_said.append(text)
    if len(_recent_said) > config.MEMORY_LINES:
        _recent_said.pop(0)


def _decode_frame(image_b64: str):
    if "," in image_b64 and image_b64.strip().lower().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    import cv2
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR


def _fire_vlm(frame, facts, sounds, question, tier):
    """Background VLM call. Questions get a guaranteed lane; others skip-if-busy."""
    is_q = tier == "question"
    if not is_q:
        if not _vlm_busy.acquire(blocking=False):
            return
    else:
        _question_busy["on"] = True
    flag.mark_vlm_fired()
    recent = list(_recent_said)
    instruction = _instruction["text"]

    def work():
        try:
            ans = brain.ask(frame, facts=facts, sounds=sounds, question=question,
                            instruction=instruction, recent=recent)
            if not VLMBrain.is_silent(ans):
                _remember_said(ans)
                with _state_lock:
                    _latest["spoken"] = ans
                    _latest["vlm_log"] = ([f"[{tier}] {ans}"] + _latest["vlm_log"])[:8]
                    _latest["ts"] = time.time()
                _pending_speech.append(ans)
        finally:
            if is_q:
                _question_busy["on"] = False
            else:
                _vlm_busy.release()

    threading.Thread(target=work, daemon=True).start()


# Things the phone should speak next (drained by the phone's polling).
_pending_speech: list[str] = []


app = FastAPI(title="BlindSpot Web")


@app.get("/health")
def health():
    return {"status": "ok", "vlm_mode": config.VLM_MODE}


@app.post("/frame")
async def frame(request: Request):
    """Phone posts a frame (+ optional question). We run the pipeline and reply
    with anything that should be spoken right now."""
    data = await request.json()
    to_speak = []

    # Decode frame (may be omitted on a pure question tick).
    frame_bgr = None
    if data.get("image"):
        try:
            frame_bgr = _decode_frame(data["image"])
        except Exception as e:
            return JSONResponse({"error": f"bad image: {e}"}, status_code=400)

    question = (data.get("question") or "").strip()
    a_words: set[str] = set()   # (browser doesn't send YAMNet; audio stays laptop-side if enabled)
    a_facts = ""

    if frame_bgr is None:
        # Question with no fresh frame -> nothing to see; still let VLM try.
        dets = []
        v_facts = ""
    else:
        dets = vision.detect(frame_bgr)
        v_facts = facts_line(dets)

    decision = flag.decide(dets, a_words, question)

    # user set a standing instruction ("keep me company", etc.)
    if decision.tier == "set_mode" and decision.instruction:
        _instruction["text"] = decision.instruction
        _recent_said.clear()
        print(f"[mode] instruction set: {decision.instruction!r}")

    # instant local speech (danger reflex / acknowledgement)
    if decision.speak_now:
        to_speak.append(decision.speak_now)
        _remember_said(decision.speak_now)

    # fire VLM if warranted
    if decision.fire_vlm and frame_bgr is not None:
        _fire_vlm(frame_bgr.copy(), v_facts, a_facts, decision.question, decision.tier)

    # collect any VLM speech that finished since last poll
    while _pending_speech:
        to_speak.append(_pending_speech.pop(0))

    # update dashboard state
    with _state_lock:
        _latest["tier"] = decision.tier
        _latest["reason"] = decision.reason
        _latest["objects"] = [
            {"label": d.label, "distance": d.distance, "side": d.horizontal,
             "conf": round(d.confidence, 2),
             "box": list(d.box)}
            for d in dets[:8]
        ]
        _latest["sounds"] = a_facts
        _latest["ts"] = time.time()
        if to_speak:
            _latest["spoken"] = to_speak[-1]

    return {"tier": decision.tier, "speak": to_speak, "objects": _latest["objects"]}


@app.get("/state")
def state():
    """Dashboard polls this (simple + robust; no websocket needed)."""
    with _state_lock:
        return dict(_latest)


# --- static pages ---
@app.get("/phone", response_class=HTMLResponse)
def phone_page():
    return FileResponse(os.path.join(_STATIC, "phone.html"))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page():
    return FileResponse(os.path.join(_STATIC, "dashboard.html"))


@app.get("/")
def root():
    return {
        "open_on_phone": "/phone",
        "open_on_laptop": "/dashboard",
        "health": "/health",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
