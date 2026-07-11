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
from blindspot.vlm import VLMBrain, too_similar

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

# Environmental sound comes from the LAPTOP microphone (AudioScene/YAMNet).
# The phone's mic belongs 100% to questions (wake word + tap-to-talk) - sharing
# it between SpeechRecognition and audio streaming broke questions entirely.
# In a demo room the laptop hears the same ambient sound anyway. Fail-soft.
_audio_scene = None
try:
    from blindspot.audio import AudioScene
    _audio_scene = AudioScene()
    _audio_scene.start()
    print("[server] audio (laptop mic, YAMNet) ON")
except Exception as e:
    print(f"[server] audio OFF ({e}) - running vision-only")

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
    "frames": 0,        # REAL phone frames processed (not dashboard polls)
    "last_frame_ts": 0.0,  # when we last received a phone frame
    "phone_connected": False,
    "running": True,    # False after /stop
}
_vlm_busy = threading.Lock()
_question_busy = {"on": False}
_recent_said: list[str] = []       # memory the VLM sees (no-repeat)
_instruction = {"text": ""}        # user's standing "behave like this" wish
_audio_warned = {"done": False}    # so we log the "no TF" note only once

# Small buffer of recent frames so the VLM gets the SHARPEST one, not whatever
# motion-blurred frame happened to arrive when we decided to fire.
from collections import deque
_frames_buf: deque = deque(maxlen=4)


def _sharpest(current):
    """Pick the sharpest recent frame (variance of Laplacian)."""
    import cv2

    def sharp(f):
        try:
            return float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                                       cv2.CV_64F).var())
        except Exception:
            return 0.0

    candidates = list(_frames_buf) + [current]
    return max(candidates, key=sharp).copy()


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
                # HARD no-repeat: the 7B rephrases the same scene instead of
                # saying NOTHING, so we enforce dedup here. Questions are never
                # suppressed - the user asked, they get an answer.
                if not is_q and too_similar(ans, _recent_said):
                    print(f"[vlm] suppressed near-duplicate: {ans!r}")
                    return
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
    if not _latest["running"]:
        return {"tier": "stopped", "speak": [], "objects": []}

    data = await request.json()
    to_speak = []

    # Decode frame (may be omitted on a pure question tick).
    frame_bgr = None
    if data.get("image"):
        try:
            frame_bgr = _decode_frame(data["image"])
        except Exception as e:
            return JSONResponse({"error": f"bad image: {e}"}, status_code=400)
        # Count REAL phone frames + mark phone connected.
        with _state_lock:
            _latest["frames"] += 1
            _latest["last_frame_ts"] = time.time()
            _latest["phone_connected"] = True
        _frames_buf.append(frame_bgr)

    question = (data.get("question") or "").strip()
    # Environmental sound: laptop mic (AudioScene/YAMNet), fail-soft.
    a_words: set[str] = set()
    a_facts = ""
    if _audio_scene is not None:
        ar = _audio_scene.latest()
        a_words = ar.label_set()
        a_facts = ar.facts_line()

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

    # fire VLM if warranted - with the SHARPEST recent frame, not a blurry one.
    # For a hazard follow-up, tell the VLM explicitly that a warning was just
    # issued so it focuses on the hazard instead of calmly describing the room.
    if decision.fire_vlm and frame_bgr is not None:
        vlm_q = decision.question
        if decision.tier in ("danger", "obstacle") and decision.speak_now:
            vlm_q = (f"Alert - I was just warned: '{decision.speak_now}' "
                     "Quickly tell me what it is and how to avoid it.")
        _fire_vlm(_sharpest(frame_bgr), v_facts, a_facts, vlm_q, decision.tier)

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


@app.post("/audio")
async def audio(request: Request):
    """Kept for compatibility. Environmental sound now comes from the LAPTOP mic
    (AudioScene) - phone audio streaming broke the phone's question mic, so the
    phone no longer sends audio. This endpoint just acknowledges and ignores."""
    return {"ok": False, "note": "environmental sound uses the laptop mic now"}


@app.post("/stop")
def stop():
    """End the session: stop processing, clear state."""
    with _state_lock:
        _latest["running"] = False
        _latest["tier"] = "stopped"
        _latest["reason"] = "session ended"
        _latest["phone_connected"] = False
    return {"ok": True}


@app.post("/start")
def start():
    """Resume a stopped session."""
    with _state_lock:
        _latest["running"] = True
        _latest["tier"] = "silent"
        _latest["reason"] = "running"
    return {"ok": True}


@app.get("/state")
def state():
    """Dashboard polls this (simple + robust; no websocket needed)."""
    with _state_lock:
        s = dict(_latest)
    # phone is "connected" only if we got a real frame in the last 2.5s
    s["phone_connected"] = (time.time() - s.get("last_frame_ts", 0.0)) < 2.5 \
                           and s.get("running", True)
    return s


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
