"""
BlindSpot - central configuration.

Every tunable knob lives here so your friend can tweak behaviour on the RTX 3050
laptop WITHOUT editing the logic modules. Read the comments; most defaults are
safe for a 6 GB laptop.

Nothing here imports heavy libraries, so it is cheap to import from anywhere.
"""

from __future__ import annotations
import os


# ---------------------------------------------------------------------------
# 1. CAMERA / VIDEO SOURCE
# ---------------------------------------------------------------------------
# 0        -> default laptop webcam
# 1, 2 ... -> other attached cameras
# "http://192.168.x.x:8080/video"  -> phone via the "IP Webcam" Android app
# "path/to/file.mp4"               -> a video file (handy for repeatable tests)
CAMERA_SOURCE: int | str = int(os.getenv("BLINDSPOT_CAMERA", "0"))

# Downscale frames before YOLO for speed. 640 is YOLO's native size and a good
# balance on a 3050. Lower = faster but misses small objects.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Target processing FPS for the fast local loop. The loop self-throttles to
# roughly this rate so we don't melt the GPU. YOLOv8n on a 3050 easily hits this.
TARGET_FPS = 15


# ---------------------------------------------------------------------------
# 2. VISION (YOLO)
# ---------------------------------------------------------------------------
# Model options (all auto-download on first run):
#   yolov8n.pt  ~6 MB   fastest, least accurate  (nano)
#   yolov8s.pt  ~22 MB  fast, good accuracy      (small)
#   yolov8m.pt  ~50 MB  best balance for 6 GB    (medium)  <-- default
#   rtdetr-x.pt ~260 MB most accurate BUT ~5-6 GB VRAM, slow on a 3050, and
#                       still only the same 80 COCO classes. Not recommended here.
#
# IMPORTANT: a bigger detector does NOT recognize *new* kinds of objects - every
# YOLO/RT-DETR model knows the same 80 COCO classes. "Recognize anything" is the
# VLM's job, not YOLO's. YOLO is just the fast, cheap "something is there" layer.
YOLO_MODEL = os.getenv("BLINDSPOT_YOLO", "yolov8m.pt")

# Ignore detections below this confidence (0-1). 0.50 is a good middle ground.
YOLO_CONFIDENCE = 0.50

# Run YOLO on GPU if available, else CPU. "cuda:0" or "cpu" or "auto".
YOLO_DEVICE = os.getenv("BLINDSPOT_DEVICE", "auto")


# ---------------------------------------------------------------------------
# 3. TTS (text to speech)
# ---------------------------------------------------------------------------
# "edge"    -> Microsoft Edge neural voices (best sounding, needs internet)
# "pyttsx3" -> fully offline robotic fallback (never fails, no internet)
TTS_ENGINE = os.getenv("BLINDSPOT_TTS", "edge")

# A natural-sounding Edge voice. Browse more with:  edge-tts --list-voices
EDGE_VOICE = "en-US-AriaNeural"
EDGE_RATE = "+0%"     # speak faster with e.g. "+15%"

# Don't repeat the exact same spoken sentence within this many seconds. Stops
# the narrator from saying "person ahead" 15 times a second.
SPEECH_COOLDOWN_SECONDS = 4.0


# ---------------------------------------------------------------------------
# 4. VLM BRAIN (Qwen2.5-VL-7B on a Hugging Face Space / rented GPU)
# ---------------------------------------------------------------------------
# "stub"   -> no network at all; returns a fake canned answer. Use this to test
#             the whole pipeline WITHOUT spending GPU money. START HERE.
# "gradio" -> talk to the Hugging Face Space via gradio_client. RECOMMENDED for
#             the real brain (works reliably on HF Spaces).
# "http"   -> POST to a custom /api/describe route (only if you self-host a server
#             that exposes one; the HF Gradio Space does NOT).
# "openai" -> talk to any OpenAI-compatible /chat/completions endpoint
#             (vLLM, LM Studio, Ollama, cloud APIs).
VLM_MODE = os.getenv("BLINDSPOT_VLM", "stub")

# For VLM_MODE="gradio": the Space id ("user/space-name") or full URL.
#   e.g. "scorp2111/blindspot-vlm"
VLM_GRADIO_SPACE = os.getenv("BLINDSPOT_VLM_SPACE", "your-username/blindspot-vlm")
# The named API endpoint exposed by the Space (see vlm_space/app.py).
VLM_GRADIO_API_NAME = "/describe"

# For VLM_MODE="http" only (custom self-hosted server, not the HF Space).
VLM_HTTP_URL = os.getenv("BLINDSPOT_VLM_URL_HTTP",
                         "https://your-username-blindspot-vlm.hf.space")

# For VLM_MODE="openai":
VLM_OPENAI_URL = os.getenv("BLINDSPOT_VLM_URL", "http://localhost:8000/v1/chat/completions")
VLM_OPENAI_MODEL = os.getenv("BLINDSPOT_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

# Shared secret. HF Spaces: set as a Space "secret" and paste here / env.
VLM_API_KEY = os.getenv("BLINDSPOT_VLM_KEY", "")

# How long to wait for the VLM before giving up (seconds). Cold Spaces are slow.
VLM_TIMEOUT = 60


# ---------------------------------------------------------------------------
# 5. AUDIO SCENE (YAMNet) - Tier 3, optional install
# ---------------------------------------------------------------------------
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
# Seconds of audio to classify at a time. YAMNet wants ~0.975s minimum.
AUDIO_WINDOW_SECONDS = 1.0
AUDIO_SAMPLE_RATE = 16000            # YAMNet is fixed at 16 kHz
# Ignore sound classes below this score (0-1).
AUDIO_CONFIDENCE = 0.30


# ---------------------------------------------------------------------------
# 6. SPEECH-TO-TEXT (Whisper) - Tier 3
# ---------------------------------------------------------------------------
# tiny / base / small ... base is the sweet spot for a 3050.
WHISPER_MODEL = "base"
# Hold this key (in the terminal) to record a question, release to transcribe.
# (Push-to-talk avoids needing wake-word detection in a hackathon.)
PUSH_TO_TALK_KEY = "space"


# ---------------------------------------------------------------------------
# 7. THE FUNNEL - tracker + flag engine (turns the 20/sec firehose into
#    rare, meaningful actions).  See blindspot/flag.py and blindspot/tracker.py.
# ---------------------------------------------------------------------------

# --- 7a. Object tracking (to detect "approaching") ---
# Two boxes count as the "same object" across frames if their overlap (IoU) is
# at least this. Lower = more forgiving matching (good for fast movement).
TRACK_IOU_MATCH = 0.20
# Drop a track after it's been unseen for this many frames.
TRACK_MAX_MISSES = 8
# How many recent frames of box-size we keep to measure growth.
APPROACH_WINDOW_FRAMES = 6
# An object is "approaching" if its box grew by at least this ratio across the
# window (1.6 = grew 60%). Higher = only warn on fast loomers.
APPROACH_GROWTH_RATIO = 1.6
# ...AND it must already occupy at least this fraction of the frame (so we don't
# warn about a tiny thing far away that happens to be growing).
APPROACH_MIN_AREA = 0.04

# --- 7b. Which objects are worth an instant danger warning ---
# Only these labels trigger the INSTANT local "approaching" warning. (Everything
# else that grows is ignored - a growing wall isn't a hazard.)
DANGER_OBJECTS = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "train", "dog", "skateboard",
}

# --- 7c. Proactive (calm) narration ---
# Enable the slow proactive VLM tick that fires on big scene changes.
PROACTIVE_ENABLED = True
# Minimum seconds between proactive VLM ticks (calm-mode rate limit).
PROACTIVE_INTERVAL_SECONDS = 8.0

# --- 7d. Global VLM rate limit (protects the paid GPU) ---
# Never fire the VLM more often than this, no matter the reason.
VLM_MIN_INTERVAL_SECONDS = 3.0

# --- 7e. Optional sound boost (only used if audio/YAMNet is running) ---
# If a danger object is approaching AND one of these sounds is heard, we treat it
# as higher-confidence (the cross-modal sight+sound cue). Purely additive: audio
# being off never breaks anything.
HAZARD_SOUNDS = {
    "vehicle", "car", "engine", "truck", "traffic",
    "horn", "honk", "siren", "emergency", "motorcycle", "bus", "train",
}
