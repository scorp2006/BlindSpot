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

# --- 7a. Object tracking (persistence, approach, obstruction) ---
# Two boxes count as the "same object" across frames if their overlap (IoU) is
# at least this. Lower = more forgiving matching (good for fast movement).
TRACK_IOU_MATCH = 0.20
# Drop a track after it's been unseen for this many frames.
TRACK_MAX_MISSES = 8
# An object must be seen this many CONSECUTIVE frames before we trust it. This
# kills 1-frame flickers and YOLO misfires (a big source of false "caution!").
TRACK_MIN_FRAMES = 3
# How many recent frames of box-size we keep to measure growth.
APPROACH_WINDOW_FRAMES = 6
# An object is "approaching" if its box grew by at least this ratio across the
# window (1.6 = grew 60%). Higher = only warn on fast loomers.
APPROACH_GROWTH_RATIO = 1.6
# ...AND it must already occupy at least this fraction of the frame (so we don't
# warn about a tiny thing far away that happens to be growing).
APPROACH_MIN_AREA = 0.04
# If ANY single box covers at least this fraction of the frame AND stays that big
# for OBSTRUCTION_MIN_FRAMES in a row, we treat the camera as OBSTRUCTED (a hand
# over the lens). Raised high + persistence-gated so a normal close-up face at
# desk distance does NOT constantly trip it.
OBSTRUCTION_AREA = 0.82
OBSTRUCTION_MIN_FRAMES = 5

# --- 7b. Safety: obstacles + approaching hazards ---
# APPROACHING hazards (box growing fast) - these get the most urgent warning.
DANGER_OBJECTS = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "train", "dog", "skateboard",
}

# OBSTACLE reflex (the "don't let them trip" rule): ANY object - chair, bag,
# box, table, backpack, person, anything - that is close AND in the walking path
# gets an instant caution, even if it's stationary and not in DANGER_OBJECTS.
# A blind user can trip on a bag; announcing it is our job.
#
# An object counts as an obstacle-in-path if:
#   - its box is at least OBSTACLE_MIN_AREA of the frame (i.e. close), AND
#   - its center is within the middle OBSTACLE_PATH_FRACTION of the frame width
#     (i.e. roughly in front, where you'd walk into it).
# Obstacle warning is a QUIET safety net, not the main voice. Only fire for
# things that are genuinely VERY close and directly ahead (a real trip risk) -
# not a laptop sitting on the desk in front of you. The VLM describes everything
# else; this reflex is just the instant "watch out" a human friend gives once.
OBSTACLE_MIN_AREA = 0.22          # must be quite large (very close) to warn
OBSTACLE_PATH_FRACTION = 0.50     # center 50% of width = "in your path"
# Never repeat the same obstacle warning within this many seconds.
OBSTACLE_SAY_ONCE_SECONDS = 15.0
# Objects that are never worth warning about as trip hazards (too small / part of
# the scene, not on the floor). Tune as needed.
OBSTACLE_IGNORE = {"tie", "clock", "kite", "frisbee"}

# --- 7c. The VLM is the mind ---
# Instead of Python deciding WHAT and WHEN to narrate, we simply OFFER the VLM a
# frame periodically (or on change) and let IT decide: speak something new, or
# reply NOTHING. Memory of recent outputs is passed so it never repeats.
#
# Enable the companion (proactive) behaviour at all.
PROACTIVE_ENABLED = True
# How often we OFFER the VLM a frame in companion mode (seconds). The VLM still
# often replies NOTHING, so this is an upper bound on how chatty it can be, not a
# guarantee it speaks. Raise to make it calmer, lower to make it more talkative.
PROACTIVE_INTERVAL_SECONDS = 6.0
# Also offer a frame immediately when the scene meaningfully changes.
PROACTIVE_ON_CHANGE = True
# How many recent spoken lines to show the VLM so it doesn't repeat itself.
MEMORY_LINES = 5
# Don't repeat the SAME danger warning within this many seconds (the reflex has a
# short memory too, so it warns once per approaching object, not every frame).
SAY_ONCE_SECONDS = 8.0

# --- 7d. Global VLM rate limit (protects the paid GPU) ---
# Never OFFER the VLM more often than this (a direct question always goes through).
VLM_MIN_INTERVAL_SECONDS = 3.0

# --- 7e. Mode-by-voice ---
# If the user's spoken input contains one of these, it's treated as setting a
# STANDING INSTRUCTION (how they want BlindSpot to behave) rather than a one-off
# question. Everything else is a normal question.
MODE_PHRASES = {
    "keep me company", "talk to me", "describe everything", "stop talking",
    "be quiet", "only warn", "only tell me", "from now on", "keep talking",
    "narrate", "don't talk", "less talking", "more detail",
}

# --- 7f. Optional sound boost (only used if audio/YAMNet is running) ---
# If a danger object is approaching AND one of these sounds is heard, we treat it
# as higher-confidence (the cross-modal sight+sound cue). Purely additive: audio
# being off never breaks anything.
HAZARD_SOUNDS = {
    "vehicle", "car", "engine", "truck", "traffic",
    "horn", "honk", "siren", "emergency", "motorcycle", "bus", "train",
}
