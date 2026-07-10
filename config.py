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
# 2. VISION (YOLOv8n)
# ---------------------------------------------------------------------------
# nano = smallest/fastest, perfect for a 6 GB card. Ultralytics auto-downloads
# the weights on first run to the working dir.
YOLO_MODEL = "yolov8n.pt"

# Ignore detections below this confidence (0-1). 0.45 keeps it from narrating junk.
YOLO_CONFIDENCE = 0.45

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
# "gradio" -> talk to a Hugging Face Gradio Space (see vlm/hf_space/).
# "openai" -> talk to any OpenAI-compatible /chat/completions endpoint
#             (vLLM, LM Studio, Ollama, cloud APIs).
VLM_MODE = os.getenv("BLINDSPOT_VLM", "stub")

# For VLM_MODE="gradio": the Space id ("user/space-name") or full URL.
VLM_GRADIO_SPACE = os.getenv("BLINDSPOT_VLM_SPACE", "your-username/blindspot-vlm")
# The named API endpoint exposed by the Space (see hf_space/app.py). Leave as is
# unless you rename it in the Space.
VLM_GRADIO_API_NAME = "/describe"

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
# 7. FUSION / TRIGGER-SEVERITY ENGINE - Tier 4
# ---------------------------------------------------------------------------
# Objects we treat as potential hazards when combined with the right sound.
HAZARD_OBJECTS = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}

# Sound classes (YAMNet labels, matched loosely) that imply motion/danger.
HAZARD_SOUNDS = {
    "vehicle", "car", "engine", "truck", "traffic",
    "horn", "honk", "siren", "emergency", "motorcycle", "bus", "train",
}

# Sound cues that imply a crowd/busy environment.
CROWD_SOUNDS = {"speech", "babble", "crowd", "chatter", "children", "hubbub"}

# A hazard object must be at least this "big" in the frame (fraction of frame
# area) to count as "close". Bigger box == closer == scarier.
HAZARD_CLOSE_AREA = 0.06

# Minimum seconds between two VLM firings, so we never spam the paid GPU.
VLM_MIN_INTERVAL_SECONDS = 3.0
