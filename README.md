# BlindSpot 👁️‍🗨️

**A real-time, proactive, multimodal assistant for the visually impaired.**
Team **Odyssey** · Hack-The-Matrix (TECHNIDHI'26) · Track 04 — Multimodal AI

BlindSpot continuously *watches* (camera) and *listens* (microphone), and speaks
to the user through voice — but only when something genuinely matters, so it never
overwhelms them. Unlike reactive "tap and ask" tools (e.g. Meta Ray-Ban), it is
**proactive and continuous**: it narrates the world on scene change, warns
instantly about obstacles/hazards while moving, answers spoken questions, and
reads text aloud.

---

## ⚠️ IMPORTANT — read before you run (the VLM brain needs a GPU/cloud)

BlindSpot is a **two-part system by design**:

| Part | Runs on | Handles |
|------|---------|---------|
| **Fast local models** | the user's laptop (a normal GPU laptop, e.g. RTX 3050 6 GB) | YOLO object detection, YAMNet sound, Whisper speech, all the decision logic, voice output |
| **VLM "brain"** — **Qwen2.5-VL-7B** | a **cloud GPU / Hugging Face Space** (≈16 GB VRAM) | rich scene understanding, reading text, answering questions |

> **The 7-billion-parameter VLM cannot run on a laptop or phone — it needs a GPU
> with ~16 GB of VRAM.** This is intentional and standard (Apple Intelligence and
> Meta's glasses do the same: light models on-device, heavy model in the cloud).
>
> **If you run this without deploying the VLM, the system is NOT broken** — the
> local half (live detection, obstacle/hazard warnings, voice) still works. The
> rich descriptions and question-answering simply won't appear until the VLM
> endpoint is connected. By default the code ships in **`stub` mode**, which
> returns placeholder brain answers so the whole pipeline runs with **zero GPU
> cost** for evaluation.
>
> To see the *full* system, deploy the VLM (5 minutes, see below) and point the
> app at it. We deploy it on a Hugging Face **GPU Space** — the exact code we use
> is included in this repo under [`vlm_space/`](vlm_space/).

---

## How it works (architecture)

```
 Camera + Microphone  (continuous)
        │
 FAST LOCAL MODELS  (laptop, every frame — free, instant)
   • YOLOv8m        → objects + positions + distance      (blindspot/vision.py)
   • Tracker        → persistence + "is it approaching?"  (blindspot/tracker.py)
   • Motion         → is the WEARER moving or still?       (blindspot/motion.py)
   • YAMNet         → 521 environmental sound classes      (blindspot/audio.py)
   • Whisper        → the user's spoken questions          (blindspot/listen.py)
        │
 DECISION / FUSION LAYER   (blindspot/flag.py)  ← the multimodal core
   • Instant local SAFETY reflex (obstacle/hazard) — can't wait for the cloud
   • Sound raises the severity of what's seen (cross-modal interaction)
   • Offers a frame to the VLM only on real scene change (not every frame)
   • Stays SILENT when nothing matters  ← the "don't overwhelm" feature
        │
 VLM BRAIN  (cloud GPU — only when triggered)   (blindspot/vlm.py → vlm_space/)
   • Qwen2.5-VL-7B: raw frame + vision facts + sound facts + question
   • Reasons across all inputs, reads text, replies in one spoken sentence
        │
 VOICE OUTPUT   (edge-tts neural on laptop, or the phone's own voice)
```

**Why this satisfies "2+ modalities that actively influence each other":** audio
changes how vision is interpreted (a vehicle you *see* + an engine you *hear* +
getting *closer* = a warning; the same vehicle in silence = parked, stay quiet),
and both are passed together into the VLM, which reasons over image + text jointly.

---

## Repository layout

```
BlindSpot/
├── README.md                  ← you are here
├── SETUP.md                   ← step-by-step laptop setup
├── run.py                     ← desktop launcher (webcam demo)
├── config.py                  ← ALL tunables (models, thresholds, VLM endpoint)
├── requirements-core.txt      ← Tiers 1 & 2 deps
├── requirements-audio.txt     ← Tier 3 deps (TensorFlow/YAMNet, Whisper)
├── blindspot/                 ← the local pipeline
│   ├── camera.py  vision.py  tracker.py  motion.py
│   ├── audio.py   listen.py  speech.py
│   ├── flag.py    vlm.py     pipeline.py  narrator.py
├── webapp/                    ← phone-as-camera web demo
│   ├── server.py              ← FastAPI: phone → laptop pipeline → voice
│   └── static/phone.html · dashboard.html
└── vlm_space/                 ← the VLM brain (DEPLOY THIS TO A GPU/CLOUD)
    ├── app.py                 ← Qwen2.5-VL-7B server (Gradio, HF-Space ready)
    ├── requirements.txt
    └── README.md              ← click-by-click deploy guide
```

---

## Running it

### 1. Local pipeline (laptop)
See **[SETUP.md](SETUP.md)** for the full, verified steps. Short version:
```bash
# Python 3.11 recommended
python -m venv .venv && .venv\Scripts\activate      # (Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-core.txt
python run.py narrator          # webcam → detection → voice (works with no VLM)
```

### 2. Phone demo (phone = camera/mic/speaker, laptop = brain)
```bash
pip install -r webapp/requirements-web.txt
python -m webapp.server         # starts on :8000
# then expose it and open on a phone:
ngrok http 8000                 # open the https URL + /phone on the phone
#                                 open the https URL + /dashboard on the laptop
```

### 3. The VLM brain (required for full functionality)
Deploy [`vlm_space/`](vlm_space/) to a Hugging Face **GPU Space** (or any GPU with
~16 GB VRAM). Full instructions: **[vlm_space/README.md](vlm_space/README.md)**.
Then point the app at it:
```bash
set BLINDSPOT_VLM=gradio
set BLINDSPOT_VLM_SPACE=<your-username>/<your-space-name>
```
Without this, `config.VLM_MODE` stays `stub` and the app runs locally with
placeholder brain replies (nothing is broken — just no cloud reasoning).

---

## What works vs. what we claim (honest scope)

✅ Live object detection with position/distance, instant obstacle & approaching
hazard warnings (only while the user is moving), motion awareness (moving vs.
still), environmental sound awareness, spoken questions, text reading and rich
scene descriptions via the VLM, a phone-as-sensor web demo, and a live dashboard.

❌ Not a certified medical/mobility device, not street-safe for unsupervised real
use, not running fully on-device. It is a **working proof-of-concept of
continuous, proactive multimodal assistance.**

---

## Models used (all pre-trained, no training from scratch)

| Job | Model | Where |
|-----|-------|-------|
| Object detection | YOLOv8m | laptop |
| Sound classification | YAMNet (521 classes) | laptop |
| Speech-to-text | Whisper (base) | laptop |
| Text-to-speech | edge-tts (neural) / browser voice | laptop / phone |
| Vision-language reasoning | **Qwen2.5-VL-7B** | **cloud GPU** |
```
