# BlindSpot 👁️‍🗨️

**Real-time multimodal assistive system for the visually impaired.**
Team **Odyssey** · Hack-The-Matrix · Track 04 (Multimodal AI)

BlindSpot watches (camera) and listens (mic) *continuously*, and speaks only when
something genuinely matters — so it doesn't overwhelm the user. Unlike "tap and
ask" tools, it is **proactive**. Audio changes how the visual scene is
interpreted (a car you *see* + an engine you *hear* + it's *close* = a warning; a
car in silence = probably parked, stay quiet). That cross-modal interaction is
the multimodal core.

> **Who runs what:** the code is authored on one machine and run on the **RTX 3050
> laptop**. The big VLM "brain" does **not** run on the laptop — it lives on a
> rented Hugging Face GPU Space. The laptop runs the fast local models and calls
> the Space only when triggered.

---

## Architecture at a glance

```
 Camera + Mic (always on)
        │
 FAST LOCAL MODELS (laptop, every frame)
   • YOLOv8n   → objects + positions        (vision.py)
   • YAMNet    → 521 sound classes          (audio.py)      [optional/Tier 3]
   • Whisper   → your spoken questions       (listen.py)     [optional/Tier 3]
        │
 FUSION / TRIGGER-SEVERITY LAYER            (fusion.py)
   • hazard? (object + matching sound + close) → fire brain + warn
   • question asked?                            → fire brain
   • nothing important?                         → STAY SILENT
        │
 VLM BRAIN (rented GPU, only when triggered) (vlm.py → vlm_space/)
   • Qwen2.5-VL-7B: raw frame + vision facts + sound facts → one sentence
        │
 TTS → VOICE (edge-tts neural)              (speech.py)
```

---

## Project layout

```
BlindSpot/
├── run.py                     # launcher: python run.py <tier>
├── config.py                  # ALL settings live here (edit this, not the logic)
├── requirements-core.txt      # Tiers 1 & 2  (install first)
├── requirements-audio.txt     # Tier 3       (install second)
├── blindspot/
│   ├── camera.py              # camera / phone / video source
│   ├── vision.py              # YOLOv8n detection + spatial phrasing
│   ├── speech.py              # edge-tts neural voice (+ offline fallback)
│   ├── narrator.py            # Tier 1 end-to-end demo
│   ├── vlm.py                 # VLM client (stub / gradio / openai)
│   ├── audio.py               # YAMNet sound classification  [Tier 3]
│   ├── listen.py              # Whisper push-to-talk          [Tier 3]
│   ├── fusion.py              # the multimodal decision engine[Tier 4]
│   └── pipeline.py            # Tier 4 everything-fused
└── vlm_space/                 # deploy this to a Hugging Face GPU Space
    ├── app.py                 # Qwen2.5-VL-7B server
    ├── requirements.txt
    └── README.md
```

---

## Setup (RTX 3050 laptop, Windows)

### Step 1 — Python 3.11 + a virtual environment
Use **Python 3.11** (or 3.10). **Not 3.13** — TensorFlow (YAMNet) has no wheels
for it and Tier 3 will fail to install.

```powershell
# check you have 3.11
py -3.11 --version

# from the repo root:
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Step 2 — Install CUDA PyTorch FIRST (so YOLO uses the GPU)
Do this **before** the requirements files, otherwise you'll get CPU-only torch.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# verify:
python -c "import torch; print('CUDA ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
You should see `CUDA ok: True NVIDIA GeForce RTX 3050`. If it says `False`, update
your NVIDIA driver and retry. (It still runs on CPU, just slower.)

### Step 3 — Core requirements (Tiers 1 & 2)
```powershell
pip install -r requirements-core.txt
```

### Step 4 — Audio + speech-to-text (Tier 3, optional)
Only when you're ready for the multimodal tier.
```powershell
pip install -r requirements-audio.txt
# Whisper needs ffmpeg on PATH:
winget install Gyan.FFmpeg
# (restart the terminal so ffmpeg is on PATH)
```

---

## Test it — MODULAR, one piece at a time 🧩

Run these **in order**. Each one works on its own. Don't move on until the
current one works.

| # | Command | What it proves | Needs |
|---|---------|----------------|-------|
| 1 | `python -m blindspot.camera` | camera opens, live window | core |
| 2 | `python -m blindspot.speech` | you hear the neural voice | core + internet |
| 3 | `python -m blindspot.vision` | boxes drawn, scene printed | core |
| 4 | `python run.py narrator` | **Tier 1 demo**: it narrates aloud | core |
| 5 | `python -m blindspot.vlm` | VLM client works (stub = free) | core |
| 6 | `python -m blindspot.audio` | prints sounds it hears | audio |
| 7 | `python -m blindspot.listen` | hold SPACE, speak, see text | audio |
| 8 | `python -m blindspot.fusion` | decision logic (no hardware) | none |
| 9 | `python run.py full` | **everything fused** | all |

Handy fallbacks for `full` if a module isn't installed yet:
```powershell
python run.py full --no-audio          # skip YAMNet
python run.py full --no-audio --no-stt # vision + voice only
python run.py full --no-window         # headless
```

---

## Turning the VLM brain on (costs GPU money — do this last)

By default `config.VLM_MODE = "stub"` → the pipeline runs **free** and returns
canned brain answers, so you can build & test everything without a GPU.

When you want the real brain:
1. Deploy `vlm_space/` to a Hugging Face GPU Space (see `vlm_space/README.md`).
2. On the laptop, point the client at it:
   ```powershell
   $env:BLINDSPOT_VLM = "gradio"
   $env:BLINDSPOT_VLM_SPACE = "your-username/your-space-name"
   ```
3. Re-run `python -m blindspot.vlm` — you should get a *real* description.
4. **Pause the Space** when done. It bills per hour while running.

---

## Common knobs (in `config.py`)

- `CAMERA_SOURCE` — `0` webcam, or `"http://<phone-ip>:8080/video"` for the phone
  IP-Webcam app, or a `.mp4` path for repeatable tests.
- `EDGE_VOICE` — try `en-US-AriaNeural`, `en-GB-SoniaNeural`, `en-US-GuyNeural`.
- `SPEECH_COOLDOWN_SECONDS` — how long before it repeats a line.
- `HAZARD_CLOSE_AREA` — how big a vehicle must look before it counts as "close".
- `VLM_MIN_INTERVAL_SECONDS` — minimum gap between paid VLM calls.

You can override most via environment variables too (see `config.py`).

---

## Honest scope

✅ Live narration, reading text / answering questions, reacting to sounds, in a
controlled room. A working **proof-of-concept of continuous multimodal
assistance**.

❌ Not street-safe for real blind users, not flawless, not running on-device or
on glasses. We don't claim it is.
