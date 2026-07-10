# BlindSpot — Laptop Setup Checklist (do this top to bottom)

**For:** the friend running it on the **RTX 3050 (6 GB) Windows laptop**.
**Goal:** get from "just cloned the repo" to "it narrates the scene out loud" with
zero guesswork.

> Follow the steps **in order**. After each step there's a ✅ **check** (what you
> should see) and a 🛠️ **if it fails**. Don't move to the next step until the
> check passes. Total time: ~20–30 min (most of it is downloads).

---

## Step 0 — What you need before starting

- [ ] Windows laptop with the **RTX 3050**
- [ ] A working **webcam** (built-in is fine)
- [ ] **Internet** (for downloads + the neural voice)
- [ ] ~5 GB free disk (models download on first run)

---

## Step 1 — Install Python 3.11

We need **Python 3.11** (or 3.10). **NOT 3.12 or 3.13** — the audio model
(TensorFlow) has no wheels for those and the install will fail later.

1. Download Python 3.11 from https://www.python.org/downloads/release/python-3119/
   (pick **Windows installer 64-bit**).
2. In the installer, **tick "Add python.exe to PATH"**, then Install.

✅ **Check** — open a **new** PowerShell window and run:
```powershell
py -3.11 --version
```
You should see `Python 3.11.x`.

🛠️ **If it fails** (`py` not found): reinstall and make sure you ticked "Add to
PATH", then open a fresh terminal.

---

## Step 2 — Get the code

```powershell
cd D:\
git clone https://github.com/scorp2006/BlindSpot.git
cd BlindSpot
```

✅ **Check**: `ls` shows `run.py`, `config.py`, `blindspot`, `README.md`.

---

## Step 3 — Create + activate a virtual environment

Keeps all the AI libraries isolated so they can't break your system Python.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

✅ **Check**: your prompt now starts with `(.venv)`.

🛠️ **If activation is blocked** ("running scripts is disabled"): run this once,
then retry the activate line:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

> ⚠️ From now on, **every** command assumes you see `(.venv)` in the prompt. If
> you open a new terminal, re-run `.\.venv\Scripts\Activate.ps1` first.

---

## Step 4 — Install CUDA PyTorch FIRST (so YOLO uses the GPU)

Do this **before** anything else, or you'll get the slow CPU version of PyTorch.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

✅ **Check**:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
You want: `CUDA: True NVIDIA GeForce RTX 3050`.

🛠️ **If it says `CUDA: False`**: update your NVIDIA driver from
https://www.nvidia.com/Download/index.aspx (or GeForce Experience), reboot, and
re-run the check. *(It still works on CPU — just slower — so you can continue and
fix this later if you're in a hurry.)*

---

## Step 5 — Install the core libraries (Tiers 1 & 2)

```powershell
pip install -r requirements-core.txt
```
This pulls YOLO, OpenCV, the neural voice, and the VLM client. Takes a few minutes.

✅ **Check**: it finishes with no red `ERROR` lines.

🛠️ **If a build error mentions a compiler / wheels**: make sure you're on Python
3.11 (Step 1) inside the venv (Step 3). 99% of install pain is wrong Python.

---

## Step 6 — Smoke-test the pieces ONE AT A TIME 🧩

This is the important part — **test each module alone** so you know exactly what
works before running everything together.

### 6a. Camera
```powershell
python -m blindspot.camera
```
✅ A window opens showing your webcam. Press **q** to close.
🛠️ Black window / error? Another app may be using the camera. Or edit
`config.py` → `CAMERA_SOURCE = 1` and retry.

### 6b. Voice (downloads nothing; needs internet for the neural voice)
```powershell
python -m blindspot.speech
```
✅ You **hear** three spoken test lines.
🛠️ No sound? Check volume/output device. If the neural voice fails, it auto-falls
back to an offline robotic voice — you'll still hear *something*.

### 6c. Vision (first run downloads the YOLOv8n weights ~6 MB)
```powershell
python -m blindspot.vision
```
✅ Webcam window with **green boxes** around objects; the scene is printed in the
terminal (e.g. `A person close ahead.`). Press **q** to quit.

### 6d. 🎉 TIER 1 — the live narrator (this is a real demo)
```powershell
python run.py narrator
```
✅ It **watches your webcam and describes the scene out loud** as things change.
Press **q** (or Ctrl+C) to stop. **If this works, you have a working demo.**

### 6e. VLM client — free "stub" mode (no GPU, no cost)
```powershell
python -m blindspot.vlm
```
✅ Prints a `(stub)` answer. This proves the brain-client wiring works **without
spending any GPU money**. (Turning on the real brain is a later step — see
`vlm_space/README.md`. Leave it in stub mode for now.)

---

## Step 7 — (LATER) Audio + voice questions (Tier 3)

Only do this once Tier 1 works and you're ready for the multimodal part.

```powershell
pip install -r requirements-audio.txt
winget install Gyan.FFmpeg
```
Then **close and reopen** PowerShell (so ffmpeg is on PATH), re-activate the venv,
and test:
```powershell
python -m blindspot.audio     # prints sounds it hears (talk, play a horn on YouTube)
python -m blindspot.listen    # hold SPACE, speak a question, release -> see the text
```
🛠️ If TensorFlow won't install: you're almost certainly not on Python 3.11.
This is exactly why we pinned it. Everything in Tiers 1–2 still works without it.

---

## Step 8 — (OPTIONAL) Phone as the camera

Walk-around demo. The **phone is just the camera**; the laptop still runs all AI.

1. Install the **"IP Webcam"** app on the phone (free, Android).
2. Open it → *Start server*. It shows a URL like `http://192.168.1.42:8080`.
3. Phone and laptop must be on the **same Wi-Fi**.
4. On the laptop, edit `config.py`:
   ```python
   CAMERA_SOURCE = "http://192.168.1.42:8080/video"   # use YOUR phone's URL
   ```
5. Run `python run.py narrator` — it now uses the phone's camera.

---

## Quick reference — everyday commands

```powershell
.\.venv\Scripts\Activate.ps1     # activate the env (do this every new terminal)
python run.py narrator           # Tier 1: live narrator (safe, always works)
python run.py full --no-audio    # everything except sound
python run.py full               # everything (needs Step 7 done)
```

## The 3 things that cause 95% of problems
1. **Wrong Python** — must be 3.11, inside the venv (prompt shows `(.venv)`).
2. **Forgot to activate the venv** in a new terminal.
3. **CUDA torch not installed first** (Step 4 before Step 5).

If stuck on any step, copy the exact error text back to the team chat — the step
number + error is enough to fix it fast.
