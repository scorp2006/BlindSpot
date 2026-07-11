# BlindSpot VLM Space — deploy the "brain" (Qwen2.5-VL-7B)

This folder is the **remote GPU half** of BlindSpot. It runs on a Hugging Face
**GPU Space**, not on the laptop. The laptop calls it over the internet.

The Space exposes **two ways in** (same model, same URL):
1. **`POST /api/describe`** — the primary path the laptop uses (robust HTTP/JSON).
2. **A Gradio UI** — for you to test manually in the browser.

---

## 1. Create the Space

1. https://huggingface.co/new-space
2. **Space name:** `blindspot-vlm` · **SDK:** `Gradio` · **Visibility:** `Public`
3. **Hardware:** choose a GPU:
   - **L4 (24 GB)** — ✅ recommended. Runs full precision (~17 GB). Leave
     `LOAD_4BIT` unset. (The "30 GB" is system RAM — plenty.)
   - A10G (24 GB) — also great, full precision.
   - T4 small (16 GB) — works, but you MUST add variable `LOAD_4BIT=1` (4-bit).

## 2. Upload the files

Upload both files from this folder to the Space (Files tab, or git push):
- `app.py`
- `requirements.txt`

## 3. (T4 only) set the 4-bit variable

Only if you picked a 16 GB T4: Space → **Settings → Variables and secrets** →
add variable `LOAD_4BIT` = `1`. **Skip this on L4/A10G.**

## 4. Wait for boot

Watch the **Logs** tab. First boot downloads ~16 GB of weights (5–15 min). Ready
when you see:
```
[space] model ready.
```

## 5. Test it (two quick checks)

**A. Health check** — open in a browser:
```
https://<your-space>.hf.space/api/health
```
Should return `{"status":"ok", ...}`.

**B. Manual describe** — the Gradio UI at `https://<your-space>.hf.space/`:
upload a photo of a sign, type `what does this say?`, click **Describe**. A
sensible sentence back = the brain works. ✅

---

## 6. Connect the laptop (your friend changes NO code)

Give your friend **one value**: the Space URL, e.g.
`https://your-username-blindspot-vlm.hf.space`

He sets two environment variables and runs:
```powershell
$env:BLINDSPOT_VLM = "http"
$env:BLINDSPOT_VLM_URL_HTTP = "https://your-username-blindspot-vlm.hf.space"
python -m blindspot.vlm      # should now print a REAL answer, not "(stub)"
```
(Public Space → **no token needed**. If you ever make it Private, also set
`$env:BLINDSPOT_VLM_KEY = "hf_xxx"`.)

Then the whole system uses the brain automatically:
```powershell
python run.py full
```

---

## The HTTP contract (for debugging)

```
POST https://<your-space>.hf.space/api/describe
Content-Type: application/json
{
  "image":    "<base64 JPEG or PNG>",      # required
  "facts":    "person(close, ahead, 0.9)", # optional YOLO hints
  "sounds":   "engine(0.4)",               # optional audio hints
  "question": "what does this sign say?"    # "" = proactive mode
}
-> { "answer": "The sign says Exit, to your right." }
```
Test from anywhere:
```bash
curl -X POST https://<your-space>.hf.space/api/health
```

---

## Cost control (hackathon budget)

- The VLM is only called **when triggered** (a question, an approaching hazard, or
  a throttled proactive tick) and never more often than
  `config.VLM_MIN_INTERVAL_SECONDS`. Even in the demo you fire it a handful of
  times, not continuously.
- **Pause the Space** (Settings → Pause) the moment you stop testing — GPU Spaces
  bill per hour while running.
- Develop everything else in `stub` mode (free); flip to `http` only to verify
  and to demo.

---

## Alternative: OpenAI-compatible server (RunPod / vLLM)

If you rent a raw GPU instead:
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000
```
```powershell
$env:BLINDSPOT_VLM = "openai"
$env:BLINDSPOT_VLM_URL = "http://YOUR_GPU_IP:8000/v1/chat/completions"
```
No code changes — the client sends the frame as an OpenAI vision data URL.
