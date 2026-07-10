# BlindSpot VLM Space — deploy the "brain" (Qwen2.5-VL-7B)

This folder is the **remote GPU half** of BlindSpot. It does **not** run on the
laptop. It runs on a rented Hugging Face **GPU Space**. The laptop talks to it
over the internet via `blindspot/vlm.py` in `gradio` mode.

> 💡 You do **not** need this to build or demo Tiers 1 and 3. Keep the laptop in
> `VLM_MODE=stub` until you're ready to spend GPU money. Deploy this only when
> you want the real point-and-ask / reasoning brain (Tier 2), and ideally just
> for final testing + the demo to keep cost at the ~$10–25 the plan budgets.

---

## Option A — Hugging Face Space (recommended, matches the code)

1. Go to https://huggingface.co/new-space
2. **Space SDK:** `Gradio`. **Space hardware:** pick a **GPU** tier.
   - `T4 small` (16 GB) works **with 4-bit** — set variable `LOAD_4BIT=1`.
   - `A10G` / `A100` runs full precision comfortably — leave `LOAD_4BIT` unset.
3. Upload the two files from this folder: `app.py` and `requirements.txt`.
4. (Optional, private Space) In **Settings → Variables and secrets**, you can
   add a token; the laptop passes `BLINDSPOT_VLM_KEY` as the HF token.
5. First boot is **slow** (it downloads ~16 GB of weights). Watch the logs until
   you see `[space] model ready.`
6. Test it right in the Space UI: upload any photo, click **Describe**.

### Point the laptop at it
On the laptop, set these (either edit `config.py` or use env vars):

```powershell
$env:BLINDSPOT_VLM = "gradio"
$env:BLINDSPOT_VLM_SPACE = "your-username/your-space-name"
# only if the Space is private:
$env:BLINDSPOT_VLM_KEY = "hf_xxx_your_token"
```

Then `python -m blindspot.vlm` on the laptop should return a real description
instead of a `(stub)` one.

---

## Option B — Any OpenAI-compatible endpoint (RunPod / vLLM / LM Studio)

If you'd rather rent a raw GPU and serve with vLLM:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000
```

Then on the laptop:

```powershell
$env:BLINDSPOT_VLM = "openai"
$env:BLINDSPOT_VLM_URL = "http://YOUR_GPU_IP:8000/v1/chat/completions"
$env:BLINDSPOT_VLM_KEY = ""   # or your key if the server requires one
```

The client (`_OpenAIBackend`) sends the frame as a base64 data URL — standard
OpenAI vision format — so no code changes are needed.

---

## Cost control (important for a hackathon budget)

- The VLM is only invoked **when triggered** (a question or a fused hazard), and
  never more often than `config.VLM_MIN_INTERVAL_SECONDS`. So even during the
  demo, you fire it a handful of times, not continuously.
- **Pause / stop the Space** the moment you're done testing. GPU Spaces bill by
  the hour while running.
- Do all pipeline development in `stub` mode (free), flip to `gradio` only to
  verify and to demo.
