# BlindSpot — Phone demo + Live dashboard

Turn the demo into a **phone experience**: the phone is the camera, mic, and
speaker; your laptop runs the AI (YOLO + flag/tracker + calls the HF Space VLM);
and a **live dashboard** on the laptop shows the jury exactly what the system
sees, hears, and decides.

```
📱 phone /phone  ──frames──►  💻 laptop server (YOLO+flag+VLM)  ──►  ☁️ HF Space VLM
      ▲  speaks   ◄──speech──          │ live state
      │                                ▼
      └────────────────────  💻 laptop /dashboard (jury view)
```

Wake word runs **in the phone's browser** ("Hey BlindSpot"), with a tap-to-talk
button as a reliable fallback. The phone speaks replies with its own voice.

---

## One-time setup (laptop)

1. Activate your venv, then install the web extras:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r webapp\requirements-web.txt
   ```
2. Point at your live VLM Space (so questions get real answers):
   ```powershell
   $env:BLINDSPOT_VLM = "gradio"
   $env:BLINDSPOT_VLM_SPACE = "scorp2111/blindspot-vlm"
   ```
   (Or leave it as `stub` to test the flow for free, no VLM answers.)
3. Install **ngrok** once: https://ngrok.com/download (needed because phone
   cameras/mics require HTTPS, which ngrok provides).

---

## Run it (every time)

**Terminal 1 — start the server:**
```powershell
python -m webapp.server
```
Wait for `[server] models ready.` (first run downloads the YOLO weights).

**Terminal 2 — expose it over HTTPS:**
```powershell
ngrok http 8000
```
Copy the `https://….ngrok-free.app` URL it prints.

**On the phone:** open `https://….ngrok-free.app/phone`
- Allow camera + microphone.
- Tap **Start**.
- Say **“Hey BlindSpot, what’s in front of me?”** — or tap the mic button.

**On the laptop (projector/screen for the jury):** open
`https://….ngrok-free.app/dashboard` (or `http://localhost:8000/dashboard`).

---

## What the jury sees on the dashboard
- The **pipeline strip** lighting up (Camera → YOLO → Decide → VLM → Voice)
- **Current decision** tier (SILENT / NARRATE / DANGER / QUESTION / …)
- **Objects detected** (label, distance, side, confidence) — live
- **Sounds heard** (if audio is enabled on the laptop)
- **VLM answers** log + the **spoken output** history

---

## Tips / troubleshooting
- **Camera won't open on phone** → you must use the **https** ngrok URL, not http.
- **Wake word flaky in a loud hall** → use the **tap-to-talk** button; it's rock
  solid. (Both feed the same pipeline.)
- **Laggy** → the phone streams ~3 fps by design; that's plenty. If ngrok feels
  slow, try same-WiFi (`http://<laptop-ip>:8000`) — but camera needs https, so
  ngrok is usually the safe choice.
- **Cost** → the VLM only fires on questions / changes / hazards, and is
  rate-limited. Pause the HF Space when you're done.
- **Free / offline flow** → set `BLINDSPOT_VLM=stub` to demo the pipeline with no
  VLM calls (narration + hazards still work; questions get a canned reply).

---

## Config knobs (in `config.py`, section 7)
- `NARRATION_MIN_INTERVAL_SECONDS` — how chatty the steady narration is
- `PROACTIVE_INTERVAL_SECONDS` — how often the VLM gives a rich update
- `SAY_ONCE_SECONDS` — how long before it may repeat the same idea
- `OBSTRUCTION_AREA` — how big a box counts as "camera blocked"
