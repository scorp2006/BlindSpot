---
title: BlindSpot VLM
emoji: 👁️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
python_version: "3.10"
pinned: false
---

# BlindSpot VLM (Qwen2.5-VL-7B)

The remote "brain" for BlindSpot — a real-time multimodal assistant for the
visually impaired. The laptop pipeline calls this Space's `POST /api/describe`
endpoint with a camera frame + sensor hints and gets back one spoken-ready
sentence.

- **HTTP:** `POST /api/describe` (primary path used by the laptop)
- **Health:** `GET /api/health`
- **UI:** the Gradio page at `/` for manual testing

Hardware: **L4 (24 GB)** recommended — runs full precision. See the project's
`vlm_space/README.md` for the full deploy + connect playbook.
