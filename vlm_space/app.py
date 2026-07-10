"""
BlindSpot VLM Space - Qwen2.5-VL-7B on a Hugging Face GPU Space.

This is the "brain". It runs on a RENTED GPU (an HF Space), NOT on the laptop.
The laptop's blindspot/vlm.py calls it over the internet.

TWO WAYS TO CALL IT (both hit the same model):
  1. HTTP POST /api/describe   <-- primary; robust, version-proof, works from
                                   anything (the laptop uses this by default).
  2. Gradio UI / Gradio API    <-- handy for you to test manually in the browser.

HTTP contract (what the laptop sends and gets):
  POST  https://<your-space>.hf.space/api/describe
  body (JSON):
    {
      "image":    "<base64 JPEG/PNG, with or without data: prefix>",
      "facts":    "person(close, ahead, 0.91); ...",   # YOLO hints (optional)
      "sounds":   "engine(0.40); ...",                  # audio hints (optional)
      "question": "what does this sign say?"            # "" = proactive mode
    }
  response (JSON):
    { "answer": "The sign says Exit, to your right." }   # or {"answer":"NOTHING"}

HARDWARE:
  - L4 (24 GB VRAM): run FULL precision (leave LOAD_4BIT unset). Fits ~17 GB. BEST.
  - T4 small (16 GB): set Space variable LOAD_4BIT=1 (fits ~8-10 GB via 4-bit).

DEPLOY: upload this file + requirements.txt to a Gradio GPU Space. First boot
downloads ~16 GB of weights (slow). Ready when logs show "[space] model ready.".
"""

import os
import io
import base64

import gradio as gr
import torch
from PIL import Image

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
LOAD_4BIT = os.getenv("LOAD_4BIT", "0") == "1"

# --------------------------------------------------------------------------
# Load model ONCE at startup.
# Qwen2.5-VL needs a transformers version that ships the Qwen2_5_VL classes.
# requirements.txt pins that; this import will fail loudly with a clear hint if
# the version is wrong.
# --------------------------------------------------------------------------
print(f"[space] loading {MODEL_ID} (4bit={LOAD_4BIT}) ...", flush=True)

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except ImportError as e:
    raise ImportError(
        "Could not import Qwen2_5_VLForConditionalGeneration. Your transformers "
        "version is too old for Qwen2.5-VL. Pin transformers>=4.49 in "
        "requirements.txt (see this repo's vlm_space/requirements.txt)."
    ) from e

_load_kwargs = dict(device_map="auto", trust_remote_code=True)
if LOAD_4BIT:
    from transformers import BitsAndBytesConfig
    _load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
else:
    # bf16 on modern GPUs (L4/A10G/A100). Falls back to fp16 if bf16 unsupported.
    _load_kwargs["torch_dtype"] = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **_load_kwargs)
model.eval()
print("[space] model ready.", flush=True)


# --------------------------------------------------------------------------
# Prompt (mirror of blindspot/vlm.build_prompt so local + remote behave the same)
# --------------------------------------------------------------------------
def build_prompt(facts: str, sounds: str, question: str) -> str:
    lines = [
        "You are BlindSpot, a calm assistant for a blind user.",
        "You are given a camera image plus sensor hints.",
        f"Objects detected (vision): {facts or 'none'}.",
        f"Sounds detected (audio): {sounds or 'none'}.",
    ]
    if question:
        lines.append(f'The user asked: "{question}"')
        lines.append("Answer that question directly using what you see. "
                     "If the user asks about text, read the text in the image aloud.")
    else:
        lines.append("No question was asked. Only speak if something genuinely "
                     "matters for safety or navigation; otherwise reply with the "
                     "single word NOTHING.")
    lines.append("Reply with ONE short spoken sentence. No preamble, no lists.")
    return "\n".join(lines)


def _decode_image(image_b64: str) -> Image.Image | None:
    """Accept base64 with or without a 'data:image/...;base64,' prefix."""
    if not image_b64:
        return None
    if "," in image_b64 and image_b64.strip().lower().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@torch.inference_mode()
def _run(image: Image.Image, facts: str, sounds: str, question: str) -> str:
    if image is None:
        return "NOTHING"
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    prompt = build_prompt(facts or "", sounds or "", question or "")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    answer = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()
    return answer or "NOTHING"


# --------------------------------------------------------------------------
# 1) Gradio UI + Gradio API (manual testing in the browser)
# --------------------------------------------------------------------------
def describe_ui(image, facts, sounds, question):
    return _run(image, facts, sounds, question)


with gr.Blocks(title="BlindSpot VLM") as demo:
    gr.Markdown(
        "# BlindSpot VLM (Qwen2.5-VL-7B)\n"
        "The remote 'brain'. The laptop calls **POST /api/describe** (JSON). "
        "You can also test manually below."
    )
    with gr.Row():
        img_in = gr.Image(type="pil", label="Camera frame")
        with gr.Column():
            facts_in = gr.Textbox(label="Vision facts (YOLO)", value="")
            sounds_in = gr.Textbox(label="Audio facts (YAMNet)", value="")
            q_in = gr.Textbox(label="User question ('' = proactive)", value="")
            out = gr.Textbox(label="Answer")
            btn = gr.Button("Describe", variant="primary")
    btn.click(describe_ui, [img_in, facts_in, sounds_in, q_in], out,
              api_name="describe")


# --------------------------------------------------------------------------
# 2) HTTP POST /api/describe  (primary path the laptop uses)
# Mounted via the FastAPI app that Gradio runs on, so it lives at the SAME URL.
# --------------------------------------------------------------------------
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/api/describe")
async def api_describe(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "body must be JSON"}, status_code=400)

    try:
        image = _decode_image(data.get("image", ""))
    except Exception as e:
        return JSONResponse({"error": f"bad image: {e}"}, status_code=400)

    answer = _run(
        image,
        data.get("facts", ""),
        data.get("sounds", ""),
        data.get("question", ""),
    )
    return {"answer": answer}


# Mount the Gradio UI onto the FastAPI app so BOTH the UI and /api/* share one
# server + one URL. gr.mount_gradio_app returns the combined app.
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    # HF Spaces expose port 7860.
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
