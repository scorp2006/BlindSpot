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

# ---------------------------------------------------------------------------
# WORKAROUND for a known Gradio/gradio_client bug:
#   TypeError: argument of type 'bool' is not iterable
# It happens in gradio_client.utils when building the /api schema: a JSON-schema
# node can legally be a bool (e.g. additionalProperties: true), but the parser
# does `if "const" in schema` assuming schema is always a dict. We patch the two
# offending functions to short-circuit when they receive a bool. This makes the
# API schema build cleanly, so gradio_client can connect. Must run BEFORE gradio
# is used.
# ---------------------------------------------------------------------------
import gradio_client.utils as _gcu

# Capture the ORIGINALS first (so our wrappers call the real ones, not themselves).
_orig_get_type = _gcu.get_type
_orig_json_schema = _gcu._json_schema_to_python_type


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "bool"
    return _orig_get_type(schema)


def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema(schema, defs)


_gcu.get_type = _safe_get_type
_gcu._json_schema_to_python_type = _safe_json_schema_to_python_type

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
def build_prompt(facts: str, sounds: str, question: str,
                 instruction: str = "", recent: str = "") -> str:
    """The VLM is the mind: it sees the image + hints, remembers what it recently
    said (no repeats), follows the user's standing instruction, and decides for
    itself whether to speak or stay silent."""
    recent_items = [r.strip() for r in (recent or "").split("||") if r.strip()]
    lines = [
        "You are BlindSpot, a warm, perceptive companion for a blind user. "
        "You are their eyes: you see the camera image and help them feel oriented "
        "and safe, like a trusted friend beside them.",
        "Sensor hints (may be imperfect, trust your own eyes more):",
        f"  - Objects (vision): {facts or 'none'}.",
        f"  - Sounds (audio): {sounds or 'none'}.",
        "Describe only what you genuinely see; never invent details.",
    ]
    if instruction:
        lines.append(f'The user has asked you to behave like this: "{instruction}" '
                     "Honor that in how and how much you speak.")
    if recent_items:
        lines.append("You have RECENTLY said the following - do NOT repeat these "
                     "ideas; only speak if you have something genuinely new, "
                     "changed, or important to add:")
        for r in recent_items[-5:]:
            lines.append(f'   • "{r}"')
    if question:
        lines.append(f'Right now the user asked: "{question}"')
        lines.append("Answer directly and helpfully using what you see. "
                     "If they ask about text, read it exactly, word for word.")
    else:
        lines.append("No question right now. You are in companion mode. If there is "
                     "something new, changed, interesting, or a gentle safety note "
                     "worth sharing, say it warmly. If the scene is essentially the "
                     "same as what you already said and there is nothing new worth "
                     "mentioning, reply with the single word NOTHING and stay quiet. "
                     "It is good and kind to stay quiet when there's nothing new.")
    lines.append("Reply with ONE short, natural spoken sentence (under 25 words), "
                 "or exactly NOTHING. No preamble, no lists.")
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
def _run(image, facts: str, sounds: str, question: str,
         instruction: str = "", recent: str = "") -> str:
    if image is None:
        return "NOTHING"
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    prompt = build_prompt(facts or "", sounds or "", question or "",
                          instruction or "", recent or "")
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
def describe_ui(image, facts, sounds, question, instruction="", recent=""):
    return _run(image, facts, sounds, question, instruction, recent)


with gr.Blocks(title="BlindSpot VLM") as demo:
    gr.Markdown(
        "# BlindSpot VLM (Qwen2.5-VL-7B)\n"
        "The remote 'brain'. The laptop calls the **/describe** API. "
        "You can also test manually below."
    )
    with gr.Row():
        img_in = gr.Image(type="pil", label="Camera frame")
        with gr.Column():
            facts_in = gr.Textbox(label="Vision facts (YOLO)", value="")
            sounds_in = gr.Textbox(label="Audio facts (YAMNet)", value="")
            q_in = gr.Textbox(label="User question ('' = proactive)", value="")
            instr_in = gr.Textbox(label="Standing instruction", value="")
            recent_in = gr.Textbox(label="Recently said (|| separated)", value="")
            out = gr.Textbox(label="Answer")
            btn = gr.Button("Describe", variant="primary")
    # The laptop's gradio_client calls this exact endpoint (api_name="describe")
    # with 6 inputs: image, facts, sounds, question, instruction, recent.
    btn.click(describe_ui,
              [img_in, facts_in, sounds_in, q_in, instr_in, recent_in], out,
              api_name="describe")


# --------------------------------------------------------------------------
# Launch - HF-native. On Hugging Face Spaces the platform launches the app with
# demo.launch() and owns the port. We do NOT run our own uvicorn/FastAPI server
# (that caused the second bind on 7861 -> "address already in use").
#
# The laptop reaches this via VLM_MODE="gradio" (gradio_client), api_name="/describe".
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # On HF Spaces, call launch() with NO host/port args - the platform sets them.
    # Passing server_name/server_port made Gradio think localhost was unreachable
    # and demand share=True -> ValueError. Bare launch() works on Spaces.
    demo.queue().launch()
