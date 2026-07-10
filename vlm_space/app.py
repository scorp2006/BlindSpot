"""
BlindSpot VLM Space - Qwen2.5-VL-7B on a Hugging Face GPU Space.

This is the "brain". It runs on a RENTED GPU (an HF Space with a T4/A10G/A100),
NOT on the laptop. The laptop's blindspot/vlm.py (gradio mode) calls this.

The exposed endpoint is named "/describe" and takes exactly 4 inputs in this
order, matching blindspot/vlm.py:
    image     : the camera frame (PIL image via gr.Image)
    facts     : YOLO vision facts (str)
    sounds    : audio facts (str)
    question  : user question, "" for proactive mode (str)
and returns ONE string (the spoken-ready sentence, or "NOTHING").

DEPLOY (see vlm_space/README.md for the click-by-click version):
  1. Create a new Space -> SDK: Gradio -> Hardware: a GPU tier
     (T4 small works; A10G is comfier for 7B).
  2. Upload this app.py and requirements.txt.
  3. Wait for it to build & the model to download (first boot is slow).
  4. Copy the Space id ("username/space-name") into the laptop's config or set
     BLINDSPOT_VLM=gradio and BLINDSPOT_VLM_SPACE=username/space-name.

Memory note: Qwen2.5-VL-7B in bf16 needs ~16-18 GB VRAM. On a 16 GB T4 use the
4-bit path (set LOAD_4BIT=1 as a Space variable) which fits in ~8-10 GB.
"""

import os
import gradio as gr
import torch
from PIL import Image

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
LOAD_4BIT = os.getenv("LOAD_4BIT", "0") == "1"

# --------------------------------------------------------------------------
# Load model once at startup.
# --------------------------------------------------------------------------
print(f"[space] loading {MODEL_ID} (4bit={LOAD_4BIT}) ...")

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

_load_kwargs = dict(device_map="auto", trust_remote_code=True)
if LOAD_4BIT:
    from transformers import BitsAndBytesConfig
    _load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
else:
    _load_kwargs["torch_dtype"] = torch.bfloat16

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **_load_kwargs)
model.eval()
print("[space] model ready.")


def build_prompt(facts: str, sounds: str, question: str) -> str:
    """Mirror of blindspot/vlm.build_prompt so behaviour matches locally."""
    lines = [
        "You are BlindSpot, a calm assistant for a blind user.",
        "You are given a camera image plus sensor hints.",
        f"Objects detected (vision): {facts or 'none'}.",
        f"Sounds detected (audio): {sounds or 'none'}.",
    ]
    if question:
        lines.append(f'The user asked: "{question}"')
        lines.append("Answer that question directly using what you see. "
                     "Read any relevant text aloud if asked.")
    else:
        lines.append("No question was asked. Only speak if something genuinely "
                     "matters for safety or navigation; otherwise say exactly "
                     "'NOTHING'.")
    lines.append("Reply with ONE short spoken sentence. No preamble, no lists.")
    return "\n".join(lines)


@torch.inference_mode()
def describe(image, facts, sounds, question):
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
    # Strip the prompt tokens, keep only the newly generated answer.
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    answer = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()
    return answer or "NOTHING"


with gr.Blocks(title="BlindSpot VLM") as demo:
    gr.Markdown("# BlindSpot VLM (Qwen2.5-VL-7B)\nThe remote 'brain'. "
                "Called by the laptop client. You can also test it manually here.")
    with gr.Row():
        img_in = gr.Image(type="pil", label="Camera frame")
        with gr.Column():
            facts_in = gr.Textbox(label="Vision facts (YOLO)", value="")
            sounds_in = gr.Textbox(label="Audio facts (YAMNet)", value="")
            q_in = gr.Textbox(label="User question ('' = proactive)", value="")
            out = gr.Textbox(label="Answer")
            btn = gr.Button("Describe", variant="primary")

    # IMPORTANT: api_name="describe" must match config.VLM_GRADIO_API_NAME.
    btn.click(describe, [img_in, facts_in, sounds_in, q_in], out, api_name="describe")

if __name__ == "__main__":
    demo.queue().launch()
