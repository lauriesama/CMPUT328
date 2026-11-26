"""ChartQA inference with a fine-tuned Pix2Struct model.

One model handles two tasks, selected by a prompt rendered above the chart:
the table prompt generates "<chart_type> <0x0A> <header> <0x0A> <row> ...", and a
question prompt generates the short answer.
"""

import json
import os
import textwrap
from functools import lru_cache

import torch
from PIL import Image, ImageDraw, ImageFont

CHART_TYPES = ("v_bar", "h_bar", "line", "pie")
TABLE_PROMPT = "Generate the chart type and underlying data table of the figure below:"
TYPE_PROMPT = "What type of chart is this?"
ROW_SEP = "<0x0A>"
COL_SEP = "|"
MAX_PREDICTED_CELLS = 128
FONT_FILE = "DejaVuSans.ttf"
CONFIG_FILE = "inference_config.json"
DEFAULT_CONFIG = {
    "max_patches": 1024,
    "table_max_new_tokens": 512,
    "qa_max_new_tokens": 32,
    "gen_batch_size": 16,
    "font_size": 36,
    "dtype": "auto",
    "type_prompt": False,
}


# ---------------------------------------------------------------- shared with training

@lru_cache(maxsize=4)
def load_font(path, size):
    return ImageFont.truetype(path, size)


def render_header(image, text, font, wrap_width=80, padding=5):
    """Draw `text` above `image`, matching Pix2Struct's VQA header rendering."""
    wrapped = "\n".join(textwrap.wrap(text, width=wrap_width)) or " "
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    _, _, right, bottom = probe.multiline_textbbox((0, 0), wrapped, font=font)
    header = Image.new("RGB", (right + 2 * padding, bottom + 2 * padding), "white")
    ImageDraw.Draw(header).multiline_text((padding, padding), wrapped, font=font, fill="black")

    width = max(header.width, image.width)
    image_height = max(1, round(image.height * width / image.width))
    header_height = max(1, round(header.height * width / header.width))
    canvas = Image.new("RGB", (width, header_height + image_height), "white")
    canvas.paste(header.resize((width, header_height)), (0, 0))
    canvas.paste(image.resize((width, image_height)), (0, header_height))
    return canvas


def _clean_cell(text):
    return " ".join(str(text).replace(COL_SEP, "/").replace(ROW_SEP, " ").split())


def serialize_table(chart_type, table):
    lines = [chart_type, f" {COL_SEP} ".join(map(_clean_cell, table["columns"]))]
    lines += [f" {COL_SEP} ".join(map(_clean_cell, row)) for row in table["rows"]]
    return f" {ROW_SEP} ".join(lines)


def parse_table(text):
    """Return (chart_type, table) from generated text; always structurally valid."""
    lines = [line.strip() for line in text.split(ROW_SEP)]
    lines = [line for line in lines if line]
    chart_type = CHART_TYPES[0]
    if lines and COL_SEP not in lines[0]:
        first = lines.pop(0)
        if first in CHART_TYPES:
            chart_type = first
    if not lines:
        return chart_type, {"columns": [], "rows": []}

    columns = [cell.strip() for cell in lines[0].split(COL_SEP)][:MAX_PREDICTED_CELLS + 1]
    if len(columns) < 2:
        return chart_type, {"columns": [], "rows": []}
    width = len(columns)
    max_rows = MAX_PREDICTED_CELLS // (width - 1)
    rows = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split(COL_SEP)]
        if len(cells) < 2:
            continue
        rows.append((cells + [""] * width)[:width])
        if len(rows) >= max_rows:
            break
    return chart_type, {"columns": columns, "rows": rows}


def encode_images(processor, images, max_patches):
    inputs = processor.image_processor(images=images, max_patches=max_patches, return_tensors="pt")
    return inputs["flattened_patches"], inputs["attention_mask"]


# ---------------------------------------------------------------- inference

def _pick_dtype(name, device):
    if name != "auto":
        return getattr(torch, name)
    if str(device).startswith("cuda") and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    # Pix2Struct (T5-style) overflows in fp16, so pre-Ampere GPUs stay in fp32.
    return torch.float32


def make_bundle(model, processor, font_path, config, device, dtype):
    cfg = {**DEFAULT_CONFIG, **config}
    model.generation_config.max_length = None
    return {"model": model, "processor": processor, "config": cfg, "device": device,
            "dtype": dtype, "font": load_font(font_path, cfg["font_size"])}


def load_model(model_dir: str, device: str):
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

    config = {}
    config_path = os.path.join(model_dir, CONFIG_FILE)
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    dtype = _pick_dtype(config.get("dtype", "auto"), device)
    processor = Pix2StructProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Pix2StructForConditionalGeneration.from_pretrained(
        model_dir, dtype=dtype, local_files_only=True).to(device).eval()
    return make_bundle(model, processor, os.path.join(model_dir, FONT_FILE), config, device, dtype)


def _generate(bundle, images, max_new_tokens):
    cfg, processor, model = bundle["config"], bundle["processor"], bundle["model"]
    texts = []
    for start in range(0, len(images), cfg["gen_batch_size"]):
        patches, mask = encode_images(processor, images[start:start + cfg["gen_batch_size"]],
                                      cfg["max_patches"])
        output = model.generate(
            flattened_patches=patches.to(bundle["device"], bundle["dtype"]),
            attention_mask=mask.to(bundle["device"]),
            max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, use_cache=True)
        texts.extend(processor.tokenizer.batch_decode(output, skip_special_tokens=True))
    return [text.strip() for text in texts]


def predict(model, images, questions):
    bundle, cfg = model, model["config"]
    font = bundle["font"]
    images = [image.convert("RGB") for image in images]

    table_texts = _generate(bundle, [render_header(image, TABLE_PROMPT, font) for image in images],
                            cfg["table_max_new_tokens"])

    qa_images, counts = [], []
    for image, image_questions in zip(images, questions):
        prompts = list(image_questions) + ([TYPE_PROMPT] if cfg["type_prompt"] else [])
        qa_images.extend(render_header(image, prompt, font) for prompt in prompts)
        counts.append(len(prompts))
    answers = _generate(bundle, qa_images, cfg["qa_max_new_tokens"]) if qa_images else []

    results, offset = [], 0
    for text, count in zip(table_texts, counts):
        chart_type, table = parse_table(text)
        chart_answers = answers[offset:offset + count]
        if cfg["type_prompt"]:
            type_answer = chart_answers.pop()
            if type_answer in CHART_TYPES:
                chart_type = type_answer
        results.append({"chart_type": chart_type, "table": table, "answers": chart_answers})
        offset += count
    return results
