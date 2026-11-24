"""CMPUT 328 Assignment 5 submission: chart type, table extraction and QA with one Pix2Struct model.

The task prompt is drawn as a text header above the chart (Pix2Struct VQA style). Two tasks:
  * table: header TABLE_PROMPT -> "<chart_type> <0x0A> col0 | col1 ... <0x0A> row0 | v | v ..."
  * qa:    header = question   -> short answer string

train.py imports the helpers below so training and inference preprocess identically.
"""

import json
import re
import sys
import textwrap
import traceback
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

CHART_TYPES = ("v_bar", "h_bar", "line", "pie")
DEFAULT_CHART_TYPE = "v_bar"
TABLE_PROMPT = "Generate the chart type and underlying data table of the figure below:"
ROW_SEP = " <0x0A> "
CELL_SEP = " | "
MAX_PREDICTED_CELLS = 128
FONT_FILE = "header_font.ttf"
CONFIG_FILE = "inference_config.json"
DEFAULT_CONFIG = {
    "max_patches": 1024,
    "header_font_size": 36,
    "header_wrap_chars": 80,
    "table_max_new_tokens": 384,
    "qa_max_new_tokens": 32,
    "qa_batch_size": 16,
    # "auto": bfloat16 on Ampere or newer, float32 otherwise (T5-style models can overflow in float16).
    "dtype": "auto",
}


# ----------------------------------------------------------------------------- preprocessing

def load_font(path, size):
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        print(f"Font {path} not found; falling back to Pillow's default font.", file=sys.stderr)
        return ImageFont.load_default(size)


def render_header(image, text, font, wrap_chars):
    """Draw `text` on a white band, scale the band to the chart width and stack it on top."""
    lines = textwrap.wrap(text, width=wrap_chars) or [" "]
    spacing = font.size // 4
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left, top, right, bottom = probe.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=spacing)
    pad = font.size // 3
    header = Image.new("RGB", (right - left + 2 * pad, bottom - top + 2 * pad), "white")
    ImageDraw.Draw(header).multiline_text((pad - left, pad - top), "\n".join(lines), font=font,
                                          fill="black", spacing=spacing)
    scale = image.width / header.width
    header = header.resize((image.width, max(1, round(header.height * scale))), Image.BICUBIC)
    canvas = Image.new("RGB", (image.width, header.height + image.height), "white")
    canvas.paste(header, (0, 0))
    canvas.paste(image, (0, header.height))
    return canvas


def encode_images(bundle, images, headers):
    """Render headers and turn images into Pix2Struct flattened patches (CPU tensors)."""
    cfg = bundle["config"]
    rendered = [render_header(image.convert("RGB"), header, bundle["font"], cfg["header_wrap_chars"])
                for image, header in zip(images, headers)]
    encoded = bundle["processor"].image_processor(images=rendered, max_patches=cfg["max_patches"],
                                                  return_tensors="pt")
    return encoded["flattened_patches"], encoded["attention_mask"]


# ----------------------------------------------------------------------------- table text format

def _clean(cell):
    return " ".join(str(cell).replace("|", "/").split())


def serialize_table_target(chart_type, table):
    lines = [chart_type, CELL_SEP.join(_clean(c) for c in table["columns"])]
    lines += [CELL_SEP.join(_clean(c) for c in row) for row in table["rows"]]
    return ROW_SEP.join(lines)


def _normalize(text):
    return " ".join(text.casefold().split())


def parse_table_output(text):
    """Parse generated text into (chart_type, table). Always returns a valid table structure."""
    lines = [line.strip() for line in re.split(r"<0x0A>|\n", text)]
    lines = [line for line in lines if line]

    chart_type = DEFAULT_CHART_TYPE
    if lines:
        first = lines[0].replace(" ", "")
        if first in CHART_TYPES:
            chart_type = first
            lines = lines[1:]
        else:
            # Missing or malformed type line: use any class name that appears near the start.
            for name in CHART_TYPES:
                if name in first:
                    chart_type = name
                    lines = lines[1:]
                    break

    empty = {"columns": [], "rows": []}
    if len(lines) < 2:
        return chart_type, empty
    header = [cell.strip() for cell in lines[0].split("|")]
    if len(header) < 2:
        return chart_type, empty

    # Drop duplicate series labels; they can only add unmatched cells.
    keep, seen = [0], set()
    for index, label in enumerate(header[1:], start=1):
        if _normalize(label) not in seen:
            seen.add(_normalize(label))
            keep.append(index)
    columns = [header[i] for i in keep]
    max_rows = MAX_PREDICTED_CELLS // (len(columns) - 1)

    rows, seen_rows = [], set()
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 2 or _normalize(cells[0]) in seen_rows:
            continue
        cells = (cells + [""] * len(header))[:len(header)]  # pad a truncated last row
        seen_rows.add(_normalize(cells[0]))
        rows.append([cells[i] for i in keep])
        if len(rows) == max_rows:
            break
    if not rows:
        return chart_type, empty
    return chart_type, {"columns": columns, "rows": rows}


# ----------------------------------------------------------------------------- model bundle

def resolve_dtype(name, device):
    if name != "auto":
        return getattr(torch, name)
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float32


def make_bundle(model, processor, font, config, device, autocast_dtype=None):
    processor.image_processor.is_vqa = False  # headers are rendered by render_header instead
    return {"model": model, "processor": processor, "font": font, "config": config,
            "device": torch.device(device), "autocast_dtype": autocast_dtype}


def load_config(model_dir):
    config = dict(DEFAULT_CONFIG)
    path = Path(model_dir) / CONFIG_FILE
    if path.is_file():
        config.update(json.loads(path.read_text(encoding="utf-8")))
    return config


def load_model(model_dir: str, device: str):
    """Load all local assets once. Return your model/bundle."""
    model_dir = Path(model_dir)
    config = load_config(model_dir)
    processor = Pix2StructProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
    model.to(device=device, dtype=resolve_dtype(config["dtype"], device)).eval()
    font = load_font(model_dir / FONT_FILE, config["header_font_size"])
    return make_bundle(model, processor, font, config, device)


# ----------------------------------------------------------------------------- inference

@torch.inference_mode()
def generate(bundle, images, headers, max_new_tokens):
    model, device = bundle["model"], bundle["device"]
    patches, mask = encode_images(bundle, images, headers)
    autocast = bundle["autocast_dtype"]
    with torch.autocast(device.type, dtype=autocast or torch.float32, enabled=autocast is not None):
        ids = model.generate(flattened_patches=patches.to(device, dtype=model.dtype),
                             attention_mask=mask.to(device), max_new_tokens=max_new_tokens,
                             num_beams=1, do_sample=False)
    return [text.strip() for text in bundle["processor"].batch_decode(ids, skip_special_tokens=True)]


def predict(model, images, questions):
    """Return one prediction dictionary per input image."""
    bundle, cfg = model, model["config"]
    images = [image.convert("RGB") for image in images]

    results = []
    try:
        outputs = generate(bundle, images, [TABLE_PROMPT] * len(images), cfg["table_max_new_tokens"])
        parsed = [parse_table_output(text) for text in outputs]
    except Exception:
        traceback.print_exc()
        parsed = [(DEFAULT_CHART_TYPE, {"columns": [], "rows": []})] * len(images)
    for chart_type, table in parsed:
        results.append({"chart_type": chart_type, "table": table, "answers": []})

    pairs = [(i, question) for i, qs in enumerate(questions) for question in qs]
    step = cfg["qa_batch_size"]
    for start in range(0, len(pairs), step):
        chunk = pairs[start:start + step]
        try:
            answers = generate(bundle, [images[i] for i, _ in chunk], [q for _, q in chunk],
                               cfg["qa_max_new_tokens"])
        except Exception:
            traceback.print_exc()
            answers = [""] * len(chunk)
        for (i, _), answer in zip(chunk, answers):
            results[i]["answers"].append(answer)
    return results
