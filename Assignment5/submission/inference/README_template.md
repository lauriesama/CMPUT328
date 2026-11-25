# ChartQA submission

- **Name:** TODO
- **CCID:** TODO
- **Student ID:** TODO

## Starting checkpoint and data

- Starting model: `google/pix2struct-base`, revision `TODO` (commit hash from the Hub)
- Prompt font: DejaVu Sans (`model/header_font.ttf`, from matplotlib's bundled fonts)
- Training data: course package `train` split only, dataset version `TODO` (from `manifest.json`).
  Validation split used only for checkpoint selection.

## Approach

One Pix2Struct-base model, fully fine-tuned on two tasks. The task prompt is drawn as a text header
above the chart image, as in Pix2Struct's VQA setup.

- **Chart type + table:** a fixed header prompt; the target is
  `<chart_type> <0x0A> col0 | col1 | ... <0x0A> row | value | ...`. The first line gives the chart
  type, and the rest is parsed into the table (duplicate labels removed, at most 128 cells).
- **QA:** the question is the header; the target is the reference answer string.

Training: AdamW, lr TODO, cosine schedule with warmup, bf16 autocast, effective batch TODO,
TODO epochs, max patches TODO, table examples repeated TODO× per chart. Greedy decoding at inference.

Approximate GPU use: TODO (e.g. ~N hours on one RTX 4070 Ti Super, 16 GB).

## Validation metrics (released evaluator)

| Split | Chart type | Table F1 | QA | Mark before time | Time (s) |
|---|---|---|---|---|---|
| val | TODO | TODO | TODO | TODO | TODO |
| public_test | TODO | TODO | TODO | TODO | TODO |

## Dependencies

`pip install -r requirements.txt`. All model files load from `model/` with `local_files_only=True`;
no network access is needed. Inference precision is set in `model/inference_config.json`
(`"auto"` = bfloat16 on Ampere or newer GPUs, float32 otherwise).

## References and AI assistance

- Lee et al., *Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding*, 2023.
- Masry et al., *ChartQA*, Findings of ACL 2022.
- Hugging Face Transformers Pix2Struct documentation and fine-tuning notebook.
- Code written with assistance from Claude (Anthropic) via Claude Code: TODO describe extent.
