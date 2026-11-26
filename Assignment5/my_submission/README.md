# ChartQA submission

- Name: Laurie Gantuangco
- CCID: gantuang
- Student ID: 1771419

## Starting checkpoint
- `google/pix2struct-base`, revision `4605659ec5d82c985955ebdea8210f32fe54a682`

## Data
- Course package `chartqa-course-v1`, version `bbd63667be55`; trained on `train.jsonl` only, `val.jsonl` used for checkpoint selection.

## Approach
A single Pix2Struct model is fine-tuned on three tasks. The task prompt is rendered as a text header above the chart (DejaVuSans font, shipped in `model/`):
- Table prompt -> `<chart_type> <0x0A> <col> | <col> ... <0x0A> <row> | <value> ...`, parsed into the table.
- "What type of chart is this?" -> one of `v_bar`, `h_bar`, `line`, `pie` (used as the chart type; falls back to the table output's first token).
- Question prompt -> short answer.

Training: 10 epochs over 12,000 table samples (each chart's table task three times) + 6,691 QA samples + 4,000 chart-type samples per epoch; AdamW (lr 5e-5, weight decay 0.01), cosine schedule with 5% warmup, batch 2 x 8 gradient accumulation (effective 16), bf16 autocast, max 1024 patches, greedy decoding. Final (epoch 10) checkpoint used. About 4.7 GPU-hours on one RTX 4070 Ti Super (16 GB), including per-epoch validation. Two earlier runs (no table repetition: 90.2; table x2 without the chart-type task: 97.7) took about 5.3 GPU-hours combined and were not used.

## Validation results
Official `evaluate.py` on `val.jsonl` (400 charts, batch size 2, RTX 4070 Ti Super, bf16):

| Metric | Value | Marks |
|---|---|---|
| Chart type (balanced acc.) | 0.998 | 10.0 / 10 |
| Table (mean cell F1) | 0.420 | 40.0 / 40 |
| QA (balanced relaxed acc.; human 0.350, augmented 0.743) | 0.547 | 50.0 / 50 |
| Total before time | | 100.0 / 100 |

Inference time: 310 s on the RTX 4070 Ti Super; 963 s on a Colab Tesla T4 (fp32, same scores, multiplier 1.0). No format or inference errors.

## Dependencies
`pip install -r requirements.txt` (exact versions). Inference runs offline from `model/`. Pre-Ampere GPUs run in fp32, Ampere+ in bf16.

## References and AI assistance
- Lee et al., Pix2Struct (2023); Masry et al., ChartQA (2022); Hugging Face Transformers documentation.
- Code written with assistance from GitHub Copilot.
