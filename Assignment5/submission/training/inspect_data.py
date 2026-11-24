"""Inspect the course dataset before training.

    python training/inspect_data.py --data-dir chartqa_course

Reports split sizes, class and QA-source counts, table sizes, image sizes, serialized target token
lengths (to choose --max-target-len / table_max_new_tokens), tokenizer round-trip failures, and
parser round-trip failures. Saves a few rendered training inputs to --out-dir for a visual check.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
from transformers import Pix2StructProcessor

from train import ROOT, default_font, read_jsonl  # also puts inference/ on sys.path
import chartqa_submission as sub  # noqa: E402


def percentiles(values, qs=(50, 90, 99, 100)):
    values = sorted(values)
    return {f"p{q}": values[min(len(values) - 1, int(len(values) * q / 100))] for q in qs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=ROOT.parent / "tmp" / "inspect")
    p.add_argument("--evaluator-dir", type=Path, default=ROOT / "evaluator")
    p.add_argument("--base-model", default="google/pix2struct-base")
    p.add_argument("--samples", type=int, default=6)
    args = p.parse_args()
    sys.path.insert(0, str(args.evaluator_dir.resolve()))
    from chartqa_metrics import table_f1

    manifest = args.data_dir / "manifest.json"
    if manifest.is_file():
        print("manifest:", manifest.read_text(encoding="utf-8")[:2000])
    tokenizer = Pix2StructProcessor.from_pretrained(args.base_model).tokenizer

    for split in ("train", "val", "public_test"):
        path = args.data_dir / f"{split}.jsonl"
        if not path.is_file():
            continue
        records = read_jsonl(path)
        types = Counter(r["chart_type"] for r in records)
        sources = Counter(qa["source"] for r in records for qa in r["qa"])
        cells = [len(r["table"]["rows"]) * (len(r["table"]["columns"]) - 1) for r in records]
        n_qa = sum(len(r["qa"]) for r in records)
        print(f"\n== {split}: {len(records)} charts, {n_qa} questions")
        print("  chart types:", dict(types))
        print("  qa sources:", dict(sources))
        print("  cells per table:", percentiles(cells))
        print("  questions per chart:", percentiles([len(r["qa"]) for r in records]))
        if split != "train":
            continue

        targets = [sub.serialize_table_target(r["chart_type"], r["table"]) for r in records]
        answers = [qa["answer"] for r in records for qa in r["qa"]]
        table_lens = [len(ids) for ids in tokenizer(targets)["input_ids"]]
        answer_lens = [len(ids) for ids in tokenizer(answers)["input_ids"]]
        print("  table target tokens:", percentiles(table_lens))
        print("  answer tokens:", percentiles(answer_lens))

        bad_tok, bad_parse = [], []
        for record, target in zip(records, targets):
            decoded = tokenizer.decode(tokenizer(target)["input_ids"], skip_special_tokens=True)
            chart_type, table = sub.parse_table_output(decoded)
            f1 = table_f1(table, record["table"])
            if sub._normalize(decoded) != sub._normalize(target):
                bad_tok.append((target, decoded))
            if chart_type != record["chart_type"] or f1 < 1.0:
                bad_parse.append((record["chart_id"], f1, decoded))
        print(f"  tokenizer round-trip mismatches: {len(bad_tok)}/{len(records)}")
        for target, decoded in bad_tok[:5]:
            print(f"    target : {target[:200]}\n    decoded: {decoded[:200]}")
        print(f"  decode+parse not perfect (upper bound on table F1 loss): {len(bad_parse)}/{len(records)}")
        for chart_id, f1, decoded in bad_parse[:5]:
            print(f"    {chart_id} F1={f1:.2f}: {decoded[:200]}")

        sizes = []
        for record in records[:300]:
            with Image.open(args.data_dir / record["image"]) as image:
                sizes.append(image.size)
        print("  image widths:", percentiles([w for w, _ in sizes]),
              "heights:", percentiles([h for _, h in sizes]))

        args.out_dir.mkdir(parents=True, exist_ok=True)
        font = sub.load_font(default_font(), sub.DEFAULT_CONFIG["header_font_size"])
        for record in records[:args.samples]:
            with Image.open(args.data_dir / record["image"]) as image:
                image = image.convert("RGB")
            for tag, header in (("table", sub.TABLE_PROMPT), ("qa", record["qa"][0]["question"])):
                rendered = sub.render_header(image, header, font, sub.DEFAULT_CONFIG["header_wrap_chars"])
                rendered.save(args.out_dir / f"{record['chart_id']}_{tag}.png")
        print(f"  rendered samples saved to {args.out_dir}")


if __name__ == "__main__":
    main()
