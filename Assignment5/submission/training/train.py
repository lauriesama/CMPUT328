"""Fine-tune google/pix2struct-base on the course ChartQA train split (table + QA multitask).

Example (from submission/):
    python training/train.py --data-dir chartqa_course --out-dir checkpoints/run1

Each chart gives one table example (repeated --table-repeat times per epoch) and one example per
question. Validation runs the real predict() from chartqa_submission.py and the course scorer, and
the best checkpoint by mark_before_time is saved to <out-dir>/best.
"""

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (Pix2StructForConditionalGeneration, Pix2StructProcessor,
                          get_cosine_schedule_with_warmup)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
import chartqa_submission as sub  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="chartqa_course/ directory")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--evaluator-dir", type=Path, default=ROOT / "evaluator",
                   help="Directory containing chartqa_metrics.py")
    p.add_argument("--base-model", default="google/pix2struct-base")
    p.add_argument("--base-revision", default=None, help="Hub revision to pin (record it in the README)")
    p.add_argument("--font", type=Path, default=None,
                   help="TTF for prompt headers (default: matplotlib's DejaVuSans)")
    p.add_argument("--max-patches", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200, help="optimizer steps")
    p.add_argument("--table-repeat", type=int, default=2, help="table examples per chart per epoch")
    p.add_argument("--max-target-len", type=int, default=512)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--train-limit", type=int, default=None, help="use only the first N train charts")
    p.add_argument("--val-limit", type=int, default=None, help="score only the first N val charts")
    p.add_argument("--eval-on-train", action="store_true",
                   help="validate on the (limited) train charts instead of val; for overfit checks")
    p.add_argument("--eval-every", type=float, default=1.0, help="epochs between validations")
    p.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def default_font():
    import matplotlib
    return Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_examples(records, table_repeat):
    examples = []
    for record in records:
        target = sub.serialize_table_target(record["chart_type"], record["table"])
        examples += [(record["image"], sub.TABLE_PROMPT, target)] * table_repeat
        examples += [(record["image"], qa["question"], qa["answer"]) for qa in record["qa"]]
    return examples


class ChartDataset(Dataset):
    def __init__(self, data_dir, examples):
        self.data_dir, self.examples = data_dir, examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        image_path, header, target = self.examples[index]
        with Image.open(self.data_dir / image_path) as image:
            return image.convert("RGB"), header, target


class Collator:
    """Runs in DataLoader workers: header rendering, patch extraction and target tokenization."""

    def __init__(self, processor, font_path, config, max_target_len):
        self.processor, self.font_path, self.config = processor, font_path, config
        self.max_target_len = max_target_len
        self.bundle = None  # built lazily in each worker; fonts and the model are not pickled

    def __call__(self, batch):
        if self.bundle is None:
            font = sub.load_font(self.font_path, self.config["header_font_size"])
            self.bundle = {"processor": self.processor, "font": font, "config": self.config}
        images, headers, targets = zip(*batch)
        patches, mask = sub.encode_images(self.bundle, images, headers)
        tokens = self.processor.tokenizer(list(targets), padding="longest", truncation=True,
                                                    max_length=self.max_target_len, return_tensors="pt")
        labels = tokens["input_ids"].masked_fill(tokens["attention_mask"] == 0, -100)
        return {"flattened_patches": patches, "attention_mask": mask, "labels": labels}


def evaluate(bundle, records, data_dir, score_records):
    bundle["model"].eval()
    predictions, started = {}, time.perf_counter()
    for offset in range(0, len(records), 4):
        batch = records[offset:offset + 4]
        images = []
        for record in batch:
            with Image.open(data_dir / record["image"]) as image:
                images.append(image.convert("RGB"))
        outputs = sub.predict(bundle, images, [[qa["question"] for qa in r["qa"]] for r in batch])
        for record, output in zip(batch, outputs):
            predictions[record["chart_id"]] = output
    elapsed = time.perf_counter() - started
    report = score_records(records, predictions, elapsed_seconds=elapsed)
    bundle["model"].train()
    return report, predictions


def save_checkpoint(bundle, font_path, out_dir, extra):
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle["model"].save_pretrained(out_dir)
    bundle["processor"].save_pretrained(out_dir)
    shutil.copyfile(font_path, out_dir / sub.FONT_FILE)
    (out_dir / sub.CONFIG_FILE).write_text(json.dumps(bundle["config"], indent=2) + "\n", encoding="utf-8")
    (out_dir / "train_info.json").write_text(json.dumps(extra, indent=2, default=str) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.evaluator_dir.resolve()))
    from chartqa_metrics import score_records

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision == "auto":
        use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    else:
        use_bf16 = args.precision == "bf16"
    amp_dtype = torch.bfloat16 if use_bf16 else None
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_records = read_jsonl(args.data_dir / "train.jsonl")[:args.train_limit]
    if args.eval_on_train:
        val_records = train_records[:args.val_limit]
    else:
        val_records = read_jsonl(args.data_dir / "val.jsonl")[:args.val_limit]
    examples = build_examples(train_records, args.table_repeat)
    print(f"train charts={len(train_records)} examples={len(examples)} | eval charts={len(val_records)}")

    processor = Pix2StructProcessor.from_pretrained(args.base_model, revision=args.base_revision)
    model = Pix2StructForConditionalGeneration.from_pretrained(args.base_model, revision=args.base_revision)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device).train()

    font_path = args.font or default_font()
    config = dict(sub.DEFAULT_CONFIG, max_patches=args.max_patches)
    font = sub.load_font(font_path, config["header_font_size"])
    bundle = sub.make_bundle(model, processor, font, config, device, autocast_dtype=amp_dtype)

    loader = DataLoader(ChartDataset(args.data_dir, examples), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        collate_fn=Collator(processor, font_path, config, args.max_target_len),
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
                        drop_last=True)
    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  fused=device.type == "cuda")
    scheduler = get_cosine_schedule_with_warmup(optimizer, min(args.warmup_steps, total_steps // 10),
                                                total_steps)
    eval_interval = max(1, round(args.eval_every * steps_per_epoch))
    print(f"optimizer steps: {steps_per_epoch}/epoch, {total_steps} total, eval every {eval_interval}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    log = (args.out_dir / "log.jsonl").open("a", encoding="utf-8")
    best, step, running, started = -1.0, 0, [], time.perf_counter()

    def run_eval():
        nonlocal best
        report, predictions = evaluate(bundle, val_records, args.data_dir, score_records)
        entry = {"step": step, "epoch": step / steps_per_epoch, "metrics": report["metrics"],
                 "marks": report["component_marks"], "mark": report["mark_before_time"],
                 "qa_by_source": report["qa_accuracy"], "eval_seconds": report["elapsed_seconds"],
                 "format_errors": report["format_errors"]}
        print(json.dumps(entry), flush=True)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        info = dict(entry, base_model=args.base_model, base_revision=args.base_revision, args=vars(args))
        save_checkpoint(bundle, font_path, args.out_dir / "last", info)
        with open(args.out_dir / "last" / "eval_predictions.jsonl", "w", encoding="utf-8") as handle:
            for chart_id, prediction in predictions.items():
                handle.write(json.dumps({"chart_id": chart_id, "prediction": prediction}) + "\n")
        if entry["mark"] > best:
            best = entry["mark"]
            shutil.copytree(args.out_dir / "last", args.out_dir / "best", dirs_exist_ok=True)
            print(f"new best mark {best:.2f} at step {step}", flush=True)

    done = False
    for epoch in range(math.ceil(total_steps / max(1, steps_per_epoch))):
        for micro, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                loss = model(**batch).loss
            (loss / args.grad_accum).backward()
            running.append(loss.item())
            if (micro + 1) % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                rate = step / (time.perf_counter() - started) * 3600
                print(f"epoch {epoch} step {step}/{total_steps} loss {sum(running) / len(running):.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} ({rate:.0f} steps/h)", flush=True)
                running = []
            if step % eval_interval == 0 or step == total_steps:
                run_eval()
            if step >= total_steps:
                done = True
                break
        if done:
            break
    print(f"finished in {(time.perf_counter() - started) / 60:.1f} min, best val mark {best:.2f}")


if __name__ == "__main__":
    main()
