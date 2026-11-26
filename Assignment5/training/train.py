"""Fine-tune Pix2Struct on the course ChartQA train split (table + QA multitask).

Run from Assignment5/:
    python training/train.py --data-dir chartqa_course --out-dir checkpoints/run1
"""

import argparse
import json
import math
import random
import shutil
import sys
import time
from itertools import zip_longest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "my_submission"))

import chartqa_submission as sub  # noqa: E402
from chartqa_metrics import read_jsonl, score_records  # noqa: E402

import matplotlib  # noqa: E402

FONT_SOURCE = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / sub.FONT_FILE


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "chartqa_course")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--base-model", default="google/pix2struct-base")
    p.add_argument("--base-revision", default="4605659ec5d82c985955ebdea8210f32fe54a682")
    p.add_argument("--max-patches", type=int, default=1024)
    p.add_argument("--table-max-len", type=int, default=512)
    p.add_argument("--qa-max-len", type=int, default=32)
    p.add_argument("--table-repeat", type=int, default=1, help="Copies of each table sample per epoch.")
    p.add_argument("--type-task", action="store_true", help="Add a chart-type question per chart and use it at inference.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--limit-train", type=int, default=0, help="Use only the first N charts (overfit/smoke tests).")
    p.add_argument("--val-limit", type=int, default=0, help="Evaluate on a fixed random subset of N val charts.")
    p.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs (and after the last).")
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--log-every", type=int, default=20, help="Optimizer steps between log lines.")
    p.add_argument("--seed", type=int, default=328)
    p.add_argument("--resume", action="store_true", help="Resume from <out-dir>/last.")
    return p.parse_args()


class ChartTasks(Dataset):
    """One sample per (chart, table task) and per (chart, question)."""

    def __init__(self, records, data_dir, max_patches, processor, table_repeat=1, type_task=False):
        self.data_dir, self.max_patches, self.processor = data_dir, max_patches, processor
        self.items = []
        for record in records:
            target = sub.serialize_table(record["chart_type"], record["table"])
            self.items.extend([(record["image"], sub.TABLE_PROMPT, target, "table")] * table_repeat)
            self.items.extend((record["image"], qa["question"], qa["answer"], "qa") for qa in record["qa"])
            if type_task:
                self.items.append((record["image"], sub.TYPE_PROMPT, record["chart_type"], "qa"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path, prompt, target, task = self.items[index]
        with Image.open(self.data_dir / image_path) as image:
            image = image.convert("RGB")
        font = sub.load_font(str(FONT_SOURCE), sub.DEFAULT_CONFIG["font_size"])
        patches, mask = sub.encode_images(self.processor, [sub.render_header(image, prompt, font)],
                                          self.max_patches)
        return patches[0], mask[0], target, task


def make_collate(tokenizer, table_max_len, qa_max_len):
    def collate(batch):
        patches = torch.stack([item[0] for item in batch])
        mask = torch.stack([item[1] for item in batch])
        max_len = table_max_len if any(item[3] == "table" for item in batch) else qa_max_len
        tokens = tokenizer([item[2] for item in batch], padding="longest", truncation=True,
                           max_length=max_len, return_tensors="pt")
        labels = tokens.input_ids.masked_fill(tokens.attention_mask == 0, -100)
        return patches, mask, labels
    return collate


def save_model_dir(model, processor, path, max_patches, type_prompt):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    shutil.copy(FONT_SOURCE, path / sub.FONT_FILE)
    config = {**sub.DEFAULT_CONFIG, "max_patches": max_patches, "type_prompt": type_prompt}
    (path / sub.CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def evaluate(model, processor, records, data_dir, args, device):
    model.eval()
    bundle = sub.make_bundle(model, processor, str(FONT_SOURCE),
                             {"max_patches": args.max_patches, "type_prompt": args.type_task},
                             device, torch.float32)
    predictions = {}
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(records), args.eval_batch_size):
            batch = records[start:start + args.eval_batch_size]
            images = []
            for record in batch:
                with Image.open(data_dir / record["image"]) as image:
                    images.append(image.convert("RGB"))
            outputs = sub.predict(bundle, images, [[qa["question"] for qa in r["qa"]] for r in batch])
            for record, output in zip(batch, outputs):
                predictions[record["chart_id"]] = output
    elapsed = time.perf_counter() - started
    model.train()
    report = score_records(records, predictions, elapsed_seconds=elapsed)
    return report, predictions


def main():
    args = parse_args()
    from transformers import (Pix2StructForConditionalGeneration, Pix2StructProcessor,
                              get_cosine_schedule_with_warmup)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    train_records = read_jsonl(args.data_dir / "train.jsonl")
    if args.limit_train:
        train_records = train_records[:args.limit_train]
    val_records = read_jsonl(args.data_dir / "val.jsonl")
    if args.val_limit:
        # Stratify by chart type so every class is present in the subset.
        shuffled = random.Random(args.seed).sample(val_records, len(val_records))
        by_type = {t: [r for r in shuffled if r["chart_type"] == t] for t in sub.CHART_TYPES}
        val_records = [r for group in zip_longest(*by_type.values()) for r in group if r][:args.val_limit]

    last_dir = args.out_dir / "last"
    source = last_dir if args.resume else args.base_model
    revision = None if args.resume else args.base_revision
    processor = Pix2StructProcessor.from_pretrained(source, revision=revision)
    model = Pix2StructForConditionalGeneration.from_pretrained(source, revision=revision).to(device)
    model.train()

    dataset = ChartTasks(train_records, args.data_dir, args.max_patches, processor, args.table_repeat,
                         args.type_task)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                        collate_fn=make_collate(processor.tokenizer, args.table_max_len, args.qa_max_len))
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    start_epoch, best_mark, step = 0, -1.0, 0
    if args.resume:
        state = torch.load(last_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_mark, step = state["epoch"], state["best_mark"], state["step"]

    log_path = args.out_dir / "log.jsonl"
    print(f"{len(dataset)} samples/epoch, {steps_per_epoch} optimizer steps/epoch, {total_steps} total",
          flush=True)

    def log(entry):
        print(json.dumps(entry), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    for epoch in range(start_epoch, args.epochs):
        running, count, started = 0.0, 0, time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for index, (patches, mask, labels) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(flattened_patches=patches.to(device), attention_mask=mask.to(device),
                             labels=labels.to(device)).loss
            (loss / args.grad_accum).backward()
            running += loss.item()
            count += 1
            if (index + 1) % args.grad_accum == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    log({"epoch": epoch + 1, "step": step, "loss": running / count,
                         "lr": scheduler.get_last_lr()[0],
                         "steps_per_hour": round(3600 * (index + 1) / args.grad_accum
                                                 / (time.perf_counter() - started))})
                    running, count = 0.0, 0

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            report, predictions = evaluate(model, processor, val_records, args.data_dir, args, device)
            mark = sum(value or 0.0 for value in report["component_marks"].values())
            log({"epoch": epoch + 1, "step": step, "val_metrics": report["metrics"],
                 "val_marks": report["component_marks"], "val_mark": mark,
                 "val_seconds": report["elapsed_seconds"], "val_format_errors": report["format_errors"]})
            if mark > best_mark:
                best_mark = mark
                save_model_dir(model, processor, args.out_dir / "best", args.max_patches, args.type_task)
                with (args.out_dir / "best_val_predictions.jsonl").open("w", encoding="utf-8") as handle:
                    for chart_id, prediction in predictions.items():
                        handle.write(json.dumps({"chart_id": chart_id, "prediction": prediction},
                                                ensure_ascii=False) + "\n")
                log({"epoch": epoch + 1, "saved_best": best_mark})

        save_model_dir(model, processor, last_dir, args.max_patches, args.type_task)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1, "best_mark": best_mark, "step": step},
                   last_dir / "trainer_state.pt")


if __name__ == "__main__":
    main()
