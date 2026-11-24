"""Run ChartQA inference and score its outputs. See README.md for examples."""

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

from chartqa_metrics import DEFAULT_GRADING, read_jsonl, score_records


def load_submission(submission):
    path = submission / "chartqa_submission.py"
    spec = importlib.util.spec_from_file_location("chartqa_submission", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prediction_fields(prediction):
    if not isinstance(prediction, dict):
        return None
    clean = {}
    for field in ("chart_type", "table", "answers"):
        value = prediction.get(field)
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError, OverflowError, RecursionError):
            value = None
        clean[field] = value
    return clean


def run_predictions(args):
    import torch
    from PIL import Image

    if args.records.resolve() == args.predictions.resolve():
        raise ValueError("Choose a predictions path distinct from the input records.")
    records = read_jsonl(args.records)
    device = torch.device(args.device)
    errors = Counter()
    elapsed = 0.0
    completed = 0

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as output:
        try:
            module = load_submission(args.submission.resolve())
            sync()
            started = time.perf_counter()
            with torch.inference_mode():
                model = module.load_model(str((args.submission / "model").resolve()), str(device))
            sync()
        except Exception as error:
            errors["load_failed"] += 1
            print(f"Load failed: {type(error).__name__}: {error}", file=sys.stderr)
        else:
            elapsed = time.perf_counter() - started
            for offset in range(0, len(records), args.batch_size):
                if elapsed >= args.max_seconds:
                    errors["inference_time_limit"] += 1
                    break
                batch = records[offset:offset + args.batch_size]
                images, questions = [], []
                for record in batch:
                    with Image.open(args.records.parent / record["image"]) as image:
                        images.append(image.convert("RGB"))
                    if "questions" in record:
                        questions.append(record["questions"])
                    else:
                        questions.append([item["question"] for item in record["qa"]])

                started = time.perf_counter()
                try:
                    with torch.inference_mode():
                        predictions = module.predict(model, images, questions)
                    sync()
                except Exception as error:
                    errors["failed_batches"] += 1
                    predictions = []
                    print(f"Batch at {offset} failed: {type(error).__name__}: {error}", file=sys.stderr)
                elapsed += time.perf_counter() - started
                if not isinstance(predictions, list):
                    errors["invalid_batch_return"] += 1
                    predictions = []
                if len(predictions) != len(batch):
                    errors["incorrect_batch_length"] += 1
                for record, prediction in zip(batch, predictions):
                    row = {"chart_id": record["chart_id"], "prediction": prediction_fields(prediction)}
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    completed += 1
                output.flush()
                if completed and (offset == 0 or (offset // args.batch_size + 1) % 50 == 0):
                    print(f"Processed {offset + len(batch)}/{len(records)} charts", flush=True)

    metadata = {"elapsed_seconds": elapsed, "input_charts": len(records),
                "prediction_rows": completed, "inference_errors": dict(errors)}
    Path(f"{args.predictions}.run.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def score_predictions(args):
    metadata_path = Path(f"{args.predictions}.run.json")
    score_inputs = (args.references, args.predictions, metadata_path)
    if args.report and args.report.resolve() in {path.resolve() for path in score_inputs}:
        raise ValueError("Choose a report path distinct from the scoring inputs.")
    grading = DEFAULT_GRADING

    records = read_jsonl(args.references)
    rows = read_jsonl(args.predictions)
    predictions = {}
    for row in rows:
        chart_id = row["chart_id"]
        if chart_id in predictions:
            raise ValueError(f"Duplicate prediction for {chart_id}")
        predictions[chart_id] = row["prediction"]
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Timing file missing: {metadata_path}. Keep the .run.json file from predict beside the predictions.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report = score_records(records, predictions, grading, metadata.get("elapsed_seconds"))
    report["grading"] = grading
    report["grading_source"] = "chartqa_metrics.py"
    report["inference_errors"] = metadata.get("inference_errors", {})
    report["unknown_prediction_ids"] = len(predictions.keys() - {r["chart_id"] for r in records})
    formatted = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(formatted, encoding="utf-8")
    print(formatted, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="Run 'predict' first, then 'score' with its predictions and timing file.")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Run inference on charts and questions.")
    predict.add_argument("--records", required=True, type=Path,
                         help="Chart JSONL. Image paths are relative to this file.")
    predict.add_argument("--submission", required=True, type=Path,
                         help="Extracted directory with chartqa_submission.py and model/.")
    predict.add_argument("--predictions", required=True, type=Path,
                         help="Output prediction JSONL. Timing is saved beside it.")
    predict.add_argument("--device", default="cuda", help="PyTorch device (default: cuda).")
    predict.add_argument("--batch-size", type=int, choices=range(1, 5), default=2,
                         help="Charts per prediction call (default: 2).")
    predict.add_argument("--max-seconds", type=float, default=DEFAULT_GRADING["hard_seconds"],
                         help="Time cap checked before each batch (default: %(default)s).")
    score = commands.add_parser("score", help="Compare predictions with the reference labels.")
    score.add_argument("--references", required=True, type=Path,
                       help="Labeled val.jsonl, public_test.jsonl, or instructor answer key.")
    score.add_argument("--predictions", required=True, type=Path,
                       help="JSONL written by predict; keep its .run.json file alongside it.")
    score.add_argument("--manifest", type=Path,
                       help="Dataset manifest path for handout compatibility; grading is built into chartqa_metrics.py.")
    score.add_argument("--report", type=Path, help="Optional path to save the JSON score report.")
    args = parser.parse_args()
    if args.command == "predict":
        if args.max_seconds <= 0:
            parser.error("The inference time limit must be positive.")
        if args.max_seconds > DEFAULT_GRADING["hard_seconds"]:
            parser.error(f"--max-seconds cannot exceed the course limit of {DEFAULT_GRADING['hard_seconds']} seconds.")
        run_predictions(args)
    else:
        score_predictions(args)


if __name__ == "__main__":
    main()
