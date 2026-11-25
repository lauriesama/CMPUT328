"""Build my_submission/ and ChartQA-<CCID>.zip from a training checkpoint.

    python training/package.py --checkpoint checkpoints/run1/best --ccid yourccid

Copies chartqa_submission.py, the inference files of the checkpoint (weights, configs, tokenizer,
processor, header font), README.md and a requirements.txt pinned to the versions installed in the
current environment. Optionally stores weights in a lower precision to reduce the archive size.
"""

import argparse
import shutil
import zipfile
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {"train_info.json", "eval_predictions.jsonl", "args.json", "log.jsonl"}
REQUIREMENTS = ("torch", "transformers", "tokenizers", "safetensors", "pillow", "numpy")
LIMIT = 2 * 1024 ** 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ccid", required=True)
    p.add_argument("--out-dir", type=Path, default=ROOT / "my_submission")
    p.add_argument("--readme", type=Path, default=ROOT / "inference" / "README_template.md")
    p.add_argument("--weights-dtype", choices=["float32", "bfloat16"], default="float32")
    args = p.parse_args()

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    model_dir = args.out_dir / "model"
    model_dir.mkdir(parents=True)
    shutil.copy(ROOT / "inference" / "chartqa_submission.py", args.out_dir)
    shutil.copy(args.readme, args.out_dir / "README.md")
    for path in args.checkpoint.iterdir():
        if path.is_file() and path.name not in SKIP:
            shutil.copy(path, model_dir)

    if args.weights_dtype != "float32":
        import torch
        from transformers import Pix2StructForConditionalGeneration
        model = Pix2StructForConditionalGeneration.from_pretrained(model_dir)
        model.to(getattr(torch, args.weights_dtype)).save_pretrained(model_dir)

    # Drop local build tags such as +cu130, which PyPI cannot resolve.
    pins = [f"{name}=={version(name).split('+')[0]}" for name in REQUIREMENTS]
    (args.out_dir / "requirements.txt").write_text("\n".join(pins) + "\n", encoding="utf-8")

    size = sum(f.stat().st_size for f in args.out_dir.rglob("*") if f.is_file())
    print(f"{args.out_dir}: {size / 1024 ** 3:.2f} GiB (limit 2 GiB)")
    if size > LIMIT:
        raise SystemExit("Submission exceeds 2 GiB; rerun with --weights-dtype bfloat16.")
    print("requirements:", ", ".join(pins))

    archive = args.out_dir.parent / f"ChartQA-{args.ccid}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(args.out_dir.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(args.out_dir))
    print(f"wrote {archive}")


if __name__ == "__main__":
    main()
