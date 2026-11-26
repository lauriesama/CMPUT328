"""Copy a trained model directory into my_submission/model and build the submission ZIP.

Run from Assignment5/:
    python training/package.py --checkpoint checkpoints/run1/best
"""

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_FILES = ("chartqa_submission.py", "requirements.txt", "README.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Directory written by train.py (best/).")
    parser.add_argument("--submission", type=Path, default=ROOT / "my_submission")
    parser.add_argument("--zip", type=Path, default=ROOT / "ChartQA-gantuang.zip")
    args = parser.parse_args()

    model_dir = args.submission / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(args.checkpoint, model_dir, ignore=shutil.ignore_patterns("trainer_state.pt"))

    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in SUBMISSION_FILES:
            archive.write(args.submission / name, name)
        for path in sorted(model_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(args.submission))
    size = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())
    print(f"model/ is {size / 2**30:.2f} GiB; wrote {args.zip}")


if __name__ == "__main__":
    main()
