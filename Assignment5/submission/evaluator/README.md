# ChartQA evaluator: student instructions

Extract `chartqa_evaluator.zip` and `chartqa_student.zip` into the same working directory. The result should contain `evaluate.py`, `chartqa_metrics.py`, and `chartqa_course/`. Put your extracted submission in `my_submission/` with `chartqa_submission.py` and `model/` directly inside it.

Install the exact versions in your submission's `requirements.txt` in a Colab GPU runtime. Run the commands below from that working directory. In a notebook, prefix each command with `!`.

```bash
python evaluate.py predict \
  --records chartqa_course/val.jsonl \
  --submission my_submission \
  --predictions runs/val_predictions.jsonl \
  --device cuda --batch-size 2 --max-seconds 2000

python evaluate.py score \
  --references chartqa_course/val.jsonl \
  --predictions runs/val_predictions.jsonl \
  --manifest chartqa_course/manifest.json \
  --report runs/val_scores.json
```

For a shorter public rehearsal, replace `val.jsonl` with `public_test.jsonl` in both commands and choose different prediction and report filenames. Run `python evaluate.py --help`, `python evaluate.py predict --help`, or `python evaluate.py score --help` for all options.

`predict` imports your `chartqa_submission.py`, calls `load_model(model_dir, device)` once, then calls `predict(model, images, questions)` in batches. Each input is an RGB PIL image paired with its question strings. The output is a JSONL file of predictions plus a neighboring `*.run.json` file containing measured seconds and inference errors. Keep both files together. A failed batch or missing output scores zero for the affected charts; successfully completed batches still count.

`score` compares predictions with labeled references and writes a JSON report. Read `metrics` for raw chart type, table, and QA results; `component_marks` for the 10/40/50 rubric; `mark_before_time`, `time_multiplier`, and `mark_out_of_100` for the final mark. `format_errors` identifies malformed or missing components; `inference_errors` records load, batch, and time-limit failures. The `--manifest` argument is retained for the assignment's command format; the manifest contains dataset metadata only. The scoring rules and thresholds are in `chartqa_metrics.py`.

The course inference limit is **2000 seconds** for 400 charts. Timing includes model loading and prediction calls. The score multiplier is 1 through **1000 seconds**, falls linearly to 0 between 1000 and 2000 seconds, and is 0 at or above 2000 seconds. The evaluator checks the limit before each batch, so an already-running batch can finish beyond it. Missing or invalid timing metadata prevents scoring. Both commands must use the same prediction path.

The final grade uses a separate withheld split. Validation and public-test results are for checking your own submission.
