"""Dataset schema and scoring rules shared by preparation and evaluation."""

import json
import math
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation


CHART_TYPES = ("v_bar", "h_bar", "line", "pie")
QA_SOURCES = ("human", "augmented")
MAX_REFERENCE_CELLS = 24
MAX_PREDICTED_CELLS = 128
SCHEMA_VERSION = "chartqa-course-v1"
DEFAULT_GRADING = {
    "chart_type": {"weight": 10, "lower": 0.45, "upper": 0.90},
    "table": {"weight": 40, "lower": 0.20, "upper": 0.40},
    "qa": {"weight": 50, "lower": 0.20, "upper": 0.50},
    "time_penalty": True,
    "fast_seconds": 1000,
    "slow_seconds": 2000,
    "hard_seconds": 2000,
}

_NUMBER = re.compile(
    r"(?:[$£€][+-]?|[+-]?[$£€]?)"
    r"(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)"
    r"(?:e[+-]?[0-9]+)?%?"
)
_NONFINITE = re.compile(r"(?:[$£€][+-]?|[+-]?[$£€]?)(?:nan|inf|infinity)%?")
_DECORATIONS = str.maketrans("", "", "$£€,%")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def parse_number(text):
    normalized = normalize_text(text)
    if _NUMBER.fullmatch(normalized) is None:
        return None
    try:
        value = Decimal(normalized.translate(_DECORATIONS))
        return value if math.isfinite(float(value)) else None
    except (InvalidOperation, ValueError, OverflowError):
        return None


def value_matches(predicted, reference):
    if not isinstance(predicted, str) or not isinstance(reference, str):
        return False
    pred_text = normalize_text(predicted)
    ref_text = normalize_text(reference)
    if _NONFINITE.fullmatch(pred_text) or _NONFINITE.fullmatch(ref_text):
        return False
    pred_number = parse_number(pred_text)
    ref_number = parse_number(ref_text)
    if pred_number is not None and ref_number is not None:
        tolerance = max(Decimal("0.05") * abs(ref_number), Decimal("0.000001"))
        return abs(pred_number - ref_number) <= tolerance
    # A syntactically numeric overflow should not match as text.
    if _NUMBER.fullmatch(pred_text) and pred_number is None:
        return False
    return bool(ref_text) and pred_text == ref_text


def table_cells(table, max_cells=MAX_PREDICTED_CELLS):
    """Return (normalized row, normalized series, value) triples, or None."""
    if not isinstance(table, dict):
        return None
    columns = table.get("columns")
    rows = table.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return None
    if not columns and not rows:
        return []
    if len(columns) < 2 or not all(isinstance(label, str) for label in columns):
        return None
    if len(rows) * (len(columns) - 1) > max_cells:
        return None
    series = [normalize_text(label) for label in columns[1:]]
    cells = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            return None
        if not all(isinstance(value, str) for value in row):
            return None
        row_label = normalize_text(row[0])
        cells.extend((row_label, name, value) for name, value in zip(series, row[1:]))
    return cells


def table_f1(predicted, reference):
    truth_cells = table_cells(reference, MAX_REFERENCE_CELLS)
    if not truth_cells:
        raise ValueError("A reference chart must contain between 1 and 24 valid cells.")
    truth = {(row, col): value for row, col, value in truth_cells}
    if len(truth) != len(truth_cells):
        raise ValueError("Reference row and series labels must give unique cell keys.")
    pred_cells = table_cells(predicted)
    if not pred_cells:
        return 0.0

    # Each reference key is unique. At most one correct prediction can use it.
    matched = set()
    for row, col, value in pred_cells:
        key = (row, col)
        if key in truth and value_matches(value, truth[key]):
            matched.add(key)
    return 2 * len(matched) / (len(pred_cells) + len(truth_cells))


def _average_groups(correct, total, groups):
    rates = {group: correct[group] / total[group] if total[group] else None
             for group in groups}
    mean = None if None in rates.values() else sum(rates.values()) / len(groups)
    return mean, rates


def score_records(references, predictions, grading=None, elapsed_seconds=None):
    """Score keyed prediction dictionaries. Missing predictions score zero."""
    if not references:
        raise ValueError("The reference file is empty.")
    grading = DEFAULT_GRADING if grading is None else grading
    class_count, class_correct = Counter(), Counter()
    source_count, source_correct = Counter(), Counter()
    errors = Counter()
    table_total = 0.0
    for record in references:
        prediction = predictions.get(record["chart_id"])
        if not isinstance(prediction, dict):
            errors["missing_or_invalid_chart_prediction"] += 1
            prediction = {}

        chart_type = record["chart_type"]
        class_count[chart_type] += 1
        pred_type = prediction.get("chart_type")
        class_correct[chart_type] += int(pred_type == chart_type)
        if pred_type not in CHART_TYPES:
            errors["invalid_chart_type"] += 1

        pred_table = prediction.get("table")
        if table_cells(pred_table) is None:
            errors["invalid_table"] += 1
        table_total += table_f1(pred_table, record["table"])

        answers = prediction.get("answers")
        if not isinstance(answers, list):
            answers = []
            errors["invalid_answer_list"] += 1
        errors["extra_answers"] += max(0, len(answers) - len(record["qa"]))
        for index, item in enumerate(record["qa"]):
            source = item["source"]
            source_count[source] += 1
            pred_answer = answers[index] if index < len(answers) else None
            if not isinstance(pred_answer, str):
                errors["missing_or_invalid_answer"] += 1
            source_correct[source] += int(value_matches(pred_answer, item["answer"]))

    type_accuracy, per_type = _average_groups(class_correct, class_count, CHART_TYPES)
    qa_accuracy, per_source = _average_groups(source_correct, source_count, QA_SOURCES)
    metrics = {"chart_type": type_accuracy, "table": table_total / len(references),
               "qa": qa_accuracy}
    marks = {}
    for name, value in metrics.items():
        component = grading[name]
        lower, upper = component["lower"], component["upper"]
        if not 0 <= lower < upper <= 1:
            raise ValueError(f"Invalid grading cutoffs for {name}.")
        marks[name] = None if value is None else component["weight"] * max(
            0.0, min(1.0, (value - lower) / (upper - lower))
        )

    total = None if None in marks.values() else sum(marks.values())
    multiplier = 1.0
    if grading["time_penalty"]:
        if elapsed_seconds is None or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("A valid measured elapsed time is required for the time penalty.")
        fast, slow = grading["fast_seconds"], grading["slow_seconds"]
        if not 0 <= fast < slow:
            raise ValueError("The time bounds must satisfy 0 <= fast < slow.")
        multiplier = max(0.0, min(1.0, (slow - elapsed_seconds) / (slow - fast)))

    return {
        "charts": len(references),
        "metrics": metrics,
        "chart_type_accuracy": per_type,
        "chart_type_counts": {k: class_count[k] for k in CHART_TYPES},
        "qa_accuracy": per_source,
        "qa_counts": {k: source_count[k] for k in QA_SOURCES},
        "component_marks": marks,
        "mark_before_time": total,
        "elapsed_seconds": elapsed_seconds,
        "time_multiplier": multiplier,
        "mark_out_of_100": None if total is None else total * multiplier,
        "format_errors": {name: count for name, count in errors.items() if count},
    }
