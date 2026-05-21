from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results_tmp" / "dices_judgement_005.json"
DEFAULT_CSV_OUTPUT = ROOT / "results_tmp" / "dices_judgement_005_comparison.csv"
DEFAULT_MD_OUTPUT = ROOT / "results_tmp" / "dices_judgement_005_comparison.md"
DEFAULT_HUMAN_CSV_OUTPUT = ROOT / "results_tmp" / "dices_judgement_005_human_comparison.csv"
DEFAULT_HUMAN_MD_OUTPUT = ROOT / "results_tmp" / "dices_judgement_005_human_comparison.md"


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"yes", "safe"}:
        return "Yes"
    if text in {"no", "unsafe"}:
        return "No"
    return None


def load_results(path: Path) -> dict[str, Any]: 
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def accuracy_stats(gold: list[str | None], preds: list[str | None]) -> dict[str, Any]:
    evaluated = [(g, p) for g, p in zip(gold, preds) if g in {"Yes", "No"} and p in {"Yes", "No"}]
    if not evaluated:
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "mcc": None,
            "correct": 0,
            "evaluated": 0,
            "coverage": 0.0,
            "safe_accuracy": None,
            "unsafe_accuracy": None,
            "safe_precision": None,
            "unsafe_precision": None,
        }

    total = len(gold)
    correct = sum(1 for g, p in evaluated if g == p)
    
    # Per-class accuracy
    gold_yes = [(g, p) for g, p in evaluated if g == "Yes"]
    gold_no = [(g, p) for g, p in evaluated if g == "No"]
    pred_yes = [(g, p) for g, p in evaluated if p == "Yes"]
    pred_no = [(g, p) for g, p in evaluated if p == "No"]
    safe_correct = sum(1 for g, p in gold_yes if p == "Yes")
    unsafe_correct = sum(1 for g, p in gold_no if p == "No")
    safe_pred_correct = sum(1 for g, p in pred_yes if g == "Yes")
    unsafe_pred_correct = sum(1 for g, p in pred_no if g == "No")
    
    safe_acc = safe_correct / len(gold_yes) if gold_yes else None
    unsafe_acc = unsafe_correct / len(gold_no) if gold_no else None
    safe_precision = safe_pred_correct / len(pred_yes) if pred_yes else None
    unsafe_precision = unsafe_pred_correct / len(pred_no) if pred_no else None
    
    def f1(precision: float | None, recall: float | None) -> float | None:
        if precision is None or recall is None or (precision + recall) == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    safe_f1 = f1(safe_precision, safe_acc)
    unsafe_f1 = f1(unsafe_precision, unsafe_acc)
    macro_f1 = None
    if safe_f1 is not None and unsafe_f1 is not None:
        macro_f1 = (safe_f1 + unsafe_f1) / 2

    # Binary MCC for Yes/Safe as the positive class
    tp = sum(1 for g, p in evaluated if g == "Yes" and p == "Yes")
    tn = sum(1 for g, p in evaluated if g == "No" and p == "No")
    fp = sum(1 for g, p in evaluated if g == "No" and p == "Yes")
    fn = sum(1 for g, p in evaluated if g == "Yes" and p == "No")
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom else None
    
    # Balanced accuracy = average of per-class accuracies
    if safe_acc is not None and unsafe_acc is not None:
        balanced_acc = (safe_acc + unsafe_acc) / 2
    else:
        balanced_acc = None

    return {
        "accuracy": correct / len(evaluated),
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "correct": correct,
        "evaluated": len(evaluated),
        "coverage": len(evaluated) / total if total else 0.0,
        "safe_accuracy": safe_acc,
        "unsafe_accuracy": unsafe_acc,
        "safe_precision": safe_precision,
        "unsafe_precision": unsafe_precision,
    }


def pairwise_outcome(gold: list[str | None], a: list[str | None], b: list[str | None]) -> dict[str, int]:
    better_a = better_b = ties = disagreements = both_valid = 0
    for g, pa, pb in zip(gold, a, b):
        if g not in {"Yes", "No"} or pa not in {"Yes", "No"} or pb not in {"Yes", "No"}:
            continue
        both_valid += 1
        correct_a = pa == g
        correct_b = pb == g
        if correct_a and correct_b:
            ties += 1
        elif correct_a and not correct_b:
            better_a += 1
            disagreements += 1
        elif correct_b and not correct_a:
            better_b += 1
            disagreements += 1
        else:
            ties += 1
    return {
        "better_a": better_a,
        "better_b": better_b,
        "ties": ties,
        "disagreements": disagreements,
        "both_valid": both_valid,
    }


def compute_correlations(pred_a: list[str | None], pred_b: list[str | None]) -> dict[str, float | None]:
    """Compute Pearson and Spearman correlation between two prediction lists (Yes=1, No=0)."""
    valid_pairs = []
    for pa, pb in zip(pred_a, pred_b):
        if pa in {"Yes", "No"} and pb in {"Yes", "No"}:
            valid_pairs.append((1 if pa == "Yes" else 0, 1 if pb == "Yes" else 0))
    
    if len(valid_pairs) < 2:
        return {"pearson": None, "spearman": None, "valid_pairs": len(valid_pairs)}
    
    a_vals, b_vals = zip(*valid_pairs)
    try:
        pearson_r, pearson_p = pearsonr(a_vals, b_vals)
        spearman_r, spearman_p = spearmanr(a_vals, b_vals)
        return {
            "pearson": pearson_r,
            "spearman": spearman_r,
            "valid_pairs": len(valid_pairs)
        }
    except Exception as e:
        return {"pearson": None, "spearman": None, "valid_pairs": len(valid_pairs), "error": str(e)}


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def shorten(text: str, limit: int = 24) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def build_method_predictions(records: list[dict[str, Any]], models: list[str]) -> tuple[list[str | None], list[str | None], dict[str, list[str | None]]]:
    gold_safety = [normalize_label(record.get("safety_gold")) for record in records]
    gold_human = [normalize_label(record.get("human_label")) for record in records]
    predictions: dict[str, list[str | None]] = {}

    for model_name in models:
        predictions[f"Judge: {shorten(model_name)}"] = [
            normalize_label(record.get("panel_outputs", {}).get(model_name, {}).get("label")) for record in records
        ]

    predictions["Majority vote"] = [normalize_label(record.get("panel_majority")) for record in records]
    predictions["Aggregator"] = [normalize_label(record.get("aggregator_label")) for record in records]
    return gold_safety, gold_human, predictions


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["Method", "Accuracy", "Balanced Acc.", "Macro F1", "MCC", "Coverage", "Correct", "Evaluated", "Safe acc.", "Unsafe acc."]
    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                row["method"],
                format_pct(row["accuracy"]),
                format_pct(row["balanced_accuracy"]),
                format_pct(row["macro_f1"]),
                f"{row['mcc']:.3f}" if row["mcc"] is not None else "n/a",
                format_pct(row["coverage"]),
                str(row["correct"]),
                str(row["evaluated"]),
                format_pct(row["safe_accuracy"]),
                format_pct(row["unsafe_accuracy"]),
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_line(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    print(render_line(headers))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(render_line(row))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "accuracy", "balanced_accuracy", "macro_f1", "mcc", "coverage", "correct", "evaluated", "safe_accuracy", "unsafe_accuracy", "safe_precision", "unsafe_precision"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| Method | Accuracy | Balanced Acc. | Macro F1 | MCC | Coverage | Correct | Evaluated | Safe acc. | Unsafe acc. |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            mcc_str = f"{row['mcc']:.3f}" if row["mcc"] is not None else "n/a"
            handle.write(
                f"| {row['method']} | {format_pct(row['accuracy'])} | {format_pct(row['balanced_accuracy'])} | {format_pct(row['macro_f1'])} | {mcc_str} | {format_pct(row['coverage'])} | {row['correct']} | {row['evaluated']} | {format_pct(row['safe_accuracy'])} | {format_pct(row['unsafe_accuracy'])} |\n"
            )


def print_correlation_table(rows: list[dict[str, Any]]) -> None:
    headers = ["Judge", "Pearson", "Spearman", "n"]
    widths = [len(h) for h in headers]
    for row in rows:
        widths[0] = max(widths[0], len(row["model"]))
        widths[1] = max(widths[1], len(f"{row['pearson']:.3f}" if row["pearson"] is not None else "n/a"))
        widths[2] = max(widths[2], len(f"{row['spearman']:.3f}" if row["spearman"] is not None else "n/a"))
        widths[3] = max(widths[3], len(str(row["valid_pairs"])))

    def render(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        pearson_str = f"{row['pearson']:.3f}" if row["pearson"] is not None else "n/a"
        spearman_str = f"{row['spearman']:.3f}" if row["spearman"] is not None else "n/a"
        print(render([row["model"], pearson_str, spearman_str, str(row["valid_pairs"])]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DICES judge results against safety_gold and human labels.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to a dices_judgement JSON file.")
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT, help="Optional CSV output path.")
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT, help="Optional Markdown output path.")
    parser.add_argument("--human-csv-output", type=Path, default=DEFAULT_HUMAN_CSV_OUTPUT, help="Optional CSV output path for human_label comparison.")
    parser.add_argument("--human-markdown-output", type=Path, default=DEFAULT_HUMAN_MD_OUTPUT, help="Optional Markdown output path for human_label comparison.")
    parser.add_argument("--no-files", action="store_true", help="Do not write CSV/Markdown files; only print to stdout.")
    args = parser.parse_args()

    data = load_results(args.input)
    records = data.get("records", []) if isinstance(data, dict) else []
    models = data.get("models", []) if isinstance(data, dict) else []

    if not records:
        raise RuntimeError(f"No records found in {args.input}")

    gold_safety, gold_human, predictions = build_method_predictions(records, models)

    # Compute accuracy against safety_gold
    summary_rows_safety: list[dict[str, Any]] = []
    for method, preds in predictions.items():
        stats = accuracy_stats(gold_safety, preds)
        summary_rows_safety.append({"method": method, **stats})

    # Compute accuracy against human_label
    summary_rows_human: list[dict[str, Any]] = []
    for method, preds in predictions.items():
        stats = accuracy_stats(gold_human, preds)
        summary_rows_human.append({"method": method, **stats})

    judge_rows_safety = [row for row in summary_rows_safety if row["method"].startswith("Judge:")]
    other_rows_safety = [row for row in summary_rows_safety if not row["method"].startswith("Judge:")]
    summary_rows_safety = judge_rows_safety + other_rows_safety

    judge_rows_human = [row for row in summary_rows_human if row["method"].startswith("Judge:")]
    other_rows_human = [row for row in summary_rows_human if not row["method"].startswith("Judge:")]
    summary_rows_human = judge_rows_human + other_rows_human

    print(f"Loaded {len(records)} evaluated samples from {args.input}")
    print(f"Safety gold labels available: {sum(1 for label in gold_safety if label in {'Yes', 'No'})}/{len(gold_safety)}")
    print(f"Human labels available: {sum(1 for label in gold_human if label in {'Yes', 'No'})}/{len(gold_human)}")
    print()

    print("=" * 100)
    print("ACCURACY AGAINST SAFETY_GOLD (Crowdworker Majority)")
    print("=" * 100)
    print_table(summary_rows_safety)

    print()
    print("=" * 100)
    print("ACCURACY AGAINST HUMAN_LABEL (JSON Annotation)")
    print("=" * 100)
    print_table(summary_rows_human)

    # Head-to-head comparisons
    print()
    print("=" * 100)
    print("HEAD-TO-HEAD COMPARISONS")
    print("=" * 100)
    
    majority_preds = predictions["Majority vote"]
    aggregator_preds = predictions["Aggregator"]
    majority_vs_aggregator = pairwise_outcome(gold_safety, majority_preds, aggregator_preds)

    print()
    print("Majority vote vs Aggregator (vs safety_gold):")
    print(
        f"Aggregator better: {majority_vs_aggregator['better_b']} | Majority better: {majority_vs_aggregator['better_a']} | "
        f"Ties: {majority_vs_aggregator['ties']} | Comparable rows: {majority_vs_aggregator['both_valid']}"
    )

    best_judge_row = max(
        (row for row in summary_rows_safety if row["method"].startswith("Judge:") and row["accuracy"] is not None),
        key=lambda row: row["accuracy"],
        default=None,
    )
    if best_judge_row is not None:
        best_judge_method = best_judge_row["method"]
        best_judge_model = None
        for model in models:
            if shorten(model) in best_judge_method:
                best_judge_model = model
                break
        
        if best_judge_model:
            judge_preds = predictions[best_judge_method]
            agg_vs_best_judge = pairwise_outcome(gold_safety, judge_preds, aggregator_preds)
            print()
            print(f"Best individual judge: {best_judge_method} ({format_pct(best_judge_row['accuracy'])})")
            print(
                f"Aggregator vs best judge: Better={agg_vs_best_judge['better_b']}, Worse={agg_vs_best_judge['better_a']}, "
                f"Ties={agg_vs_best_judge['ties']}, Comparable={agg_vs_best_judge['both_valid']}"
            )

    # Correlation analysis
    print()
    print("=" * 100)
    print("CORRELATION ANALYSIS: Aggregator vs Each Judge")
    print("=" * 100)
    print()
    print("Pearson and Spearman correlations between aggregator predictions and each judge:")
    print()
    
    corr_rows = []
    for model_name in models:
        judge_method = f"Judge: {shorten(model_name)}"
        judge_preds = predictions[judge_method]
        corrs = compute_correlations(aggregator_preds, judge_preds)
        
        corr_rows.append({
            "model": shorten(model_name, limit=30),
            "pearson": corrs.get("pearson"),
            "spearman": corrs.get("spearman"),
            "valid_pairs": corrs.get("valid_pairs"),
        })

    corr_rows.sort(
        key=lambda row: (
            row["pearson"] is None,
            row["pearson"] if row["pearson"] is not None else float("inf"),
            row["spearman"] if row["spearman"] is not None else float("inf"),
        )
    )
    print_correlation_table(corr_rows)

    if not args.no_files:
        write_csv(args.csv_output, summary_rows_safety)
        write_markdown(args.markdown_output, summary_rows_safety)
        write_csv(args.human_csv_output, summary_rows_human)
        write_markdown(args.human_markdown_output, summary_rows_human)
        print()
        print(f"Saved CSV summary (vs safety_gold) to: {args.csv_output}")
        print(f"Saved Markdown summary (vs safety_gold) to: {args.markdown_output}")
        print(f"Saved CSV summary (vs human_label) to: {args.human_csv_output}")
        print(f"Saved Markdown summary (vs human_label) to: {args.human_markdown_output}")


if __name__ == "__main__":
    main()