#!/usr/bin/env python3
"""
Lightweight evaluator for QAGS judgement JSON that can run in parallel
with the judgement writer. It tolerates partially-written files by
extracting completed JSON objects from the `records` array.

Usage examples:
    python llm-experiments/qags/evaluate_qags_results.py \
        --input results_tmp/qags_cnndm/qags_judgement.json \
        --aggregator-input results_tmp/qags_cnndm/meta_aggregator.json

The script prints per-model accuracies and writes a small summary JSON.
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def try_full_load(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return None


def iter_records_from_partial(content):
    """Yield JSON objects parsed from a partially-complete `records` array.

    This finds the `"records"` array in the file content and then
    extracts any complete JSON objects found inside it by balancing braces.
    """
    key = '"records"'
    idx = content.find(key)
    if idx == -1:
        return
    arr_start = content.find("[", idx)
    if arr_start == -1:
        return

    i = arr_start + 1
    n = len(content)
    while i < n:
        # skip whitespace and commas
        while i < n and content[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        if content[i] != '{':
            # no object start — could be end of array
            break
        start = i
        depth = 0
        while i < n:
            ch = content[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    i += 1
                    obj_text = content[start:i]
                    try:
                        obj = json.loads(obj_text)
                        yield obj
                    except Exception:
                        # skip unparsable object
                        pass
                    break
            i += 1
        # continue scanning after the end of the object


def open_partial_records(path):
    # Try full JSON load first (fast path)
    full = try_full_load(path)
    if full is not None and isinstance(full, dict) and "records" in full:
        for r in full["records"]:
            yield r
        return

    # Fallback: read raw and attempt to recover completed objects
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    for r in iter_records_from_partial(content):
        yield r


def load_summary(path):
    full = try_full_load(path)
    if full is not None:
        return full
    return None


def load_records(path):
    full = try_full_load(path)
    if full is not None and isinstance(full, dict) and "records" in full:
        records = full.get("records", [])
        if isinstance(records, list):
            return [r for r in records if isinstance(r, dict)]
    return [r for r in open_partial_records(path) if isinstance(r, dict)]


def compute_metrics(records):
    totals = {}
    correct = {}
    tp = {}
    tn = {}
    pos = {}
    neg = {}
    pred_yes = {}
    pred_no = {}
    processed = 0
    overall_pos = 0
    overall_neg = 0
    overall_other = 0
    for r in records:
        processed += 1
        maj = r.get("majority_human")
        if maj is None:
            overall_other += 1
        else:
            maj_l = str(maj).strip().lower()
            if maj_l == "yes":
                overall_pos += 1
            elif maj_l == "no":
                overall_neg += 1
            else:
                overall_other += 1
        model_outputs = r.get("model_outputs", {})
        for mname, mout in model_outputs.items():
            # Skip if model returned an error
            if mout is None:
                continue
            if isinstance(mout, dict) and mout.get("error"):
                # treat error as no prediction (skip)
                continue
            label = None
            if isinstance(mout, dict):
                label = mout.get("label")
            else:
                # sometimes flattened fields exist as strings
                label = mout
            if label is None:
                continue
            lm = mname
            totals.setdefault(lm, 0)
            correct.setdefault(lm, 0)
            tp.setdefault(lm, 0)
            tn.setdefault(lm, 0)
            pos.setdefault(lm, 0)
            neg.setdefault(lm, 0)

            totals[lm] += 1
            pred = str(label).strip().lower()
            if pred == "yes":
                pred_yes[lm] = pred_yes.get(lm, 0) + 1
            elif pred == "no":
                pred_no[lm] = pred_no.get(lm, 0) + 1
            if maj is not None:
                maj_l = str(maj).strip().lower()
                if maj_l in ("yes", "no"):
                    if maj_l == "yes":
                        pos[lm] += 1
                        if pred == "yes":
                            tp[lm] += 1
                    else:
                        neg[lm] += 1
                        if pred == "no":
                            tn[lm] += 1
                if pred == maj_l:
                    correct[lm] += 1

    metrics = {}
    for m in totals:
        t = totals[m]
        c = correct.get(m, 0)
        p = pos.get(m, 0)
        n = neg.get(m, 0)
        tpp = tp.get(m, 0)
        tnn = tn.get(m, 0)
        acc = c / t if t > 0 else None
        balanced = None
        if p > 0 and n > 0:
            tpr = tpp / p
            tnr = tnn / n
            balanced = 0.5 * (tpr + tnr)
        py = pred_yes.get(m, 0)
        pn = pred_no.get(m, 0)

        # Cohen's kappa for this model
        kappa = None
        if t > 0:
            po = c / t
            pe = 0.0
            if t > 0:
                pe = ((p / t) * (py / t)) + ((n / t) * (pn / t))
            if pe != 1:
                kappa = (po - pe) / (1 - pe)

        metrics[m] = {
            "total": t,
            "correct": c,
            "accuracy": acc,
            "pos_count": p,
            "neg_count": n,
            "pred_yes": py,
            "pred_no": pn,
            "tp": tpp,
            "tn": tnn,
            "balanced_accuracy": balanced,
            "kappa": kappa,
        }
    overall = {"yes": overall_pos, "no": overall_neg, "other": overall_other}
    return metrics, processed, overall


def compute_metrics_from_predictions(rows, prediction_getter):
    correct = 0
    total = 0
    pos = 0
    neg = 0
    tp = 0
    tn = 0
    pred_yes = 0
    pred_no = 0
    other = 0

    for row in rows:
        gold = row.get("majority_human")
        pred = prediction_getter(row)
        if gold is None:
            other += 1
            continue
        gold_l = str(gold).strip().lower()
        pred_l = None if pred is None else str(pred).strip().lower()
        if gold_l not in ("yes", "no"):
            other += 1
            continue
        if pred_l not in ("yes", "no"):
            other += 1
            continue

        total += 1
        if pred_l == "yes":
            pred_yes += 1
        else:
            pred_no += 1
        if gold_l == "yes":
            pos += 1
            if pred_l == "yes":
                tp += 1
        else:
            neg += 1
            if pred_l == "no":
                tn += 1

        if pred_l == gold_l:
            correct += 1

    balanced = None
    if pos > 0 and neg > 0:
        balanced = 0.5 * ((tp / pos) + (tn / neg))

    kappa = None
    if total > 0:
        po = correct / total
        pe = ((pos / total) * (pred_yes / total)) + ((neg / total) * (pred_no / total))
        if pe != 1:
            kappa = (po - pe) / (1 - pe)

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else None,
        "pos_count": pos,
        "neg_count": neg,
        "pred_yes": pred_yes,
        "pred_no": pred_no,
        "tp": tp,
        "tn": tn,
        "balanced_accuracy": balanced,
        "kappa": kappa,
        "other": other,
    }


def print_metric_table(title, metrics_dict):
    print(title)
    if not metrics_dict:
        print("  No metrics available.")
        return

    rows = []
    for name, stats in metrics_dict.items():
        acc = stats.get("accuracy")
        bal = stats.get("balanced_accuracy")
        rows.append(
            [
                name,
                f"{acc:.4f}" if acc is not None else "N/A",
                f"{stats.get('correct',0)}/{stats.get('total',0)}",
                str(stats.get("pos_count", 0)),
                str(stats.get("neg_count", 0)),
                str(stats.get("pred_yes", 0)),
                str(stats.get("pred_no", 0)),
                str(stats.get("tp", 0)),
                str(stats.get("tn", 0)),
                f"{bal:.4f}" if bal is not None else "N/A",
                f"{stats.get('kappa'):.4f}" if stats.get("kappa") is not None else "N/A",
            ]
        )

    headers = ["Model", "Accuracy", "Correct/Total", "GoldYes", "GoldNo", "PredYes", "PredNo", "TP", "TN", "Balanced", "Kappa"]
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    widths = [max(len(str(x)) for x in col) + 2 for col in cols]

    def fmt_row(items):
        return "".join(str(v).ljust(w) for v, w in zip(items, widths))

    print(fmt_row(headers))
    print("".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def collect_error_sample_ids(records, error_key=None):
    errors_by_model: dict[str, list[Any]] = {}
    for row in records:
        sample_id = row.get("sample_id")
        if error_key is not None:
            if row.get(error_key):
                errors_by_model.setdefault(error_key, []).append(sample_id)
            continue

        model_outputs = row.get("model_outputs", {})
        if not isinstance(model_outputs, dict):
            continue
        for model_name, info in model_outputs.items():
            if isinstance(info, dict) and info.get("error"):
                errors_by_model.setdefault(model_name, []).append(sample_id)
    return errors_by_model


def print_error_report(title, errors_by_model):
    print(title)
    if not errors_by_model:
        print("  No non-null errors found.")
        return

    for model_name, sample_ids in sorted(errors_by_model.items(), key=lambda item: item[0]):
        unique_ids = []
        seen = set()
        for sample_id in sample_ids:
            if sample_id in seen:
                continue
            seen.add(sample_id)
            unique_ids.append(sample_id)
        print(f"  {model_name}: {len(unique_ids)} -> {unique_ids}")


def write_csv_summary(output_path, metrics, majority_vote_metrics=None, aggregator_name=None, aggregator_metrics=None):
    rows = []
    for name, stats in metrics.items():
        rows.append(("individual", name, stats))
    if majority_vote_metrics is not None:
        rows.append(("majority_vote", "Majority Vote", majority_vote_metrics))
    if aggregator_metrics is not None:
        rows.append(("aggregator", aggregator_name or "aggregator", aggregator_metrics))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kind",
                "name",
                "accuracy",
                "correct",
                "total",
                "pos_count",
                "neg_count",
                "pred_yes",
                "pred_no",
                "tp",
                "tn",
                "balanced_accuracy",
                "kappa",
                "other",
            ],
        )
        writer.writeheader()
        for kind, name, stats in rows:
            writer.writerow(
                {
                    "kind": kind,
                    "name": name,
                    "accuracy": stats.get("accuracy"),
                    "correct": stats.get("correct"),
                    "total": stats.get("total"),
                    "pos_count": stats.get("pos_count"),
                    "neg_count": stats.get("neg_count"),
                    "pred_yes": stats.get("pred_yes"),
                    "pred_no": stats.get("pred_no"),
                    "tp": stats.get("tp"),
                    "tn": stats.get("tn"),
                    "balanced_accuracy": stats.get("balanced_accuracy"),
                    "kappa": stats.get("kappa"),
                    "other": stats.get("other"),
                }
            )


def compute_majority_vote_label(row, model_outputs_key="model_outputs"):
    """Compute majority vote label for a sample from all models."""
    model_outputs = row.get(model_outputs_key, {})
    if not isinstance(model_outputs, dict):
        return None
    
    labels = []
    for model_name, info in model_outputs.items():
        if isinstance(info, dict) and not info.get("error"):
            label = info.get("label")
            if label in ("yes", "no"):
                labels.append(label)
    
    if not labels:
        return None
    from collections import Counter
    counts = Counter(labels)
    most_common_label, _ = counts.most_common(1)[0]
    return most_common_label


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-model accuracy from qags_judgement.json (partial-safe)")
    parser.add_argument("--dataset", choices=["cnndm", "xsum"], default="cnndm", help="Dataset to evaluate (cnndm or xsum)")
    args = parser.parse_args()

    DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "results_tmp" / f"qags_{args.dataset}" / "qags_judgement.json"
    DEFAULT_AGG = Path(__file__).resolve().parents[2] / "results_tmp" / "qags_xsum" / "gemini_aggregator.json"
    print("\n DEFAULT_AGG PATH : ", DEFAULT_AGG)
    input_path = DEFAULT_INPUT
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(2)

    input_summary = load_summary(input_path)
    input_records = load_records(input_path)
    if not input_records:
        print(f"No records found in {input_path}")
        sys.exit(2)

    # Aggregator path is hard-coded to the xsum gemini aggregator file.
    aggregator_path = DEFAULT_AGG
    aggregator_summary = load_summary(aggregator_path) if aggregator_path.exists() else None
    aggregator_records = load_records(aggregator_path) if aggregator_path.exists() else []

    input_by_id = {row.get("sample_id"): row for row in input_records}
    merged_rows = []
    for row in input_records:
        merged = dict(row)
        sample_id = row.get("sample_id")
        if sample_id in input_by_id:
            merged.update(input_by_id[sample_id])
        if aggregator_records:
            for agg_row in aggregator_records:
                if agg_row.get("sample_id") == sample_id:
                    merged["gemini_aggregator_label"] = agg_row.get("meta_aggregator_label")
                    merged["gemini_aggregator_reason"] = agg_row.get("meta_aggregator_reason")
                    merged["gemini_aggregator_error"] = agg_row.get("meta_aggregator_error")
                    break
        merged_rows.append(merged)

    metrics, processed, overall = compute_metrics(merged_rows)

    # Compute majority vote metrics
    majority_vote_metrics = compute_metrics_from_predictions(
        merged_rows,
        lambda row: compute_majority_vote_label(row),
    )

    aggregator_name = None
    if aggregator_summary and isinstance(aggregator_summary, dict):
        aggregator_name = aggregator_summary.get("aggregator_model")
    if not aggregator_name:
        # aggregator_name = "meta/llama-3.3-70b-instruct-maas"
        aggregator_name = "gemini-2.5-flash-lite"


    aggregator_metrics = None
    if aggregator_records:
        aggregator_metrics = compute_metrics_from_predictions(
            merged_rows,
            lambda row: row.get("gemini_aggregator_label"),
        )

    input_error_report = collect_error_sample_ids(input_records)
    aggregator_error_report = collect_error_sample_ids(aggregator_records, error_key="meta_aggregator_error") if aggregator_records else {}

    now = datetime.utcnow().isoformat() + "Z"
    summary = {
        "timestamp": now,
        "input": str(input_path),
        "dataset": (input_summary or {}).get("dataset") if isinstance(input_summary, dict) else None,
        "aggregator_input": str(aggregator_path) if aggregator_path else None,
        "aggregator_name": aggregator_name,
        "n_records_processed": processed,
        "metrics": metrics,
        "aggregator_metrics": aggregator_metrics,
    }

    # Print concise results
    dataset_name = (input_summary or {}).get("dataset") if isinstance(input_summary, dict) else None
    if not dataset_name:
        dataset_name = input_path.parent.name.replace("qags_", "") or "unknown"

    print(f"Dataset: {dataset_name}")
    print(f"Aggregator: {aggregator_name}")
    print(f"Processed records: {processed}")
    print(f"Label distribution (majority_human): yes={overall.get('yes',0)}, no={overall.get('no',0)}, other={overall.get('other',0)}")
    print("\n")
    print_metric_table("Individual model metrics:", metrics)
    print("\n")
    if majority_vote_metrics is not None:
        print_metric_table("Majority vote metrics:", {"Majority Vote": majority_vote_metrics})
        print("\n")
    print_error_report("Error sample IDs in individual model outputs:", input_error_report)
    if aggregator_metrics is not None:
        print("\n")
        print_metric_table(f"Aggregator metrics ({aggregator_name}):", {aggregator_name: aggregator_metrics})
        print("\n")
        print_error_report("Error sample IDs in aggregator output:", aggregator_error_report)
        print("\n")
    else:
        print("Aggregator metrics: not available (missing gemini_aggregator.json or no matching records).")

    csv_output_path = input_path.with_name(f"{input_path.stem}_metrics.csv")
    write_csv_summary(
        csv_output_path,
        metrics,
        majority_vote_metrics=majority_vote_metrics,
        aggregator_name=aggregator_name,
        aggregator_metrics=aggregator_metrics,
    )
    print(f"CSV summary saved to: {csv_output_path}")

    # No file output requested — print only.


if __name__ == "__main__":
    main()
