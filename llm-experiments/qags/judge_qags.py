'''
Run LLM judges on QAGS factual-consistency data.

Usage:
    python llm-experiments/qags/judge_qags.py --dataset cnndm
    python llm-experiments/qags/judge_qags.py --dataset cnndm --samples 10
    python llm-experiments/qags/judge_qags.py --dataset cnndm --sample-ids 5 12 99
    python llm-experiments/qags/judge_qags.py --dataset cnndm --no-resume
'''
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm-experiments" / "pipeline"))
from utils.api import SERVICE_ACCOUNT_PATH, PROJECT_ID, dispatch, init_clients  # noqa: E402

MODELS = [
    "MiniMax-M2.7",
    "GPT-OSS-120b",
    "Qwen3.6-35B",
    "gemini-2.5-flash",
    "meta/llama-3.3-70b-instruct-maas",
    "google/gemma-4-26b-a4b-it-maas",
    "Apertus-8B",
]

SEED                   = 42
DEFAULT_CHECKPOINT_EVERY = 5


# ---------------------------------------------------------------------------
# Label parsing — QAGS uses yes/no
# ---------------------------------------------------------------------------

def normalize_label(text: Any) -> str | None:
    if text is None:
        return None
    v = str(text).strip().lower()
    if v in {"yes", "supported", "true"}:
        return "yes"
    if v in {"no", "unsupported", "false"}:
        return "no"
    return None


def extract_label_and_reason(raw_output: str | None) -> tuple[str | None, str | None]:
    if not raw_output:
        return None, None
    m = re.search(r"\b(yes|no)\b", raw_output, flags=re.IGNORECASE)
    label = normalize_label(m.group(1)) if m else None
    if label is None:
        return None, None
    lines = [l.strip() for l in raw_output.splitlines() if l.strip()]
    reason = " ".join(lines[1:]).strip() if len(lines) > 1 else None
    return label, reason


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_qags_data(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    instances = data.get("instances", []) if isinstance(data, dict) else []
    out: list[dict[str, Any]] = []
    for ex in instances:
        try:
            prompt         = ex["instance"].strip()
            majority_human = normalize_label(ex["annotations"]["Factual Consistency"]["majority_human"])
            individual     = [normalize_label(v) for v in ex["annotations"]["Factual Consistency"]["individual_human_scores"]]
        except (KeyError, TypeError, AttributeError):
            continue
        if not prompt or majority_human is None:
            continue
        out.append({
            "sample_id":              ex.get("id"),
            "prompt":                 prompt,
            "majority_human":         majority_human,
            "individual_human_scores": individual,
        })
    return out


def build_prompt(example: dict[str, Any]) -> str:
    return (
        example["prompt"]
        + "\n\nPlease answer with only 'yes' or 'no'.\n"
          "You may add a short explanation on the next line."
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_bacc(rows: list[dict[str, Any]], key: str) -> float | None:
    tp = fp = tn = fn = 0
    for row in rows:
        gold = row.get("majority_human")
        pred = row.get(key)
        if pred not in {"yes", "no"} or gold not in {"yes", "no"}:
            continue
        if gold == "yes"  and pred == "yes":  tp += 1
        elif gold == "no" and pred == "yes":  fp += 1
        elif gold == "no" and pred == "no":   tn += 1
        elif gold == "yes" and pred == "no":  fn += 1
    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    tnr = tn / (tn + fp) if (tn + fp) > 0 else None
    return (tpr + tnr) / 2 if (tpr is not None and tnr is not None) else None


def compute_accuracy(rows: list[dict[str, Any]], key: str) -> float | None:
    correct = total = 0
    for row in rows:
        pred = row.get(key)
        gold = row.get("majority_human")
        if pred not in {"yes", "no"} or gold not in {"yes", "no"}:
            continue
        total += 1
        if pred == gold:
            correct += 1
    return correct / total if total > 0 else None


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def save_votes_csv(
    records: list[dict[str, Any]],
    models: list[str],
    dataset: str,
) -> None:
    """Save raw judge votes to results_tmp/qags/votes/votes_{dataset}.csv.

    Columns: sample_id, gold_label (1=yes/0=no), then one column per model (1/0/empty).
    Mirrors the format of results_tmp/memerag_ext/votes/votes_{lang}.csv.
    """
    label_to_int = {"yes": 1, "no": 0}
    votes_dir    = ROOT / "results_tmp" / "qags" / "votes"
    votes_dir.mkdir(parents=True, exist_ok=True)
    out_path     = votes_dir / f"votes_{dataset}.csv"

    fieldnames = ["sample_id", "gold_label"] + models
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            gold = label_to_int.get(rec.get("majority_human"), "")
            row: dict[str, Any] = {
                "sample_id":  rec.get("sample_id", ""),
                "gold_label": gold,
            }
            for m in models:
                lbl = (rec.get("model_outputs") or {}).get(m, {}).get("label")
                row[m] = label_to_int.get(lbl, "")
            writer.writerow(row)

    print(f"  Votes CSV    → {out_path.relative_to(ROOT)}  ({len(records)} samples × {len(models)} models)")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    try:
        tmp.replace(path)
    except PermissionError:
        ts = int(time.time())
        fallback = path.with_suffix(path.suffix + f".{ts}.bak")
        try:
            tmp.replace(fallback)
            print(f"[WARN] PermissionError on {path}; backup saved to {fallback}")
        except Exception as exc:
            print(f"[ERROR] Failed to save checkpoint: {exc}")
    except Exception as exc:
        print(f"[ERROR] Unexpected error saving JSON: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM judges on QAGS data.")
    parser.add_argument("--dataset",  choices=["cnndm", "xsum"], default="cnndm")
    parser.add_argument("--samples",  type=int, default=None,
                        help="Limit to first N examples (omit = all).")
    parser.add_argument("--all",      action="store_true",
                        help="Explicit shortcut for running all rows.")
    parser.add_argument("--sample-ids", type=int, nargs="+", default=None,
                        help="Re-run (or patch) specific sample IDs.")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                        help="Save checkpoint after this many processed samples.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignoring any existing output file.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override default output directory.")
    args = parser.parse_args()

    random.seed(SEED)

    data_file = ROOT / "data" / "qags" / f"{args.dataset}.json"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    all_data = load_qags_data(data_file)
    if not all_data:
        raise RuntimeError(f"No valid examples in {data_file}")

    requested_ids: set[int] = set(args.sample_ids) if args.sample_ids else set()

    # Determine candidate rows
    if requested_ids:
        candidate_data = [r for r in all_data if r["sample_id"] in requested_ids]
        if not candidate_data:
            raise ValueError(f"No samples found for IDs: {sorted(requested_ids)}")
    elif args.samples is not None and not args.all:
        candidate_data = all_data[:min(args.samples, len(all_data))]
    else:
        candidate_data = all_data

    output_dir  = args.output_dir or (ROOT / "results_tmp" / "qags" / args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qags_judgement.json"

    # Load existing for resume
    completed_records_dict: dict[int, dict[str, Any]] = {}
    if output_path.exists() and not args.no_resume:
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("records"), list):
                for r in existing["records"]:
                    if isinstance(r, dict) and r.get("sample_id") is not None:
                        completed_records_dict[r["sample_id"]] = r
        except Exception:
            pass

    # Per-sample: which models are still missing?
    remaining: list[tuple[dict[str, Any], list[str]]] = []
    for row in candidate_data:
        sid              = row["sample_id"]
        existing_outputs = completed_records_dict.get(sid, {}).get("model_outputs", {})
        if requested_ids and sid in requested_ids:
            missing = [
                m for m in MODELS
                if m not in existing_outputs
                or existing_outputs[m].get("label") is None
                or existing_outputs[m].get("error")
            ]
        else:
            missing = [m for m in MODELS if m not in existing_outputs]
        if missing:
            remaining.append((row, missing))

    print(f"Loaded      : {len(all_data)} QAGS examples ({args.dataset})")
    print(f"Candidates  : {len(candidate_data)}")
    print(f"Already done: {len(candidate_data) - len(remaining)}")
    print(f"Needs work  : {len(remaining)}")
    print(f"Output      : {output_path}")

    # Set up Google credentials for Vertex models
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
    os.environ["GOOGLE_CLOUD_PROJECT"]            = PROJECT_ID
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"]       = "True"

    genai = HttpOptions = None
    try:
        from google import genai         # type: ignore
        from google.genai.types import HttpOptions  # type: ignore
    except Exception as exc:
        needs_vertex = any(
            m not in ("MiniMax-M2.7", "GPT-OSS-120b", "Qwen3.6-35B", "Apertus-8B")
            for m in MODELS
        )
        if needs_vertex:
            print(f"[ERROR] Could not import google-genai: {exc}")
            return

    clients = init_clients(MODELS)

    def checkpoint() -> None:
        records = list(completed_records_dict.values())
        save_json(output_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset":   args.dataset,
            "data_file": str(data_file),
            "n_samples": len(records),
            "models":    MODELS,
            "records":   records,
        })

    try:
        for idx, (sample, missing_models) in enumerate(remaining, start=1):
            sid    = sample["sample_id"]
            prompt = build_prompt(sample)
            print(f"\n[{idx}/{len(remaining)}] sample_id={sid}  gold={sample['majority_human']}  "
                  f"running {len(missing_models)} model(s): {missing_models}")

            model_order = list(missing_models)
            random.shuffle(model_order)

            existing_record  = completed_records_dict.get(sid)
            merged_outputs: dict[str, dict[str, Any]] = dict(
                (existing_record or {}).get("model_outputs", {})
            )

            for model_name in model_order:
                raw_text, error = dispatch(model_name, prompt, clients, genai, HttpOptions)
                label, reason   = extract_label_and_reason(raw_text) if raw_text else (None, None)
                merged_outputs[model_name] = {
                    "label":                  label,
                    "reason":                 reason,
                    "raw":                    raw_text,
                    "error":                  error,
                    "correct_majority_human": label == sample["majority_human"] if label in {"yes", "no"} else None,
                }
                print(f"  {model_name} -> {label} ({'error' if error else 'ok'})")

            if existing_record is not None:
                existing_record["model_outputs"] = merged_outputs
                existing_order: list[str] = existing_record.get("model_order") or []
                for m in merged_outputs:
                    if m not in existing_order:
                        existing_order.append(m)
                existing_record["model_order"] = existing_order
            else:
                completed_records_dict[sid] = {
                    "sample_id":               sid,
                    "majority_human":          sample["majority_human"],
                    "individual_human_scores": sample["individual_human_scores"],
                    "prompt":                  sample["prompt"],
                    "model_order":             model_order,
                    "model_outputs":           merged_outputs,
                }

            if idx % max(args.checkpoint_every, 1) == 0 or idx == len(remaining):
                checkpoint()
                print(f"  [checkpoint] {len(completed_records_dict)}/{len(candidate_data)} saved")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving checkpoint...")
        checkpoint()
        return

    # Auto-retry: one extra pass for any errored model call
    retry_queue: list[tuple[dict[str, Any], list[str]]] = []
    for row in candidate_data:
        sid     = row["sample_id"]
        outputs = completed_records_dict.get(sid, {}).get("model_outputs", {})
        errored = [
            m for m in MODELS
            if isinstance(outputs.get(m), dict)
            and (outputs[m].get("label") is None or outputs[m].get("error"))
        ]
        if errored:
            retry_queue.append((row, errored))

    if retry_queue:
        n_err = sum(len(ms) for _, ms in retry_queue)
        print(f"\n[AUTO-RETRY] {n_err} errored call(s) across {len(retry_queue)} sample(s) — retrying once...")
        for retry_idx, (sample, error_models) in enumerate(retry_queue, start=1):
            sid    = sample["sample_id"]
            prompt = build_prompt(sample)
            print(f"\n  [{retry_idx}/{len(retry_queue)}] sample_id={sid} | retrying: {error_models}")
            existing_record = completed_records_dict[sid]
            merged_outputs  = dict(existing_record.get("model_outputs", {}))
            for model_name in error_models:
                raw_text, error = dispatch(model_name, prompt, clients, genai, HttpOptions)
                label, reason   = extract_label_and_reason(raw_text) if raw_text else (None, None)
                merged_outputs[model_name] = {
                    "label":                  label,
                    "reason":                 reason,
                    "raw":                    raw_text,
                    "error":                  error,
                    "correct_majority_human": label == sample["majority_human"] if label in {"yes", "no"} else None,
                }
                print(f"    {model_name} -> {label} ({'error' if error else 'ok'})")
            existing_record["model_outputs"] = merged_outputs
        checkpoint()
        print("  [auto-retry done]")

    # Final summary + metrics
    completed_records = list(completed_records_dict.values())
    for row in completed_records:
        for m, info in row.get("model_outputs", {}).items():
            row[f"__{m}"] = info.get("label")

    print(f"\n{'='*60}\nRESULTS ({args.dataset.upper()})\n{'='*60}")
    print(f"  {'Model':<44} {'BAcc':>7}  {'Acc':>7}")
    print(f"  {'-'*60}")
    metrics: dict[str, Any] = {}
    for m in MODELS:
        bacc = compute_bacc(completed_records, f"__{m}")
        acc  = compute_accuracy(completed_records, f"__{m}")
        metrics[m] = {"bacc": bacc, "accuracy": acc}
        bacc_s = "n/a" if bacc is None else f"{bacc*100:.2f}%"
        acc_s  = "n/a" if acc  is None else f"{acc*100:.2f}%"
        print(f"  {m:<44} {bacc_s:>7}  {acc_s:>7}")

    print(f"\n  Label dist: {Counter(r['majority_human'] for r in completed_records)}")
    print(f"  Saved to  : {output_path}")
    save_votes_csv(completed_records, MODELS, args.dataset)

    # Missing / error report
    total_records = len(completed_records)
    missing_per: dict[str, list] = {}
    error_per:   dict[str, list] = {}
    for r in completed_records:
        sid     = r.get("sample_id")
        outputs = r.get("model_outputs", {})
        for m in MODELS:
            if m not in outputs:
                missing_per.setdefault(m, []).append(sid)
            elif isinstance(outputs[m], dict) and (outputs[m].get("label") is None or outputs[m].get("error")):
                error_per.setdefault(m, []).append(sid)

    if missing_per:
        all_ids = sorted({sid for ids in missing_per.values() for sid in ids})
        print(f"\n[MISSING] Models absent from some records (out of {total_records}):")
        for m, ids in missing_per.items():
            print(f"  {m}: {len(ids)} missing — ids: {ids[:10]}{'...' if len(ids)>10 else ''}")
        print(f"\nRecover: python llm-experiments/qags/judge_qags.py --dataset {args.dataset} --sample-ids {' '.join(str(i) for i in all_ids)}")
    else:
        print(f"\n[OK] All {total_records} records have outputs for every model.")

    if error_per:
        all_ids = sorted({sid for ids in error_per.values() for sid in ids})
        print(f"\n[ERRORS] Models with null label or API error:")
        for m, ids in error_per.items():
            print(f"  {m}: {len(ids)} error(s) — ids: {ids[:10]}{'...' if len(ids)>10 else ''}")
        print(f"\nRetry: python llm-experiments/qags/judge_qags.py --dataset {args.dataset} --sample-ids {' '.join(str(i) for i in all_ids)}")
    else:
        print("[OK] No errors — all model responses have valid labels.")


if __name__ == "__main__":
    main()
