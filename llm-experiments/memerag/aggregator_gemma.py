from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

'''
python aggregator_gemma.py --lang en --all --exclude gemini-2.5-flash-lite --no-resume
'''

ROOT = Path(__file__).resolve().parents[2]
INPUT_JSON_PATH = ROOT / "results_tmp" / "memerag_ext" / "en" / "memerag_judgement_en.json"
SAVE_JSON_PATH  = ROOT / "results_tmp" / "memerag_ext" / "en" / "aggregator_gemma_en.json"

# Maps model ID to its LOO subfolder name — add new models here for future LOO runs
EXCL_SUBDIR: dict[str, str] = {
    "gemini-2.5-flash-lite":            "excl_gemini_lite",
    "gemini-2.5-flash":                 "excl_gemini_flash",
    "google/gemma-4-26b-a4b-it-maas":  "excl_gemma",
    "meta/llama-3.3-70b-instruct-maas": "excl_llama",
    "gpt-5.4-mini":                     "excl_gpt_mini",
    "gpt-5.4-mini-2026-03-17":          "excl_gpt_mini_march",
}

SERVICE_ACCOUNT_PATH = r"C:\Users\Deepali\Downloads\llm-juries-dd7439c15063.json"
PROJECT_ID = "llm-juries"
DEFAULT_LOCATION = "global"
GOOGLE_GENAI_USE_VERTEXAI = "True"

AGGREGATOR_MODEL = "google/gemma-4-26b-a4b-it-maas"
DEFAULT_CHECKPOINT_EVERY = 5

PROMPT_TEMPLATE = (
    "You are a strict factual-consistency evaluator.\n"
    "Task: Decide whether the answer segment is fully supported by the evidence passages.\n\n"
    "Use the provided judge outputs as evidence, but do not blindly follow them.\n"
    "If the judges disagree, prefer the reasoning most directly grounded in the passages.\n\n"
    "Output format:\n"
    "<Answer>Supported</Answer> or <Answer>Not Supported</Answer>\n"
    "<Reasoning>Your concise reasoning here</Reasoning>\n\n"
    "ORIGINAL TASK:\n{task_prompt}\n\n"
    "JUDGE RESPONSES:\n{judge_responses}\n"
)

TASK_PROMPT_TEMPLATE = (
    "Evidence Passages:\n{context}\n\n"
    "Question:\n{query}\n\n"
    "Answer Segment:\n{answer_segment}"
)


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "supported":
        return "Supported"
    if text in {"not supported", "unsupported"}:
        return "Not Supported"
    return None


def model_location_for(model_name: str) -> str:
    if "meta" in model_name.lower():
        return "us-central1"
    return DEFAULT_LOCATION


def extract_label_and_reason(raw_output: str | None) -> tuple[str | None, str | None]:
    if not raw_output:
        return None, None

    answer_match = re.search(
        r"<Answer>\s*(supported|not supported)\s*</Answer>",
        raw_output,
        flags=re.IGNORECASE,
    )
    if answer_match:
        label = normalize_label(answer_match.group(1))
    else:
        lowered = raw_output.lower()
        if "not supported" in lowered:
            label = "Not Supported"
        elif "supported" in lowered:
            label = "Supported"
        else:
            label = None

    if label is None:
        return None, None

    reasoning_match = re.search(
        r"<Reasoning>(.*?)</Reasoning>", raw_output, flags=re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        return label, reasoning_match.group(1).strip()

    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if len(lines) > 1:
        return label, " ".join(lines[1:]).strip()

    return label, None


def compute_bacc(rows: list[dict[str, Any]], pred_key: str) -> float | None:
    pos_total = pos_correct = neg_total = neg_correct = 0
    for row in rows:
        gold = row.get("gold_label")
        pred = row.get(pred_key)
        if gold not in {"Supported", "Not Supported"}:
            continue
        if pred not in {"Supported", "Not Supported"}:
            continue
        if gold == "Supported":
            pos_total += 1
            if pred == "Supported":
                pos_correct += 1
        else:
            neg_total += 1
            if pred == "Not Supported":
                neg_correct += 1
    if pos_total == 0 or neg_total == 0:
        return None
    return 0.5 * (pos_correct / pos_total + neg_correct / neg_total)


def compute_cohen_kappa(rows: list[dict[str, Any]], pred_key: str) -> float | None:
    labels = ["Supported", "Not Supported"]
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = 0
    agree = 0
    gold_counts: Counter = Counter()
    pred_counts: Counter = Counter()

    for row in rows:
        gold = row.get("gold_label")
        pred = row.get(pred_key)
        if gold not in label_to_idx or pred not in label_to_idx:
            continue
        n += 1
        gold_counts[gold] += 1
        pred_counts[pred] += 1
        if gold == pred:
            agree += 1

    if n == 0:
        return None

    po = agree / n
    pe = sum((gold_counts[label] / n) * (pred_counts[label] / n) for label in labels)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def try_full_load(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def iter_records_from_partial(content: str):
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
        while i < n and content[i] in " \t\r\n,":
            i += 1
        if i >= n or content[i] != "{":
            break
        start = i
        depth = 0
        while i < n:
            ch = content[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    try:
                        yield json.loads(content[start:i])
                    except Exception:
                        pass
                    break
            i += 1


def load_records_safe(path: Path) -> list[dict[str, Any]]:
    full = try_full_load(path)
    if full is not None and isinstance(full.get("records"), list):
        return [r for r in full["records"] if isinstance(r, dict)]
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        content = handle.read()
    return [r for r in iter_records_from_partial(content) if isinstance(r, dict)]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    try:
        temp_path.replace(path)
    except PermissionError:
        timestamp = int(time.time())
        fallback = path.with_suffix(path.suffix + f".{timestamp}.bak")
        try:
            temp_path.replace(fallback)
            print(f"[WARN] PermissionError on {path}; backup saved to {fallback}")
        except Exception as exc:
            print(f"[ERROR] Failed to save checkpoint: {exc}")
    except Exception as exc:
        print(f"[ERROR] Unexpected error saving JSON: {exc}")


def build_aggregator_prompt(record: dict[str, Any], exclude: str | None = None) -> str:
    context_texts = record.get("context_texts", [])
    context = "\n".join(
        f"{i + 1}. {text}" for i, text in enumerate(context_texts)
    )
    task_prompt = TASK_PROMPT_TEMPLATE.format(
        context=context,
        query=record.get("query", ""),
        answer_segment=record.get("answer_segment", ""),
    )

    model_outputs = record.get("model_outputs", {})
    judge_blocks: list[str] = []
    for idx, (model_name, judge) in enumerate(model_outputs.items(), start=1):
        # Skip excluded model — it is neither a proposer nor visible to the aggregator
        if model_name == exclude:
            continue
        label = judge.get("label") if isinstance(judge, dict) else None
        reason = judge.get("reason") if isinstance(judge, dict) else None
        judge_blocks.append(
            f"{idx}. Model: {model_name}\n"
            f"   Label: {label}\n"
            f"   Reasoning: {reason}"
        )

    return PROMPT_TEMPLATE.format(
        task_prompt=task_prompt,
        judge_responses="\n".join(judge_blocks),
    )


def build_records(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    for record in raw_records:
        if not isinstance(record, dict):
            continue
        sample_id = record.get("sample_id")
        gold_label = normalize_label(record.get("gold_label"))
        model_outputs = record.get("model_outputs", {})
        if sample_id is None or gold_label is None or not isinstance(model_outputs, dict):
            continue

        cleaned_outputs: dict[str, dict[str, Any]] = {}
        for model_name, info in model_outputs.items():
            if not isinstance(info, dict):
                continue
            cleaned_outputs[model_name] = {
                "label": normalize_label(info.get("label")),
                "reason": info.get("reason"),
                "raw": info.get("raw"),
                "error": info.get("error"),
                "location": info.get("location"),
            }

        processed.append(
            {
                "sample_id": sample_id,
                "query_id": record.get("query_id"),
                "sentence_id": record.get("sentence_id"),
                "query": record.get("query"),
                "answer_segment": record.get("answer_segment"),
                "context_texts": record.get("context_texts", []),
                "gold_label": gold_label,
                "gold_labels_all": record.get("gold_labels_all", []),
                "model_outputs": cleaned_outputs,
                "panel_majority": normalize_label(record.get("panel_majority")),
            }
        )
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemma as an aggregator over saved MEMERAG judge outputs."
    )
    parser.add_argument(
        "--lang", type=str, choices=["en", "es", "de", "fr", "hi"], default=None,
        help="Language shortcut — auto-sets --input and --output using ROOT paths.",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to saved MEMERAG judge outputs JSON. Overrides --lang.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Base path to write aggregator results. Overrides --lang. Excl suffix added automatically.",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Limit number of samples to process. Omit to run all.",
    )
    parser.add_argument(
        "--sample-id", type=int, default=None,
        help="Run only one specific sample_id.",
    )
    parser.add_argument(
        "--sample-ids", type=int, nargs="+", default=None,
        help="Run a list of sample_id values. Example: --sample-ids 786 3351",
    )
    parser.add_argument("--all", action="store_true", help="Run all available rows.")
    parser.add_argument(
        "--exclude", type=str, default=None,
        help="Exclude a model from the proposer panel (leave-one-out). E.g. --exclude gemini-2.5-flash-lite",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
        help="Write a checkpoint after this many processed samples.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh instead of resuming from an existing output file.",
    )
    args = parser.parse_args()

    requested_ids: set[int] = set()
    if args.sample_id is not None:
        requested_ids.add(args.sample_id)
    if args.sample_ids:
        requested_ids.update(args.sample_ids)

    # Resolve input/output from --lang if explicit paths not given
    if args.lang:
        lang_dir = ROOT / "results_tmp" / "memerag_ext" / args.lang
        if args.input is None:
            args.input = lang_dir / f"memerag_judgement_{args.lang}.json"
        if args.output is None:
            args.output = lang_dir / f"aggregator_gemma_{args.lang}.json"
    else:
        if args.input is None:
            args.input = INPUT_JSON_PATH
        if args.output is None:
            args.output = SAVE_JSON_PATH

    raw_records = load_records_safe(args.input)
    records = build_records(raw_records)
    if not records:
        raise RuntimeError(f"No valid records found in {args.input}")

    if requested_ids:
        records = [r for r in records if r.get("sample_id") in requested_ids]
        if not records:
            raise RuntimeError(f"No records found for sample_id(s)={sorted(requested_ids)}")

    if args.samples is not None and not args.all:
        records = records[: args.samples]

    # Compute final output path first so resume check reads the correct file
    output_path = args.output
    if args.exclude:
        excl_folder = EXCL_SUBDIR.get(args.exclude, "excl_" + args.exclude.split("/")[-1].replace(".", "_"))
        output_path = args.output.parent / excl_folder / args.output.name

    resume_records: list[dict[str, Any]] = []
    if output_path.exists() and not args.no_resume:
        try:
            existing = try_full_load(output_path)
            if existing and isinstance(existing.get("records"), list):
                resume_records = [r for r in existing["records"] if isinstance(r, dict)]
        except Exception:
            resume_records = []

    if requested_ids:
        resume_records = [r for r in resume_records if r.get("sample_id") not in requested_ids]

    processed_ids = {r.get("sample_id") for r in resume_records}
    remaining = [r for r in records if r.get("sample_id") not in processed_ids]

    print(f"Loaded {len(records)} MEMERAG records from {args.input}")
    print(f"Aggregator model: {AGGREGATOR_MODEL}")
    print(f"Excluded proposer: {args.exclude or 'none'}")
    print(f"Already completed: {len(processed_ids)}")
    print(f"Remaining to process: {len(remaining)}")
    print(f"Writing JSON results to {output_path}")

    try:
        from google import genai
        from google.genai.types import HttpOptions
    except Exception as exc:
        print(f"[ERROR] Could not import google-genai: {exc}")
        return

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = GOOGLE_GENAI_USE_VERTEXAI

    agg_client = genai.Client(
        http_options=HttpOptions(api_version="v1"),
        vertexai=True,
        project=PROJECT_ID,
        location=model_location_for(AGGREGATOR_MODEL),
    )

    completed_records = list(resume_records)

    def checkpoint() -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_input": str(args.input),
            "output_file": str(output_path),
            "aggregator_model": AGGREGATOR_MODEL,
            "checkpoint_every": args.checkpoint_every,
            "n_source_records": len(records),
            "n_completed": len(completed_records),
            "records": completed_records,
        }
        save_json(output_path, payload)

    try:
        for index, sample in enumerate(remaining, start=len(processed_ids) + 1):
            sample_id = sample.get("sample_id")
            print(f"[{index}/{len(records)}] sample_id={sample_id}")

            agg_prompt = build_aggregator_prompt(sample, exclude=args.exclude)
            agg_raw = agg_label = agg_reason = agg_error = None
            for attempt in range(3):
                try:
                    response = agg_client.models.generate_content(
                        model=AGGREGATOR_MODEL, contents=agg_prompt
                    )
                    text = getattr(response, "text", None)
                    if text is None:
                        if attempt < 2:
                            print(f"  [RETRY] Empty response. Retrying {attempt + 2}/3...")
                            time.sleep(2 ** attempt)
                            continue
                        agg_error = "Empty response: model returned no text after all retries"
                        break
                    agg_raw = text
                    agg_label, agg_reason = extract_label_and_reason(agg_raw)
                    break
                except Exception as exc:
                    agg_error = str(exc)
                    break

            print(f"  gemma aggregator -> {agg_label} ({'error' if agg_error else 'ok'})")

            result_record = dict(sample)
            result_record.update(
                {
                    "aggregator_model": AGGREGATOR_MODEL,
                    "aggregator_prompt": agg_prompt,
                    "aggregator_label": agg_label,
                    "aggregator_reason": agg_reason,
                    "aggregator_raw": agg_raw,
                    "aggregator_error": agg_error,
                    "aggregator_correct_gold": (
                        agg_label == sample.get("gold_label")
                        if agg_label in {"Supported", "Not Supported"}
                        else None
                    ),
                }
            )
            completed_records.append(result_record)

            if len(completed_records) % max(args.checkpoint_every, 1) == 0 or index == len(records):
                checkpoint()
                print(f"  [checkpoint] {len(completed_records)}/{len(records)} saved")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving checkpoint before exit...")
        checkpoint()
        print(f"Saved {len(completed_records)}/{len(records)} samples to {output_path}")
        return

    checkpoint()

    for row in completed_records:
        row["__aggregator"] = row.get("aggregator_label")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_input": str(args.input),
        "output_file": str(output_path),
        "aggregator_model": AGGREGATOR_MODEL,
        "excluded_model": args.exclude,
        "n_samples": len(completed_records),
        "records": completed_records,
        "metrics": {
            "gemma_aggregator": {
                "balanced_accuracy": compute_bacc(completed_records, "__aggregator"),
                "cohen_kappa": compute_cohen_kappa(completed_records, "__aggregator"),
            }
        },
    }

    save_json(output_path, summary)
    print(f"Saved results to: {output_path}")
    print(f"Records saved: {len(completed_records)}")


if __name__ == "__main__":
    main()
