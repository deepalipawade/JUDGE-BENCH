from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# No need to pass --dataset argument, give appropriate json PATH

ROOT = Path(__file__).resolve().parents[2]
# INPUT_JSON_PATH = ROOT / "results_tmp" / "qags_cnndm" / "qags_judgement.json"
# SAVE_JSON_PATH = ROOT / "results_tmp" / "qags_cnndm" / "meta_aggregator.json"

INPUT_JSON_PATH = ROOT / "results_tmp" / "qags_xsum" / "qags_judgement.json"
SAVE_JSON_PATH = ROOT / "results_tmp" / "qags_xsum" / "gemini_aggregator.json"

SERVICE_ACCOUNT_PATH = r"C:\Users\Deepali\Downloads\llm-juries-dd7439c15063.json"
PROJECT_ID = "llm-juries"
DEFAULT_LOCATION = "us-central1"
GOOGLE_GENAI_USE_VERTEXAI = "True"

# AGGREGATOR_MODEL = "meta/llama-3.3-70b-instruct-maas"
AGGREGATOR_MODEL = "gemini-2.5-flash-lite"
DEFAULT_CHECKPOINT_EVERY = 5


PROMPT_TEMPLATE = (
    "You are a strict factual-consistency evaluator for QAGS.\n"
    "Task: Decide whether the sentence is supported by the article.\n\n"
    "Use the provided judge outputs as evidence, but do not blindly follow them.\n"
    "If the judges disagree, prefer the reasoning that is most directly grounded in the article and sentence.\n\n"
    "Output format:\n"
    "<Answer>yes</Answer> or <Answer>no</Answer>\n"
    "<Reasoning>Your concise reasoning here</Reasoning>\n\n"
    "Where:\n"
    "yes = supported by the article\n"
    "no = not supported by the article\n\n"
    "ARTICLE / SENTENCE PROMPT:\n{prompt}\n\n"
    "JUDGE RESPONSES:\n{judge_responses}\n"
)


def normalize_label(text: Any) -> str | None:
    if text is None:
        return None
    value = str(text).strip().lower()
    if value in {"yes", "supported", "true"}:
        return "yes"
    if value in {"no", "unsupported", "false"}:
        return "no"
    return None


def extract_label_and_reason(raw_output: str | None) -> tuple[str | None, str | None]:
    if not raw_output:
        return None, None

    answer_match = re.search(r"<Answer>\s*(yes|no)\s*</Answer>", raw_output, flags=re.IGNORECASE)
    if answer_match:
        label = answer_match.group(1).lower()
    else:
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if not lines:
            return None, None
        first_line = lines[0]
        match = re.search(r"\b(yes|no)\b", first_line, flags=re.IGNORECASE)
        label = match.group(1).lower() if match else None

    if label is None:
        return None, None

    reasoning_match = re.search(r"<Reasoning>(.*?)</Reasoning>", raw_output, flags=re.DOTALL)
    if reasoning_match:
        return label, reasoning_match.group(1).strip()

    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if len(lines) > 1:
        return label, " ".join(lines[1:]).strip()

    return label, None


def model_location_for(model_name: str) -> str:
    if "meta" in model_name.lower():
        return "us-central1"
    return DEFAULT_LOCATION


def load_input(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def try_full_load(path: Path) -> dict[str, Any] | None:
    try:
        return load_input(path)
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
        if i >= n:
            break
        if content[i] != "{":
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
                    obj_text = content[start:i]
                    try:
                        yield json.loads(obj_text)
                    except Exception:
                        pass
                    break
            i += 1


def load_records_partial_safe(path: Path) -> list[dict[str, Any]]:
    full = try_full_load(path)
    if full is not None and isinstance(full, dict) and isinstance(full.get("records"), list):
        return [record for record in full["records"] if isinstance(record, dict)]

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        content = handle.read()

    return [record for record in iter_records_from_partial(content) if isinstance(record, dict)]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    try:
        temp_path.replace(path)
    except PermissionError as exc:
        timestamp = int(time.time())
        fallback = path.with_suffix(path.suffix + f".{timestamp}.bak")
        try:
            temp_path.replace(fallback)
            print(f"[WARN] Could not replace {path!s} due to PermissionError; saved backup to {fallback!s}")
        except Exception as exc2:
            print(f"[ERROR] Failed to save checkpoint file: {exc2}")
    except Exception as exc:
        print(f"[ERROR] Unexpected error while saving JSON checkpoint: {exc}")


def build_aggregator_prompt(sample: dict[str, Any]) -> str:
    judge_blocks: list[str] = []
    model_outputs = sample.get("model_outputs", {})
    for index, model_name in enumerate(model_outputs.keys(), start=1):
        judge = model_outputs.get(model_name, {})
        judge_blocks.append(
            f"{index}. Model: {model_name}\n"
            f"   Label: {judge.get('label')}\n"
            f"   Reasoning: {judge.get('reason')}"
        )

    return PROMPT_TEMPLATE.format(
        prompt=sample.get("prompt", ""),
        judge_responses="\n".join(judge_blocks),
    )


def build_records(source: dict[str, Any]) -> list[dict[str, Any]]:
    records = source.get("records", []) if isinstance(source, dict) else []
    processed: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue

        sample_id = record.get("sample_id")
        prompt = record.get("prompt")
        majority_human = normalize_label(record.get("majority_human"))
        individual_human_scores = [normalize_label(v) for v in record.get("individual_human_scores", [])]
        model_outputs = record.get("model_outputs", {})

        if sample_id is None or not prompt or majority_human is None or not isinstance(model_outputs, dict):
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
                "prompt": prompt,
                "majority_human": majority_human,
                "individual_human_scores": individual_human_scores,
                "model_outputs": cleaned_outputs,
            }
        )

    return processed


def compute_accuracy(rows: list[dict[str, Any]], key: str) -> float | None:
    correct = 0
    total = 0
    for row in rows:
        pred = row.get(key)
        gold = row.get("majority_human")
        if pred not in {"yes", "no"} or gold not in {"yes", "no"}:
            continue
        total += 1
        if pred == gold:
            correct += 1
    if total == 0:
        return None
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Meta as an aggregator over saved QAGS judge outputs.")
    parser.add_argument("--input", type=Path, default=INPUT_JSON_PATH, help="Path to the saved QAGS judge outputs JSON.")
    parser.add_argument("--output", type=Path, default=SAVE_JSON_PATH, help="Path to write Meta aggregator results.")
    parser.add_argument("--samples", type=int, default=None, help="Optional limit on number of samples to process. Omit to run all.")
    parser.add_argument("--sample-id", type=int, default=None, help="Run only one specific sample_id (overrides --samples).")
    parser.add_argument(
        "--sample-ids",
        type=int,
        nargs="+",
        default=None,
        help="Run a list of sample_id values (overrides --samples). Example: --sample-ids 17 19 37",
    )
    parser.add_argument("--all", action="store_true", help="Shortcut for running all available rows.")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY, help="Write a checkpoint after this many processed samples.")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh instead of resuming from an existing output file.")
    args = parser.parse_args()

    requested_sample_ids: set[int] = set()
    if args.sample_id is not None:
        requested_sample_ids.add(args.sample_id)
    if args.sample_ids:
        requested_sample_ids.update(args.sample_ids)

    records = build_records({"records": load_records_partial_safe(args.input)})
    if not records:
        raise RuntimeError(f"No valid records found in {args.input}")

    if requested_sample_ids:
        records = [record for record in records if record.get("sample_id") in requested_sample_ids]
        if not records:
            raise RuntimeError(f"No valid record found for sample_id(s)={sorted(requested_sample_ids)} in {args.input}")

    if args.samples is not None and args.samples >= 0 and not args.all:
        records = records[: args.samples]

    resume_records: list[dict[str, Any]] = []
    if args.output.exists() and not args.no_resume:
        try:
            existing = load_input(args.output)
            existing_records = existing.get("records", []) if isinstance(existing, dict) else []
            if isinstance(existing_records, list):
                resume_records = [record for record in existing_records if isinstance(record, dict)]
        except Exception:
            resume_records = []

    if requested_sample_ids:
        resume_records = [record for record in resume_records if record.get("sample_id") not in requested_sample_ids]

    processed_by_id = {record.get("sample_id") for record in resume_records}
    remaining_records = [record for record in records if record.get("sample_id") not in processed_by_id]

    print(f"Loaded {len(records)} QAGS records from {args.input}")
    print(f"Meta aggregator model: {AGGREGATOR_MODEL}")
    print(f"Already completed: {len(processed_by_id)}")
    print(f"Remaining to process: {len(remaining_records)}")
    print(f"Writing JSON results to {args.output}")
    if requested_sample_ids:
        if len(requested_sample_ids) == 1:
            print(f"Single-sample rerun mode enabled for sample_id={next(iter(requested_sample_ids))}")
        else:
            print(f"Multi-sample rerun mode enabled for sample_ids={sorted(requested_sample_ids)}")

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
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_input": str(args.input),
            "output_file": str(args.output),
            "aggregator_model": AGGREGATOR_MODEL,
            "checkpoint_every": args.checkpoint_every,
            "n_source_records": len(records),
            "n_completed": len(completed_records),
            "records": completed_records,
        }
        save_json(args.output, payload)

    try:
        for index, sample in enumerate(remaining_records, start=len(processed_by_id) + 1):
            remaining = len(records) - index
            sample_id = sample.get("sample_id")
            print(f"[{index}/{len(records)}] sample_id={sample_id} | remaining={remaining}")

            agg_prompt = build_aggregator_prompt(sample)
            try:
                response = agg_client.models.generate_content(model=AGGREGATOR_MODEL, contents=agg_prompt)
                agg_raw = getattr(response, "text", None)
                agg_label, agg_reason = extract_label_and_reason(agg_raw)
                agg_error = None
            except Exception as exc:
                agg_raw = None
                agg_label = None
                agg_reason = None
                agg_error = str(exc)

            print(f"  Meta aggregator -> {agg_label} ({'error' if agg_error else 'ok'})")

            result_record = dict(sample)
            result_record.update(
                {
                    "meta_aggregator_model": AGGREGATOR_MODEL,
                    "meta_aggregator_prompt": agg_prompt,
                    "meta_aggregator_label": agg_label,
                    "meta_aggregator_reason": agg_reason,
                    "meta_aggregator_raw": agg_raw,
                    "meta_aggregator_error": agg_error,
                    "meta_aggregator_correct_majority_human": agg_label == sample.get("majority_human") if agg_label in {"yes", "no"} else None,
                }
            )
            completed_records.append(result_record)

            if len(completed_records) % max(args.checkpoint_every, 1) == 0 or index == len(records):
                checkpoint()
                print(f"  [checkpoint] saved {len(completed_records)}/{len(records)} samples to {args.output}")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving checkpoint before exit...")
        checkpoint()
        print(f"Saved {len(completed_records)}/{len(records)} samples to {args.output}")
        return

    checkpoint()

    for row in completed_records:
        row["__meta_aggregator"] = row.get("meta_aggregator_label")

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source_input": str(args.input),
        "output_file": str(args.output),
        "aggregator_model": AGGREGATOR_MODEL,
        "n_samples": len(completed_records),
        "records": completed_records,
        "metrics": {
            "meta_aggregator": {
                "accuracy_vs_majority_human": compute_accuracy(completed_records, "__meta_aggregator"),
            }
        },
    }

    save_json(args.output, summary)

    acc = summary["metrics"]["meta_aggregator"]["accuracy_vs_majority_human"]
    acc_text = "n/a" if acc is None else f"{acc:.4f}"
    print("\nMeta aggregator accuracy vs majority_human:")
    print(f"  {AGGREGATOR_MODEL}: {acc_text}")
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
