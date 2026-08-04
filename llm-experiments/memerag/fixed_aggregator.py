from __future__ import annotations

'''
Run a single fixed model as aggregator over saved MEMERAG judge outputs.
The aggregator reads all proposer outputs and makes a final label decision.

Usage:
    python fixed_aggregator.py --lang en --all
    python fixed_aggregator.py --lang en --all --model MiniMax-M2.7
    python fixed_aggregator.py --lang en --all --exclude gemini-2.5-flash-lite
    python fixed_aggregator.py --lang en --sample-ids 533 786 3168
'''

import argparse
import json
import os
import re
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

SERVICE_ACCOUNT_PATH = r"C:\Users\Deepali\Downloads\Thesis\Ivaxi LLMs\llm-juries-503009-babf6fe232e6.json"
PROJECT_ID = "llm-juries-503009"
DEFAULT_LOCATION = "global"
GOOGLE_GENAI_USE_VERTEXAI = "True"

# Default aggregator model — override at runtime with --model
AGGREGATOR_MODEL = "google/gemma-4-26b-a4b-it-maas"

DEFAULT_CHECKPOINT_EVERY = 5
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0

BLABLADOR_BASE_URL = "https://api.blablador.fz-juelich.de/v1/"
BLABLADOR_MAX_TOKENS = 6000

BLABLADOR_API_IDS: dict[str, str] = {
    "MiniMax-M2.7":  "01 - MiniMax-M2.7 - our best model as of April, 2026",
    "GPT-OSS-120b":  "01 - GPT-OSS-120b - an open model released by OpenAI in August 2025",
    "Qwen3.6-35B":   "08 - Qwen3.6-35B-A3B-FP8 - Multimodal model from Apr 2026",
    "Apertus-8B":    "15 - Apertus-8B-Instruct-2509 - A new swiss model from September 2025",
}

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "supported":
        return "Supported"
    if text in {"not supported", "unsupported"}:
        return "Not Supported"
    return None


def is_openai_model(model_name: str) -> bool:
    return model_name.startswith("gpt-")


def is_blablador_model(model_name: str) -> bool:
    return model_name in BLABLADOR_API_IDS


def model_location_for(model_name: str) -> str:
    if "meta" in model_name.lower():
        return "us-central1"
    return DEFAULT_LOCATION


def extract_label_and_reason(raw_output: str | None) -> tuple[str | None, str | None]:
    if not raw_output:
        return None, None
    answer_match = re.search(
        r"<Answer>\s*(supported|not supported)\s*</Answer>",
        raw_output, flags=re.IGNORECASE,
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
    reasoning_match = re.search(r"<Reasoning>(.*?)</Reasoning>", raw_output, flags=re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        return label, reasoning_match.group(1).strip()
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    return label, " ".join(lines[1:]).strip() if len(lines) > 1 else None


def compute_bacc(rows: list[dict[str, Any]], pred_key: str) -> float | None:
    pos_total = pos_correct = neg_total = neg_correct = 0
    for row in rows:
        gold = row.get("gold_label")
        pred = row.get(pred_key)
        if gold not in {"Supported", "Not Supported"} or pred not in {"Supported", "Not Supported"}:
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
    n = agree = gold_sup = gold_not = pred_sup = pred_not = 0
    for row in rows:
        gold = row.get("gold_label")
        pred = row.get(pred_key)
        if gold not in {"Supported", "Not Supported"} or pred not in {"Supported", "Not Supported"}:
            continue
        n += 1
        if gold == "Supported":
            gold_sup += 1
        else:
            gold_not += 1
        if pred == "Supported":
            pred_sup += 1
        else:
            pred_not += 1
        if gold == pred:
            agree += 1
    if n == 0:
        return None
    po = agree / n
    pe = (gold_sup / n) * (pred_sup / n) + (gold_not / n) * (pred_not / n)
    return (po - pe) / (1 - pe) if pe != 1.0 else (1.0 if po == 1.0 else 0.0)


# ---------------------------------------------------------------------------
# API call functions
# ---------------------------------------------------------------------------

def call_vertex_with_retry(
    genai: Any, HttpOptions: Any, model_name: str, prompt: str,
    max_retries: int = MAX_RETRIES, base_delay: float = RETRY_BASE_DELAY,
) -> tuple[str | None, str | None]:
    for attempt in range(max_retries):
        try:
            client = genai.Client(
                http_options=HttpOptions(api_version="v1"),
                vertexai=True,
                project=PROJECT_ID,
                location=model_location_for(model_name),
            )
            response = client.models.generate_content(model=model_name, contents=prompt)
            text = getattr(response, "text", None)
            if text is None:
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                return None, "Empty response after all retries"
            return text, None
        except Exception as exc:
            err = str(exc)
            is_rate_limited = "429" in err or "RESOURCE_EXHAUSTED" in err
            if attempt < max_retries - 1 and is_rate_limited:
                time.sleep(base_delay * (2 ** attempt))
                continue
            return None, err
    return None, "Max retries exceeded"


def call_openai_with_retry(
    client: Any, model_name: str, prompt: str,
    max_retries: int = MAX_RETRIES, base_delay: float = RETRY_BASE_DELAY,
) -> tuple[str | None, str | None]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,
                temperature=0,
            )
            text = response.choices[0].message.content
            if text is None:
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                return None, "Empty response after all retries"
            return text, None
        except Exception as exc:
            err = str(exc)
            if attempt < max_retries - 1 and ("429" in err or "rate_limit" in err.lower()):
                time.sleep(base_delay * (2 ** attempt))
                continue
            return None, err
    return None, "Max retries exceeded"


def call_blablador_with_retry(
    client: Any, api_id: str, prompt: str,
    max_retries: int = MAX_RETRIES, base_delay: float = RETRY_BASE_DELAY,
) -> tuple[str | None, str | None]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=api_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=BLABLADOR_MAX_TOKENS,
                temperature=0,
            )
            text = response.choices[0].message.content
            if text is None:
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                return None, "Empty response after all retries"
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text, None
        except Exception as exc:
            err = str(exc)
            if attempt < max_retries - 1 and ("429" in err or "rate_limit" in err.lower()):
                time.sleep(base_delay * (2 ** attempt))
                continue
            return None, err
    return None, "Max retries exceeded"


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_records_safe(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data.get("records"), list):
            return [r for r in data["records"] if isinstance(r, dict)]
    except Exception:
        pass
    return []


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    try:
        temp_path.replace(path)
    except PermissionError:
        fallback = path.with_suffix(f".{int(time.time())}.bak")
        temp_path.replace(fallback)
        print(f"[WARN] PermissionError on {path}; saved to {fallback}")
    except Exception as exc:
        print(f"[ERROR] Failed to save: {exc}")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_aggregator_prompt(record: dict[str, Any], exclude: str | None = None) -> str:
    context_texts = record.get("context_texts", [])
    context = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(context_texts))
    task_prompt = TASK_PROMPT_TEMPLATE.format(
        context=context,
        query=record.get("query", ""),
        answer_segment=record.get("answer_segment", ""),
    )
    model_outputs = record.get("model_outputs", {})
    judge_blocks: list[str] = []
    for idx, (model_name, judge) in enumerate(model_outputs.items(), start=1):
        if model_name == exclude:
            continue
        label = judge.get("label") if isinstance(judge, dict) else None
        reason = judge.get("reason") if isinstance(judge, dict) else None
        judge_blocks.append(
            f"{idx}. Model: {model_name}\n"
            f"   Label: {label}\n"
            f"   Reasoning: {reason}"
        )
    return PROMPT_TEMPLATE.format(task_prompt=task_prompt, judge_responses="\n".join(judge_blocks))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed model as aggregator over MEMERAG judge outputs.")
    parser.add_argument("--lang", type=str, choices=["en", "es", "de", "fr", "hi"], default=None)
    parser.add_argument("--model", type=str, default=None,
                        help=f"Aggregator model name. Defaults to AGGREGATOR_MODEL ({AGGREGATOR_MODEL}).")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--sample-id", type=int, default=None)
    parser.add_argument("--sample-ids", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Exclude a model from the proposer panel (LOO).")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    aggregator_model = args.model or AGGREGATOR_MODEL
    model_short = aggregator_model.split("/")[-1]

    requested_ids: set[int] = set()
    if args.sample_id is not None:
        requested_ids.add(args.sample_id)
    if args.sample_ids:
        requested_ids.update(args.sample_ids)

    if args.lang:
        lang_dir = ROOT / "results_tmp" / "memerag_ext" / args.lang
        if args.input is None:
            args.input = lang_dir / f"memerag_judgement_{args.lang}.json"
        if args.output is None:
            args.output = lang_dir / f"fixed_agg_{model_short}_{args.lang}.json"
    else:
        if args.input is None or args.output is None:
            raise ValueError("Provide --lang or both --input and --output.")

    output_path = args.output
    if args.exclude:
        excl_folder = "excl_" + args.exclude.split("/")[-1].lower().replace(".", "_").replace("-", "_")
        output_path = args.output.parent / excl_folder / args.output.name

    raw_records = load_records_safe(args.input)
    if not raw_records:
        raise RuntimeError(f"No valid records found in {args.input}")

    if requested_ids:
        raw_records = [r for r in raw_records if r.get("sample_id") in requested_ids]
        if not raw_records:
            raise RuntimeError(f"No records found for sample_id(s)={sorted(requested_ids)}")
    elif args.samples is not None and not args.all:
        raw_records = raw_records[: args.samples]

    # Resume
    resume_records: list[dict[str, Any]] = []
    if output_path.exists() and not args.no_resume:
        resume_records = load_records_safe(output_path)

    if requested_ids:
        resume_records = [r for r in resume_records if r.get("sample_id") not in requested_ids]

    processed_ids = {r.get("sample_id") for r in resume_records}
    remaining = [r for r in raw_records if r.get("sample_id") not in processed_ids]

    print(f"Loaded {len(raw_records)} records from {args.input}")
    print(f"Aggregator model : {aggregator_model}")
    print(f"Excluded proposer: {args.exclude or 'none'}")
    print(f"Already done     : {len(processed_ids)}")
    print(f"Remaining        : {len(remaining)}")
    print(f"Output           : {output_path}")

    # Init clients
    genai = HttpOptions = None
    openai_client = blablador_client = None

    if is_blablador_model(aggregator_model):
        from openai import OpenAI
        blablador_key = os.environ.get("BLABLADOR_API_KEY")
        if not blablador_key:
            raise ValueError("BLABLADOR_API_KEY not set")
        blablador_client = OpenAI(base_url=BLABLADOR_BASE_URL, api_key=blablador_key)
    elif is_openai_model(aggregator_model):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        openai_client = OpenAI(api_key=api_key)
    else:
        from google import genai as _genai
        from google.genai.types import HttpOptions as _HttpOptions
        genai, HttpOptions = _genai, _HttpOptions
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
        os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = GOOGLE_GENAI_USE_VERTEXAI

    completed_records = list(resume_records)

    def checkpoint() -> None:
        save_json(output_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_input": str(args.input),
            "aggregator_model": aggregator_model,
            "excluded_model": args.exclude,
            "n_samples": len(completed_records),
            "records": completed_records,
        })

    try:
        for index, record in enumerate(remaining, start=len(processed_ids) + 1):
            sample_id = record.get("sample_id")
            print(f"\n[{index}/{len(raw_records)}] sample_id={sample_id}")

            prompt = build_aggregator_prompt(record, exclude=args.exclude)

            if is_blablador_model(aggregator_model):
                api_id = BLABLADOR_API_IDS[aggregator_model]
                raw_text, error = call_blablador_with_retry(blablador_client, api_id, prompt)
            elif is_openai_model(aggregator_model):
                raw_text, error = call_openai_with_retry(openai_client, aggregator_model, prompt)
            else:
                raw_text, error = call_vertex_with_retry(genai, HttpOptions, aggregator_model, prompt)

            agg_label, agg_reason = extract_label_and_reason(raw_text) if raw_text else (None, None)
            print(f"  -> {agg_label} ({'error' if error else 'ok'})")

            result = dict(record)
            result.update({
                "aggregator_model": aggregator_model,
                "aggregator_prompt": prompt,
                "aggregator_label": agg_label,
                "aggregator_reason": agg_reason,
                "aggregator_raw": raw_text,
                "aggregator_error": error,
                "aggregator_correct_gold": (
                    agg_label == record.get("gold_label")
                    if agg_label in {"Supported", "Not Supported"} else None
                ),
                "__aggregator": agg_label,
            })
            completed_records.append(result)

            if index % max(args.checkpoint_every, 1) == 0 or index == len(raw_records):
                checkpoint()
                print(f"  [checkpoint] {len(completed_records)}/{len(raw_records)} saved")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving checkpoint...")
        checkpoint()
        return

    # Final save with metrics
    for row in completed_records:
        row["__aggregator"] = row.get("aggregator_label")

    bacc = compute_bacc(completed_records, "__aggregator")
    kappa = compute_cohen_kappa(completed_records, "__aggregator")

    save_json(output_path, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_input": str(args.input),
        "aggregator_model": aggregator_model,
        "excluded_model": args.exclude,
        "n_samples": len(completed_records),
        "metrics": {
            aggregator_model: {
                "balanced_accuracy": bacc,
                "cohen_kappa": kappa,
            }
        },
        "records": completed_records,
    })

    # Results table
    col = 42
    print(f"\nBalanced Accuracy / Cohen's Kappa:")
    print(f"  {'Model':<{col}} {'BAcc':>7}  {'Kappa':>7}")
    print("  " + "-" * (col + 18))
    print(f"  {aggregator_model:<{col}} {bacc:.4f if bacc else 'n/a':>7}  {kappa:.4f if kappa else 'n/a':>7}")

    print(f"\nSaved to: {output_path}")
    print(f"Label distribution: {Counter(r.get('gold_label') for r in completed_records)}")

    # Error summary
    failed = {
        str(r.get("sample_id", "")): r.get("aggregator_error", "null label")
        for r in completed_records
        if r.get("aggregator_label") is None or r.get("aggregator_error")
    }
    if failed:
        print(f"\n[ERRORS] {len(failed)} sample(s) with errors or null labels:")
        for sid, err in failed.items():
            print(f"  sample_id={sid}: {err}")
        print(f"\nRetry with:")
        print(f"  python fixed_aggregator.py --lang {args.lang} --model {aggregator_model} --sample-ids {' '.join(failed.keys())}")
    else:
        print("\n[OK] All samples completed successfully.")


if __name__ == "__main__":
    main()
