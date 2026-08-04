'''
Command args to run:
python judge_memerag.py --lang en --all
python judge_memerag.py --lang es --all
python judge_memerag.py --lang de --all
python judge_memerag.py --lang fr --all
python judge_memerag.py --lang hi --all
'''

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
DATA_ROOT = ROOT / "MEMERAG-main" / "data"

SERVICE_ACCOUNT_PATH = r"C:\Users\Deepali\Downloads\Thesis\Ivaxi LLMs\llm-juries-503009-babf6fe232e6.json"
PROJECT_ID = "llm-juries-503009"
DEFAULT_LOCATION = "global"
GOOGLE_GENAI_USE_VERTEXAI = "True"

BLABLADOR_BASE_URL = "https://api.blablador.fz-juelich.de/v1/"
BLABLADOR_MAX_TOKENS = 6000  # high enough for thinking models (MiniMax, Qwen3.x)

# Maps clean display name (used everywhere: JSON, CSV, keys) → full Blablador API ID
BLABLADOR_API_IDS: dict[str, str] = {
    "MiniMax-M2.7":  "01 - MiniMax-M2.7 - our best model as of April, 2026",
    "GPT-OSS-120b":  "01 - GPT-OSS-120b - an open model released by OpenAI in August 2025",
    "Qwen3.5-122B":  "02 - Qwen3.5-122B-A10B-FP8, general purpose large model",
    "Qwen3.6-35B":   "08 - Qwen3.6-35B-A3B-FP8 - Multimodal model from Apr 2026",
    "Apertus-8B":    "15 - Apertus-8B-Instruct-2509 - A new swiss model from September 2025",
}

MODELS = [
    "meta/llama-3.3-70b-instruct-maas",
    "google/gemma-4-26b-a4b-it-maas",
    "gemini-2.5-flash",
    # "gemini-2.5-flash-lite",
    "gpt-5.4-mini",
    # "gpt-5.4-mini-2026-03-17",
    "MiniMax-M2.7",
    "GPT-OSS-120b",
    "Qwen3.6-35B",
    "Apertus-8B",
    # "Qwen3.5-122B",  # Not used because of connection issues

]

# Models excluded from accuracy tables, best-model selection, and missing/error analysis.
# Add or remove model names here to control what appears in the output.
IGNORE_MODELS: set[str] = {
    "gpt-5.4-mini",
    "gpt-5.4-mini-2026-03-17",
}

SEED = 42
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
DEFAULT_CHECKPOINT_EVERY = 5

SYSTEM_PROMPT = (
    "Given a set of evidence passages, and an answer, determine if the answer is fully supported by the evidence passages or not. "
    "Analyze each sentence of the answer carefully and verify that all information it contains is explicitly stated in or can be directly inferred from the evidence passages.\n\n"
    'Output "Not Supported" if ANY of the following are true:\n\n'
    "The answer contains any information not explicitly stated in or directly inferable from the passages.\n"
    "The answer contradicts any information in the passages.\n"
    "The answer introduces any new information not found in the passages.\n"
    "The answer misrepresents or inaccurately paraphrases information from the passages.\n"
    "The answer draws conclusions not logically supported by the given information.\n"
    "The answer changes the level of certainty, specificity, or nuance from what is expressed in the passages.\n"
    "The answer does not directly address the specific aspect asked about in the question.\n"
    "The answer conflates or misrepresents separate pieces of information when summarizing multiple passages.\n\n"
    "Output Supported otherwise.\n\n"
    "write your reasoning in between the tags <rationale></rationale> and Provide your final answer in <answer></answer> tags."
)

TASK_PROMPT = (
    "Evidence Passages:\n\n"
    "{context}\n\n"
    "Question:\n"
    "{query}\n\n"
    "Answer:\n"
    "{answer_segment}\n\n"
    'Now provided your label directly as "Supported" or "Not Supported".'
)


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower()
    if label == "supported":
        return "Supported"
    if label == "not supported":
        return "Not Supported"
    return None


def normalize_annotation_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def majority_vote_label(values: list[Any]) -> str | None:
    labels = [normalize_label(value) for value in values]
    valid = [label for label in labels if label in {"Supported", "Not Supported"}]
    if not valid:
        return None
    counts = Counter(valid)
    if counts["Supported"] == counts["Not Supported"]:
        return None
    return counts.most_common(1)[0][0]


def majority_vote_label_from_normalized(values: list[str | None]) -> str | None:
    valid = [label for label in values if label in {"Supported", "Not Supported"}]
    if not valid:
        return None
    counts = Counter(valid)
    if counts["Supported"] == counts["Not Supported"]:
        return None
    return counts.most_common(1)[0][0]


def model_location_for(model_name: str) -> str:
    if "meta" in model_name.lower():
        return "us-central1"
    return DEFAULT_LOCATION


def is_openai_model(model_name: str) -> bool:
    return model_name.startswith("gpt-")


def is_blablador_model(model_name: str) -> bool:
    return model_name in BLABLADOR_API_IDS


def load_memerag_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            query_id = record.get("query_id")
            query = str(record.get("query", "")).strip()
            context = record.get("context", [])
            answers = record.get("answer", [])

            if not query_id or not query or not isinstance(context, list) or not isinstance(answers, list):
                continue

            context_texts = []
            for passage in context:
                if isinstance(passage, dict):
                    text = str(passage.get("text", "")).strip()
                    if text:
                        context_texts.append(text)

            for s_idx, answer in enumerate(answers):
                if not isinstance(answer, dict):
                    continue
                sentence = str(answer.get("sentence", "")).strip()
                if not sentence:
                    continue

                factuality_all = normalize_annotation_list(answer.get("factuality"))
                relevance_all = normalize_annotation_list(answer.get("relevance"))
                fine_grained_all = normalize_annotation_list(answer.get("fine_grained_factuality"))
                comments_all = normalize_annotation_list(answer.get("comments"))

                gold_label = normalize_label(answer.get("factuality"))
                if gold_label is None:
                    gold_label = majority_vote_label_from_normalized(
                        [normalize_label(item) for item in factuality_all]
                    )

                sentence_id = answer.get("sentence_id", s_idx)
                rows.append(
                    {
                        "sample_id": f"{query_id}#s{sentence_id}",
                        "query_id": query_id,
                        "sentence_id": sentence_id,
                        "query": query,
                        "context_texts": context_texts,
                        "answer_segment": sentence,
                        "gold_label": gold_label,
                        "gold_labels_all": factuality_all,
                        "factuality_all": factuality_all,
                        "relevance_all": relevance_all,
                        "fine_grained_factuality_all": fine_grained_all,
                        "comments_all": comments_all,
                        "fine_grained_factuality": fine_grained_all[0] if fine_grained_all else answer.get("fine_grained_factuality"),
                        "relevance": relevance_all[0] if relevance_all else answer.get("relevance"),
                        "comments": comments_all[0] if comments_all else answer.get("comments"),
                    }
                )

    return rows


def build_prompt(row: dict[str, Any]) -> str:
    context = "\n".join(f"{index + 1}. {text}" for index, text in enumerate(row["context_texts"]))
    return SYSTEM_PROMPT + "\n\n" + TASK_PROMPT.format(
        query=row["query"],
        context=context,
        answer_segment=row["answer_segment"],
    )


def extract_label_and_reason(raw_output: str | None) -> tuple[str | None, str | None]:
    if not raw_output:
        return None, None

    answer_match = re.search(r"<answer>\s*(supported|not supported)\s*</answer>", raw_output, flags=re.IGNORECASE)
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

    rationale_match = re.search(r"<rationale>(.*?)</rationale>", raw_output, flags=re.IGNORECASE | re.DOTALL)
    if rationale_match:
        return label, rationale_match.group(1).strip()

    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if len(lines) > 1:
        return label, " ".join(lines[1:]).strip()

    return label, None


def call_model_with_retry(
    client: Any,
    model: str,
    prompt: str,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
) -> tuple[str | None, str | None]:
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            text = getattr(response, "text", None)
            if text is None:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  [RETRY] Empty response (text=None). Waiting {delay:.1f}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(delay)
                    continue
                return None, "Empty response: model returned no text after all retries"
            return text, None
        except Exception as exc:
            error_msg = str(exc)
            is_rate_limited = "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg
            if attempt < max_retries - 1 and is_rate_limited:
                delay = base_delay * (2 ** attempt)
                print(f"  [RETRY] Rate-limited. Waiting {delay:.1f}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(delay)
                continue
            return None, error_msg

    return None, "Max retries exceeded"


def call_openai_with_retry(
    client: Any,
    model: str,
    prompt: str,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
) -> tuple[str | None, str | None]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,
                temperature=0,
            )
            text = response.choices[0].message.content
            if text is None:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  [RETRY] Empty response. Waiting {delay:.1f}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(delay)
                    continue
                return None, "Empty response after all retries"
            return text, None
        except Exception as exc:
            error_msg = str(exc)
            is_rate_limited = "429" in error_msg or "rate_limit" in error_msg.lower()
            if attempt < max_retries - 1 and is_rate_limited:
                delay = base_delay * (2 ** attempt)
                print(f"  [RETRY] Rate-limited. Waiting {delay:.1f}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(delay)
                continue
            return None, error_msg
    return None, "Max retries exceeded"


def call_blablador_with_retry(
    client: Any,
    model: str,
    prompt: str,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
) -> tuple[str | None, str | None]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=BLABLADOR_MAX_TOKENS,
                temperature=0,
            )
            text = response.choices[0].message.content
            if text is None:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  [RETRY] Empty response. Waiting {delay:.1f}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(delay)
                    continue
                return None, "Empty response after all retries"
            # Strip <think>...</think> blocks produced by reasoning models (MiniMax, Qwen3.x)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text, None
        except Exception as exc:
            error_msg = str(exc)
            is_rate_limited = "429" in error_msg or "rate_limit" in error_msg.lower()
            if attempt < max_retries - 1 and is_rate_limited:
                delay = base_delay * (2 ** attempt)
                print(f"  [RETRY] Rate-limited. Waiting {delay:.1f}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(delay)
                continue
            return None, error_msg
    return None, "Max retries exceeded"


def majority_vote(votes: list[str | None]) -> str | None:
    valid = [vote for vote in votes if vote in {"Supported", "Not Supported"}]
    if not valid:
        return None
    counts = Counter(valid)
    if counts["Supported"] == counts["Not Supported"]:
        return random.choice(["Supported", "Not Supported"])
    return counts.most_common(1)[0][0]


def impute_label(gold_label: str | None, pred_label: str | None) -> str | None:
    if gold_label not in {"Supported", "Not Supported"}:
        return None
    if pred_label in {"Supported", "Not Supported"}:
        return pred_label
    return "Not Supported" if gold_label == "Supported" else "Supported"


# def compute_bacc(rows: list[dict[str, Any]], pred_key: str) -> float | None:    
#     pos_total = 0
#     neg_total = 0
#     pos_correct = 0
#     neg_correct = 0

#     for row in rows:
#         gold = row.get("gold_label")
#         pred = row.get(pred_key)
#         if gold not in {"Supported", "Not Supported"} or pred not in {"Supported", "Not Supported"}:
#             continue
#         if gold == "Supported":
#             pos_total += 1
#             if pred == "Supported":
#                 pos_correct += 1
#         else:
#             neg_total += 1
#             if pred == "Not Supported":
#                 neg_correct += 1

#     if pos_total == 0 or neg_total == 0:
#         return None
#     return 0.5 * ((pos_correct / pos_total) + (neg_correct / neg_total))

def compute_bacc(rows: list[dict[str, Any]], pred_key: str) -> float | None:
    pos_total = 0
    neg_total = 0
    pos_correct = 0
    neg_correct = 0

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

        elif gold == "Not Supported":
            neg_total += 1
            if pred == "Not Supported":
                neg_correct += 1

    print(
        f"[DEBUG {pred_key}] "
        f"pos_total={pos_total}, "
        f"neg_total={neg_total}, "
        f"pos_correct={pos_correct}, "
        f"neg_correct={neg_correct}"
    )

    if pos_total == 0 or neg_total == 0:
        return None

    return 0.5 * (
        (pos_correct / pos_total)
        + (neg_correct / neg_total)
    )

def print_response_analysis(records: list[dict[str, Any]], models: list[str]) -> None:
    total = len(records)
    print(f"\nResponse Analysis ({total} total records):")
    header = f"  {'Model':<40} {'Attempts':>8} {'Success':>8} {'Null(empty)':>12} {'Error':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model in models:
        attempts = success = null_count = error_count = 0
        for rec in records:
            info = rec.get("model_outputs", {}).get(model)
            if info is None:
                continue
            attempts += 1
            if info.get("error"):
                error_count += 1
            elif info.get("label") is None:
                null_count += 1
            else:
                success += 1
        print(f"  {model:<40} {attempts:>8} {success:>8} {null_count:>12} {error_count:>8}")
    print()


def compute_cohen_kappa(rows: list[dict[str, Any]], pred_key: str) -> float | None:
    n = agree = 0
    gold_sup = gold_not = pred_sup = pred_not = 0
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


def write_metrics_csv(path: Path, records: list[dict[str, Any]], models: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for model in models:
        key = f"__{model}"
        bacc = compute_bacc(records, key)
        kappa = compute_cohen_kappa(records, key)
        total = sum(1 for r in records if r.get(key) in {"Supported", "Not Supported"})
        rows.append({"kind": "individual", "name": model, "balanced_accuracy": bacc, "cohen_kappa": kappa, "total_valid": total})
    mv_key = "__panel_majority"
    rows.append({
        "kind": "majority_vote",
        "name": "panel_majority",
        "balanced_accuracy": compute_bacc(records, mv_key),
        "cohen_kappa": compute_cohen_kappa(records, mv_key),
        "total_valid": sum(1 for r in records if r.get(mv_key) in {"Supported", "Not Supported"}),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "name", "balanced_accuracy", "cohen_kappa", "total_valid"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV metrics saved to: {path}")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AG+COT MEMERAG judges on MEMERAG JSONL data.")
    parser.add_argument("--lang", required=True, choices=["en", "es", "de", "fr", "hi"], help="Target language for evaluation.")

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of sentence-level examples to run. Omit to run all rows.",
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default=None,
        help="Run only a specific sample_id/query_id."
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        nargs="+",
        default=None,
        help="Run a list of sample_ids. Example: --sample_ids 786 3351 7",
    )
    parser.add_argument("--all", action="store_true", help="Shortcut for running all available rows.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Write a checkpoint after this many processed samples.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from an existing output file.",
    )
    args = parser.parse_args()

    random.seed(SEED)

    data_file = DATA_ROOT / "memerag_ext" / f"{args.lang}.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    all_data = load_memerag_jsonl(data_file)
    if not all_data:
        raise RuntimeError(f"No valid examples loaded from {data_file}")

    requested_ids: set[str] = set()
    if args.sample_id is not None:
        requested_ids.add(str(args.sample_id))
    if args.sample_ids:
        requested_ids.update(str(i) for i in args.sample_ids)

    # Determine candidate rows from the source JSONL
    if requested_ids:
        candidate_data = [row for row in all_data if str(row["sample_id"]) in requested_ids]
        if not candidate_data:
            raise ValueError(f"No samples found for requested IDs: {sorted(requested_ids)}")
    elif args.samples is not None and not args.all:
        random.shuffle(all_data)
        candidate_data = all_data[:min(args.samples, len(all_data))]
    else:
        candidate_data = all_data

    output_dir = ROOT / "results_tmp" / "memerag_ext" / args.lang
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"memerag_judgement_{args.lang}.json"

    # Load existing output for resume / patch
    resume_records: list[dict[str, Any]] = []
    if output_path.exists() and not args.no_resume:
        try:
            with output_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, dict) and isinstance(existing.get("records"), list):
                resume_records = [r for r in existing["records"] if isinstance(r, dict)]
        except Exception:
            resume_records = []

    # Model-level resume: mutable lookup dict keyed by sample_id
    completed_records_dict: dict[str, dict[str, Any]] = {
        str(r.get("sample_id")): r for r in resume_records
    }

    # For each candidate sample, find which models are still missing
    remaining_with_missing: list[tuple[dict[str, Any], list[str]]] = []
    for row in candidate_data:
        sid = str(row["sample_id"])
        existing_outputs = completed_records_dict.get(sid, {}).get("model_outputs", {})
        if requested_ids and sid in requested_ids:
            # Retry mode: re-run models that are missing OR previously failed/returned null
            missing = [
                m for m in MODELS
                if m not in existing_outputs
                or existing_outputs[m].get("label") is None
                or existing_outputs[m].get("error")
            ]
        else:
            missing = [m for m in MODELS if m not in existing_outputs]
        if missing:
            remaining_with_missing.append((row, missing))

    print(f"Loaded {len(all_data)} MEMERAG examples from {data_file}")
    print(f"Samples fully done: {len(candidate_data) - len(remaining_with_missing)}")
    print(f"Samples needing at least one model: {len(remaining_with_missing)}")
    if requested_ids:
        print(f"Patch mode: re-running {len(requested_ids)} sample_id(s)")
    print(f"Writing JSON results to {output_path}")

    genai = None
    HttpOptions = None
    try:
        from google import genai
        from google.genai.types import HttpOptions
    except Exception as exc:
        needs_google = any(not is_openai_model(m) and not is_blablador_model(m) for m in MODELS)
        if needs_google:
            print(f"[ERROR] Could not import google-genai: {exc}")
            return

    if genai is not None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
        os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = GOOGLE_GENAI_USE_VERTEXAI

    openai_client = None
    if any(is_openai_model(m) for m in MODELS):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set (required for GPT models)")
        openai_client = OpenAI(api_key=api_key)

    blablador_client = None
    if any(is_blablador_model(m) for m in MODELS):
        from openai import OpenAI
        blablador_key = os.environ.get("BLABLADOR_API_KEY")
        if not blablador_key:
            raise ValueError("BLABLADOR_API_KEY environment variable not set (required for Blablador models)")
        blablador_client = OpenAI(base_url=BLABLADOR_BASE_URL, api_key=blablador_key)

    def build_summary() -> dict[str, Any]:
        completed_records = list(completed_records_dict.values())
        # BUG FIX: MODELS only contains the models for the current run (e.g. just GPT models when
        # adding new models to existing records). Collect ALL models ever seen across all records so
        # metrics are computed for every judge, not just the ones in MODELS for this run.
        all_models_seen: list[str] = []
        seen_set: set[str] = set()
        for row in completed_records:
            for m in row.get("model_outputs", {}).keys():
                if m not in seen_set:
                    all_models_seen.append(m)
                    seen_set.add(m)
        summary: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_name": "memerag_ext",
            "lang": args.lang,
            "data_file": str(data_file),
            "n_samples": len(completed_records),
            "models": all_models_seen,
            "records": completed_records,
            "metrics": {},
        }
        for model_name in all_models_seen:
            metric_key = f"__{model_name}"
            for row in completed_records:
                row[metric_key] = row["model_outputs"].get(model_name, {}).get("label")
            summary["metrics"][model_name] = {
                "balanced_accuracy": compute_bacc(completed_records, metric_key),
            }
        for row in completed_records:
            row["__panel_majority"] = row.get("panel_majority")
        summary["metrics"]["panel_majority"] = {
            "balanced_accuracy": compute_bacc(completed_records, "__panel_majority"),
        }
        # Find best individual model by balanced accuracy (excluding panel_majority and ignored models)
        best_model = None
        best_bacc = -1.0
        for m in all_models_seen:
            if m in IGNORE_MODELS:
                continue
            bacc = summary["metrics"][m].get("balanced_accuracy")
            if bacc is not None and bacc > best_bacc:
                best_bacc = bacc
                best_model = m
        summary["best_model"] = best_model
        summary["best_model_bacc"] = round(best_bacc, 4) if best_model else None
        return summary

    def checkpoint() -> None:
        summary = build_summary()
        for k in ("metrics", "best_model", "best_model_bacc"):
            summary.pop(k, None)
        save_json(output_path, summary)

    try:
        for index, (sample, missing_models) in enumerate(remaining_with_missing, start=1):
            prompt = build_prompt(sample)
            sid = str(sample["sample_id"])
            print(f"\n[{index}/{len(remaining_with_missing)}] sample_id={sid} | running {len(missing_models)} model(s): {missing_models}")

            model_order = list(missing_models)
            random.shuffle(model_order)
            print(f"Model order: {', '.join(model_order)}")

            existing_record = completed_records_dict.get(sid)
            merged_outputs: dict[str, dict[str, Any]] = dict(
                (existing_record or {}).get("model_outputs", {})
            )

            for model_name in model_order:
                if is_openai_model(model_name):
                    raw_text, error = call_openai_with_retry(openai_client, model_name, prompt)
                    location = "openai"
                elif is_blablador_model(model_name):
                    api_id = BLABLADOR_API_IDS[model_name]
                    raw_text, error = call_blablador_with_retry(blablador_client, api_id, prompt)
                    location = "blablador"
                else:
                    location = model_location_for(model_name)
                    client = genai.Client(
                        http_options=HttpOptions(api_version="v1"),
                        vertexai=True,
                        project=PROJECT_ID,
                        location=location,
                    )
                    raw_text, error = call_model_with_retry(client, model_name, prompt)

                label, reason = extract_label_and_reason(raw_text) if raw_text else (None, None)
                merged_outputs[model_name] = {
                    "label": label,
                    "reason": reason,
                    "raw": raw_text,
                    "error": error,
                    "location": location,
                    "eval_label": impute_label(sample.get("gold_label"), label),
                    "correct_gold": label == sample.get("gold_label") if label in {"Supported", "Not Supported"} else None,
                }
                print(f"  {model_name} -> {label} ({'error' if error else 'ok'})")

            votes = [info["label"] for info in merged_outputs.values()]
            panel_majority = majority_vote(votes)

            if existing_record is not None:
                existing_record["model_outputs"] = merged_outputs
                existing_record["panel_majority"] = panel_majority
                existing_record["panel_majority_eval_label"] = impute_label(sample.get("gold_label"), panel_majority)
                # BUG FIX: model_order was set during the original run and only contained the
                # original 4 models. Append any new models so model_order reflects all models
                # that actually have outputs in this record.
                existing_order: list[str] = existing_record.get("model_order") or []
                for m in merged_outputs:
                    if m not in existing_order:
                        existing_order.append(m)
                existing_record["model_order"] = existing_order
            else:
                completed_records_dict[sid] = {
                    "sample_id": sample.get("sample_id"),
                    "query_id": sample.get("query_id"),
                    "sentence_id": sample.get("sentence_id"),
                    "query": sample.get("query"),
                    "answer_segment": sample.get("answer_segment"),
                    "context_texts": sample.get("context_texts"),
                    "gold_label": sample.get("gold_label"),
                    "gold_labels_all": sample.get("gold_labels_all"),
                    "factuality_all": sample.get("factuality_all"),
                    "relevance_all": sample.get("relevance_all"),
                    "fine_grained_factuality_all": sample.get("fine_grained_factuality_all"),
                    "comments_all": sample.get("comments_all"),
                    "fine_grained_factuality": sample.get("fine_grained_factuality"),
                    "relevance": sample.get("relevance"),
                    "comments": sample.get("comments"),
                    "prompt": prompt,
                    "model_order": model_order,
                    "model_outputs": merged_outputs,
                    "panel_majority": panel_majority,
                    "panel_majority_eval_label": impute_label(sample.get("gold_label"), panel_majority),
                }

            if index % max(args.checkpoint_every, 1) == 0 or index == len(remaining_with_missing):
                checkpoint()
                print(f"  [checkpoint] saved {len(completed_records_dict)}/{len(candidate_data)} samples")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving checkpoint before exit...")
        checkpoint()
        print(f"Saved {len(completed_records_dict)}/{len(candidate_data)} samples to {output_path}")
        return

    # Auto-retry: one extra pass for any model that errored in the main run
    retry_queue: list[tuple[dict[str, Any], list[str]]] = []
    seen_retry_sids: set[str] = set()
    for row in candidate_data:
        sid = str(row["sample_id"])
        if sid in seen_retry_sids:
            continue
        outputs = completed_records_dict.get(sid, {}).get("model_outputs", {})
        errored = [
            m for m in MODELS
            if isinstance(outputs.get(m), dict)
            and (outputs[m].get("label") is None or outputs[m].get("error"))
        ]
        if errored:
            retry_queue.append((row, errored))
            seen_retry_sids.add(sid)

    if retry_queue:
        n_errors = sum(len(ms) for _, ms in retry_queue)
        print(f"\n[AUTO-RETRY] {n_errors} errored response(s) across {len(retry_queue)} sample(s) — retrying once...")
        for retry_idx, (sample, error_models) in enumerate(retry_queue, start=1):
            prompt      = build_prompt(sample)
            sid         = str(sample["sample_id"])
            print(f"\n  [{retry_idx}/{len(retry_queue)}] sample_id={sid} | retrying: {error_models}")
            existing_record = completed_records_dict[sid]
            merged_outputs  = dict(existing_record.get("model_outputs", {}))

            for model_name in error_models:
                if is_openai_model(model_name):
                    raw_text, error = call_openai_with_retry(openai_client, model_name, prompt)
                    location = "openai"
                elif is_blablador_model(model_name):
                    api_id = BLABLADOR_API_IDS[model_name]
                    raw_text, error = call_blablador_with_retry(blablador_client, api_id, prompt)
                    location = "blablador"
                else:
                    location = model_location_for(model_name)
                    retry_client = genai.Client(
                        http_options=HttpOptions(api_version="v1"),
                        vertexai=True, project=PROJECT_ID, location=location,
                    )
                    raw_text, error = call_model_with_retry(retry_client, model_name, prompt)

                label, reason = extract_label_and_reason(raw_text) if raw_text else (None, None)
                merged_outputs[model_name] = {
                    "label": label, "reason": reason, "raw": raw_text, "error": error,
                    "location": location,
                    "eval_label": impute_label(sample.get("gold_label"), label),
                    "correct_gold": label == sample.get("gold_label")
                        if label in {"Supported", "Not Supported"} else None,
                }
                print(f"    {model_name} -> {label} ({'error' if error else 'ok'})")

            votes = [info["label"] for info in merged_outputs.values()]
            existing_record["model_outputs"] = merged_outputs
            existing_record["panel_majority"] = majority_vote(votes)
            existing_record["panel_majority_eval_label"] = impute_label(
                sample.get("gold_label"), existing_record["panel_majority"]
            )

        checkpoint()
        print(f"  [auto-retry done] checkpoint saved")

    final = build_summary()
    save_data = {k: v for k, v in final.items() if k not in ("metrics", "best_model", "best_model_bacc")}
    save_json(output_path, save_data)
    completed_records = list(completed_records_dict.values())

    # All models seen across records; filter out ignored ones for display/analysis
    all_models_in_summary = [m for m in final["metrics"] if m != "panel_majority"]
    active_models = [m for m in all_models_in_summary if m not in IGNORE_MODELS]
    print_response_analysis(completed_records, active_models)

    # Build rows sorted by bacc descending; panel_majority always last
    metric_rows: list[tuple[str, float | None, float | None]] = []
    for model_name in active_models:
        bacc = final["metrics"][model_name]["balanced_accuracy"]
        kappa = compute_cohen_kappa(completed_records, f"__{model_name}")
        metric_rows.append((model_name, bacc, kappa))
    metric_rows.sort(key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
    panel_bacc = final["metrics"]["panel_majority"]["balanced_accuracy"]
    panel_kappa = compute_cohen_kappa(completed_records, "__panel_majority")
    metric_rows.append(("panel_majority", panel_bacc, panel_kappa))

    print("\nBalanced Accuracy / Cohen's Kappa  (Gap = BAcc − Kappa; lower gap = less class-bias):")
    col = 42
    print(f"  {'Model':<{col}} {'BAcc':>7}  {'Kappa':>7}  {'Gap':>7}")
    print("  " + "-" * (col + 28))
    for model_name, bacc, kappa in metric_rows:
        bacc_str  = "n/a" if bacc  is None else f"{bacc:.4f}"
        kappa_str = "n/a" if kappa is None else f"{kappa:.4f}"
        gap_str   = "n/a" if (bacc is None or kappa is None) else f"{bacc - kappa:.4f}"
        print(f"  {model_name:<{col}} {bacc_str:>7}  {kappa_str:>7}  {gap_str:>7}")

    print(f"\nBest individual model (excl. panel_majority): {final.get('best_model')}  bacc={final.get('best_model_bacc')}")
    print(f"Label distribution: {Counter(r['gold_label'] for r in completed_records)}")
    print(f"Saved results to: {output_path}")

    # Per-model breakdown: missing (absent from model_outputs) vs errors (null label / API error)
    total_records = len(completed_records)
    missing_per_model: dict[str, list[str]] = {}
    error_per_model: dict[str, list[str]] = {}
    for model_name in active_models:
        for r in completed_records:
            sid = str(r.get("sample_id", ""))
            outputs = r.get("model_outputs", {})
            if model_name not in outputs:
                missing_per_model.setdefault(model_name, []).append(sid)
            else:
                info = outputs[model_name]
                if isinstance(info, dict) and (info.get("label") is None or info.get("error")):
                    error_per_model.setdefault(model_name, []).append(sid)

    if missing_per_model:
        all_missing_ids: set[str] = set()
        print(f"\n[MISSING] Models with no response entry (out of {total_records} records):")
        for model_name, ids in missing_per_model.items():
            all_missing_ids.update(ids)
            print(f"  {model_name}: {len(ids)} missing")
            print(f"    sample_ids: {' '.join(ids)}")
        print(f"\nRecover missing samples (uncomment affected models in MODELS first):")
        print(f"  python judge_memerag.py --lang {args.lang} --sample_ids {' '.join(sorted(all_missing_ids))}")
    else:
        print(f"\n[OK] No missing responses — all models present in all {total_records} records.")

    if error_per_model:
        all_error_ids: set[str] = set()
        print(f"\n[ERRORS] Models with null label or API error:")
        for model_name, ids in error_per_model.items():
            all_error_ids.update(ids)
            print(f"  {model_name}: {len(ids)} error(s)")
            print(f"    sample_ids: {' '.join(ids)}")
        print(f"\nRetry errored samples with:")
        print(f"  python judge_memerag.py --lang {args.lang} --sample_ids {' '.join(sorted(all_error_ids))}")
    else:
        print("\n[OK] No errors — all model responses have valid labels.")


if __name__ == "__main__":
    main()