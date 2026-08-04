"""Random aggregator for MEMERAG.

For each sample, randomly picks one model to act as aggregator.
The aggregator sees all OTHER models' outputs and makes a final label decision.
This is identical to aggregator_gemma.py but the aggregator is chosen randomly per sample.

--exclude: removes a model from BOTH roles — it is neither a proposer nor an aggregator.

Usage:
    python random_aggregator.py --lang en --all --no-resume # to start fresh
    python random_aggregator.py --lang en --all --seed 0
    python random_aggregator.py --lang en --all --exclude gemini-2.5-flash  # LOO
"""
from __future__ import annotations

import argparse
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
RESULTS_ROOT = ROOT / "results_tmp" / "memerag_ext"


SERVICE_ACCOUNT_PATH = r"C:\Users\Deepali\Downloads\Thesis\Ivaxi LLMs\llm-juries-503009-babf6fe232e6.json"
PROJECT_ID = "llm-juries-503009"
DEFAULT_LOCATION = "global"
GOOGLE_GENAI_USE_VERTEXAI = "True"

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


BLABLADOR_BASE_URL = "https://api.blablador.fz-juelich.de/v1/"
BLABLADOR_MAX_TOKENS = 6000

BLABLADOR_API_IDS: dict[str, str] = {
    "MiniMax-M2.7":  "01 - MiniMax-M2.7 - our best model as of April, 2026",
    "GPT-OSS-120b":  "01 - GPT-OSS-120b - an open model released by OpenAI in August 2025",
    "Qwen3.6-35B":   "08 - Qwen3.6-35B-A3B-FP8 - Multimodal model from Apr 2026",
    "Apertus-8B":    "15 - Apertus-8B-Instruct-2509 - A new swiss model from September 2025",
}


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


def call_vertex(genai: Any, HttpOptions: Any, model_name: str, prompt: str) -> tuple[str | None, str | None]:
    for attempt in range(3):
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
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return None, "Empty response after retries"
            return text, None
        except Exception as exc:
            err = str(exc)
            if attempt < 2 and ("429" in err or "RESOURCE_EXHAUSTED" in err):
                time.sleep(2 ** attempt)
                continue
            return None, err
    return None, "Max retries exceeded"


def call_blablador(blablador_client: Any, api_id: str, prompt: str) -> tuple[str | None, str | None]:
    for attempt in range(3):
        try:
            response = blablador_client.chat.completions.create(
                model=api_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=BLABLADOR_MAX_TOKENS,
                temperature=0,
            )
            text = response.choices[0].message.content
            if text is None:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return None, "Empty response after retries"
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text, None
        except Exception as exc:
            err = str(exc)
            if attempt < 2 and ("429" in err or "rate_limit" in err.lower()):
                time.sleep(2 ** attempt)
                continue
            return None, err
    return None, "Max retries exceeded"


def call_openai(openai_client: Any, model_name: str, prompt: str) -> tuple[str | None, str | None]:
    for attempt in range(3):
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,
                temperature=0,
            )
            text = response.choices[0].message.content
            if text is None:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return None, "Empty response after retries"
            return text, None
        except Exception as exc:
            err = str(exc)
            if attempt < 2 and ("429" in err or "rate_limit" in err.lower()):
                time.sleep(2 ** attempt)
                continue
            return None, err
    return None, "Max retries exceeded"


def build_aggregator_prompt(record: dict[str, Any], proposer_models: list[str]) -> str:
    context_texts = record.get("context_texts", [])
    context = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(context_texts))
    task_prompt = TASK_PROMPT_TEMPLATE.format(
        context=context,
        query=record.get("query", ""),
        answer_segment=record.get("answer_segment", ""),
    )
    model_outputs = record.get("model_outputs", {})
    judge_blocks: list[str] = []
    for idx, model_name in enumerate(proposer_models, start=1):
        info = model_outputs.get(model_name, {})
        label = info.get("label") if isinstance(info, dict) else None
        reason = info.get("reason") if isinstance(info, dict) else None
        judge_blocks.append(
            f"{idx}. Model: {model_name}\n"
            f"   Label: {label}\n"
            f"   Reasoning: {reason}"
        )
    return PROMPT_TEMPLATE.format(
        task_prompt=task_prompt,
        judge_responses="\n".join(judge_blocks),
    )


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    try:
        temp.replace(path)
    except Exception as exc:
        print(f"[WARN] Could not replace {path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-aggregator for MEMERAG: randomly picks aggregator per sample.")
    parser.add_argument("--lang", required=True, choices=["en", "es", "de", "fr", "hi"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Run all samples.")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--exclude", type=str, default=None,
                        help="Exclude model from both proposer and aggregator roles (LOO).")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = args.input or (RESULTS_ROOT / args.lang / f"memerag_judgement_{args.lang}.json")
    if args.exclude:
        excl_folder = "excl_" + args.exclude.split("/")[-1].lower().replace(".", "_").replace("-", "_")
        output_path = args.output or (RESULTS_ROOT / args.lang / excl_folder / f"random_aggregator_{args.lang}.json")
    else:
        output_path = args.output or (RESULTS_ROOT / args.lang / f"random_aggregator_{args.lang}.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    all_records: list[dict[str, Any]] = data.get("records", [])
    if not all_records:
        raise RuntimeError(f"No records in {input_path}")

    if args.samples and not args.all:
        all_records = all_records[:args.samples]

    # Resume logic
    resume_records: list[dict[str, Any]] = []
    if output_path.exists() and not args.no_resume:
        try:
            with output_path.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
            resume_records = [r for r in existing.get("records", []) if isinstance(r, dict)]
        except Exception:
            resume_records = []

    processed_ids = {r.get("sample_id") for r in resume_records}
    remaining = [r for r in all_records if r.get("sample_id") not in processed_ids]

    print(f"Loaded {len(all_records)} records from {input_path}")
    print(f"Excluded model: {args.exclude or 'none'}")
    print(f"Already done: {len(processed_ids)} | Remaining: {len(remaining)}")
    print(f"Output: {output_path}")

    # Init Vertex AI
    genai = HttpOptions = None
    try:
        from google import genai as _genai
        from google.genai.types import HttpOptions as _HttpOptions
        genai, HttpOptions = _genai, _HttpOptions
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
        os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = GOOGLE_GENAI_USE_VERTEXAI
    except ImportError:
        pass

    # Init OpenAI
    openai_client = None
    try:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            openai_client = OpenAI(api_key=api_key)
    except ImportError:
        pass

    # Init Blablador
    blablador_client = None
    try:
        from openai import OpenAI as _OAI
        blablador_key = os.environ.get("BLABLADOR_API_KEY")
        if blablador_key:
            blablador_client = _OAI(base_url=BLABLADOR_BASE_URL, api_key=blablador_key)
    except ImportError:
        pass

    completed_records = list(resume_records)
    aggregator_choice_counts: Counter = Counter()

    def checkpoint() -> None:
        save_json(output_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lang": args.lang,
            "seed": args.seed,
            "excluded_model": args.exclude,
            "n_samples": len(completed_records),
            "aggregator_choice_counts": dict(aggregator_choice_counts),
            "records": completed_records,
        })

    try:
        for index, record in enumerate(remaining, start=len(processed_ids) + 1):
            sample_id = record.get("sample_id")
            gold_label = normalize_label(record.get("gold_label"))
            model_outputs = record.get("model_outputs", {})

            # All models in this sample with valid labels, minus excluded
            eligible_models = [
                m for m, info in model_outputs.items()
                if isinstance(info, dict)
                and normalize_label(info.get("label")) is not None
                and m != args.exclude
            ]

            if len(eligible_models) < 2:
                print(f"[{index}] sample_id={sample_id} — skip: fewer than 2 eligible models")
                completed_records.append({
                    "sample_id": sample_id, "gold_label": gold_label,
                    "aggregator_model": None, "aggregator_label": None,
                    "aggregator_error": "insufficient models",
                })
                continue

            # Randomly pick one model as aggregator; all eligible models (incl. aggregator's own) go into the prompt
            aggregator_model = random.choice(eligible_models)
            aggregator_choice_counts[aggregator_model] += 1

            print(f"\n[{index}/{len(all_records)}] sample_id={sample_id}")
            print(f"  Aggregator: {aggregator_model}")
            print(f"  Panel     : {eligible_models}")

            prompt = build_aggregator_prompt(record, eligible_models)

            if is_blablador_model(aggregator_model):
                raw, error = call_blablador(blablador_client, BLABLADOR_API_IDS[aggregator_model], prompt)
            elif is_openai_model(aggregator_model):
                raw, error = call_openai(openai_client, aggregator_model, prompt)
            else:
                raw, error = call_vertex(genai, HttpOptions, aggregator_model, prompt)

            agg_label, agg_reason = extract_label_and_reason(raw) if raw else (None, None)
            print(f"  -> {agg_label} ({'error' if error else 'ok'})")

            completed_records.append({
                "sample_id": sample_id,
                "gold_label": gold_label,
                "aggregator_model": aggregator_model,
                "panel_models": eligible_models,
                "aggregator_prompt": prompt,
                "aggregator_label": agg_label,
                "aggregator_reason": agg_reason,
                "aggregator_raw": raw,
                "aggregator_error": error,
                "__aggregator": agg_label,
            })

            if index % max(args.checkpoint_every, 1) == 0 or index == len(all_records):
                checkpoint()
                print(f"  [checkpoint] {len(completed_records)}/{len(all_records)} saved")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] saving...")
        checkpoint()
        return

    bacc = compute_bacc(completed_records, "__aggregator")
    kappa = compute_cohen_kappa(completed_records, "__aggregator")

    # --- Results table ---
    bacc_str  = f"{bacc:.4f}"  if bacc  is not None else "n/a"
    kappa_str = f"{kappa:.4f}" if kappa is not None else "n/a"
    col_w = max(len("random_aggregator"), 20)
    print(f"\nResults (seed={args.seed}, lang={args.lang}, excl={args.exclude}):")
    print(f"\n  {'Model':<{col_w}}  {'Bal. Acc':>10}  {'Kappa':>8}")
    print(f"  {'-'*col_w}  {'-'*10}  {'-'*8}")
    print(f"  {'random_aggregator':<{col_w}}  {bacc_str:>10}  {kappa_str:>8}")
    print(f"\n  Aggregator selection counts: {dict(aggregator_choice_counts)}")

    # --- Error summary ---
    errors   = [r for r in completed_records if r.get("aggregator_error")]
    missing  = [r for r in completed_records if r.get("aggregator_label") is None and not r.get("aggregator_error")]

    if errors:
        print(f"\n[ERRORS] {len(errors)} record(s) with errors:")
        for r in errors:
            print(f"  sample_id={r.get('sample_id')}  model={r.get('aggregator_model')}  err={r.get('aggregator_error')}")
        err_ids = " ".join(str(r["sample_id"]) for r in errors)
        model_short = args.exclude.split("/")[-1] if args.exclude else "none"
        print(f"  Retry: python random_aggregator.py --lang {args.lang} --seed {args.seed} --exclude {model_short} --sample_ids {err_ids}")

    if missing:
        print(f"\n[MISSING] {len(missing)} record(s) with no label:")
        for r in missing:
            print(f"  sample_id={r.get('sample_id')}  reason={r.get('aggregator_error', 'unknown')}")

    if not errors and not missing:
        print("\n[OK] All records have labels.")

    save_json(output_path, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lang": args.lang,
        "seed": args.seed,
        "excluded_model": args.exclude,
        "n_samples": len(completed_records),
        "metrics": {
            "random_aggregator": {"balanced_accuracy": bacc, "cohen_kappa": kappa}
        },
        "aggregator_choice_counts": dict(aggregator_choice_counts),
        "records": completed_records,
    })
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
