from __future__ import annotations

import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

# CLI notes:
# - `--dataset` selects `cnndm` by default or `xsum`
# - `--samples` / `--all` control how many examples to judge
# - `--checkpoint` saves intermediate JSON snapshots during long runs, default=5
# - `--output-dir` lets you redirect the saved JSON results

# Vertex / GenAI settings.
SERVICE_ACCOUNT_PATH = r"C:\Users\Deepali\Downloads\llm-juries-dd7439c15063.json"
PROJECT_ID = "llm-juries"
DEFAULT_LOCATION = "global"
GOOGLE_GENAI_USE_VERTEXAI = "True"

# Keep the strongest judge first for later comparison across models.
MODELS = [
    "meta/llama-3.3-70b-instruct-maas",
    "google/gemma-4-26b-a4b-it-maas",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

DEFAULT_DATASET = "cnndm"
DEFAULT_N_SAMPLES = 3
SEED = 42

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


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

    answer_match = re.search(r"\b(yes|no)\b", raw_output, flags=re.IGNORECASE)
    label = normalize_label(answer_match.group(1)) if answer_match else None

    if label is None:
        return None, None

    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    reason = " ".join(lines[1:]).strip() if len(lines) > 1 else None
    return label, reason


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
            return getattr(response, "text", None), None
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


def model_location_for(model_name: str) -> str:
    if "meta" in model_name.lower():
        return "us-central1"
    return DEFAULT_LOCATION


def get_next_output_path(base_path: Path) -> Path:
    return base_path


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def load_qags_data(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    instances = data.get("instances", []) if isinstance(data, dict) else []
    processed: list[dict[str, Any]] = []
    for ex in instances:
        try:
            prompt = ex["instance"].strip()
            majority_human = normalize_label(ex["annotations"]["Factual Consistency"]["majority_human"])
            individual_human_scores = [normalize_label(v) for v in ex["annotations"]["Factual Consistency"]["individual_human_scores"]]
        except (KeyError, TypeError, AttributeError):
            continue

        if not prompt or majority_human is None:
            continue

        processed.append(
            {
                "id": ex.get("id"),
                "prompt": prompt,
                "majority_human": majority_human,
                "individual_human_scores": individual_human_scores,
            }
        )

    return processed


def build_prompt(example: dict[str, Any]) -> str:
    return example["prompt"] + (
        "\n\nPlease answer with only 'yes' or 'no'.\n"
        "You may add a short explanation on the next line."
    )


def print_sample_header(sample: dict[str, Any]) -> None:
    print("\n--- QAGS MODEL JUDGEMENT ---")
    print(f"Sample id: {sample['id']}")
    print(f"Majority human: {sample.get('majority_human')}")
    print(f"Individual human scores: {sample.get('individual_human_scores')}")
    print(f"Prompt chars: {len(build_prompt(sample))}")


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


def shuffled_model_order() -> list[str]:
    order = list(MODELS)
    random.shuffle(order)
    return order


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run individual LLM judgments on QAGS.")
    parser.add_argument("--dataset", choices=["cnndm", "xsum"], default=DEFAULT_DATASET)
    parser.add_argument(
        "--samples",
        type=int,
        dest="samples",
        help="Number of rows to run. If omitted, run all rows (default). To run a small sample pass e.g. --samples 3.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--sample-id", type=int, default=None, help="Run only one specific sample_id (overrides --samples).")
    parser.add_argument("--all", action="store_true", help="Shortcut for running all available rows (redundant when --samples omitted).")
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=5,
        help="Save a checkpoint JSON after this many examples (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the JSON results file should be written.",
    )
    args = parser.parse_args()

    random.seed(SEED)

    data_file = ROOT / "data" / "qags" / f"{args.dataset}.json"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    data = load_qags_data(data_file)
    if not data:
        raise RuntimeError(f"No valid examples loaded from {data_file}")

    try:
        from google import genai
        from google.genai.types import HttpOptions
    except Exception as exc:
        print(f"[ERROR] Could not import google-genai: {exc}")
        return

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = GOOGLE_GENAI_USE_VERTEXAI

    # Default behaviour: run all instances when `--samples` is omitted.
    if not hasattr(args, "samples"):
        n_samples = len(data)
    else:
        n_samples = len(data) if args.all or args.samples < 0 else min(args.samples, len(data))
    samples = data[:n_samples]

    if args.sample_id is not None:
        samples = [sample for sample in samples if sample.get("id") == args.sample_id]
        if not samples:
            raise RuntimeError(f"No valid example found for sample_id={args.sample_id} in {data_file}")
        n_samples = len(samples)

    output_dir = args.output_dir or (ROOT / "results_tmp" / f"qags_{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = get_next_output_path(output_dir / "qags_judgement.json")

    existing_summary: dict[str, Any] | None = None
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            existing_summary = json.load(handle)

    print(f"Loaded {len(data)} QAGS examples from {data_file}")
    print(f"Running individual models on {n_samples} / {len(data)} rows")
    print(f"Writing JSON results to {output_path}")

    results: list[dict[str, Any]] = []
    start_index = 0
    if existing_summary and isinstance(existing_summary.get("records"), list):
        results = list(existing_summary["records"])
        if args.sample_id is not None:
            results = [record for record in results if record.get("sample_id") != args.sample_id]
            start_index = 0
        else:
            start_index = len(results)
        if start_index:
            print(f"Resuming from existing checkpoint with {start_index} completed rows.")

    if args.sample_id is not None:
        print(f"Single-sample rerun mode enabled for sample_id={args.sample_id}")

    for index, sample in enumerate(samples[start_index:], start=start_index):
        prompt = build_prompt(sample)
        print(f"\n[{index + 1}/{n_samples}] Processing sample_id={sample['id']}")
        print_sample_header(sample)

        model_outputs: dict[str, dict[str, Any]] = {}
        model_order = shuffled_model_order()
        print(f"Model order: {', '.join(model_order)}")
        for model_name in model_order:
            location = model_location_for(model_name)
            client = genai.Client(
                http_options=HttpOptions(api_version="v1"),
                vertexai=True,
                project=PROJECT_ID,
                location=location,
            )

            raw_text, error = call_model_with_retry(client, model_name, prompt)
            label, reason = extract_label_and_reason(raw_text) if raw_text else (None, None)

            model_outputs[model_name] = {
                "label": label,
                "reason": reason,
                "raw": raw_text,
                "error": error,
                "location": location,
                "correct_majority_human": label == sample.get("majority_human") if label in {"yes", "no"} else None,
            }
            print(f"{model_name} -> {label} ({'error' if error else 'ok'})")

        results.append(
            {
                "sample_id": sample["id"],
                "majority_human": sample.get("majority_human"),
                "individual_human_scores": sample.get("individual_human_scores"),
                "prompt": sample.get("prompt"),
                "model_order": model_order,
                "model_outputs": model_outputs,
            }
        )

        if args.checkpoint > 0 and (len(results) % args.checkpoint == 0 or len(results) == n_samples):
            checkpoint_payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "dataset": args.dataset,
                "data_file": str(data_file),
                "n_samples": len(results),
                "models": MODELS,
                "records": results,
            }
            save_json(output_path, checkpoint_payload)
            print(f"Checkpoint saved to: {output_path}")

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": args.dataset,
        "data_file": str(data_file),
        "n_samples": len(results),
        "models": MODELS,
        "records": results,
        "metrics": {
            model_name: {
                "accuracy_vs_majority_human": compute_accuracy(results, f"__{model_name}")
            }
            for model_name in MODELS
        },
    }

    # Store model-level predictions at the top level of each record so the summary can be computed easily.
    for row in results:
        for model_name, info in row["model_outputs"].items():
            row[f"__{model_name}"] = info.get("label")

    # Recompute metrics after populating the helper keys.
    for model_name in MODELS:
        summary["metrics"][model_name]["accuracy_vs_majority_human"] = compute_accuracy(results, f"__{model_name}")

    save_json(output_path, summary)

    print("\nModel accuracies vs majority_human:")
    for model_name in MODELS:
        acc = summary["metrics"][model_name]["accuracy_vs_majority_human"]
        acc_text = "n/a" if acc is None else f"{acc:.4f}"
        print(f"  {model_name}: {acc_text}")

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()