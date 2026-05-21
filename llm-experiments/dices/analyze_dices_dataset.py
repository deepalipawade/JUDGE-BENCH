from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "dices" / "dices_350_crowdsourced.json"
SAFETY_GOLD_PATH = ROOT / "data" / "dices" / "safety_gold" / "diverse_safety_adversarial_dialog_350.csv"


def normalize_label(text: Any) -> str | None:
    if text is None:
        return None
    value = str(text).strip().lower()
    if value in {"yes", "safe"}:
        return "Yes"
    if value in {"no", "unsafe"}:
        return "No"
    return None


def normalize_multi_value(text: Any) -> str | None:
    if text is None:
        return None

    value = str(text).strip()
    if not value or value == "[]":
        return None

    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            items = [str(item).strip() for item in parsed if str(item).strip()]
            if not items:
                return None
            if len(items) == 1:
                return items[0]
            return ", ".join(items)

    return value


def short_text(text: Any, limit: int = 180) -> str:
    if text is None:
        return ""
    cleaned = str(text).replace("\n", " ").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def load_dices_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    instances = raw.get("instances", []) if isinstance(raw, dict) else []
    rows: list[dict[str, Any]] = []

    for item in instances:
        try:
            instance = item["instance"]
            user_prompt = instance["user_prompt"].strip()
            response = instance["response"].strip()
        except (KeyError, TypeError, AttributeError):
            continue

        if not user_prompt or not response:
            continue

        rows.append(
            {
                "id": item.get("id"),
                "user_prompt": user_prompt,
                "response": response,
                "human_label": normalize_label(item.get("annotations", {}).get("safety", {}).get("majority_human")),
            }
        )

    return rows


def load_safety_gold(path: Path) -> dict[int, dict[str, Any]]:
    votes_by_item: dict[int, list[str]] = defaultdict(list)
    reasons_by_item: dict[int, list[str]] = defaultdict(list)
    degree_by_item: dict[int, list[str]] = defaultdict(list)
    harm_by_item: dict[int, list[str]] = defaultdict(list)
    raw_rows = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_rows += 1
            raw_id = row.get("item_id") or row.get("id")
            try:
                item_id = int(raw_id)
            except (TypeError, ValueError):
                continue

            label = normalize_label(row.get("safety_gold") or row.get("gold_label") or row.get("gold") or row.get("label"))
            if label is not None:
                votes_by_item[item_id].append(label)

            reason = normalize_multi_value(row.get("safety_gold_reason"))
            if reason:
                reasons_by_item[item_id].append(reason)

            degree = normalize_multi_value(row.get("degree_of_harm"))
            if degree:
                degree_by_item[item_id].append(degree)

            harm = normalize_multi_value(row.get("harm_type"))
            if harm:
                harm_by_item[item_id].append(harm)

    merged: dict[int, dict[str, Any]] = {}
    for item_id, labels in votes_by_item.items():
        label_counts = Counter(labels)
        merged[item_id] = {
            "safety_gold": label_counts.most_common(1)[0][0],
            "vote_counts": dict(label_counts),
            "n_votes": sum(label_counts.values()),
            "safety_gold_reason": Counter(reasons_by_item[item_id]).most_common(1)[0][0] if reasons_by_item[item_id] else None,
            "degree_of_harm": Counter(degree_by_item[item_id]).most_common(1)[0][0] if degree_by_item[item_id] else None,
            "harm_type": Counter(harm_by_item[item_id]).most_common(1)[0][0] if harm_by_item[item_id] else None,
        }

    merged["_meta"] = {
        "raw_rows": raw_rows,
        "n_items": len(votes_by_item),
    }
    return merged


def merge_rows(json_rows: list[dict[str, Any]], safety_gold: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in json_rows:
        row_id = row.get("id")
        merged_row = dict(row)
        if isinstance(row_id, int) and row_id in safety_gold:
            merged_row.update(safety_gold[row_id])
        merged.append(merged_row)
    return merged


def prompt_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    prompt_lengths = [len(r["user_prompt"]) for r in rows]
    response_lengths = [len(r["response"]) for r in rows]
    return {
        "prompt_mean": statistics.mean(prompt_lengths) if prompt_lengths else 0.0,
        "prompt_median": statistics.median(prompt_lengths) if prompt_lengths else 0.0,
        "response_mean": statistics.mean(response_lengths) if response_lengths else 0.0,
        "response_median": statistics.median(response_lengths) if response_lengths else 0.0,
    }


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_record(row: dict[str, Any], index: int) -> None:
    print(f"[{index}] id={row.get('id')} human={row.get('human_label')} safety_gold={row.get('safety_gold')}")
    print(f"    degree_of_harm: {row.get('degree_of_harm')} | harm_type: {row.get('harm_type')}")
    reason = row.get("safety_gold_reason")
    if reason:
        print(f"    safety_gold_reason: {short_text(reason, 220)}")
    print(f"    prompt: {short_text(row.get('user_prompt'), 260)}")
    print(f"    response: {short_text(row.get('response'), 260)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the DICES-350 dataset and its safety_gold annotations.")
    parser.add_argument("--samples", type=int, default=5, help="Number of random merged examples to print.")
    parser.add_argument("--disagreements", type=int, default=5, help="Number of human vs safety_gold disagreements to print.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sample selection.")
    args = parser.parse_args()

    random.seed(args.seed)

    json_rows = load_dices_json(DATA_FILE)
    safety_gold = load_safety_gold(SAFETY_GOLD_PATH)
    merged_rows = merge_rows(json_rows, safety_gold)

    matched_rows = [row for row in merged_rows if row.get("safety_gold") in {"Yes", "No"}]
    disagreement_rows = [row for row in matched_rows if row.get("human_label") in {"Yes", "No"} and row.get("human_label") != row.get("safety_gold")]

    print_section("Dataset Overview")
    print(f"JSON examples loaded: {len(json_rows)}")
    print(f"CSV raw rows: {safety_gold.get('_meta', {}).get('raw_rows', 0)}")
    print(f"CSV unique item_ids: {safety_gold.get('_meta', {}).get('n_items', 0)}")
    print(f"Merged examples with safety_gold: {len(matched_rows)}")

    human_counts = Counter(row.get("human_label") for row in matched_rows if row.get("human_label") in {"Yes", "No"})
    gold_counts = Counter(row.get("safety_gold") for row in matched_rows if row.get("safety_gold") in {"Yes", "No"})
    agreement = sum(1 for row in matched_rows if row.get("human_label") in {"Yes", "No"} and row.get("human_label") == row.get("safety_gold"))
    comparable = sum(1 for row in matched_rows if row.get("human_label") in {"Yes", "No"} and row.get("safety_gold") in {"Yes", "No"})

    print(f"Human label distribution: {dict(human_counts)}")
    print(f"Safety gold distribution: {dict(gold_counts)}")
    print(f"Human vs safety_gold agreement: {agreement}/{comparable} = {agreement / comparable:.3f}" if comparable else "Human vs safety_gold agreement: n/a")

    stats = prompt_stats(matched_rows)
    print(f"Prompt length mean/median: {stats['prompt_mean']:.1f} / {stats['prompt_median']:.1f}")
    print(f"Response length mean/median: {stats['response_mean']:.1f} / {stats['response_median']:.1f}")

    print_section("Most Common Safety Gold Reasons and Harm Types")
    reason_counts = Counter(row.get("safety_gold_reason") for row in matched_rows if row.get("safety_gold_reason"))
    harm_counts = Counter(row.get("harm_type") for row in matched_rows if row.get("harm_type"))
    degree_counts = Counter(row.get("degree_of_harm") for row in matched_rows if row.get("degree_of_harm"))

    print("Top safety_gold_reason values:")
    for reason, count in reason_counts.most_common(5):
        print(f"  {count:>3}  {short_text(reason, 120)}")

    print("Top harm_type values:")
    for harm, count in harm_counts.most_common(8):
        print(f"  {count:>3}  {harm}")

    print("Degree of harm distribution:")
    for degree, count in degree_counts.most_common():
        print(f"  {count:>3}  {degree}")

    print_section("Random Example Rows")
    examples = random.sample(matched_rows, min(args.samples, len(matched_rows)))
    for index, row in enumerate(examples, start=1):
        print_record(row, index)

    print_section("Human vs Safety Gold Disagreements")
    if disagreement_rows:
        for index, row in enumerate(disagreement_rows[: args.disagreements], start=1):
            print_record(row, index)
    else:
        print("No disagreements found in the merged rows.")

    print_section("Label Balance By Harm Type")
    by_harm: dict[str, Counter] = defaultdict(Counter)
    for row in matched_rows:
        harm = row.get("harm_type") or "<missing>"
        label = row.get("safety_gold") or "<missing>"
        by_harm[harm][label] += 1

    for harm, counter in sorted(by_harm.items(), key=lambda item: sum(item[1].values()), reverse=True)[:10]:
        print(f"{harm}: {dict(counter)}")


if __name__ == "__main__":
    main()