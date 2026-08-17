"""
Qualitative analysis: cases where the aggregator LLM chose the minority label.

This script is about LLM-based aggregation only — NOT majority voting or any
algorithmic aggregator (DS, IWMV, ISP, MACE).  The pipeline has two stages:
  1. Seven judge LLMs (PROPOSERS in config.py) each independently label an item.
  2. A separate *aggregator* LLM reads all seven labels + reasoning and produces
     a final label.

A "minority override" is when the aggregator LLM's final label disagrees with
the raw majority vote of the 7 judge LLMs — i.e., the aggregator read the judge
outputs and still sided with the smaller group.

Uses PROPOSERS and IGNORE_MODELS from config.py so that only the actual active
judges are counted — not extra models that happen to appear in model_outputs.

Usage
-----
    python llm-experiments/pipeline/llm_agg_minority_overrides.py

To use for a different dataset/language, edit DATASET / LANG in config.py, or
override them via --dataset and --lang flags:

    python llm-experiments/pipeline/llm_agg_minority_overrides.py --lang de
    python llm-experiments/pipeline/llm_agg_minority_overrides.py --all       # all languages
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from config import DATASET, IGNORE_MODELS, LANG, PROPOSERS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]  # JUDGE-BENCH root

LABEL_MAP = {"Supported": 1, "Not Supported": 0}
DECODE = {v: k for k, v in LABEL_MAP.items()}

# Active judges: PROPOSERS minus anything in IGNORE_MODELS
ACTIVE_MODELS: list[str] = [m for m in PROPOSERS if m not in IGNORE_MODELS]

# Language lists per dataset
LANG_SETS = {
    "memerag":     ["en", "de", "es", "fr", "hi"],
    "memerag_ext": ["en", "de", "es", "fr", "hi"],
    "qags":        ["cnndm", "xsum"],
}

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_best_bacc_file(results_dir: Path, lang: str) -> Path | None:
    lang_dir = results_dir / lang
    matches = sorted(lang_dir.glob("llm_agg_best_bacc_*.json"))
    return matches[0] if matches else None


def analyze_overrides(path: Path, active_models: list[str]) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    agg_model = data.get("aggregator_model", "?")
    records = data.get("records", [])

    overrides = []
    total = 0

    for r in records:
        agg_label = r.get("aggregator_label")
        if not agg_label:
            continue
        total += 1

        outputs = r.get("model_outputs", {})
        valid: list[int] = []
        for m in active_models:
            raw = outputs.get(m)
            label = raw.get("label") if isinstance(raw, dict) else raw
            enc = LABEL_MAP.get(label)
            if enc is not None:
                valid.append(enc)

        if not valid:
            continue

        n1 = sum(valid)
        n0 = len(valid) - n1
        if n1 == n0:
            continue  # tie — no clear majority to override

        majority_enc = 1 if n1 > n0 else 0
        agg_enc = LABEL_MAP.get(agg_label)
        if agg_enc is None or agg_enc == majority_enc:
            continue  # not an override

        gold = r.get("gold_label", "?")
        gold_enc = LABEL_MAP.get(gold)

        overrides.append({
            "sid":         r.get("sample_id", "?"),
            "n1":          n1,
            "n0":          n0,
            "votes":       "%d:%d" % (n1, n0),
            "majority":    DECODE.get(majority_enc, "?"),
            "agg":         agg_label,
            "gold":        gold,
            "agg_correct": agg_enc == gold_enc,
            "maj_correct": majority_enc == gold_enc,
        })

    return {"agg_model": agg_model, "total": total, "overrides": overrides}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_lang_results(lang: str, result: dict) -> None:
    short_model = result["agg_model"].split("/")[-1]
    n_ov = len(result["overrides"])
    total = result["total"]
    pct = 100 * n_ov / total if total else 0.0
    n_correct = sum(1 for o in result["overrides"] if o["agg_correct"])
    n_wrong = n_ov - n_correct

    print("\n[%s] aggregator=%s  overrides=%d / %d (%.1f%%)  correct=%d  wrong=%d" % (
        lang.upper(), short_model, n_ov, total, pct, n_correct, n_wrong))

    if result["overrides"]:
        fmt = "  %-20s %-8s %-18s %-18s %-18s %-10s %-10s"
        print(fmt % ("sample_id", "votes", "majority", "agg_choice", "gold", "agg_ok?", "mv_ok?"))
        print("  " + "-" * 100)
        for o in result["overrides"]:
            print(fmt % (
                str(o["sid"]), o["votes"], o["majority"], o["agg"],
                o["gold"],
                "CORRECT" if o["agg_correct"] else "WRONG",
                "correct" if o["maj_correct"] else "wrong",
            ))
    else:
        print("  (none)")


def print_summary(all_results: dict[str, dict]) -> None:
    from collections import Counter

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_ov = sum(len(r["overrides"]) for r in all_results.values())
    total_correct = sum(
        sum(1 for o in r["overrides"] if o["agg_correct"])
        for r in all_results.values()
    )
    total_wrong = total_ov - total_correct

    print("Total overrides across all languages: %d" % total_ov)
    if total_ov:
        print("  Correct (agg was right to override): %d (%.0f%%)" % (
            total_correct, 100 * total_correct / total_ov))
        print("  Wrong   (agg was wrong to override): %d (%.0f%%)" % (
            total_wrong, 100 * total_wrong / total_ov))

    direction: Counter = Counter()
    for r in all_results.values():
        for o in r["overrides"]:
            direction[("majority=" + o["majority"], "agg=" + o["agg"])] += 1

    if direction:
        print("\nOverride direction breakdown (aggregator bias):")
        for pair, count in direction.most_common():
            print("  %s  ->  %s : %d" % (pair[0], pair[1], count))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang",    default=None, help="Override config.py LANG")
    parser.add_argument("--dataset", default=None, help="Override config.py DATASET")
    parser.add_argument("--all",     action="store_true", help="Run all languages for the dataset")
    args = parser.parse_args()

    dataset = args.dataset or DATASET
    results_dir = ROOT / "results_tmp" / (dataset + "_ext" if not (ROOT / "results_tmp" / dataset).exists() else dataset)
    # fallback: try with _ext suffix if base name doesn't exist
    if not results_dir.exists():
        alt = ROOT / "results_tmp" / dataset
        results_dir = alt if alt.exists() else results_dir

    if args.all:
        langs = LANG_SETS.get(dataset, LANG_SETS.get(dataset + "_ext", [LANG]))
    else:
        langs = [args.lang or LANG]

    print("Minority-override analysis: LLM aggregator chose label with fewer judge votes")
    print("(LLM-based aggregation only — NOT majority vote or any algorithmic aggregator)")
    print("Dataset      : %s" % dataset)
    print("Active judges: %s" % ", ".join(ACTIVE_MODELS))
    print("Results dir  : %s" % results_dir)
    print("=" * 80)

    all_results: dict[str, dict] = {}
    for lang in langs:
        path = find_best_bacc_file(results_dir, lang)
        if path is None:
            print("\n[%s] no llm_agg_best_bacc_*.json found — skipping" % lang.upper())
            continue
        result = analyze_overrides(path, ACTIVE_MODELS)
        all_results[lang] = result
        print_lang_results(lang, result)

    if len(all_results) > 1:
        print_summary(all_results)


if __name__ == "__main__":
    main()
