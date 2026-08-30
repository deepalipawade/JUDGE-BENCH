"""
Cases where BOTH majority vote AND the best-BAcc LLM aggregator are wrong.

These are potential "majority influence" cases — the aggregator read the judge
reasoning and echoed the majority's wrong conclusion rather than correcting it.

Each case is tagged:
  followed majority  — aggregator agreed with the (wrong) majority
  overrode majority  — aggregator independently chose a different (also wrong) label

Usage:
    python llm-experiments/pipeline/llm_agg_both_wrong.py --lang en
    python llm-experiments/pipeline/llm_agg_both_wrong.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import IGNORE_MODELS, PROPOSERS

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results_tmp" / "memerag_ext"

ACTIVE    = [m for m in PROPOSERS if m not in IGNORE_MODELS]
LABEL_MAP = {"Supported": 1, "Not Supported": 0}
DECODE    = {1: "Supported", 0: "Not Supported"}

LANGS = ["en", "de", "es", "fr", "hi"]


def find_best_bacc_file(lang: str) -> Path | None:
    matches = sorted((RESULTS_DIR / lang).glob("llm_agg_best_bacc_*.json"))
    return matches[0] if matches else None


def analyze(lang: str) -> list[dict]:
    path = find_best_bacc_file(lang)
    if not path:
        print(f"[{lang.upper()}] no best_bacc file found", file=sys.stderr)
        return []

    data    = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    results = []

    for r in records:
        agg_label = r.get("aggregator_label")
        gold      = r.get("gold_label")
        if not agg_label or not gold:
            continue

        outputs = r.get("model_outputs", {})
        valid   = []
        for m in ACTIVE:
            raw = outputs.get(m)
            lbl = raw.get("label") if isinstance(raw, dict) else raw
            enc = LABEL_MAP.get(lbl)
            if enc is not None:
                valid.append(enc)

        if not valid:
            continue

        n1 = sum(valid); n0 = len(valid) - n1
        if n1 == n0:
            continue

        majority_enc = 1 if n1 > n0 else 0
        gold_enc     = LABEL_MAP.get(gold)
        agg_enc      = LABEL_MAP.get(agg_label)

        if (majority_enc != gold_enc) and (agg_enc != gold_enc):
            results.append({
                "sid":      r.get("sample_id", "?"),
                "votes":    "%d:%d" % (n1, n0),
                "majority": DECODE[majority_enc],
                "agg":      agg_label,
                "gold":     gold,
                "followed": agg_enc == majority_enc,
            })

    return results


def print_results(lang: str, results: list[dict]) -> None:
    path = find_best_bacc_file(lang)
    agg_model = json.loads(path.read_text())["aggregator_model"].split("/")[-1] if path else "?"

    print(f"\n[{lang.upper()}] aggregator={agg_model}  both-wrong cases: {len(results)}")
    if not results:
        return

    followed = sum(1 for o in results if o["followed"])
    overrode = len(results) - followed
    print(f"  followed majority: {followed}  |  independently wrong: {overrode}")
    print()

    fmt = "  %-15s %-8s %-20s %-20s %-20s %s"
    print(fmt % ("sample_id", "votes", "majority", "agg_choice", "gold", "note"))
    print("  " + "-" * 95)
    for o in results:
        tag = "followed majority" if o["followed"] else "overrode majority (both wrong)"
        print(fmt % (str(o["sid"]), o["votes"], o["majority"], o["agg"], o["gold"], tag))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", default="en", choices=LANGS)
    parser.add_argument("--all",  action="store_true", help="Run all languages")
    args = parser.parse_args()

    langs = LANGS if args.all else [args.lang]

    print("Both-wrong analysis: majority vote AND best-BAcc LLM aggregator wrong")
    print("=" * 90)

    total_both_wrong = 0
    total_followed   = 0
    for lang in langs:
        results = analyze(lang)
        print_results(lang, results)
        total_both_wrong += len(results)
        total_followed   += sum(1 for o in results if o["followed"])

    if len(langs) > 1:
        print(f"\n{'='*90}")
        print(f"TOTAL both-wrong across all languages : {total_both_wrong}")
        print(f"  aggregator followed majority         : {total_followed} ({100*total_followed//total_both_wrong if total_both_wrong else 0}%)")
        print(f"  aggregator independently wrong       : {total_both_wrong - total_followed}")


if __name__ == "__main__":
    main()
