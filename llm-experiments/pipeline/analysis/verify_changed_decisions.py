"""
Independent verification that helped + confused == actual label changes.

Directly compares model_outputs[agg_model]["label"] vs aggregator_label
for every record — no correctness classification needed.

Usage:
    python llm-experiments/pipeline/analysis/verify_changed_decisions.py --all
    python llm-experiments/pipeline/analysis/verify_changed_decisions.py --lang en
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import IGNORE_MODELS, PROPOSERS

ROOT        = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results_tmp" / "memerag_ext"
LANGS       = ["en", "de", "es", "fr", "hi"]


def find_best_bacc_file(lang: str) -> Path | None:
    matches = sorted((RESULTS_DIR / lang).glob("llm_agg_best_bacc_*.json"))
    return matches[0] if matches else None


def verify(lang: str) -> None:
    path = find_best_bacc_file(lang)
    if not path:
        print(f"[{lang.upper()}] no best_bacc file found")
        return

    data      = json.loads(path.read_text(encoding="utf-8"))
    agg_model = data.get("aggregator_model", "?")
    records   = data.get("records", [])

    # Method A: direct label comparison
    changed_direct = 0
    same_direct    = 0
    skipped        = 0

    for r in records:
        agg_label = r.get("aggregator_label")
        outputs   = r.get("model_outputs", {})
        raw       = outputs.get(agg_model)
        ind_lbl   = raw.get("label") if isinstance(raw, dict) else raw

        if agg_label is None or ind_lbl is None:
            skipped += 1
            continue

        if agg_label != ind_lbl:
            changed_direct += 1
        else:
            same_direct += 1

    total_direct = changed_direct + same_direct

    # Method B: helped + confused from correctness analysis
    helped = confused = both_right = both_wrong = 0
    for r in records:
        gold      = r.get("gold_label")
        agg_label = r.get("aggregator_label")
        outputs   = r.get("model_outputs", {})
        raw       = outputs.get(agg_model)
        ind_lbl   = raw.get("label") if isinstance(raw, dict) else raw

        if not gold or not agg_label or ind_lbl is None:
            continue

        ind_ok = ind_lbl == gold
        agg_ok = agg_label == gold

        if     ind_ok and     agg_ok: both_right += 1
        elif   ind_ok and not agg_ok: confused   += 1
        elif not ind_ok and  agg_ok:  helped     += 1
        else:                          both_wrong += 1

    changed_indirect = helped + confused

    # Report
    print(f"\n[{lang.upper()}]  aggregator = {agg_model.split('/')[-1]}")
    print(f"  {'─'*55}")
    print(f"  Method A (direct label diff):   changed = {changed_direct:>4}  same = {same_direct:>4}  skipped = {skipped}")
    print(f"  Method B (helped + confused):   changed = {changed_indirect:>4}  (helped={helped}, confused={confused})")
    match = changed_direct == changed_indirect
    print(f"  Match: {'✓ YES' if match else '✗ NO — DISCREPANCY'}")
    if not match:
        print(f"  Difference: {abs(changed_direct - changed_indirect)} — likely skipped records with missing gold label")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--lang", choices=LANGS, default="en")
    grp.add_argument("--all",  action="store_true")
    args  = parser.parse_args()
    langs = LANGS if args.all else [args.lang]

    print("Verifying: helped + confused == actual label changes")
    print("=" * 60)
    for lang in langs:
        verify(lang)


if __name__ == "__main__":
    main()
