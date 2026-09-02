"""
Cases where the strong LLM was RIGHT as an individual judge but WRONG as aggregator
on the same sample — i.e., reading the noisy panel's reasoning confused it.

For each language, reports:
  - Sample ID
  - Gold label
  - Individual judge label (correct)
  - Aggregator label (wrong)
  - Vote split of the panel (how noisy was the input)

Usage:
    python llm-experiments/pipeline/llm_agg_judge_confused.py
    python llm-experiments/pipeline/llm_agg_judge_confused.py --lang de
    python llm-experiments/pipeline/llm_agg_judge_confused.py --all
    python llm-experiments/pipeline/llm_agg_judge_confused.py --all --save
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
LANGS     = ["en", "de", "es", "fr", "hi"]


def find_best_bacc_file(lang: str) -> Path | None:
    matches = sorted((RESULTS_DIR / lang).glob("llm_agg_best_bacc_*.json"))
    return matches[0] if matches else None


def vote_split(outputs: dict) -> str:
    vals = []
    for m in ACTIVE:
        raw = outputs.get(m)
        lbl = raw.get("label") if isinstance(raw, dict) else raw
        enc = LABEL_MAP.get(lbl)
        if enc is not None:
            vals.append(enc)
    if not vals:
        return "?"
    n1 = sum(vals); n0 = len(vals) - n1
    return f"{max(n1,n0)}-{min(n1,n0)}"


def analyze(lang: str) -> dict:
    path = find_best_bacc_file(lang)
    if not path:
        print(f"[{lang.upper()}] no best_bacc file found", file=sys.stderr)
        return {}

    data      = json.loads(path.read_text(encoding="utf-8"))
    agg_model = data.get("aggregator_model", "?")
    records   = data.get("records", [])

    cases = []
    for r in records:
        sid       = r.get("sample_id", "?")
        gold      = r.get("gold_label")
        agg_label = r.get("aggregator_label")
        outputs   = r.get("model_outputs", {})

        if not gold or not agg_label:
            continue

        # Individual judge label for this model on this sample
        raw      = outputs.get(agg_model)
        ind_lbl  = raw.get("label") if isinstance(raw, dict) else raw

        if ind_lbl is None:
            continue

        ind_correct = ind_lbl == gold
        agg_correct = agg_label == gold

        if ind_correct and not agg_correct:
            cases.append({
                "sample_id":   str(sid),
                "gold":        gold,
                "judge_label": ind_lbl,
                "agg_label":   agg_label,
                "split":       vote_split(outputs),
            })

    return {
        "lang":       lang,
        "agg_model":  agg_model,
        "total":      len(records),
        "cases":      cases,
    }


def fmt_results(result: dict) -> str:
    lines = []
    lang      = result["lang"]
    agg_model = result["agg_model"].split("/")[-1]
    cases     = result["cases"]
    total     = result["total"]

    lines.append(f"\n{'='*80}")
    lines.append(
        f"[{lang.upper()}]  model = {agg_model}  |  "
        f"confused cases = {len(cases)} / {total} "
        f"({100*len(cases)/total:.1f}%)"
    )
    lines.append(f"{'='*80}")

    if not cases:
        lines.append("  (none)")
        return "\n".join(lines)

    # Count split breakdown
    from collections import Counter
    split_counts = Counter(c["split"] for c in cases)

    lines.append(f"\n  Split breakdown of confused cases:")
    for sp in sorted(split_counts, key=lambda s: -int(s.split("-")[0])):
        lines.append(f"    {sp}: {split_counts[sp]}")

    hdr = f"\n  {'sample_id':<22} {'gold':<16} {'as judge':<16} {'as agg':<16} {'panel split'}"
    lines.append(hdr)
    lines.append("  " + "-"*75)
    for c in cases:
        lines.append(
            f"  {c['sample_id']:<22} {c['gold']:<16} {c['judge_label']:<16} "
            f"{c['agg_label']:<16} {c['split']}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang",  default="en", choices=LANGS)
    parser.add_argument("--all",   action="store_true")
    parser.add_argument("--save",  action="store_true",
                        help="Save output to results_tmp/<lang>/llm_agg_judge_confused_<lang>.txt")
    args  = parser.parse_args()
    langs = LANGS if args.all else [args.lang]

    print("Judge-as-individual CORRECT but same model as AGGREGATOR WRONG")
    print("Hypothesis: reading noisy panel reasoning confused the model\n")

    all_results = []
    for lang in langs:
        result = analyze(lang)
        if result:
            all_results.append(result)
            print(fmt_results(result))
            if args.save:
                out = RESULTS_DIR / lang / f"llm_agg_judge_confused_{lang}.txt"
                out.write_text(fmt_results(result), encoding="utf-8")
                print(f"\n  Saved: {out}", file=sys.stderr)

    # Cross-language summary table
    if len(all_results) > 1:
        print(f"\n\n{'='*80}")
        print("CROSS-LANGUAGE SUMMARY")
        print(f"{'='*80}")
        hdr = f"  {'Lang':<6} {'Model':<28} {'Confused':<10} {'Total':<8} {'%'}"
        print(hdr)
        print("  " + "-"*60)
        for r in all_results:
            pct = 100 * len(r["cases"]) / r["total"] if r["total"] else 0
            print(f"  {r['lang'].upper():<6} {r['agg_model'].split('/')[-1]:<28} "
                  f"{len(r['cases']):<10} {r['total']:<8} {pct:.1f}%")


if __name__ == "__main__":
    main()
