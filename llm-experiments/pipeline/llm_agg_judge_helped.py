"""
Cases where the strong LLM was WRONG as an individual judge but RIGHT as aggregator
on the same sample — i.e., reading the panel's reasoning corrected its own mistake.

Complement of llm_agg_judge_confused.py.

Usage:
    python llm-experiments/pipeline/llm_agg_judge_helped.py
    python llm-experiments/pipeline/llm_agg_judge_helped.py --lang de
    python llm-experiments/pipeline/llm_agg_judge_helped.py --all
    python llm-experiments/pipeline/llm_agg_judge_helped.py --all --save
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

    helped  = []   # judge wrong, agg right
    confused = 0   # judge right, agg wrong (for reference rate)
    both_wrong = 0
    both_right = 0

    for r in records:
        sid       = r.get("sample_id", "?")
        gold      = r.get("gold_label")
        agg_label = r.get("aggregator_label")
        outputs   = r.get("model_outputs", {})

        if not gold or not agg_label:
            continue

        raw     = outputs.get(agg_model)
        ind_lbl = raw.get("label") if isinstance(raw, dict) else raw
        if ind_lbl is None:
            continue

        ind_correct = ind_lbl == gold
        agg_correct = agg_label == gold

        if not ind_correct and agg_correct:
            helped.append({
                "sample_id":   str(sid),
                "gold":        gold,
                "judge_label": ind_lbl,
                "agg_label":   agg_label,
                "split":       vote_split(outputs),
            })
        elif ind_correct and not agg_correct:
            confused += 1
        elif not ind_correct and not agg_correct:
            both_wrong += 1
        else:
            both_right += 1

    total_judge_wrong = len(helped) + both_wrong   # cases where judge was wrong
    return {
        "lang":             lang,
        "agg_model":        agg_model,
        "total":            len(records),
        "helped":           helped,
        "n_confused":       confused,
        "n_both_wrong":     both_wrong,
        "n_both_right":     both_right,
        "total_judge_wrong": total_judge_wrong,
    }


def fmt_results(result: dict) -> str:
    lines = []
    lang      = result["lang"]
    agg_model = result["agg_model"].split("/")[-1]
    helped    = result["helped"]
    total     = result["total"]
    n_jw      = result["total_judge_wrong"]
    pct_fixed = 100 * len(helped) / n_jw if n_jw else 0

    lines.append(f"\n{'='*80}")
    lines.append(
        f"[{lang.upper()}]  model = {agg_model}  |  "
        f"aggregation helped = {len(helped)} / {n_jw} judge-wrong cases "
        f"({pct_fixed:.1f}% fixed)"
    )
    lines.append(
        f"         (judge wrong+agg wrong={result['n_both_wrong']}  |  "
        f"judge right+agg wrong={result['n_confused']}  |  "
        f"both right={result['n_both_right']})"
    )
    lines.append(f"{'='*80}")

    if not helped:
        lines.append("  (none)")
        return "\n".join(lines)

    split_counts = Counter(c["split"] for c in helped)
    lines.append(f"\n  Split breakdown of helped cases:")
    for sp in sorted(split_counts, key=lambda s: -int(s.split("-")[0])):
        lines.append(f"    {sp}: {split_counts[sp]}")

    hdr = f"\n  {'sample_id':<22} {'gold':<16} {'as judge':<16} {'as agg':<16} {'panel split'}"
    lines.append(hdr)
    lines.append("  " + "-"*75)
    for c in helped:
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
    parser.add_argument("--save",  action="store_true")
    args  = parser.parse_args()
    langs = LANGS if args.all else [args.lang]

    print("Aggregation HELPED: judge wrong individually, but RIGHT as aggregator")
    print("Hypothesis: panel reasoning corrected the model's own mistake\n")

    all_results = []
    for lang in langs:
        result = analyze(lang)
        if result:
            all_results.append(result)
            print(fmt_results(result))
            if args.save:
                out = RESULTS_DIR / lang / f"llm_agg_judge_helped_{lang}.txt"
                out.write_text(fmt_results(result), encoding="utf-8")
                print(f"\n  Saved: {out}", file=sys.stderr)

    if len(all_results) > 1:
        print(f"\n\n{'='*80}")
        print("CROSS-LANGUAGE SUMMARY")
        print(f"{'='*80}")
        hdr = (f"  {'Lang':<6} {'Model':<28} {'Helped':>8} {'JudgeWrong':>12} "
               f"{'%Fixed':>8} {'Confused':>10} {'BothWrong':>11}")
        print(hdr)
        print("  " + "-"*80)
        for r in all_results:
            n_jw  = r["total_judge_wrong"]
            pct   = 100 * len(r["helped"]) / n_jw if n_jw else 0
            print(
                f"  {r['lang'].upper():<6} {r['agg_model'].split('/')[-1]:<28} "
                f"{len(r['helped']):>8} {n_jw:>12} {pct:>7.1f}% "
                f"{r['n_confused']:>10} {r['n_both_wrong']:>11}"
            )


if __name__ == "__main__":
    main()
