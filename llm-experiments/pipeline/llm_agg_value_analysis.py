"""
Does aggregation add value over the individual judge?

2x2 breakdown per language:
  ┌──────────────────┬──────────────┬──────────────┐
  │                  │  Agg RIGHT   │  Agg WRONG   │
  ├──────────────────┼──────────────┼──────────────┤
  │  Judge RIGHT     │  Both right  │  Confused    │  ← aggregation HURT
  │  Judge WRONG     │  Helped      │  Both wrong  │  ← aggregation added no value
  └──────────────────┴──────────────┴──────────────┘

"Value added" = Helped − Confused  (net cases where aggregation changed outcome for better)

Usage:
    python llm-experiments/pipeline/llm_agg_value_analysis.py --all
    python llm-experiments/pipeline/llm_agg_value_analysis.py --lang en
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


def analyze(lang: str) -> dict | None:
    path = find_best_bacc_file(lang)
    if not path:
        print(f"[{lang.upper()}] no best_bacc file found", file=sys.stderr)
        return None

    data      = json.loads(path.read_text(encoding="utf-8"))
    agg_model = data.get("aggregator_model", "?")
    records   = data.get("records", [])

    both_right = 0   # judge right, agg right
    confused   = 0   # judge right, agg wrong  → aggregation HURT
    helped     = 0   # judge wrong, agg right  → aggregation HELPED
    both_wrong = 0   # judge wrong, agg wrong  → aggregation added no value
    skipped    = 0

    for r in records:
        gold      = r.get("gold_label")
        agg_label = r.get("aggregator_label")
        outputs   = r.get("model_outputs", {})

        if not gold or not agg_label:
            skipped += 1
            continue

        raw     = outputs.get(agg_model)
        ind_lbl = raw.get("label") if isinstance(raw, dict) else raw
        if ind_lbl is None:
            skipped += 1
            continue

        ind_ok = ind_lbl == gold
        agg_ok = agg_label == gold

        if     ind_ok and     agg_ok: both_right += 1
        elif   ind_ok and not agg_ok: confused   += 1
        elif not ind_ok and  agg_ok:  helped     += 1
        else:                          both_wrong += 1

    total = both_right + confused + helped + both_wrong
    return {
        "lang":       lang,
        "model":      agg_model.split("/")[-1],
        "total":      total,
        "skipped":    skipped,
        "both_right": both_right,
        "confused":   confused,
        "helped":     helped,
        "both_wrong": both_wrong,
    }


def print_table(results: list[dict]) -> None:
    W = 82

    print("\n" + "="*W)
    print("  Aggregation value analysis — does reading the panel help or hurt?")
    print("="*W)

    for r in results:
        lang  = r["lang"].upper()
        model = r["model"]
        N     = r["total"]

        br = r["both_right"]
        cf = r["confused"]
        hp = r["helped"]
        bw = r["both_wrong"]

        judge_right = br + cf
        judge_wrong = hp + bw
        net         = hp - cf

        print(f"\n  [{lang}]  {model}  (n={N})")
        print(f"  {'─'*70}")
        print(f"  {'':30}  {'Agg RIGHT':>14}  {'Agg WRONG':>14}")
        print(f"  {'─'*70}")
        print(f"  {'Judge RIGHT':30}  {br:>8} ({br/N*100:4.1f}%)  {cf:>8} ({cf/N*100:4.1f}%)"
              f"   ← {cf} cases HURT by aggregation")
        print(f"  {'Judge WRONG':30}  {hp:>8} ({hp/N*100:4.1f}%)  {bw:>8} ({bw/N*100:4.1f}%)"
              f"   ← {bw} cases aggregation added NO value")
        print(f"  {'─'*70}")

        pct_no_value = 100 * bw / judge_wrong if judge_wrong else 0
        pct_helped   = 100 * hp / judge_wrong if judge_wrong else 0
        pct_hurt     = 100 * cf / judge_right if judge_right else 0

        print(f"\n  Of {judge_wrong} judge-wrong cases  →  fixed: {hp} ({pct_helped:.1f}%)  |  "
              f"no value: {bw} ({pct_no_value:.1f}%)")
        print(f"  Of {judge_right} judge-right cases  →  hurt:  {cf} ({pct_hurt:.1f}%)")
        print(f"  Net cases changed for better: {net:+d}  "
              f"({'positive' if net > 0 else 'negative' if net < 0 else 'zero'} contribution)")

    # ── Cross-language summary ────────────────────────────────────────────────
    print(f"\n\n{'='*W}")
    print("  CROSS-LANGUAGE SUMMARY")
    print(f"{'='*W}")
    hfmt = f"  {'Lang':<5} {'Model':<26} {'N':>5}  {'BothRight':>10} {'Helped':>8} {'Confused':>10} {'BothWrong':>11}  {'Net':>5}  {'%NoValue':>9}"
    print(hfmt)
    print("  " + "─"*76)
    for r in results:
        N   = r["total"]
        net = r["helped"] - r["confused"]
        jw  = r["helped"] + r["both_wrong"]
        pct_no_value = 100 * r["both_wrong"] / jw if jw else 0
        print(
            f"  {r['lang'].upper():<5} {r['model']:<26} {N:>5}  "
            f"{r['both_right']:>7} ({r['both_right']/N*100:4.1f}%)  "
            f"{r['helped']:>4} ({r['helped']/N*100:4.1f}%)  "
            f"{r['confused']:>6} ({r['confused']/N*100:4.1f}%)  "
            f"{r['both_wrong']:>7} ({r['both_wrong']/N*100:4.1f}%)  "
            f"{net:>+4}   {pct_no_value:>6.1f}%"
        )

    # ── Aggregated totals ─────────────────────────────────────────────────────
    total_N  = sum(r["total"]      for r in results)
    total_br = sum(r["both_right"] for r in results)
    total_cf = sum(r["confused"]   for r in results)
    total_hp = sum(r["helped"]     for r in results)
    total_bw = sum(r["both_wrong"] for r in results)
    total_jw = total_hp + total_bw
    net_all  = total_hp - total_cf

    print("  " + "─"*76)
    print(
        f"  {'ALL':<5} {'':26} {total_N:>5}  "
        f"{total_br:>7} ({total_br/total_N*100:4.1f}%)  "
        f"{total_hp:>4} ({total_hp/total_N*100:4.1f}%)  "
        f"{total_cf:>6} ({total_cf/total_N*100:4.1f}%)  "
        f"{total_bw:>7} ({total_bw/total_N*100:4.1f}%)  "
        f"{net_all:>+4}   {100*total_bw/total_jw:>6.1f}%"
    )
    print(f"\n  %NoValue = of cases where judge was wrong, aggregation also wrong (no correction)")
    print(f"  Net      = Helped − Confused  (positive = aggregation net-beneficial)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--lang", choices=LANGS, default="en")
    grp.add_argument("--all",  action="store_true")
    parser.add_argument("--save", action="store_true")
    args  = parser.parse_args()
    langs = LANGS if args.all else [args.lang]

    results = [r for lang in langs if (r := analyze(lang)) is not None]
    if results:
        print_table(results)
        if args.save and len(results) > 1:
            out = RESULTS_DIR / "llm_agg_value_analysis.txt"
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_table(results)
            out.write_text(buf.getvalue(), encoding="utf-8")
            print(f"\n  Saved: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
