"""
Aggregator behaviour by vote split — EN (best-BAcc LLM aggregator).

For each vote split (7-0, 6-1, 5-2, 4-3), shows:
  - How often the aggregator FOLLOWED the majority
  - How often it OVERRODE (went with minority)
  - Win/loss rate for each behaviour

Hypothesis: aggregator mostly follows majority, but becomes unpredictable (higher
override rate) at 4-3 splits where the evidence is genuinely ambiguous.

Usage:
    python llm-experiments/pipeline/llm_agg_split_analysis.py
    python llm-experiments/pipeline/llm_agg_split_analysis.py --lang de
    python llm-experiments/pipeline/llm_agg_split_analysis.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import IGNORE_MODELS, LANG, PROPOSERS

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results_tmp" / "memerag_ext"

ACTIVE    = [m for m in PROPOSERS if m not in IGNORE_MODELS]
LABEL_MAP = {"Supported": 1, "Not Supported": 0}
DECODE    = {1: "Supported", 0: "Not Supported"}
LANGS     = ["en", "de", "es", "fr", "hi"]


def split_label(n_maj: int, n_min: int) -> str:
    return f"{n_maj}-{n_min}"


def find_best_bacc_file(lang: str) -> Path | None:
    matches = sorted((RESULTS_DIR / lang).glob("llm_agg_best_bacc_*.json"))
    return matches[0] if matches else None


def analyze(lang: str) -> dict:
    path = find_best_bacc_file(lang)
    if not path:
        print(f"[{lang.upper()}] no best_bacc file found", file=sys.stderr)
        return {}

    data    = json.loads(path.read_text(encoding="utf-8"))
    agg_model = data.get("aggregator_model", "?")
    records = data.get("records", [])

    # stats[split_label][behaviour][outcome] = list of sample_ids
    # behaviour: "followed" | "overrode"
    # outcome:   "win" | "loss"
    stats: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: {"followed": {"win": [], "loss": []},
                 "overrode": {"win": [], "loss": []}}
    )
    skipped = 0

    for r in records:
        agg_label = r.get("aggregator_label")
        gold      = r.get("gold_label")
        sid       = r.get("sample_id", "?")
        if not agg_label or not gold:
            skipped += 1
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
            skipped += 1
            continue

        n1 = sum(valid); n0 = len(valid) - n1
        if n1 == n0:
            skipped += 1
            continue  # tie — skip

        n_maj = max(n1, n0); n_min = min(n1, n0)
        split = split_label(n_maj, n_min)

        majority_enc = 1 if n1 > n0 else 0
        agg_enc      = LABEL_MAP.get(agg_label)
        gold_enc     = LABEL_MAP.get(gold)

        behaviour = "followed" if agg_enc == majority_enc else "overrode"
        outcome   = "win"      if agg_enc == gold_enc     else "loss"

        stats[split][behaviour][outcome].append(str(sid))

    return {"agg_model": agg_model, "stats": stats, "skipped": skipped,
            "total": len(records)}


def print_results(lang: str, result: dict) -> None:
    if not result:
        return

    agg = result["agg_model"].split("/")[-1]
    print(f"\n{'='*85}")
    print(f"[{lang.upper()}]  aggregator = {agg}  |  total records = {result['total']}")
    print(f"{'='*85}")

    # Define canonical split order (most to least agreement)
    split_order = ["7-0", "6-1", "5-2", "4-3", "3-4", "2-5", "1-6", "0-7"]
    # normalise: always use maj-min ordering
    stats = result["stats"]
    present = sorted(stats.keys(),
                     key=lambda s: (-int(s.split("-")[0]), int(s.split("-")[1])))

    hdr = f"  {'Split':<8} {'Total':>6}  {'Followed maj':>14}  {'  correct':>9}  {'  wrong':>7}  " \
          f"{'Overrode (minority)':>20}  {'  correct':>9}  {'  wrong':>7}  {'Override%':>9}"
    print(hdr)
    print("  " + "-" * 97)

    all_splits_total = 0
    all_override_total = 0

    for split in present:
        s = stats[split]
        f_win  = len(s["followed"]["win"])
        f_loss = len(s["followed"]["loss"])
        o_win  = len(s["overrode"]["win"])
        o_loss = len(s["overrode"]["loss"])
        total  = f_win + f_loss + o_win + o_loss
        overrides = o_win + o_loss
        override_pct = 100 * overrides / total if total else 0
        all_splits_total   += total
        all_override_total += overrides

        print(f"  {split:<8} {total:>6}  {'followed':>14}  {f_win:>9}  {f_loss:>7}  "
              f"{'overrode':>20}  {o_win:>9}  {o_loss:>7}  {override_pct:>8.1f}%")

    overall_pct = 100 * all_override_total / all_splits_total if all_splits_total else 0
    print("  " + "-" * 97)
    print(f"  {'TOTAL':<8} {all_splits_total:>6}  {'':>14}  {'':>9}  {'':>7}  "
          f"{'':>20}  {'':>9}  {'':>7}  {overall_pct:>8.1f}%")

    # Detailed sample IDs for overrides, grouped by split
    print(f"\n  Override sample IDs by split:")
    for split in present:
        wins  = stats[split]["overrode"]["win"]
        losses= stats[split]["overrode"]["loss"]
        if not wins and not losses:
            continue
        print(f"\n  [{split}]  overrides={len(wins)+len(losses)}  "
              f"correct={len(wins)}  wrong={len(losses)}")
        if wins:
            print(f"    CORRECT: {', '.join(wins)}")
        if losses:
            print(f"    WRONG  : {', '.join(losses)}")


def save_output(lang: str, result: dict, out_path: Path) -> None:
    """Write the printed output to a text file."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_results(lang, result)
    out_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"  Saved: {out_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", default=LANG, choices=LANGS)
    parser.add_argument("--all",  action="store_true")
    parser.add_argument("--save", action="store_true",
                        help="Also write output to results_tmp/<lang>/llm_agg_split_analysis_<lang>.txt")
    args = parser.parse_args()

    langs = LANGS if args.all else [args.lang]

    print("Aggregator behaviour by vote split")
    print("Hypothesis: aggregator follows majority at clear splits, gets unpredictable at 4-3")

    for lang in langs:
        result = analyze(lang)
        print_results(lang, result)
        if args.save and result:
            out = RESULTS_DIR / lang / f"llm_agg_split_analysis_{lang}.txt"
            save_output(lang, result, out)


if __name__ == "__main__":
    main()
