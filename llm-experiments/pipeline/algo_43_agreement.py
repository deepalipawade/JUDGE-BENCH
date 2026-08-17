"""
Algo agreement analysis on ALL 4-3 vote split cases.

For each 4-3 case, checks whether OWI / ISP / DS sides with the
4-model majority or the 3-model minority — and whether that choice
was correct.

Tables produced per language:
  1. Agreement breakdown — how often each method sides with majority
     vs minority, and accuracy in each scenario.
  2. Overall accuracy on all 4-3 cases (majority vs DS vs OWI vs ISP).

Saves: results_tmp/memerag_ext/agg_algo_43_agreement.md

Usage:
    python llm-experiments/pipeline/algo_43_agreement.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results_tmp" / "memerag_ext"
LANGS        = ["en", "de", "es", "fr", "hi"]

PROPOSERS = [
    "Qwen3.6-35B",
    "google/gemma-4-26b-a4b-it-maas",
    "MiniMax-M2.7",
    "GPT-OSS-120b",
    "meta/llama-3.3-70b-instruct-maas",
    "gemini-2.5-flash",
    "Apertus-8B",
]
LABEL_MAP = {"Supported": 1, "Not Supported": 0}
ALGO_METHODS = ["majority", "owi", "isp", "ds"]
METHOD_LABELS = {
    "majority": "Majority vote",
    "owi":      "OWI",
    "isp":      "ISP",
    "ds":       "Dawid-Skene",
}


def load_algo_labels(lang: str) -> dict[str, dict]:
    p = RESULTS_ROOT / lang / f"algo_agg_{lang}.json"
    if not p.exists():
        return {}
    records = json.loads(p.read_text(encoding="utf-8")).get("records", [])
    result = {}
    for rec in records:
        sid = rec.get("sample_id")
        if sid:
            result[sid] = {
                "majority": LABEL_MAP.get(rec.get("__majority")),
                "owi":      LABEL_MAP.get(rec.get("__owi")),
                "isp":      LABEL_MAP.get(rec.get("__isp")),
                "ds":       LABEL_MAP.get(rec.get("__ds")),
            }
    return result


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def pad(s: str, w: int) -> str:
        return s.ljust(w)

    return (
        ["| " + " | ".join(pad(h, widths[i]) for i, h in enumerate(headers)) + " |",
         "|-" + "-|-".join("-" * w for w in widths) + "-|"]
        + ["| " + " | ".join(pad(c, widths[i]) for i, c in enumerate(row)) + " |"
           for row in rows]
    )


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100*n/d:.0f}%)" if d else "—"


def analyze_lang(lang: str) -> str | None:
    judg_p = RESULTS_ROOT / lang / f"memerag_judgement_{lang}.json"
    if not judg_p.exists():
        return None

    records   = json.loads(judg_p.read_text(encoding="utf-8")).get("records", [])
    algo_lbls = load_algo_labels(lang)

    stats: dict[str, dict[str, int]] = {
        m: {"majo_sides": 0, "mino_sides": 0,
            "majo_correct": 0, "mino_correct": 0,
            "total": 0}
        for m in ALGO_METHODS
    }

    for rec in records:
        sid  = rec.get("sample_id")
        gold = LABEL_MAP.get(rec.get("gold_label"))
        if gold is None or sid not in algo_lbls:
            continue

        outs   = rec.get("model_outputs", {})
        labels = {}
        for m in PROPOSERS:
            out = outs.get(m)
            if out:
                lbl = LABEL_MAP.get(out.get("label"))
                if lbl is not None:
                    labels[m] = lbl

        if len(labels) < 7:
            continue

        n_sup = sum(labels.values())
        if n_sup not in (3, 4):
            continue

        majority_lbl = 1 if n_sup == 4 else 0
        algo         = algo_lbls[sid]

        for method in ALGO_METHODS:
            pred = algo.get(method)
            if pred is None:
                continue

            sides_majority = (pred == majority_lbl)
            correct        = (pred == gold)

            stats[method]["total"] += 1
            if sides_majority:
                stats[method]["majo_sides"] += 1
                if correct:
                    stats[method]["majo_correct"] += 1
            else:
                stats[method]["mino_sides"] += 1
                if correct:
                    stats[method]["mino_correct"] += 1

    total_43 = stats["majority"]["total"]
    if total_43 == 0:
        return None

    lines = [f"\n## {lang.upper()}  ({total_43} total 4-3 cases)\n"]

    # Single table
    lines.append(
        "### 4-3 split behaviour\n\n"
        "> Note: 'Follows majority' and 'Correct when follows majority' are different. "
        "A method can blindly follow majority 100% of the time and still be wrong ~50% "
        "of the time, because majority itself is unreliable on 4-3 cases.\n"
    )
    rows = []
    for m in ALGO_METHODS:
        s       = stats[m]
        t       = s["total"]
        correct = s["majo_correct"] + s["mino_correct"]
        if t == 0:
            continue
        rows.append([
            METHOD_LABELS[m],
            pct(s["majo_sides"],   t),
            pct(s["majo_correct"], s["majo_sides"]) if s["majo_sides"] else "—",
            pct(s["mino_correct"], s["mino_sides"]) if s["mino_sides"] else "—",
            pct(correct, t),
        ])
    lines += md_table(
        ["Method",
         "Follows majority",
         "Correct when follows majority",
         "Correct when breaks from majority",
         "Overall accuracy"],
        rows,
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    lines = ["# Algo Agreement on 4-3 Vote Split Cases\n"]
    lines.append(
        "## What this analysis measures\n\n"
        "In a 4-3 vote split, 4 judges vote one way and 3 vote the other. "
        "Majority vote always follows the 4. This analysis asks: do weighted "
        "aggregation algorithms (OWI, ISP, Dawid-Skene) do the same — or do "
        "they sometimes break from the majority and side with the 3?\n\n"
        "**Column definitions:**\n\n"
        "- **Follows majority** — how often the method produces the same label "
        "as the 4-model majority. Majority vote is always 100% by definition.\n\n"
        "- **Correct when follows majority** — of the cases where the method "
        "agreed with majority, what % was that decision actually right? "
        "*Important: following majority is not the same as being correct.* "
        "We already know from the split analysis that majority vote is only ~38–57% "
        "correct on 4-3 cases — so agreeing with majority can still mean being wrong.\n\n"
        "- **Correct when breaks from majority** — of the cases where the method "
        "disagreed with majority (sided with the 3-model minority), what % was "
        "the method right? A high value here means the algorithm has learned to "
        "identify when the minority is actually correct — i.e. it can recover "
        "cases that majority vote gets wrong.\n\n"
        "- **Overall accuracy** — across all 4-3 cases, what % did the method "
        "get right regardless of which side it took.\n"
    )

    for lang in LANGS:
        section = analyze_lang(lang)
        if section:
            lines.append(section)
        else:
            lines.append(f"\n## {lang.upper()} — no data found\n")

    report = "\n".join(lines)
    print(report)

    out = RESULTS_ROOT / "agg_algo_43_agreement.md"
    out.write_text(report, encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
