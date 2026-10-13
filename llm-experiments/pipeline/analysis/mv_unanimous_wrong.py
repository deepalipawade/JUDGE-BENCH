"""
Unanimous wrong case analysis.

A unanimous wrong case = all 7 judges voted the same label AND were wrong.
Majority vote, OWI, ISP, Dawid-Skene all operate on labels — when all 7 agree
they produce the same wrong label by definition.

Only an LLM aggregator (which reads the actual judge reasoning, not just labels)
has any chance of catching these cases.

This script checks: for each unanimous wrong sample, did best_bacc LLM /
random LLM get it right?

Saves: results_tmp/memerag_ext/unanimous_wrong_analysis.md

Usage:
    python llm-experiments/pipeline/mv_unanimous_wrong.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results_tmp" / "memerag_ext"
LANGS        = ["en", "de", "es", "fr", "hi"]

LABEL_MAP = {"Supported": 1, "Not Supported": 0}

# Best-BAcc LLM aggregator file per language
BEST_BACC_FILE = {
    "en": "llm_agg_best_bacc_qwen3_6_35b_en.json",
    "de": "llm_agg_best_bacc_gemma_4_26b_a4b_it_maas_de.json",
    "es": "llm_agg_best_bacc_qwen3_6_35b_es.json",
    "fr": "llm_agg_best_bacc_qwen3_6_35b_fr.json",
    "hi": "llm_agg_best_bacc_gemma_4_26b_a4b_it_maas_hi.json",
}


def load_agg_labels(path: Path) -> dict[str, int | None]:
    """Return {sample_id: label (0/1)} from an llm_agg file."""
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8")).get("records", [])
    return {
        r["sample_id"]: LABEL_MAP.get(r.get("aggregator_label"))
        for r in records
        if "sample_id" in r
    }


def load_algo_labels(lang: str) -> dict[str, dict[str, int | None]]:
    """Return {sample_id: {majority, owi, isp, ds}} from algo_agg file."""
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


def tick(correct: bool | None) -> str:
    if correct is None:
        return "—"
    return "✓ correct" if correct else "✗ wrong"


def analyze_lang(lang: str) -> str | None:
    consensus_p = RESULTS_ROOT / lang / f"analysis_consensus_{lang}.json"
    if not consensus_p.exists():
        return None

    consensus     = json.loads(consensus_p.read_text(encoding="utf-8"))
    unan_wrong    = consensus.get("summary", {}).get("unanimous_wrong", {})
    sample_ids    = unan_wrong.get("sample_ids", [])
    n_cases       = unan_wrong.get("n_cases", 0)

    if not sample_ids:
        return f"\n## {lang.upper()} — no unanimous wrong cases\n"

    best_bacc_labels = load_agg_labels(
        RESULTS_ROOT / lang / BEST_BACC_FILE[lang]
    )
    random_labels = load_agg_labels(
        RESULTS_ROOT / lang / f"llm_agg_random_seed42_{lang}.json"
    )
    algo_labels = load_algo_labels(lang)

    # Load gold labels from judgement file
    judg_p  = RESULTS_ROOT / lang / f"memerag_judgement_{lang}.json"
    gold_map: dict[str, int | None] = {}
    if judg_p.exists():
        for rec in json.loads(judg_p.read_text(encoding="utf-8")).get("records", []):
            sid = rec.get("sample_id")
            if sid:
                gold_map[sid] = LABEL_MAP.get(rec.get("gold_label"))

    lines = [f"\n## {lang.upper()}  ({n_cases} cases)\n"]

    summary = {"best_bacc": 0, "random": 0, "owi": 0, "isp": 0, "ds": 0}

    for sid in sample_ids:
        gold  = gold_map.get(sid)
        algo  = algo_labels.get(sid, {})

        if best_bacc_labels.get(sid) == gold:  summary["best_bacc"] += 1
        if random_labels.get(sid)    == gold:  summary["random"]    += 1
        if algo.get("owi")           == gold:  summary["owi"]       += 1
        if algo.get("isp")           == gold:  summary["isp"]       += 1
        if algo.get("ds")            == gold:  summary["ds"]        += 1

    rows = [
        ["Majority vote",  f"0/{n_cases}"],
        ["OWI",            f"{summary['owi']}/{n_cases}"],
        ["ISP",            f"{summary['isp']}/{n_cases}"],
        ["Dawid-Skene",    f"{summary['ds']}/{n_cases}"],
        ["Random LLM",     f"{summary['random']}/{n_cases}"],
        ["Best-BAcc LLM",  f"{summary['best_bacc']}/{n_cases}"],
    ]
    lines += md_table(["Method", "Recovered correctly"], rows)
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    lines = ["# Unanimous Wrong Case Analysis\n"]
    lines.append(
        "All 7 judges agreed and were wrong. "
        "Majority vote = 0% recovery by definition. "
        "OWI/ISP/DS only read labels — unanimous input → same wrong output. "
        "Only LLM aggregators (read reasoning) can recover.\n"
    )

    for lang in LANGS:
        section = analyze_lang(lang)
        if section:
            lines.append(section)

    report = "\n".join(lines)
    print(report)

    out = RESULTS_ROOT / "unanimous_wrong_analysis.md"
    out.write_text(report, encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
