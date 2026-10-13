"""
Tests whether 4-3 vote splits correlate with response length/complexity.

Hypothesis (from RAG-groundedness literature): longer answer segments
concentrate in the ambiguous middle of the grounding scale, leading to
more disagreement — which shows up as 4-3 splits in our panel.

Measures per item:
  - answer_segment length (chars, words)
  - total evidence passage length (chars)
  - query length (words)
  - answer/evidence ratio (compression ratio proxy)

Groups items by vote split: unanimous (7-0), near (6-1 / 5-2), contested (4-3)
Reports: mean/median per group + Mann-Whitney U (contested vs unanimous)

Usage:
    python llm-experiments/pipeline/analysis/response_length_split_analysis.py --all
    python llm-experiments/pipeline/analysis/response_length_split_analysis.py --lang en
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import IGNORE_MODELS, PROPOSERS

ROOT        = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results_tmp" / "memerag_ext"
LANGS       = ["en", "de", "es", "fr", "hi"]
ACTIVE      = [m for m in PROPOSERS if m not in IGNORE_MODELS]
LABEL_MAP   = {"Supported": 1, "Not Supported": 0}


def find_judgement_file(lang: str) -> Path | None:
    candidates = sorted((RESULTS_DIR / lang).glob("memerag_judgement_*.json"))
    return candidates[0] if candidates else None


def vote_split_label(outputs: dict) -> str:
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
    maj, min_ = max(n1, n0), min(n1, n0)
    return f"{maj}-{min_}"


def split_group(split: str) -> str:
    if split == "7-0":
        return "7-0 (unanimous)"
    elif split in ("6-1", "5-2"):
        return "6-1 / 5-2 (near)"
    elif split == "4-3":
        return "4-3 (contested)"
    return "other"


def analyze(lang: str) -> dict | None:
    path = find_judgement_file(lang)
    if not path:
        print(f"[{lang.upper()}] no judgement file found", file=sys.stderr)
        return None

    records = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("records", [])

    groups: dict[str, list[dict]] = defaultdict(list)

    for r in records:
        outputs = r.get("model_outputs", {})
        split   = vote_split_label(outputs)
        group   = split_group(split)
        if group == "other":
            continue

        answer   = r.get("answer_segment", "") or ""
        query    = r.get("query", "") or ""
        passages = r.get("context_texts", []) or []
        evidence = " ".join(str(p) for p in passages)

        ans_chars  = len(answer)
        ans_words  = len(answer.split())
        evid_chars = len(evidence)
        qry_words  = len(query.split())
        ratio      = ans_chars / evid_chars if evid_chars > 0 else 0.0

        groups[group].append({
            "split":      split,
            "ans_chars":  ans_chars,
            "ans_words":  ans_words,
            "evid_chars": evid_chars,
            "qry_words":  qry_words,
            "ratio":      ratio,
        })

    return {"lang": lang, "groups": dict(groups)}


def mannwhitney_u(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U (no scipy — manual rank-sum)."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = {i: i + 1 for i in range(len(combined))}
    r1 = sum(ranks[i] for i, (_, g) in enumerate(combined) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u  = min(u1, u2)
    # Normal approximation
    mu    = n1 * n2 / 2
    sigma = ((n1 * n2 * (n1 + n2 + 1)) / 12) ** 0.5
    z     = (u - mu) / sigma if sigma > 0 else 0.0
    # two-tailed p from z
    import math
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return u, p


def print_results(result: dict) -> None:
    lang   = result["lang"].upper()
    groups = result["groups"]
    order  = ["7-0 (unanimous)", "6-1 / 5-2 (near)", "4-3 (contested)"]

    print(f"\n[{lang}]")
    print(f"  {'Group':<22} {'n':>5}  {'ans_chars (med)':>16}  {'ans_words (med)':>16}  {'ans/evid ratio (med)':>22}")
    print("  " + "─" * 85)
    for g in order:
        items = groups.get(g, [])
        if not items:
            continue
        ac = sorted(x["ans_chars"]  for x in items)
        aw = sorted(x["ans_words"]  for x in items)
        rt = sorted(x["ratio"]      for x in items)
        med = lambda lst: lst[len(lst)//2]
        print(f"  {g:<22} {len(items):>5}  "
              f"{np.mean(ac):>8.0f} ({med(ac):>6.0f})  "
              f"{np.mean(aw):>8.1f} ({med(aw):>5.1f})  "
              f"{np.mean(rt):>10.4f} ({med(rt):>8.4f})")

    # Mann-Whitney: 4-3 vs 7-0
    contested  = groups.get("4-3 (contested)", [])
    unanimous  = groups.get("7-0 (unanimous)", [])
    if contested and unanimous:
        for metric, key in [("ans_chars", "ans_chars"), ("ans_words", "ans_words"), ("ratio", "ratio")]:
            a = [x[key] for x in contested]
            b = [x[key] for x in unanimous]
            u, p = mannwhitney_u(a, b)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  Mann-Whitney 4-3 vs 7-0  [{metric}]:  U={u:.0f}  p={p:.4f}  {sig}")


def make_plot(results: list[dict], out_path: Path) -> None:
    order  = ["7-0 (unanimous)", "6-1 / 5-2 (near)", "4-3 (contested)"]
    colors = {"7-0 (unanimous)": "#4878CF", "6-1 / 5-2 (near)": "#F4A936", "4-3 (contested)": "#C44E52"}
    langs  = [r["lang"].upper() for r in results]
    n_langs = len(langs)

    fig, axes = plt.subplots(1, n_langs, figsize=(3.5 * n_langs, 5), sharey=False)
    if n_langs == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        data   = [result["groups"].get(g, []) for g in order]
        values = [[x["ans_words"] for x in grp] for grp in data]
        bp = ax.boxplot(
            [v for v in values if v],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
        )
        valid_order = [g for g, v in zip(order, values) if v]
        for patch, g in zip(bp["boxes"], valid_order):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.75)

        ax.set_title(result["lang"].upper(), fontsize=11, fontweight="bold")
        ax.set_xticks(range(1, len(valid_order) + 1))
        ax.set_xticklabels([g.split(" ")[0] for g in valid_order], fontsize=9)
        ax.set_ylabel("Answer length (words)", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Answer length by vote split — does length predict disagreement?",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--lang", choices=LANGS, default="en")
    grp.add_argument("--all",  action="store_true")
    args  = parser.parse_args()
    langs = LANGS if args.all else [args.lang]

    print("Response length vs vote split — testing RAG-groundedness hypothesis")
    print("=" * 70)

    results = []
    for lang in langs:
        r = analyze(lang)
        if r:
            results.append(r)
            print_results(r)

    if results:
        out_dir = RESULTS_DIR / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        make_plot(results, out_dir / "response_length_by_split.png")


if __name__ == "__main__":
    main()
