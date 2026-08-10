"""
Compute analytical outputs and save as JSON files with 'analysis_' prefix.

Outputs (per language):
  analysis_judge_bias_{lang}.json            — TPR/TNR/FPR/FNR/BAcc + positive bias per judge
  analysis_consensus_{lang}.json             — per-sample vote agreement + dissenter identity
  analysis_agg_override_{lang}.json          — when LLM agg agrees/overrides majority
  analysis_agg_features_{lang}.json          — final synthesis: all per-model features joined +
                                               2 new: consensus_agreement_rate, agg_bacc_from_random
                                               + Spearman correlation summary

Output (cross-language):
  analysis_crosslang_kappa.json              — judge-pair kappa across all languages

Usage:
    python pipeline/analyze.py
    python pipeline/analyze.py --lang de
    python pipeline/analyze.py --lang en de fr
    python pipeline/analyze.py --all
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results_tmp" / "memerag_ext"
PIPELINE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PIPELINE_DIR))
from utils.io import load_json

LABELS    = {"Supported", "Not Supported"}
ALL_LANGS = ["en", "de", "es", "fr", "hi"]

# Algo and LLM-method keys — everything else is a judge
_ALGO_KEYS = {"majority", "owi", "isp", "ds", "MV", "OW-I", "ISP", "Dawid-Skene"}


# ---------------------------------------------------------------------------
# Stat helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out   = [0.0] * len(xs)
    for rank, idx in enumerate(order):
        out[idx] = rank + 1.0
    return out


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num    = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denom  = (sum((rx[i] - mx) ** 2 for i in range(n)) *
              sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return round(num / denom, 4) if denom else None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    spec = importlib.util.spec_from_file_location("config", PIPELINE_DIR / "config.py")
    cfg  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(cfg)                    # type: ignore[union-attr]
    return {
        "lang":          getattr(cfg, "LANG", "en"),
        "proposers":     list(getattr(cfg, "PROPOSERS", [])),
        "ignore_models": set(getattr(cfg, "IGNORE_MODELS", set())),
    }


def _active(proposers: list[str], ignore: set[str]) -> list[str]:
    return [m for m in proposers if m not in ignore]


def _save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# 1. Judge bias — TPR/TNR/FPR/FNR/BAcc + positive-label tendency
# ---------------------------------------------------------------------------

def analyze_judge_bias(lang: str, proposers: list[str], ignore: set[str]) -> None:
    p = RESULTS_ROOT / lang / f"memerag_judgement_{lang}.json"
    if not p.exists():
        print(f"  [SKIP] judge_bias — {p.name} not found")
        return

    records = [r for r in load_json(p).get("records", []) if isinstance(r, dict)]
    models  = _active(proposers, ignore)
    n       = len(records)

    judges: dict[str, dict] = {}
    for model in models:
        tp = fp = tn = fn = missing = 0
        for rec in records:
            gold = rec.get("gold_label")
            out  = (rec.get("model_outputs") or {}).get(model)
            lbl  = out.get("label") if isinstance(out, dict) else None
            if lbl not in LABELS:
                missing += 1
                continue
            if   gold == "Supported"     and lbl == "Supported":     tp += 1
            elif gold == "Supported"     and lbl == "Not Supported":  fn += 1
            elif gold == "Not Supported" and lbl == "Not Supported":  tn += 1
            elif gold == "Not Supported" and lbl == "Supported":      fp += 1

        valid = tp + fp + tn + fn
        tpr   = tp / (tp + fn) if (tp + fn) else None
        tnr   = tn / (tn + fp) if (tn + fp) else None
        bacc  = (tpr + tnr) / 2 if (tpr is not None and tnr is not None) else None
        # positive_bias: how often does this judge say "Supported" (regardless of gold)
        pos_bias = (tp + fp) / valid if valid else None

        judges[model] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_missing":     missing,
            "coverage":      round(valid / n, 4) if n else 0,
            "tpr":           round(tpr,       4) if tpr       is not None else None,
            "tnr":           round(tnr,       4) if tnr       is not None else None,
            "fpr":           round(1 - tnr,   4) if tnr       is not None else None,
            "fnr":           round(1 - tpr,   4) if tpr       is not None else None,
            "bacc":          round(bacc,      4) if bacc      is not None else None,
            "positive_bias": round(pos_bias,  4) if pos_bias  is not None else None,
        }

    # Sort by BAcc descending for readability
    judges = dict(sorted(judges.items(), key=lambda x: -(x[1]["bacc"] or 0)))

    out = {
        "lang":       lang,
        "n_samples":  n,
        "n_proposers": len(models),
        "note": (
            "Positive class = 'Supported'. "
            "TPR = recall (sensitivity). TNR = specificity. "
            "FPR = 1-TNR. FNR = 1-TPR. "
            "positive_bias = fraction of valid outputs labeled 'Supported'."
        ),
        "judges": judges,
    }
    _save(RESULTS_ROOT / lang / f"analysis_judge_bias_{lang}.json", out)


# ---------------------------------------------------------------------------
# 2. Consensus — per-sample vote agreement + dissenter identity analysis
# ---------------------------------------------------------------------------

def analyze_consensus(lang: str, proposers: list[str], ignore: set[str]) -> None:
    p = RESULTS_ROOT / lang / f"memerag_judgement_{lang}.json"
    if not p.exists():
        print(f"  [SKIP] consensus — {p.name} not found")
        return

    records = [r for r in load_json(p).get("records", []) if isinstance(r, dict)]
    models  = _active(proposers, ignore)

    per_sample: list[dict] = []
    split_counts: dict[str, int] = {}

    # Accumulators for dissenter / contrarian analysis
    dissenter_total:   dict[str, int] = {m: 0 for m in models}  # in minority at all
    dissenter_by_split: dict[str, dict[str, int]] = {}            # split → model → count
    correct_vs_majority: dict[str, int] = {m: 0 for m in models} # right when majority wrong
    unanimous_wrong_ids: list[str] = []

    n_unanimous = n_tie = n_majority_correct = 0

    for rec in records:
        gold   = rec.get("gold_label")
        sid    = rec.get("sample_id", "")
        votes: dict[str, str] = {}  # model → label

        for m in models:
            out = (rec.get("model_outputs") or {}).get(m)
            lbl = out.get("label") if isinstance(out, dict) else None
            if lbl in LABELS:
                votes[m] = lbl

        if not votes:
            continue

        c         = Counter(votes.values())
        n_sup     = c.get("Supported", 0)
        n_not_sup = c.get("Not Supported", 0)
        n_valid   = len(votes)

        top_count = max(n_sup, n_not_sup)
        top_label = "Supported" if n_sup >= n_not_sup else "Not Supported"
        min_label = "Not Supported" if top_label == "Supported" else "Supported"
        is_tie    = (n_sup == n_not_sup)
        is_unan   = (top_count == n_valid)
        agreement = top_count / n_valid

        maj_correct: bool | None = (top_label == gold) if (gold in LABELS and not is_tie) else None

        if is_unan: n_unanimous += 1
        if is_tie:  n_tie       += 1
        if maj_correct: n_majority_correct += 1

        hi, lo    = max(n_sup, n_not_sup), min(n_sup, n_not_sup)
        split_key = f"{hi}-{lo}"
        split_counts[split_key] = split_counts.get(split_key, 0) + 1

        # --- who are the dissenters (voted against majority)? ---
        if not is_tie and not is_unan:
            minority_models = [m for m, lbl in votes.items() if lbl == min_label]
            if split_key not in dissenter_by_split:
                dissenter_by_split[split_key] = {}
            for m in minority_models:
                dissenter_total[m] += 1
                dissenter_by_split[split_key][m] = dissenter_by_split[split_key].get(m, 0) + 1

        # --- correct contrarians: right when majority was wrong ---
        if maj_correct is False and gold in LABELS:
            for m, lbl in votes.items():
                if lbl == gold:
                    correct_vs_majority[m] += 1

        # --- unanimous but wrong ---
        if is_unan and maj_correct is False:
            unanimous_wrong_ids.append(sid)

        per_sample.append({
            "sample_id":       sid,
            "gold_label":      gold,
            "n_valid_votes":   n_valid,
            "n_supported":     n_sup,
            "n_not_supported": n_not_sup,
            "vote_split":      split_key,
            "majority_label":  top_label if not is_tie else None,
            "is_unanimous":    is_unan,
            "is_tie":          is_tie,
            "majority_correct": maj_correct,
            "minority_models": [] if (is_tie or is_unan) else
                               [m for m, lbl in votes.items() if lbl == min_label],
        })

    n = len(per_sample)

    # Sort dissenters by total minority count descending
    dissenter_summary = dict(sorted(
        {m: {
            "n_minority_votes":   dissenter_total[m],
            "pct_of_samples":     round(dissenter_total[m] / n * 100, 2) if n else 0,
        } for m in models}.items(),
        key=lambda x: -x[1]["n_minority_votes"]
    ))

    # Sort correct-contrarian by count descending
    contrarian_summary = dict(sorted(
        {m: {
            "n_correct_vs_majority": correct_vs_majority[m],
            "pct_of_samples":        round(correct_vs_majority[m] / n * 100, 2) if n else 0,
        } for m in models}.items(),
        key=lambda x: -x[1]["n_correct_vs_majority"]
    ))

    # Sort minority models within each split level by count descending
    dissenter_by_split_sorted = {
        sk: dict(sorted(mv.items(), key=lambda x: -x[1]))
        for sk, mv in sorted(
            dissenter_by_split.items(),
            key=lambda x: (-int(x[0].split("-")[0]), -int(x[0].split("-")[1]))
        )
    }

    summary = {
        "n_samples":             n,
        "n_proposers":           len(models),
        "n_unanimous":           n_unanimous,
        "pct_unanimous":         round(n_unanimous / n * 100, 2) if n else 0,
        "n_tie":                 n_tie,
        "pct_tie":               round(n_tie / n * 100, 2) if n else 0,
        "n_majority_correct":    n_majority_correct,
        "pct_majority_correct":  round(n_majority_correct / n * 100, 2) if n else 0,
        "mean_agreement_rate":   round(sum(
                                     max(s["n_supported"], s["n_not_supported"]) / s["n_valid_votes"]
                                     for s in per_sample if s["n_valid_votes"]) / n, 4) if n else 0,
        "vote_split_distribution": dict(
            sorted(split_counts.items(),
                   key=lambda x: (-int(x[0].split("-")[0]), -int(x[0].split("-")[1])))
        ),
        "unanimous_wrong": {
            "n_cases":        len(unanimous_wrong_ids),
            "pct_of_samples": round(len(unanimous_wrong_ids) / n * 100, 2) if n else 0,
            "sample_ids":     unanimous_wrong_ids,
        },
        "dissenter_counts": {
            "note": "How often each model voted against the majority (non-unanimous, non-tie samples only).",
            "by_model": dissenter_summary,
            "by_split": dissenter_by_split_sorted,
        },
        "correct_contrarians": {
            "note": "How often each model voted correctly when the majority was wrong.",
            "by_model": contrarian_summary,
        },
    }
    out = {
        "lang":       lang,
        "summary":    summary,
        "per_sample": per_sample,
    }
    _save(RESULTS_ROOT / lang / f"analysis_consensus_{lang}.json", out)


# ---------------------------------------------------------------------------
# 3. Aggregator override — how often does LLM agg agree/override majority,
#    and when it overrides, does it help or hurt?
# ---------------------------------------------------------------------------

def _pick_agg_file(lang_dir: Path, method: str) -> Path | None:
    """Most recently modified llm_agg file for the given method prefix."""
    candidates = list(lang_dir.glob(f"llm_agg_{method}_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.stat().st_mtime)


def analyze_aggregator_override(lang: str, proposers: list[str], ignore: set[str]) -> None:
    lang_dir = RESULTS_ROOT / lang
    models   = _active(proposers, ignore)

    method_results: dict[str, dict] = {}

    for method in ["random", "best_bacc", "worse_bacc", "ds_rank"]:
        agg_file = _pick_agg_file(lang_dir, method)
        if agg_file is None:
            continue

        records = [
            r for r in load_json(agg_file).get("records", [])
            if isinstance(r, dict) and r.get("aggregator_label") in LABELS
        ]
        if not records:
            continue

        agree = override_improved = override_hurt = override_neutral = 0

        for rec in records:
            gold      = rec.get("gold_label")
            agg_lbl   = rec.get("aggregator_label")

            # Majority from ALL active proposers whose labels are available in this record
            votes = []
            for m in models:
                out = (rec.get("model_outputs") or {}).get(m)
                lbl = out.get("label") if isinstance(out, dict) else None
                if lbl in LABELS:
                    votes.append(lbl)

            if not votes:
                continue

            c   = Counter(votes)
            top = c.most_common(1)[0]
            # Conservative tie-break: "Not Supported" wins ties
            maj = top[0] if top[1] > len(votes) / 2 else "Not Supported"

            if agg_lbl == maj:
                agree += 1
            else:
                # Override: agg disagrees with majority
                agg_ok = (agg_lbl == gold)
                maj_ok = (maj == gold)
                if agg_ok and not maj_ok:
                    override_improved += 1   # agg right, majority wrong → net gain
                elif maj_ok and not agg_ok:
                    override_hurt     += 1   # majority right, agg wrong → net loss
                else:
                    override_neutral  += 1   # both right or both wrong

        total      = len(records)
        n_override = override_improved + override_hurt + override_neutral

        method_results[method] = {
            "file":              agg_file.name,
            "n_valid_records":   total,
            "n_agree_with_majo": agree,
            "pct_agree":         round(agree / total * 100, 2) if total else 0,
            "n_override":        n_override,
            "pct_override":      round(n_override / total * 100, 2) if total else 0,
            "override_improved": override_improved,
            "override_hurt":     override_hurt,
            "override_neutral":  override_neutral,
            "override_net_gain": override_improved - override_hurt,
        }

    if not method_results:
        print(f"  [SKIP] aggregator_override — no llm_agg files found for {lang}")
        return

    out = {
        "lang": lang,
        "note": (
            "majority recomputed from active PROPOSERS per record. "
            "override = aggregator label differs from majority. "
            "improved = agg correct AND majority wrong. "
            "hurt = majority correct AND agg wrong. "
            "neutral = both correct or both wrong."
        ),
        "methods": method_results,
    }
    _save(lang_dir / f"analysis_agg_override_{lang}.json", out)


# ---------------------------------------------------------------------------
# 4. Cross-language judge-pair kappa
# ---------------------------------------------------------------------------

def _is_judge(name: str) -> bool:
    return name not in _ALGO_KEYS and not name.startswith("llm_")


def analyze_crosslang_kappa(proposers: list[str], ignore: set[str]) -> None:
    model_set = set(_active(proposers, ignore))

    # Collect kappa per canonical (judge_a, judge_b) pair per language
    pair_kappas: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)

    for lang in ALL_LANGS:
        kp = RESULTS_ROOT / lang / f"method_kappa_{lang}.json"
        if not kp.exists():
            continue
        mk = load_json(kp).get("method_pairwise_kappa", {})
        for key, val in mk.items():
            parts = key.split(" vs ", 1)
            if len(parts) != 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if not (_is_judge(a) and _is_judge(b)):
                continue
            if a not in model_set or b not in model_set:
                continue
            canonical = tuple(sorted([a, b]))
            pair_kappas[canonical][lang] = round(float(val), 4)

    pairs_out: list[dict] = []
    judge_vals: dict[str, list[float]] = defaultdict(list)

    for (a, b), lang_kappas in pair_kappas.items():
        vals = list(lang_kappas.values())
        mean = round(sum(vals) / len(vals), 4)
        pairs_out.append({
            "judge_a":    a,
            "judge_b":    b,
            "per_lang":   lang_kappas,
            "n_langs":    len(vals),
            "mean_kappa": mean,
            "min_kappa":  round(min(vals), 4),
            "max_kappa":  round(max(vals), 4),
            "range":      round(max(vals) - min(vals), 4),
        })
        for v in vals:
            judge_vals[a].append(v)
            judge_vals[b].append(v)

    pairs_out.sort(key=lambda x: -x["mean_kappa"])

    per_judge = {
        name: {
            "mean_kappa_all_pairs":  round(sum(vs) / len(vs), 4),
            "n_observations":        len(vs),
        }
        for name, vs in judge_vals.items()
    }
    per_judge = dict(sorted(per_judge.items(), key=lambda x: -x[1]["mean_kappa_all_pairs"]))

    out = {
        "note": (
            "Judge-vs-judge pairs only (algo methods excluded). "
            "mean_kappa averaged over all available languages. "
            "per_judge summary averages kappa across all pairings and all languages."
        ),
        "per_judge_summary": per_judge,
        "judge_pairs":       pairs_out,
    }
    out_path = RESULTS_ROOT / "analysis_crosslang_kappa.json"
    _save(out_path, out)


# ---------------------------------------------------------------------------
# 5. Aggregator features — final synthesis file
#    Reads from the 3 already-saved per-lang files + method_kappa + judge JSON
#    Computes 2 new things: consensus_agreement_rate, agg_bacc_from_random
# ---------------------------------------------------------------------------

def analyze_aggregator_features(lang: str, proposers: list[str], ignore: set[str]) -> None:
    lang_dir = RESULTS_ROOT / lang
    models   = _active(proposers, ignore)

    # ── 1. Judge bias (individual accuracy) ──────────────────────────────────
    bias_path = lang_dir / f"analysis_judge_bias_{lang}.json"
    if not bias_path.exists():
        print(f"  [SKIP] aggregator_features — run analyze first (missing judge_bias)")
        return
    bias_data: dict = load_json(bias_path).get("judges", {})

    # ── 2. Consensus (dissent + contrarian counts) ────────────────────────────
    cons_path = lang_dir / f"analysis_consensus_{lang}.json"
    if not cons_path.exists():
        print(f"  [SKIP] aggregator_features — run analyze first (missing consensus)")
        return
    cons_summary      = load_json(cons_path).get("summary", {})
    dissenter_map     = cons_summary.get("dissenter_counts", {}).get("by_model", {})
    contrarian_map    = cons_summary.get("correct_contrarians", {}).get("by_model", {})
    n_samples         = cons_summary.get("n_samples", 0)

    # ── 3. Method kappa (DS rank + pairwise judge kappas) ────────────────────
    kappa_path = lang_dir / f"method_kappa_{lang}.json"
    if not kappa_path.exists():
        print(f"  [SKIP] aggregator_features — method_kappa_{lang}.json not found")
        return
    kappa_data   = load_json(kappa_path)
    pairwise     = kappa_data.get("method_pairwise_kappa", {})
    ds_analysis  = kappa_data.get("dawid_skene_reliability_analysis", {})
    ds_rank_map  = ds_analysis.get("ds_rank", {})
    ds_score_map = ds_analysis.get("reliability_score", {})

    mean_kappa_map: dict[str, float | None] = {}
    for m in models:
        vals = []
        for other in models:
            if other == m:
                continue
            v = pairwise.get(f"{m} vs {other}") or pairwise.get(f"{other} vs {m}")
            if v is not None:
                vals.append(float(v))
        mean_kappa_map[m] = round(sum(vals) / len(vals), 4) if vals else None

    # ── 4. NEW: consensus_agreement_rate ─────────────────────────────────────
    # When N-1 other proposers agree on a label, how often does this model agree?
    judge_path = lang_dir / f"memerag_judgement_{lang}.json"
    ca_map: dict[str, dict[str, int]] = {m: {"agree": 0, "total": 0} for m in models}
    n_needed = len(models) - 1  # strict high-consensus threshold

    if judge_path.exists():
        jrecords = [r for r in load_json(judge_path).get("records", []) if isinstance(r, dict)]
        for rec in jrecords:
            votes: dict[str, str] = {}
            for m in models:
                out = (rec.get("model_outputs") or {}).get(m)
                lbl = out.get("label") if isinstance(out, dict) else None
                if lbl in LABELS:
                    votes[m] = lbl
            for m in models:
                if m not in votes:
                    continue
                others_votes = [l for o, l in votes.items() if o != m]
                if not others_votes:
                    continue
                c = Counter(others_votes)
                top_lbl, top_cnt = c.most_common(1)[0]
                if top_cnt >= n_needed:            # N-1 others agree
                    ca_map[m]["total"] += 1
                    if votes[m] == top_lbl:
                        ca_map[m]["agree"] += 1

    # ── 5. NEW: agg_bacc from all LLM agg methods ────────────────────────────
    # random:     each model acts as agg on a random subset → BAcc per model
    # best_bacc / worse_bacc / ds_rank: one model handles ALL samples

    def _bacc_from_pairs(pairs: list[tuple[str, str]]) -> float | None:
        tp = fp = tn = fn = 0
        for pred, gold in pairs:
            if   gold == "Supported"     and pred == "Supported":     tp += 1
            elif gold == "Supported"     and pred == "Not Supported":  fn += 1
            elif gold == "Not Supported" and pred == "Not Supported":  tn += 1
            elif gold == "Not Supported" and pred == "Supported":      fp += 1
        tpr  = tp / (tp + fn) if (tp + fn) else None
        tnr  = tn / (tn + fp) if (tn + fp) else None
        return round((tpr + tnr) / 2, 4) if (tpr is not None and tnr is not None) else None

    def _load_agg_perf(method: str) -> dict[str, dict]:
        """Returns {short_model_name: {agg_bacc, n_as_agg}} for a given method."""
        f = _pick_agg_file(lang_dir, method)
        if f is None:
            return {}
        by_model: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for rec in load_json(f).get("records", []):
            if not isinstance(rec, dict):
                continue
            agg_m = rec.get("aggregator_model")
            pred  = rec.get("aggregator_label")
            gold  = rec.get("gold_label")
            if agg_m and pred in LABELS and gold in LABELS:
                by_model[agg_m].append((pred, gold))
        result = {}
        for agg_m, pairs in by_model.items():
            result[agg_m.split("/")[-1]] = {
                "agg_bacc": _bacc_from_pairs(pairs),
                "n_as_agg": len(pairs),
            }
        return result

    agg_perf_random    = _load_agg_perf("random")
    agg_perf_best      = _load_agg_perf("best_bacc")
    agg_perf_worse     = _load_agg_perf("worse_bacc")
    agg_perf_ds        = _load_agg_perf("ds_rank")

    # ── Assemble per-model feature table ─────────────────────────────────────
    per_model: dict[str, dict] = {}
    for m in models:
        short    = m.split("/")[-1]
        b        = bias_data.get(m, {})
        dis      = dissenter_map.get(m, {})
        contra   = contrarian_map.get(m, {})
        ca       = ca_map.get(m, {})
        n_diss   = dis.get("n_minority_votes", 0)
        n_contra = contra.get("n_correct_vs_majority", 0)
        ca_rate  = round(ca["agree"] / ca["total"], 4) if ca.get("total") else None

        # Which methods selected this model as aggregator?
        selected_by = [
            meth for meth, perf in [
                ("best_bacc",  agg_perf_best),
                ("worse_bacc", agg_perf_worse),
                ("ds_rank",    agg_perf_ds),
            ] if short in perf
        ]

        per_model[short] = {
            # From judge_bias
            "individual_bacc":              b.get("bacc"),
            "tpr":                          b.get("tpr"),
            "tnr":                          b.get("tnr"),
            "positive_bias":                b.get("positive_bias"),
            # From consensus
            "dissent_rate_pct":             dis.get("pct_of_samples"),
            "n_minority_votes":             n_diss,
            "correct_dissent_rate":         round(n_contra / n_diss, 4) if n_diss else None,
            "n_correct_vs_majority":        n_contra,
            # From method_kappa
            "mean_kappa_with_others":       mean_kappa_map.get(m),
            "ds_rank":                      ds_rank_map.get(m),
            "ds_reliability_score":         ds_score_map.get(m),
            # New — consensus agreement rate
            "consensus_agreement_rate":     ca_rate,
            "n_high_consensus_samples":     ca.get("total"),
            # Agg BAcc from all methods (null if model was not used for that method)
            "selected_by_methods":          selected_by,
            "agg_bacc_from_random":         agg_perf_random.get(short, {}).get("agg_bacc"),
            "n_samples_as_agg_random":      agg_perf_random.get(short, {}).get("n_as_agg"),
            "agg_bacc_from_best_bacc":      agg_perf_best.get(short,   {}).get("agg_bacc"),
            "n_samples_as_agg_best_bacc":   agg_perf_best.get(short,   {}).get("n_as_agg"),
            "agg_bacc_from_worse_bacc":     agg_perf_worse.get(short,  {}).get("agg_bacc"),
            "n_samples_as_agg_worse_bacc":  agg_perf_worse.get(short,  {}).get("n_as_agg"),
            "agg_bacc_from_ds_rank":        agg_perf_ds.get(short,     {}).get("agg_bacc"),
            "n_samples_as_agg_ds_rank":     agg_perf_ds.get(short,     {}).get("n_as_agg"),
        }

    # ── Method-level summary (overall BAcc per aggregation strategy) ──────────
    def _method_summary(perf: dict[str, dict], agg_file: Path | None = None) -> dict:
        if not perf:
            return {}
        model_selected = list(perf.keys())
        n_total        = sum(v["n_as_agg"] for v in perf.values())

        if len(model_selected) == 1:
            # best_bacc / worse_bacc / ds_rank — one model handles all samples
            overall_bacc = perf[model_selected[0]]["agg_bacc"]
        else:
            # random — combine ALL records from the agg file to get the strategy-level BAcc
            overall_bacc = None
            if agg_file is not None:
                all_pairs: list[tuple[str, str]] = []
                for rec in load_json(agg_file).get("records", []):
                    if not isinstance(rec, dict):
                        continue
                    pred = rec.get("aggregator_label")
                    gold = rec.get("gold_label")
                    if pred in LABELS and gold in LABELS:
                        all_pairs.append((pred, gold))
                overall_bacc = _bacc_from_pairs(all_pairs)

        return {
            "model_selected": model_selected[0] if len(model_selected) == 1 else model_selected,
            "agg_bacc":       overall_bacc,
            "n_samples":      n_total,
        }

    random_file = _pick_agg_file(lang_dir, "random")
    method_summary = {
        "random":     _method_summary(agg_perf_random, agg_file=random_file),
        "best_bacc":  _method_summary(agg_perf_best),
        "worse_bacc": _method_summary(agg_perf_worse),
        "ds_rank":    _method_summary(agg_perf_ds),
    }

    # ── Spearman correlation: each feature vs agg_bacc_from_random ───────────
    target_key   = "agg_bacc_from_random"
    feature_keys = [
        "individual_bacc", "dissent_rate_pct", "correct_dissent_rate",
        "mean_kappa_with_others", "ds_reliability_score",
        "consensus_agreement_rate", "positive_bias",
    ]
    valid_models = [m for m in per_model if per_model[m][target_key] is not None]

    correlations: dict[str, dict] = {}
    for fk in feature_keys:
        pairs = [
            (per_model[m][fk], per_model[m][target_key])
            for m in valid_models if per_model[m][fk] is not None
        ]
        if len(pairs) < 3:
            correlations[fk] = {"spearman_r": None, "n": len(pairs), "note": "too few points"}
            continue
        xs, ys = zip(*pairs)
        correlations[fk] = {
            "spearman_r": _spearman(list(xs), list(ys)),
            "n":          len(pairs),
        }

    out = {
        "lang":        lang,
        "n_proposers": len(models),
        "note": (
            "agg_bacc_from_random: BAcc for this model across the random subset where it was selected. "
            "agg_bacc_from_best/worse/ds_rank: BAcc over ALL samples for the one model chosen by that method (null for others). "
            "consensus_agreement_rate: when N-1 others agree, fraction this model also agrees. "
            "correct_dissent_rate: of times in minority, fraction model was correct. "
            "Spearman correlation uses n=n_proposers — treat directionally."
        ),
        "method_summary": method_summary,
        "per_model":      per_model,
        "correlation_with_agg_bacc": {
            "target":   target_key,
            "n_models": len(valid_models),
            "note":     "Correlating structural features against agg_bacc_from_random (each model's BAcc when randomly selected as aggregator).",
            "features": correlations,
        },
    }
    _save(lang_dir / f"analysis_agg_features_{lang}.json", out)


# ---------------------------------------------------------------------------
# Majority accuracy by vote split + minority quality at 4-3 failures
# ---------------------------------------------------------------------------

def analyze_vote_confidence(lang: str, proposers: list[str], ignore: set[str]) -> None:
    """
    Q1: Does vote margin predict majority accuracy?
        → per split level (7-0, 6-1, 5-2, 4-3): n_samples, n_correct, accuracy_pct

    Q2: At 6-1, 5-2, 4-3 splits where majority is WRONG:
        - is the minority side stronger BAcc?
        - did best_bacc / random aggregator recover the correct answer?
        Saves per-split JSON summary + CSV to {lang}/plots/.
    """
    lang_dir  = RESULTS_ROOT / lang
    active    = [m for m in proposers if m not in ignore]
    n_prop    = len(active)
    if n_prop < 2:
        print(f"  [SKIP] vote_confidence — too few proposers")
        return

    random_file = _pick_agg_file(lang_dir, "random")
    if random_file is None:
        print(f"  [SKIP] vote_confidence — no random agg file for {lang}")
        return
    best_file = _pick_agg_file(lang_dir, "best_bacc")

    def _build_lkp(filepath: Path | None) -> dict:
        if filepath is None:
            return {}
        return {str(r["sample_id"]): r for r in load_json(filepath).get("records", []) if isinstance(r, dict)}

    random_lkp = _build_lkp(random_file)
    best_lkp   = _build_lkp(best_file)

    # Load individual BAcc per model
    bias_file = lang_dir / f"analysis_judge_bias_{lang}.json"
    model_bacc: dict[str, float] = {}
    if bias_file.exists():
        for m, v in load_json(bias_file).get("judges", {}).items():
            if isinstance(v, dict) and "bacc" in v:
                model_bacc[m] = v["bacc"]

    TARGET_SPLITS = {(6, 1), (5, 2), (4, 3)}

    split_stats: dict[str, dict] = {}
    failures: dict[str, list[dict]] = {"6-1": [], "5-2": [], "4-3": []}

    for rec in load_json(random_file).get("records", []):
        if not isinstance(rec, dict):
            continue
        gold = rec.get("gold_label")
        if gold not in LABELS:
            continue
        sid = str(rec.get("sample_id"))

        outputs = rec.get("model_outputs", {})
        votes: dict[str, str] = {}
        for m in active:
            raw = outputs.get(m)
            lbl = raw.get("label") if isinstance(raw, dict) else raw
            if lbl in LABELS:
                votes[m] = lbl

        if len(votes) < 2:
            continue

        counts   = Counter(votes.values())
        majority = counts.most_common(1)[0][0]
        n_maj    = counts[majority]
        n_min    = n_prop - n_maj
        split_key = f"{n_maj}-{n_min}" if n_maj >= n_min else f"{n_min}-{n_maj}"

        if split_key not in split_stats:
            split_stats[split_key] = {"n": 0, "correct": 0}
        split_stats[split_key]["n"] += 1
        if majority == gold:
            split_stats[split_key]["correct"] += 1

        hi_lo = (max(n_maj, n_min), min(n_maj, n_min))
        if hi_lo in TARGET_SPLITS and majority != gold:
            sk         = f"{hi_lo[0]}-{hi_lo[1]}"
            maj_models = [m for m, lbl in votes.items() if lbl == majority]
            min_models = [m for m, lbl in votes.items() if lbl != majority]
            maj_bacc   = [model_bacc[m] for m in maj_models if m in model_bacc]
            min_bacc   = [model_bacc[m] for m in min_models if m in model_bacc]
            maj_mean   = round(sum(maj_bacc) / len(maj_bacc), 4) if maj_bacc else None
            min_mean   = round(sum(min_bacc) / len(min_bacc), 4) if min_bacc else None

            br           = best_lkp.get(sid, {})
            best_label   = br.get("aggregator_label")
            best_correct = (best_label == gold) if best_label in LABELS else None

            rr           = random_lkp.get(sid, {})
            rand_model   = rr.get("aggregator_model")
            rand_label   = rr.get("aggregator_label")
            rand_correct = (rand_label == gold) if rand_label in LABELS else None

            failures[sk].append({
                "sample_id":            rec.get("sample_id"),
                "gold_label":           gold,
                "majority_label":       majority,
                "majority_models":      maj_models,
                "minority_models":      min_models,
                "majority_mean_bacc":   maj_mean,
                "minority_mean_bacc":   min_mean,
                "minority_was_stronger": (
                    (min_mean > maj_mean)
                    if min_mean is not None and maj_mean is not None else None
                ),
                "best_bacc_label":      best_label,
                "best_bacc_correct":    best_correct,
                "random_agg_model":     rand_model.split("/")[-1] if rand_model else None,
                "random_agg_label":     rand_label,
                "random_correct":       rand_correct,
            })

    # Q1 summary
    q1_summary: dict[str, dict] = {}
    for sk in sorted(split_stats.keys(), reverse=True):
        d = split_stats[sk]
        n, c = d["n"], d["correct"]
        q1_summary[sk] = {
            "n_samples":    n,
            "n_correct":    c,
            "accuracy_pct": round(c / n * 100, 1) if n else None,
        }

    # Q2 summary per split
    q2_by_split: dict[str, dict] = {}
    for sk, case_list in failures.items():
        n_fail = len(case_list)
        n_min_s  = sum(1 for f in case_list if f["minority_was_stronger"])
        n_best_c = sum(1 for f in case_list if f["best_bacc_correct"])
        n_rand_c = sum(1 for f in case_list if f["random_correct"])
        q2_by_split[sk] = {
            "n_cases":               n_fail,
            "n_minority_stronger":   n_min_s,
            "pct_minority_stronger": round(n_min_s  / n_fail * 100, 1) if n_fail else None,
            "best_bacc_correct":     n_best_c,
            "best_bacc_pct":         round(n_best_c / n_fail * 100, 1) if n_fail else None,
            "random_correct":        n_rand_c,
            "random_pct":            round(n_rand_c / n_fail * 100, 1) if n_fail else None,
            "cases":                 case_list,
        }

    out = {
        "lang": lang,
        "note": (
            "Q1: majority accuracy at each vote split level. "
            "Q2: at 6-1, 5-2, 4-3 splits where majority was wrong — "
            "was minority side stronger BAcc? did best_bacc/random recover it?"
        ),
        "q1_majority_accuracy_by_split": q1_summary,
        "q2_majority_wrong_by_split":    q2_by_split,
    }
    _save(lang_dir / f"analysis_vote_confidence_{lang}.json", out)

    # Save per-split CSVs to {lang}/plots/
    def _enc(val: str | None) -> str:
        if val == "Supported":     return "1"
        if val == "Not Supported": return "0"
        return "" if val is None else str(val)

    def _bacc(val: float | None) -> str:
        return f"{val * 100:.2f}" if val is not None else ""

    plots_dir = lang_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "gold_label", "majority_label",
        "majority_models", "maj_mean_bacc",
        "minority_models", "min_mean_bacc", "minority_was_stronger",
        "best_bacc_label", "best_bacc_correct",
        "random_agg_model", "random_agg_label", "random_correct",
    ]
    for sk, case_list in failures.items():
        if not case_list:
            continue
        n_fail   = len(case_list)
        n_best_c = sum(1 for f in case_list if f["best_bacc_correct"])
        n_rand_c = sum(1 for f in case_list if f["random_correct"])
        csv_path = plots_dir / f"{sk.replace('-', '_')}_failures_{lang}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for c in case_list:
                w.writerow({
                    "sample_id":            c["sample_id"],
                    "gold_label":           _enc(c["gold_label"]),
                    "majority_label":       _enc(c["majority_label"]),
                    "majority_models":      " | ".join(m.split("/")[-1] for m in c["majority_models"]),
                    "maj_mean_bacc":        _bacc(c["majority_mean_bacc"]),
                    "minority_models":      " | ".join(m.split("/")[-1] for m in c["minority_models"]),
                    "min_mean_bacc":        _bacc(c["minority_mean_bacc"]),
                    "minority_was_stronger": c["minority_was_stronger"],
                    "best_bacc_label":      _enc(c["best_bacc_label"]),
                    "best_bacc_correct":    c["best_bacc_correct"],
                    "random_agg_model":     c["random_agg_model"] or "",
                    "random_agg_label":     _enc(c["random_agg_label"]),
                    "random_correct":       c["random_correct"],
                })
            fh.write(
                f"\nNOTE: Each row = a case where majority ({sk.split('-')[0]} judges) voted wrong."
                f" labels: 1=Supported 0=Not Supported. bacc in xx.yy format."
                f" correct = aggregator label matched gold label.\n"
                f"SUMMARY — of {n_fail} failure cases: "
                f"best_bacc recovered correct answer {n_best_c}/{n_fail} ({round(n_best_c/n_fail*100,1) if n_fail else 0}%) times  |  "
                f"random recovered correct answer {n_rand_c}/{n_fail} ({round(n_rand_c/n_fail*100,1) if n_fail else 0}%) times\n"
            )
        print(f"    CSV: {csv_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg           = _load_config()
    proposers     = cfg["proposers"]
    ignore        = cfg["ignore_models"]
    default_lang  = cfg["lang"]

    parser = argparse.ArgumentParser(
        description="Compute analytical outputs and save as analysis_*.json files."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--lang", nargs="+", choices=ALL_LANGS,
        help=f"Language(s) to analyze (config default: {default_lang})",
    )
    group.add_argument("--all", action="store_true", help="Analyze all languages")
    args = parser.parse_args()

    langs = ALL_LANGS if args.all else (args.lang or [default_lang])

    for lang in langs:
        print(f"\n[{lang.upper()}]")
        analyze_judge_bias           (lang, proposers, ignore)
        analyze_consensus            (lang, proposers, ignore)
        analyze_aggregator_override  (lang, proposers, ignore)
        analyze_aggregator_features  (lang, proposers, ignore)
        analyze_vote_confidence      (lang, proposers, ignore)

    print("\n[CROSS-LANG]")
    analyze_crosslang_kappa(proposers, ignore)


if __name__ == "__main__":
    main()
