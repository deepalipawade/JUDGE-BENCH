"""
judge_logprob_newsroom.py

Graded (1-5) summary-quality judge using LOCAL HuggingFace models.
Extracts the full log-probability distribution over the five rating tokens
at the label decision point.

NEWSROOM is a GRADED task, not binary and not ranking: 420 (article, summary)
pairs, each scored 1-5 by 3 crowdworkers on four independent metrics.
See results_tmp/newsroom/README.md for dataset notes and the gold-label caveat.

Key difference from judge_logprob_memerag.py: the label space is 5-way, so each
judgment yields a distribution rather than a single p. That gives two quantities
binary never did:

    expected_rating = sum(k * p(k))   continuous prediction
    entropy         = -sum(p log p)   real uncertainty measure

Three modes (same shape as the MEMERAG script):

  DEFAULT (bare-rating):
    Prompt asks for a single digit 1-5, no explanation.
    Logprob read at step 0 - uncertainty BEFORE any reasoning.
    Output: logprob_judgement_{metric}.json

  --rationale:
    <rationale>...</rationale><answer>N</answer><confidence>...</confidence>
    Logprob read at the digit inside <answer>.
    Output: logprob_judgement_{metric}_rationale.json

  --chain:
    <answer_initial>N</answer_initial> -> rationale -> <answer>N</answer>
    Pre-reasoning logprob at <answer_initial>, post at <answer>.
    Output: logprob_judgement_{metric}_chain.json

Metric codes:  info = Informativeness   rel = Relevance
               flu  = Fluency          coh = Coherence

Usage:
    # PILOT FIRST - validates that models emit a bare digit at step 0
    python judge_logprob_newsroom.py --metric rel --samples 10

    # full run, one metric
    python judge_logprob_newsroom.py --metric rel

    # force the digit position if label_mass comes back low
    python judge_logprob_newsroom.py --metric rel --samples 10 --prefill

    # other modes
    python judge_logprob_newsroom.py --metric flu --rationale
    python judge_logprob_newsroom.py --metric flu --chain

    # single model / single instance
    python judge_logprob_newsroom.py --metric info --models microsoft/Phi-4
    python judge_logprob_newsroom.py --metric info --instance-id 1
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT      = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "newsroom" / "newsroom.json"

CHECKPOINT_EVERY = 5
MAX_RETRIES      = 3
RETRY_BASE_DELAY = 2.0

RATINGS = ["1", "2", "3", "4", "5"]

# Short lowercase codes you type -> the exact metric name used inside
# newsroom.json. Long lowercase forms are accepted too.
METRIC_ALIASES = {
    "info": "Informativeness", "informativeness": "Informativeness",
    "rel":  "Relevance",       "relevance":       "Relevance",
    "flu":  "Fluency",         "fluency":         "Fluency",
    "coh":  "Coherence",       "coherence":       "Coherence",
}
METRIC_CODES = ["info", "rel", "flu", "coh"]          # for help text / filenames
SHORT_FOR    = {v: k for k, v in METRIC_ALIASES.items() if k in METRIC_CODES}

# Text appended after the generation prompt to force the next token into digit
# position. Only used with --prefill (bare mode).
PREFILL_TEXT = "Rating: "

# ── Prompts ───────────────────────────────────────────────────────────────────
# The TASK prompt is NOT written here - each metric ships its own prompt inside
# newsroom.json (with a {{ instance }} placeholder). We substitute into it so the
# results stay comparable to the JUDGE-BENCH reference numbers. Only the SYSTEM
# prompt is ours, and in bare mode its whole job is to force a bare digit.

SYSTEM_PROMPT_BARE = (
    "You are evaluating the quality of a machine-generated summary against its "
    "source article.\n\n"
    "You will be given a rating question and a scale from 1 (low) to 5 (high). "
    "Judge only the specific quality the question asks about - ignore every other "
    "aspect of the summary.\n\n"
    "Use the full scale. A 3 means genuinely middling, not 'unsure'. Do not "
    "default to 4.\n\n"
    "Respond with EXACTLY ONE CHARACTER: the digit 1, 2, 3, 4, or 5.\n"
    "No words, no punctuation, no explanation, no restating the question."
)

SYSTEM_PROMPT_RATIONALE = (
    "You are evaluating the quality of a machine-generated summary against its "
    "source article.\n\n"
    "You will be given a rating question and a scale from 1 (low) to 5 (high). "
    "Judge only the specific quality the question asks about - ignore every other "
    "aspect of the summary. Use the full scale; a 3 means genuinely middling.\n\n"
    "Write your reasoning inside <rationale></rationale> tags.\n"
    "Give your rating inside <answer></answer> tags as a single digit, "
    "e.g. <answer>4</answer>.\n\n"
    "Then state your confidence in that rating, based on how clear-cut the "
    "evidence in the summary and article is - not on how strongly you feel.\n"
    "Output exactly one of these three fixed scores - do not interpolate:\n\n"
    "  High   -> <confidence>High</confidence>   <confidence_score>0.95</confidence_score>\n"
    "            The summary clearly sits at one point on the scale; another careful\n"
    "            reader would almost certainly pick the same digit.\n\n"
    "  Medium -> <confidence>Medium</confidence> <confidence_score>0.65</confidence_score>\n"
    "            The summary sits between two adjacent points; a careful reader might\n"
    "            reasonably pick the digit either side of yours.\n\n"
    "  Low    -> <confidence>Low</confidence>    <confidence_score>0.35</confidence_score>\n"
    "            The quality is genuinely ambiguous or the article is hard to assess;\n"
    "            your digit is a best guess.\n\n"
    "Output format (always all four tags, in this order):\n"
    "<rationale>...</rationale>\n"
    "<answer>N</answer>\n"
    "<confidence>High OR Medium OR Low</confidence>\n"
    "<confidence_score>0.95 OR 0.65 OR 0.35</confidence_score>\n"
)

SYSTEM_PROMPT_CHAIN = (
    "You are evaluating the quality of a machine-generated summary against its "
    "source article.\n\n"
    "You will be given a rating question and a scale from 1 (low) to 5 (high). "
    "Judge only the specific quality the question asks about - ignore every other "
    "aspect of the summary. Use the full scale; a 3 means genuinely middling.\n\n"
    "Output format (always all five tags, in this order):\n\n"
    "1. Your initial rating BEFORE reasoning (gut reaction), a single digit:\n"
    "   <answer_initial>N</answer_initial>\n\n"
    "2. Your reasoning about the summary against the article:\n"
    "   <rationale>...</rationale>\n\n"
    "3. Your final rating - you MAY revise the initial digit if reasoning changed\n"
    "   your mind:\n"
    "   <answer>N</answer>\n\n"
    "4. Your confidence in the final rating, based on how clear-cut the evidence is:\n"
    "   <confidence>High OR Medium OR Low</confidence>\n"
    "   <confidence_score>0.95 OR 0.65 OR 0.35</confidence_score>\n\n"
    "  High   = another careful reader would almost certainly pick the same digit\n"
    "  Medium = a careful reader might pick the digit either side of yours\n"
    "  Low    = genuinely ambiguous; your digit is a best guess\n"
)

# Appended after the dataset's own metric prompt, to pin the output position.
TASK_SUFFIX_BARE      = "\n\nYour rating (a single digit, 1 to 5):"
TASK_SUFFIX_RATIONALE = "\n\nProvide your rationale, rating, and confidence using the tags specified above."
TASK_SUFFIX_CHAIN     = "\n\nProvide your initial rating, rationale, final rating, and confidence using the tags specified above."

# ── Model list ────────────────────────────────────────────────────────────────
# Falcon3 and OLMo-2 are excluded: on MEMERAG they showed format-following
# failures and near-chance accuracy. Re-add if you want them back.
MODELS = [
    "Qwen/Qwen3-8B",
    "microsoft/Phi-4",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",   # gated
    "google/gemma-2-9b-it",               # gated
    # "tiiuae/Falcon3-7B-Instruct",
    # "allenai/OLMo-2-1124-7B-Instruct",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_newsroom(path: Path, metric: str) -> tuple[list[dict[str, Any]], str]:
    """
    Load NEWSROOM instances for one metric.

    Returns (rows, metric_prompt_template). The template contains a
    '{{ instance }}' placeholder taken verbatim from the dataset.

    Gold handling (see README): majority_human is ABSENT for 28-46% of
    instances depending on metric, because 3 annotators on a 5-point scale
    often fail to produce a majority. mean_human is therefore the primary gold
    and is always present; majority_human is carried through as optional.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    prompt_template = None
    for ann in data.get("annotations", []):
        if ann.get("metric") == metric:
            prompt_template = ann.get("prompt")
            break
    if prompt_template is None:
        available = [a.get("metric") for a in data.get("annotations", [])]
        sys.exit(f"Metric {metric!r} not found. Available: {available}")

    rows: list[dict[str, Any]] = []
    for inst in data.get("instances", []):
        body = str(inst.get("instance", "")).strip()
        ann  = inst.get("annotations", {}).get(metric)
        if not body or not isinstance(ann, dict):
            continue
        mean_human = ann.get("mean_human")
        if mean_human is None:
            continue
        rows.append({
            "sample_id":              str(inst.get("id")),
            "metric":                 metric,
            "instance":               body,
            "mean_human":             float(mean_human),
            "majority_human":         ann.get("majority_human"),
            "individual_human_scores": ann.get("individual_human_scores", []),
        })
    return rows, prompt_template


# ── Rating extraction from generated text ─────────────────────────────────────

_DIGIT_RE = re.compile(r"[1-5]")


def _first_digit(text: str | None) -> int | None:
    if not text:
        return None
    m = _DIGIT_RE.search(text)
    return int(m.group(0)) if m else None


def extract_rating_bare(text: str | None) -> tuple[int | None, None]:
    """Bare mode: the output should be a single digit. Take the first 1-5 found."""
    return _first_digit(text), None


def extract_rating_rationale(
    text: str | None,
) -> tuple[int | None, str | None, str | None, float | None]:
    if not text:
        return None, None, None, None
    m = re.search(r"<answer>\s*([1-5])\s*</answer>", text, re.IGNORECASE)
    rating = int(m.group(1)) if m else _first_digit(text)
    rm = re.search(r"<rationale>(.*?)</rationale>", text, re.IGNORECASE | re.DOTALL)
    rationale = rm.group(1).strip() if rm else None
    cm = re.search(r"<confidence>\s*(high|medium|low)\s*</confidence>", text, re.IGNORECASE)
    confidence = cm.group(1).capitalize() if cm else None
    sm = re.search(r"<confidence_score>\s*([0-9.]+)\s*</confidence_score>", text, re.IGNORECASE)
    confidence_score = float(sm.group(1)) if sm else None
    return rating, rationale, confidence, confidence_score


def extract_rating_chain(
    text: str | None,
) -> tuple[int | None, int | None, str | None, str | None, float | None]:
    if not text:
        return None, None, None, None, None
    mi = re.search(r"<answer_initial>\s*([1-5])\s*</answer_initial>", text, re.IGNORECASE)
    rating_initial = int(mi.group(1)) if mi else None
    mf = re.search(r"<answer>\s*([1-5])\s*</answer>", text, re.IGNORECASE)
    rating_final = int(mf.group(1)) if mf else _first_digit(text)
    rm = re.search(r"<rationale>(.*?)</rationale>", text, re.IGNORECASE | re.DOTALL)
    rationale = rm.group(1).strip() if rm else None
    cm = re.search(r"<confidence>\s*(high|medium|low)\s*</confidence>", text, re.IGNORECASE)
    confidence = cm.group(1).capitalize() if cm else None
    sm = re.search(r"<confidence_score>\s*([0-9.]+)\s*</confidence_score>", text, re.IGNORECASE)
    confidence_score = float(sm.group(1)) if sm else None
    return rating_initial, rating_final, rationale, confidence, confidence_score


# ── Token resolution ──────────────────────────────────────────────────────────

def resolve_rating_tokens(tok: Any, labels: list[str]) -> dict[str, dict]:
    """
    Map each rating digit to the token id the model would actually emit.

    IMPORTANT: do not blindly take ids[0]. SentencePiece tokenizers (Mistral and
    others) prepend a whitespace-marker token when encoding a bare string, so
    ids[0] is the SAME for every digit. Five identical ids means softmaxing five
    copies of one logit, which yields a perfectly uniform 0.2/0.2/0.2/0.2/0.2
    "distribution" - plausible-looking and completely meaningless.

    So: find the first position at which the five encodings actually differ, and
    hard-fail if no such position exists rather than emitting junk.
    """
    encs: dict[str, tuple[str, list[int]]] = {}
    for label in labels:
        cands = []
        for variant in (label, " " + label):
            ids = tok.encode(variant, add_special_tokens=False)
            if ids:
                cands.append((variant, ids))
        if not cands:
            raise RuntimeError(f"Tokenizer produced no ids for {label!r}")
        encs[label] = min(cands, key=lambda c: len(c[1]))

    for label, (variant, ids) in encs.items():
        print(f"    {label!r:4s} encodes as {ids}  (variant {variant!r})")

    max_len = max(len(ids) for _, ids in encs.values())
    pos = None
    for i in range(max_len):
        at_i = [ids[i] for _, ids in encs.values() if i < len(ids)]
        if len(at_i) == len(labels) and len(set(at_i)) == len(labels):
            pos = i
            break

    if pos is None:
        raise RuntimeError(
            "Could not find a token position where all rating digits differ.\n"
            f"  encodings: { {l: ids for l, (_, ids) in encs.items()} }\n"
            "  Using these ids would produce a uniform, meaningless distribution."
        )
    if pos != 0:
        print(f"    [note] token position 0 is shared across digits "
              f"(tokenizer prefix) - using position {pos} instead")

    resolved = {}
    for label, (variant, ids) in encs.items():
        resolved[label] = {"variant": variant, "token_id": ids[pos], "position": pos}
    print(f"    resolved ids: { {l: r['token_id'] for l, r in resolved.items()} }")
    return resolved


# ── Logprob extraction ────────────────────────────────────────────────────────

def extract_step0_logprobs(
    scores: tuple[torch.Tensor, ...],
    rating_token_ids: dict[str, int],
) -> tuple[int, dict[str, float], dict[str, float], float]:
    """
    Read the 5-way rating distribution at step 0 (first generated token).
    Identical in shape to the MEMERAG version - it was already generic over N
    labels, so nothing here is binary-specific.
    """
    logits    = scores[0][0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    raw_lp    = {l: log_probs[tid].item() for l, tid in rating_token_ids.items()}
    sel       = torch.stack([logits[t] for t in rating_token_ids.values()]).float()
    normed    = torch.softmax(sel, dim=-1).tolist()
    norm_probs = dict(zip(rating_token_ids.keys(), normed))
    label_mass = sum(math.exp(lp) for lp in raw_lp.values())
    return 0, raw_lp, norm_probs, label_mass


def find_rating_token_step(
    generated_ids: list[int],
    scores: tuple[torch.Tensor, ...],
    tok: Any,
    rating_token_ids: dict[str, int],
    tag: str = "<answer>",
) -> tuple[int | None, dict[str, float], dict[str, float], float]:
    """
    Find the step where the rating digit appears inside `tag`, and read the
    5-way distribution there. Used by --rationale and --chain.

    Decodes incrementally and triggers at the first step whose text after `tag`
    contains a 1-5 digit. Falls back to an exact token-id scan if the tag never
    appears in the decoded text.
    """
    tag_lo  = tag.lower()
    tag_len = len(tag_lo)
    cumulative = ""
    for step, token_id in enumerate(generated_ids):
        cumulative += tok.decode([token_id], skip_special_tokens=False)
        lo  = cumulative.lower()
        pos = lo.rfind(tag_lo)
        if pos == -1:
            continue
        after = lo[pos + tag_len:]
        # Only trigger once a digit has actually been emitted after the tag,
        # and only while nothing else substantive has been written yet.
        if _DIGIT_RE.search(after) and len(after.strip()) <= 3:
            if step >= len(scores):
                break
            logits    = scores[step][0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            raw_lp    = {l: log_probs[tid].item() for l, tid in rating_token_ids.items()}
            sel       = torch.stack([logits[t] for t in rating_token_ids.values()]).float()
            normed    = torch.softmax(sel, dim=-1).tolist()
            norm_probs = dict(zip(rating_token_ids.keys(), normed))
            label_mass = sum(math.exp(lp) for lp in raw_lp.values())
            return step, raw_lp, norm_probs, label_mass

    # Fallback: first step whose argmax is one of the rating tokens
    wanted = set(rating_token_ids.values())
    for step, token_id in enumerate(generated_ids):
        if token_id in wanted and step < len(scores):
            logits    = scores[step][0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            raw_lp    = {l: log_probs[tid].item() for l, tid in rating_token_ids.items()}
            sel       = torch.stack([logits[t] for t in rating_token_ids.values()]).float()
            normed    = torch.softmax(sel, dim=-1).tolist()
            norm_probs = dict(zip(rating_token_ids.keys(), normed))
            label_mass = sum(math.exp(lp) for lp in raw_lp.values())
            return step, raw_lp, norm_probs, label_mass

    return None, {}, {}, 0.0


def derived_scores(norm_probs: dict[str, float]) -> dict[str, float | int | None]:
    """
    expected_rating  = sum(k * p(k))      continuous, the main model score
    argmax_rating    = argmax_k p(k)      discrete prediction
    entropy          = -sum(p log p)      nats; max = ln(5) = 1.609
    entropy_norm     = entropy / ln(5)    in [0,1], easier to plot
    """
    if not norm_probs:
        return {"expected_rating": None, "argmax_rating": None,
                "entropy": None, "entropy_norm": None}
    exp_r = sum(int(k) * v for k, v in norm_probs.items())
    argmx = int(max(norm_probs.items(), key=lambda kv: kv[1])[0])
    ent   = -sum(v * math.log(v + 1e-12) for v in norm_probs.values())
    return {
        "expected_rating": round(exp_r, 4),
        "argmax_rating":   argmx,
        "entropy":         round(ent, 4),
        "entropy_norm":    round(ent / math.log(len(norm_probs)), 4),
    }


# ── IO helpers ────────────────────────────────────────────────────────────────

def save_output(output_path: Path, header: dict, records: list[dict]) -> None:
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({**header, "records": list(records)}, fh, indent=2, ensure_ascii=False)


def load_existing(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}
    try:
        with output_path.open("r", encoding="utf-8") as fh:
            existing = json.load(fh)
        return {
            str(r["sample_id"]): r
            for r in existing.get("records", [])
            if isinstance(r, dict) and r.get("sample_id")
        }
    except Exception as e:
        print(f"  [resume] Could not load existing output ({e}), starting fresh.")
        return {}


def apply_chat_template_safe(tok: Any, messages: list[dict]) -> str:
    """
    Apply chat template with two fallbacks:
    1. TypeError -> enable_thinking unsupported (non-Qwen); retry without it.
    2. system role unsupported (Gemma etc.) -> merge system into the user turn.
    """
    for use_thinking in (True, False):
        kwargs = {"enable_thinking": False} if use_thinking else {}
        for merge_system in (False, True):
            msgs = messages
            if merge_system:
                merged = messages[0]["content"] + "\n\n" + messages[1]["content"]
                msgs = [{"role": "user", "content": merged}]
            try:
                return tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **kwargs
                )
            except TypeError:
                break
            except Exception as e:
                if "system" in str(e).lower():
                    continue
                raise
    raise RuntimeError("apply_chat_template failed with all fallback attempts")


# ── Per-model run ─────────────────────────────────────────────────────────────

def run_one_model(
    model_name: str,
    candidate_data: list[dict],
    prompt_template: str,
    all_records: dict[str, dict],
    output_path: Path,
    header: dict,
    no_resume: bool,
    max_new_tokens: int,
    checkpoint_every: int,
    use_rationale: bool = False,
    use_chain: bool = False,
    prefill: bool = False,
) -> None:
    remaining = []
    for row in candidate_data:
        sid = str(row["sample_id"])
        existing_rec = all_records.get(sid, {})
        existing_out = existing_rec.get("model_outputs", {}).get(model_name, {})
        if no_resume or model_name not in existing_rec.get("model_outputs", {}) \
                or existing_out.get("rating") is None:
            remaining.append(row)

    print(f"  Remaining for {model_name}: {len(remaining)}")
    if not remaining:
        print("  Nothing to do - skipping model load.\n")
        return

    print(f"  Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"  Loaded on: {model.device}")

    if use_chain:
        system_prompt, task_suffix, mode_label = SYSTEM_PROMPT_CHAIN, TASK_SUFFIX_CHAIN, "chain"
    elif use_rationale:
        system_prompt, task_suffix, mode_label = SYSTEM_PROMPT_RATIONALE, TASK_SUFFIX_RATIONALE, "rationale"
    else:
        system_prompt, task_suffix, mode_label = SYSTEM_PROMPT_BARE, TASK_SUFFIX_BARE, "bare-rating"
    print(f"  Mode: {mode_label}" + ("  (+prefill)" if prefill and not (use_chain or use_rationale) else ""))

    print("  Resolving rating tokens:")
    rating_tokens    = resolve_rating_tokens(tok, RATINGS)
    rating_token_ids = {lab: info["token_id"] for lab, info in rating_tokens.items()}

    for idx, row in enumerate(remaining):
        sid = str(row["sample_id"])
        print(f"\n  [{idx+1}/{len(remaining)}] id={sid}  "
              f"mean_human={row['mean_human']}  majority={row.get('majority_human')}")

        rating = rating_initial = None
        rationale = raw_text = None
        confidence = confidence_score = None
        step, raw_lp, norm_probs, label_mass = None, {}, {}, 0.0
        step_i, raw_lp_i, norm_probs_i, label_mass_i = None, {}, {}, 0.0
        error_msg = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                user_content = prompt_template.replace("{{ instance }}", row["instance"]) + task_suffix
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ]
                prompt_str = apply_chat_template_safe(tok, messages)
                # Force the digit position: with the prefill in place, the very
                # next token the model emits has to be the rating itself.
                if prefill and not (use_chain or use_rationale):
                    prompt_str = prompt_str + PREFILL_TEXT

                inputs = tok(prompt_str, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                prompt_len    = inputs["input_ids"].shape[1]
                generated_ids = out.sequences[0][prompt_len:].tolist()
                raw_text      = tok.decode(generated_ids, skip_special_tokens=True)

                if use_chain:
                    rating_initial, rating, rationale, confidence, confidence_score = \
                        extract_rating_chain(raw_text)
                    step_i, raw_lp_i, norm_probs_i, label_mass_i = find_rating_token_step(
                        generated_ids, out.scores, tok, rating_token_ids, tag="<answer_initial>"
                    )
                    step, raw_lp, norm_probs, label_mass = find_rating_token_step(
                        generated_ids, out.scores, tok, rating_token_ids, tag="<answer>"
                    )
                    if step_i is None: print("    [warn] <answer_initial> digit not found")
                    if step   is None: print("    [warn] <answer> digit not found")
                elif use_rationale:
                    rating, rationale, confidence, confidence_score = \
                        extract_rating_rationale(raw_text)
                    step, raw_lp, norm_probs, label_mass = find_rating_token_step(
                        generated_ids, out.scores, tok, rating_token_ids
                    )
                    if step is None: print("    [warn] digit not found inside <answer>")
                else:
                    rating, rationale = extract_rating_bare(raw_text)
                    step, raw_lp, norm_probs, label_mass = extract_step0_logprobs(
                        out.scores, rating_token_ids
                    )

                error_msg = None
                break

            except Exception as e:
                error_msg = str(e)
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    print(f"    [retry {attempt}/{MAX_RETRIES}] {type(e).__name__}: {e} - waiting {delay}s")
                    torch.cuda.empty_cache()
                    time.sleep(delay)
                else:
                    print(f"    [failed after {MAX_RETRIES} attempts] {type(e).__name__}: {e}")

        der   = derived_scores(norm_probs)
        der_i = derived_scores(norm_probs_i) if use_chain else {}

        if error_msg:
            print("    Skipping sample - saved as error record.")
        else:
            dist = "  ".join(f"{k}:{norm_probs.get(k, 0):.3f}" for k in RATINGS)
            print(f"    rating={rating}  E[r]={der['expected_rating']}  "
                  f"H={der['entropy_norm']}  mass={label_mass:.4f}")
            print(f"    p: {dist}")
            if label_mass > 1.001:
                print(f"    [ERROR] label_mass={label_mass:.4f} exceeds 1.0 - the rating token "
                      "ids are not distinct, so this distribution is meaningless.")
            elif label_mass < 0.30:
                print("    [warn] LOW label_mass - model likely did not emit a bare digit here. "
                      "If this is common, rerun with --prefill.")
            if raw_text is not None and not raw_text.strip()[:1].isdigit() and not (use_chain or use_rationale):
                print(f"    [warn] output did not start with a digit: {raw_text[:60]!r}")

        if sid not in all_records:
            all_records[sid] = {
                k: row[k] for k in (
                    "sample_id", "metric", "instance", "mean_human",
                    "majority_human", "individual_human_scores",
                )
            }
            all_records[sid]["model_outputs"] = {}

        out_rec = {
            "rating":                rating,
            "rationale":             rationale,
            "verbalized_confidence": confidence,
            "verbalized_confidence_score": confidence_score,
            "raw":                   raw_text,
            "logprob_rating":        {k: raw_lp.get(k) for k in RATINGS} if raw_lp else {},
            "p_rating":              {k: norm_probs.get(k) for k in RATINGS} if norm_probs else {},
            "expected_rating":       der["expected_rating"],
            "argmax_rating":         der["argmax_rating"],
            "entropy":               der["entropy"],
            "entropy_norm":          der["entropy_norm"],
            "label_mass":            label_mass,
            "answer_token_step":     step,
            "error":                 error_msg,
        }
        if use_chain:
            out_rec.update({
                "rating_initial":          rating_initial,
                "flipped":                 (rating_initial != rating)
                                           if (rating_initial is not None and rating is not None) else None,
                "p_rating_initial":        {k: norm_probs_i.get(k) for k in RATINGS} if norm_probs_i else {},
                "expected_rating_initial": der_i.get("expected_rating"),
                "argmax_rating_initial":   der_i.get("argmax_rating"),
                "entropy_initial":         der_i.get("entropy"),
                "entropy_norm_initial":    der_i.get("entropy_norm"),
                "label_mass_initial":      label_mass_i or None,
                "answer_initial_token_step": step_i,
            })
        all_records[sid]["model_outputs"][model_name] = out_rec

        if (idx + 1) % checkpoint_every == 0:
            save_output(output_path, header, list(all_records.values()))
            print(f"    [checkpoint] {len(all_records)} records -> {output_path.name}")

    save_output(output_path, header, list(all_records.values()))

    # ── Per-model summary ────────────────────────────────────────────────────
    done = [r for r in all_records.values() if model_name in r.get("model_outputs", {})]
    if done:
        pairs = [(r["model_outputs"][model_name], r) for r in done]
        valid = [(o, r) for o, r in pairs if o.get("expected_rating") is not None]

        if valid:
            mae_exp = sum(abs(o["expected_rating"] - r["mean_human"]) for o, r in valid) / len(valid)
            mae_arg = sum(abs(o["argmax_rating"]  - r["mean_human"]) for o, r in valid) / len(valid)
            print(f"\n  {model_name}")
            print(f"    MAE vs mean_human   expected_rating {mae_exp:.3f}   argmax {mae_arg:.3f}")

            # Spearman against mean_human - the headline metric for graded tasks
            try:
                from scipy.stats import spearmanr, kendalltau
                xs = [o["expected_rating"] for o, _ in valid]
                ys = [r["mean_human"]      for _, r in valid]
                rho, _ = spearmanr(xs, ys)
                tau, _ = kendalltau(xs, ys)
                print(f"    Spearman rho {rho:.3f}   Kendall tau {tau:.3f}   (expected_rating vs mean_human, n={len(valid)})")
            except ImportError:
                print("    [scipy not available - skipping correlations]")

            # exact match on the subset that HAS a majority
            with_maj = [(o, r) for o, r in valid if r.get("majority_human") is not None]
            if with_maj:
                exact = sum(1 for o, r in with_maj if o["argmax_rating"] == r["majority_human"])
                print(f"    Exact match vs majority_human: {exact}/{len(with_maj)} "
                      f"= {exact/len(with_maj)*100:.1f}%  "
                      f"({len(valid)-len(with_maj)} instances have no majority)")

        masses = [o.get("label_mass") for o, _ in pairs if o.get("label_mass") is not None]
        if masses:
            low = sum(1 for m in masses if m < 0.30)
            print(f"    label_mass: mean {sum(masses)/len(masses):.4f}   "
                  f"min {min(masses):.4f}   below 0.30: {low}/{len(masses)}")
            if low > len(masses) * 0.1:
                print("    [WARN] >10% of judgments have low label_mass. The confidence "
                      "numbers are unreliable - rerun this mode with --prefill.")

    del model, tok
    torch.cuda.empty_cache()
    print(f"  Unloaded {model_name}.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log-prob graded (1-5) summary judge for NEWSROOM (local HF models)")
    parser.add_argument("--metric", default="info", type=str.lower, metavar="CODE",
                        help="info | rel | flu | coh   (default: info). "
                             "Full names also accepted.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model(s) to run. Defaults to the MODELS list in this script.")
    parser.add_argument("--instance-id", default=None, help="Run a single instance by id")
    parser.add_argument("--samples", type=int, default=None,
                        help="Random subset size - use for a pilot, e.g. --samples 10")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore existing output")
    parser.add_argument("--rationale", action="store_true",
                        help="rationale -> answer -> confidence. Logprob at the <answer> digit.")
    parser.add_argument("--chain", action="store_true",
                        help="answer_initial -> rationale -> answer -> confidence. "
                             "Pre-reasoning logprob at <answer_initial>, post at <answer>.")
    parser.add_argument("--prefill", action="store_true",
                        help=f"Append {PREFILL_TEXT!r} after the generation prompt so the next "
                             "token must be the digit. Use if label_mass is low. Bare mode only.")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Default: 8 in bare mode, 512 in rationale/chain mode.")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    args = parser.parse_args()

    if args.metric not in METRIC_ALIASES:
        sys.exit(f"Unknown metric {args.metric!r}. Use one of: {', '.join(METRIC_CODES)}")
    metric      = METRIC_ALIASES[args.metric]      # canonical name, for the JSON lookup
    metric_code = SHORT_FOR[metric]                # short code, for the output filename

    if not DATA_FILE.exists():
        sys.exit(f"Dataset not found: {DATA_FILE}\n"
                 f"Run: cd data/newsroom && bash download_original_data.sh && python convert.py")

    all_data, prompt_template = load_newsroom(DATA_FILE, metric)
    if not all_data:
        sys.exit("No valid instances loaded.")

    if args.instance_id:
        candidate_data = [r for r in all_data if r["sample_id"] == str(args.instance_id)]
        if not candidate_data:
            sys.exit(f"Instance id not found: {args.instance_id}")
    elif args.samples is not None:
        import random; random.seed(42)
        shuffled = list(all_data); random.shuffle(shuffled)
        candidate_data = shuffled[: args.samples]
    else:
        candidate_data = all_data

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 512 if (args.rationale or args.chain) else 8

    model_list = args.models if args.models else MODELS

    if args.chain:
        suffix, mode_str = "_chain", "chain (answer_initial -> rationale -> answer -> confidence)"
    elif args.rationale:
        suffix, mode_str = "_rationale", "rationale -> answer -> confidence"
    else:
        suffix, mode_str = "", "bare-rating (logprob at step 0)"

    output_dir = ROOT / "results_tmp" / "newsroom"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"logprob_judgement_{metric_code}{suffix}.json"

    n_majority = sum(1 for r in candidate_data if r.get("majority_human") is not None)
    print(f"Dataset: {len(candidate_data)} instances  (metric: {metric} [{metric_code}])")
    print(f"  with majority_human: {n_majority}/{len(candidate_data)} "
          f"({len(candidate_data)-n_majority} have none - mean_human is the primary gold)")
    print(f"Mode: {mode_str}")
    if args.prefill and not (args.rationale or args.chain):
        print(f"Prefill: ON - appending {PREFILL_TEXT!r} to force the digit position")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"Models to run ({len(model_list)}): {model_list}")
    print(f"Output: {output_path}\n")
    print(f"Metric prompt template:\n  {prompt_template[:160]}...\n")

    all_records = {} if args.no_resume else load_existing(output_path)
    if all_records:
        print(f"[resume] Loaded {len(all_records)} existing records\n")

    header = {
        "dataset":    "newsroom",
        "task":       "graded",
        "metric":      metric,
        "metric_code": metric_code,
        "scale":      {"worst": 1, "best": 5},
        "mode":       mode_str,
        "prefill":    bool(args.prefill and not (args.rationale or args.chain)),
        "models":     model_list,
        "n_instances": len(candidate_data),
        "gold_note":  "mean_human is primary (always present); majority_human "
                      "is absent where the 3 annotators had no majority",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    for i, model_name in enumerate(model_list, 1):
        print(f"\n{'='*78}\n[{i}/{len(model_list)}] {model_name}\n{'='*78}")
        try:
            run_one_model(
                model_name=model_name,
                candidate_data=candidate_data,
                prompt_template=prompt_template,
                all_records=all_records,
                output_path=output_path,
                header=header,
                no_resume=args.no_resume,
                max_new_tokens=max_new_tokens,
                checkpoint_every=args.checkpoint_every,
                use_rationale=args.rationale,
                use_chain=args.chain,
                prefill=args.prefill,
            )
        except Exception as e:
            print(f"  [model failed] {type(e).__name__}: {e}")
            save_output(output_path, header, list(all_records.values()))
            torch.cuda.empty_cache()

    save_output(output_path, header, list(all_records.values()))
    print(f"\nDone. {len(all_records)} records -> {output_path}")


if __name__ == "__main__":
    main()
