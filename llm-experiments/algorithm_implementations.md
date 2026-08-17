# Aggregation Algorithm Implementations

This document describes how IWMV, ISP, OWI, DS, and MACE are implemented in this
codebase, what each one computes, and what it outputs.
Corrected from the original mislabeled `isp()` implementation.

**Source files:**
- `llm-experiments/pipeline/utils/aggregation.py` — IWMV, ISP, OWI, Dawid-Skene (hand-coded)
- `llm-experiments/pipeline/utils/mace.py` — MACE wrapper around crowd-kit

---

## Vote encoding

All algorithms work on integer-encoded votes:

| Label        | Integer |
|--------------|---------|
| Supported    | 1       |
| Not Supported| 0       |
| Missing      | None    |

---

## IWMV — Iterative Weighted Majority Vote

**Function:** `run_iwmv(models, matrix)`

**What it computes:** Each judge's "accuracy" is estimated against a soft consensus
label, then converted to a log-odds weight. Votes are re-tallied with those weights,
producing a new soft consensus, and the process repeats until convergence.
This is self-supervised — no gold labels are used.

**Step by step:**

1. **Init:** Start with soft majority vote as the initial pseudo-labels.
   Each item gets a soft label: `0.8` if majority said 1, `0.2` if majority said 0,
   `0.5` if tied or no votes.

2. **M-step (estimate accuracy):** For each judge `m`, compute how often its votes
   agree with the current soft labels:
   ```
   acc_m = mean over items of: soft_label if voted 1, (1 - soft_label) if voted 0
   ```
   Clamped to `[0.51, 1 - eps]` to keep log-odds defined and ensure above-chance.

3. **Weight:** Convert accuracy to log-odds weight:
   ```
   w_m = log(acc_m / (1 - acc_m))
   ```
   Higher accuracy → higher weight. All weights are positive (since acc > 0.5).

4. **E-step (update soft labels):** For each item, compute a weighted score:
   ```
   score_i = sum_m (w_m if voted 1, -w_m if voted 0)
   soft_label_i = sigmoid(score_i)
   ```

5. **Repeat** steps 2–4 until `max(|soft_new - soft_old|) < eps` or max iterations reached.

6. **Hard labels:** `1` if soft > 0.5, `0` if soft < 0.5, `None` if exactly 0.5 (tie).

**Returns:** `(labels, per_model_accuracy, per_model_weight)`

**Key property:** Same core assumption as majority vote — "if judges agree, they're
right." Cannot detect or correct for shared systematic bias.

---

## ISP — Inverse Surprising Popularity (Algorithm 2, Ai/Pan et al. arXiv:2510.01499)

**Function:** `run_isp(models, matrix)`

**What it computes:** For each item, asks: "Is label `s` more popular than you'd
predict from each judge's correlation with the others?" If judges who independently
tend to agree on `s` are all saying `s`, that's unsurprising. If they're agreeing
*more than their pairwise correlations would predict*, that's surprisingly popular —
a stronger signal that `s` is correct.

### Step 1: Build conditional probability table

For every ordered pair of judges `(i, j)` and every label `l ∈ {0, 1}`, estimate:
```
P(judge_i says 1 | judge_j said l) = count(items where j=l AND i=1) / count(items where j=l)
```
This gives 84 probabilities for 7 judges: `7 × 6 ordered pairs × 2 labels`.

If `count(j=l) = 0` (judge j never said label l across all items), falls back to `0.5`
as an uninformative default. **On MEMERAG, the fallback rate is 0% on all languages —
every pair has real data.**

### Step 2: Compute AdvISP scores per item

For each item and each candidate label `s ∈ {0, 1}`:

```
vote_count(s)  = number of valid judges who actually said s on this item

SISP(s, i)     = mean over all j≠i of P(judge_i says s | judge_j says opposite-of-what-j-voted)
               = mean_{j≠i} P(i=s | j=(1 - vote_j))

sisp_sum(s)    = sum over valid judges i of SISP(s, i)

AdvISP(s)      = vote_count(s) - sisp_sum(s)
```

The `SISP` counterfactual asks: "If each other judge had said the opposite of what
they actually said, how often would judge i say `s`?" Subtracting this from the
actual vote count measures how much more popular `s` is than expected from correlations.

**Mathematical identity:** `AdvISP(0) = -AdvISP(1)` always (because
`sisp_sum(0) + sisp_sum(1) = n_valid`, a provable identity from probability axioms).
So the decision is entirely determined by the sign of `AdvISP(1)`.

### Step 3: Predict

```
label = 1  if AdvISP(1) > 0
label = 0  if AdvISP(1) < 0
label = None if AdvISP(1) = 0 (exact tie, rare)
```

**Returns:** `(labels, conditional_prob_table)`

Note: ISP has no per-model accuracy or weights — it aggregates at the item level,
not by estimating judge quality.

### Observed behavior on MEMERAG

ISP agrees with majority vote on **100% of items** across all 5 languages, including
every 4-3 split item. This is not a bug or a sparsity problem. It is a structural
consequence of the judge panel's correlation structure:

- All 7 LLM judges are strongly positively correlated with each other:
  `P(i=1 | j=1) ≈ 0.85–0.95` and `P(i=1 | j=0) ≈ 0.10–0.35` for every pair.
- When correlation is uniform and positive, the counterfactual correction always
  moves `sisp_sum(1)` in the same direction as `vote_count(1)`, so the sign of
  `AdvISP(1)` always matches the majority.

ISP adds independent signal over majority vote only when the panel has
*differentiated* correlation — e.g., a correlated sub-cluster voting wrong together
while independent judges vote correctly. LLM judges on a binary classification task
typically do not have this structure.

---

## OWI — Optimal Weighted I (OW-I in the paper)

**Function:** `run_owi(models, matrix)`

**What it computes:** Uses real ISP labels as a fixed pseudo-ground-truth, estimates
each judge's accuracy against those labels, converts to a log-odds weight, then does
a single weighted vote. Unlike IWMV, there is no iteration — it is one pass.

**Step by step:**

1. **Get ISP pseudo-labels:** Call `run_isp(models, matrix)` to get a label per item.
   These serve as the reference "ground truth" for accuracy estimation.

2. **Estimate per-model accuracy:**
   For each judge `m`, count how many items where both the ISP label and the judge's
   vote are non-None, and the judge agreed with ISP:
   ```
   acc_m = count(judge_m agrees with ISP label) / count(both non-None)
   ```
   Clamped to `[0.5 + eps, 1 - eps]`.

3. **Compute log-odds weights:**
   ```
   w_m = log(acc_m / (1 - acc_m))
   ```

4. **Single weighted vote:**
   For each item:
   ```
   score_1 = sum of w_m for judges who voted 1
   score_0 = sum of w_m for judges who voted 0
   label   = 1 if score_1 > score_0, else 0, else None
   ```

**Returns:** `(labels, per_model_accuracy, per_model_weight)`

**Relationship to ISP and IWMV:**

| Property                   | IWMV               | ISP                      | OWI                       |
|----------------------------|--------------------|--------------------------|---------------------------|
| Uses gold labels           | No                 | No                       | No                        |
| Per-model weights          | Yes (iterated)     | No                       | Yes (one pass)            |
| Reference for accuracy     | Own soft labels    | Pairwise cond. probs     | ISP pseudo-labels         |
| Iterations                 | Until convergence  | Single pass              | Single pass               |
| Output: per-model stats    | accuracy + weight  | None (item-level only)   | accuracy + weight         |

---

## MACE — Multi-Annotator Competence Estimation (Hovy et al. 2013)

**Function:** `run_mace(models, matrix)` in `utils/mace.py`

**Implementation:** We do **not** implement MACE from scratch. We call
`crowdkit.aggregation.MACE` from the crowd-kit library (Toloka, 2021).
Our wrapper in `utils/mace.py` handles the DataFrame format crowd-kit expects,
extracts the outputs we need, and converts them back to our internal format.

**Why crowd-kit rather than from scratch:** MACE's variational Bayes EM involves
Beta-Dirichlet priors and multiple random restarts for stability — a well-tested
library implementation is more reliable than a hand-coded version for this.

**What MACE models:**

Each annotator has two latent variables:
- `spamming` (ψ_i): probability that annotator i is in "spamming mode" (ignores the item, picks randomly)
- `strategy` (θ_i): probability distribution over labels when not spamming

For each annotation, the generative model is:
```
with prob ψ_i  → annotator spams:  label ~ Uniform({0, 1})
with prob 1-ψ_i → annotator is honest: label ~ θ_i
```

EM (or variational Bayes) infers ψ_i, θ_i, and the latent true label for each item.

**Parameters we use:**
```
n_restarts    = 10     # random restarts to avoid local optima
n_iter        = 50     # EM/VB iterations per restart
method        = "vb"   # variational Bayes (vs "em" = standard EM)
smoothing     = 0.1    # Laplace-style label smoothing on θ
alpha = beta  = 0.5    # Beta(0.5, 0.5) prior on spamming probability
random_state  = 42     # seed for reproducibility
```

`method="vb"` (variational Bayes) is preferred over `"em"` because it provides
uncertainty estimates and is less prone to degenerate solutions.

**Input format:** crowd-kit expects a DataFrame with columns `task`, `worker`, `label`.
We build this from our `{model: [int|None]}` matrix, skipping `None` entries.

**Returns:** `(labels, competence, entropy_per_item)`

- `labels`: hard prediction per item (argmax of posterior label distribution)
- `competence`: per-model non-spamming probability = `spamming_[:, 1]` from crowd-kit.
  Range [0, 1]. Higher = more reliable. This is `1 - ψ_i`, not a direct accuracy estimate.
- `entropy_per_item`: Shannon entropy of `mace.probas_` (posterior label distribution)
  per item. Higher = MACE is more uncertain about that item's true label.

**Key difference from DS:** Dawid-Skene models annotator accuracy as sensitivity (α)
and specificity (β) — class-conditional correctness rates. MACE instead models
annotators as sometimes spamming (random) and sometimes honest. DS captures
class-specific bias; MACE captures global reliability vs. noise.

---

## Pipeline keys

| Config key  | Function called  | What it is                              |
|-------------|------------------|-----------------------------------------|
| `"majority"`| `run_majority`   | Unweighted majority vote                |
| `"iwmv"`    | `run_iwmv`       | Iterative weighted MV (no gold)         |
| `"isp"`     | `run_isp`        | Pairwise-conditional ISP                |
| `"owi"`     | `run_owi`        | ISP-seeded one-pass weighted vote       |
| `"ds"`      | `run_dawid_skene`| Dawid-Skene EM (bounded, from scratch)  |
| `"mace"`    | `run_mace`       | Variational MACE via crowd-kit          |
