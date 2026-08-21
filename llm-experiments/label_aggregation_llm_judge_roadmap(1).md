# Label Aggregation for LLM-as-a-Judge: A Learning & Experiment Roadmap

## Goal

The specific problem is:

> **Given multiple LLM judges that produce binary factuality labels (0/1), and no ground-truth labels, how can we infer the latent label as reliably as possible?**

You do **not** need to study crowdsourcing broadly. Focus on **label aggregation / truth inference**.

---

## 1. The core problem

You have:

- Items: \(i = 1,\ldots,N\)
- Judges: \(j = 1,\ldots,M\)
- Observed judgments: \(y_{ij} \in \{0,1\}\)
- Latent factuality: \(z_i \in \{0,1\}\)
- No observed ground truth \(z_i\)

The goal is to infer:

\[
P(z_i=1 \mid y_{i1},...,y_{iM})
\]

or ultimately:

\[
\hat z_i
\]

Example:

| Claim | Judge A | Judge B | Judge C | Judge D |
|---|---:|---:|---:|---:|
| x₁ | 1 | 1 | 0 | 1 |
| x₂ | 0 | 0 | 1 | 0 |
| x₃ | 1 | 0 | 1 | 1 |

The central challenge is that **the truth is latent**.

---

# 2. The conceptual hierarchy

```text
Multiple LLM judges
        |
        v
Binary judgments (0/1)
        |
        v
Simple baselines
  - Majority vote
  - Weighted vote
        |
        v
Agreement analysis
  - Pairwise agreement
  - Cohen's kappa
  - Krippendorff's alpha
        |
        v
Latent-truth models
  - Dawid–Skene
  - GLAD
  - MACE
        |
        v
Inference / estimation
  - EM
  - Bayesian methods
        |
        v
LLM-specific complications
  - Judge heterogeneity
  - Item difficulty
  - Correlated judges
  - Systematic biases
        |
        v
External validation
  - Small trusted ground-truth subset
  - Synthetic data with known truth
```

---

# 3. What each concept means

| Concept | What it answers |
|---|---|
| Crowdsourcing | Who/how do we obtain judgments? |
| Annotation | What judgment does a worker/judge assign? |
| Label aggregation | How do we combine multiple judgments? |
| Majority vote | A simple aggregation rule |
| Dawid–Skene | A probabilistic latent-truth aggregation model |
| GLAD | A model incorporating judge ability and item difficulty |
| MACE | A model designed to identify unreliable/spamming annotators |
| EM | An algorithm used to estimate latent variables/model parameters |
| Bayesian aggregation | Aggregation while explicitly modeling uncertainty |

### Important distinction

**Dawid–Skene and EM are not two competing aggregation techniques.**

- **Dawid–Skene = model**
- **EM = one algorithm for fitting/inference in that model**

---

# 4. The learning path

## Stage 0 — Formalize the problem

Be completely comfortable with:

\[
y_{ij} = \text{observed judgment}
\]

\[
z_i = \text{latent factuality}
\]

The fundamental task is:

\[
\text{multiple noisy observations} \rightarrow \text{latent truth}
\]

This is the foundation for everything else.

---

## Stage 1 — Majority vote

**Learn and test this first.**

For binary labels:

\[
\hat z_i =
\mathbb{1}
\left(
\sum_j y_{ij} > M/2
\right)
\]

Why it matters:

Every sophisticated aggregation method needs to answer:

> **Why are you better than majority vote?**

If Dawid–Skene cannot beat majority vote under your conditions, that is itself an important result.

---

## Stage 2 — Understand judge agreement

Inspect how your LLM judges behave.

Useful measurements:

- Raw agreement
- Pairwise agreement
- Cohen's kappa
- Krippendorff's alpha
- Fraction of unanimous items
- Disagreement rate

The most important question is:

> **Do the judges actually behave differently?**

If all judges are almost identical, sophisticated aggregation has little room to help.

---

## Stage 3 — Dawid–Skene

This should be your **first serious aggregation model**.

The basic idea:

> Different judges have different, unknown reliability, while the true label is also unknown.

For binary factuality, a judge has quantities resembling:

\[
P(y_{ij}=1 \mid z_i=1)
\]

and

\[
P(y_{ij}=0 \mid z_i=0)
\]

These correspond roughly to sensitivity and specificity.

The circularity is the key:

- If we knew judge reliability, we could estimate the truth.
- If we knew the truth, we could estimate judge reliability.
- We know neither.

Dawid–Skene solves this jointly.

---

## Stage 4 — EM

Learn EM **through Dawid–Skene**, rather than studying generic EM first.

### E-step

Estimate:

> Given current judge reliabilities, how likely is each item to be factual?

### M-step

Estimate:

> Given the inferred truths, how reliable is each judge?

Repeat until convergence.

---

# 5. Stage 5 — Synthetic experiments

This is **extremely important**.

Before relying on real LLM data, construct a world where you know the truth.

```text
True labels
    ↓
simulate judges
    ↓
different accuracies
    ↓
different biases
    ↓
different disagreement
    ↓
aggregation
    ↓
compare with known truth
```

Example judge qualities:

- Judge A: 95% accurate
- Judge B: 90%
- Judge C: 70%
- Judge D: 60%

Compare:

- Majority vote
- Weighted vote
- Dawid–Skene

against the known simulated truth.

---

# 6. Stage 6 — Judge heterogeneity

Deliberately create different judge scenarios.

### A — Equal judges

All judges have similar accuracy.

Expected: majority vote should be strong.

### B — One excellent judge

One judge is much better than the others.

Question: can aggregation discover and exploit that?

### C — Systematic bias

One judge consistently makes a particular type of mistake.

Question: can aggregation detect/downweight the judge?

### D — Random judge

One judge is nearly random.

Question: can the method learn to ignore it?

### E — Adversarial judge

One judge systematically tends to disagree with the truth.

Question: how robust is the aggregation?

This is where methods such as MACE become interesting.

---

# 7. Stage 7 — Item difficulty

Some factuality items are inherently harder than others.

A judge might have:

```text
Easy items: 98%
Hard items: 65%
```

This motivates models such as **GLAD**, which incorporate both:

- judge ability
- item difficulty

Key experiment:

> Does modeling item difficulty improve aggregation over Dawid–Skene?

---

# 8. Stage 8 — Bayesian aggregation

Only move here after understanding Dawid–Skene/GLAD/MACE.

Instead of:

> Claim 17 = TRUE

you can have:

> Claim 17 = TRUE with probability 0.82

Conceptually:

\[
P(z_i=1 \mid \text{all judgments})
\]

This can be useful downstream, but don't start here.

---

# 9. Stage 9 — The major LLM-specific problem: correlated judges

Classical aggregation often benefits from multiple approximately independent annotators.

But:

> GPT + Claude + Gemini are not necessarily independent sources of evidence.

They may share:

- training data
- knowledge
- biases
- reasoning patterns
- evaluation tendencies
- systematic failure modes

A majority vote can therefore be misleading when several judges share the same error.

This is potentially one of the most interesting parts of your project.

---

# 10. Stage 10 — More advanced models

Only after the basics are established, explore:

- Correlated-annotator models
- Hierarchical Bayesian models
- Spectral/matrix methods
- Learned aggregation
- Neural aggregation
- Calibration-based weighting
- Judge specialization
- Prompt-level aggregation
- Multiple samples from the same LLM
- Other ensemble methods

These are extensions, not the starting point.

---

# 11. The actual experimental ladder

## Experiment 1 — Majority vote

```text
LLM judgments
      ↓
majority
      ↓
label
```

This is your baseline.

## Experiment 2 — Simple weighted vote

Give judges different weights and ask:

> Can simple weighting already improve over majority vote?

## Experiment 3 — Dawid–Skene

Run DS without ground truth.

Estimate:

- latent factuality
- judge confusion matrices

Compare directly with majority vote.

## Experiment 4 — Synthetic ground truth

Create simulated data with known truths and judge behavior.

Measure:

- Accuracy
- Precision
- Recall
- F1
- Calibration
- Log loss
- Brier score

## Experiment 5 — Real LLM judges

Compare:

```text
Majority
   vs
Weighted vote
   vs
Dawid–Skene
   vs
GLAD
   vs
MACE
```

## Experiment 6 — Judge correlation

Measure how dependent the LLM judges are, then deliberately create correlated judge groups.

Question:

> Does an aggregation method still work when judges are not independent?

## Experiment 7 — Small trusted ground-truth set

Even if your main setting has no ground truth, having a small expert-verified subset is highly valuable.

For example:

```text
100,000 claims
       |
       +-- 99,000 no ground truth
       |
       +-- 1,000 expert verified
```

Use the expert set to evaluate which aggregation method actually recovers reality.

---

# 12. What to learn deeply vs. lightly

## Learn deeply

1. Majority vote
2. Annotator/judge agreement
3. Dawid–Skene
4. EM
5. Synthetic noisy-label experiments
6. GLAD / item difficulty
7. MACE / unreliable annotators
8. Judge correlation/dependence
9. Uncertainty and Bayesian aggregation

## Know they exist, but don't study yet

10. Spectral methods
11. More exotic Bayesian models
12. Neural aggregation
13. Learned/meta aggregators
14. Preference/ranking aggregation

---

# 13. Why factuality is a good starting point

Binary factuality gives you a clean research problem:

\[
\boxed{
\text{Multiple noisy binary judgments}
\rightarrow
\text{latent binary truth}
}
\]

You avoid the extra complexity of preference/ranking problems such as:

> Which answer is better?

Those lead into another literature, including Bradley–Terry, Thurstone, Plackett–Luce, and ranking/preference aggregation.

You can explore those later.

---

# 14. The most important conceptual warning

> **Without ground truth, an aggregation algorithm is not literally discovering truth. It is making assumptions about how the observed judges were generated and using those assumptions to infer a latent truth.**

This matters especially for LLM judges.

An aggregation method may implicitly assume:

- judges are sufficiently independent
- each judge has relatively stable reliability
- there is a meaningful latent truth
- disagreements are mostly noise rather than systematic bias

Those assumptions may be much less realistic for LLM judges than for human annotators.

That is potentially where your research becomes interesting.

---

# 15. Recommended starting point

If you want the shortest path to being productive, use this sequence:

```text
Phase 1
  Binary labels → Majority vote → Agreement
Phase 2
  Dawid–Skene → EM
Phase 3
  Synthetic data → Known truth → Heterogeneous judges
Phase 4
  GLAD → MACE
Phase 5
  Real LLM judges → Judge correlation → Expert subset
Phase 6
  Bayesian uncertainty → Correlated-judge models → Advanced methods
```

A clean initial research question:

> **When multiple LLM judges independently evaluate factuality as a binary 0/1 task, when do model-based label aggregation methods outperform majority voting in recovering latent factuality?**

Useful follow-up questions:

- Does judge heterogeneity matter?
- Does item difficulty matter?
- Does the number of judges matter?
- How much judge correlation hurts aggregation?
- Can an aggregation method identify weak judges?
- When does weighting outperform majority?
- Does probabilistic aggregation provide better calibrated uncertainty?
- How much does a small trusted ground-truth set change conclusions?

---

# 16. Minimal mental model

```text
Observed LLM judgments
          |
          v
    Aggregation model
          |
   +------+------+------+
   |             |      |
Majority      DS/EM   GLAD/MACE
   |             |      |
   +------+------+------+
          |
          v
     Latent label
```

The deeper research question is:

> **Which assumptions and aggregation strategies best recover the underlying factuality when the judges themselves are noisy, heterogeneous, and potentially correlated?**
