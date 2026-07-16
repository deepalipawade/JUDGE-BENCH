# Selection Quality Framework for Aggregator Selection

*Applying the "Selection Bottleneck" (When Agents Disagree) framework to our
Dawid–Skene-based aggregator-judge selection.*

---

## 1. Why this framework is relevant

Our task: from a panel of ~4 LLM judges (each outputs reasoning + a binary
yes/no label), select **one judge to act as the aggregator** — it reads all
judges' reasoning + labels and produces a final verdict. We want to select that
aggregator judge **without ground-truth labels**, using Dawid–Skene reliability
scores.

The Selection Bottleneck paper studies exactly the quantity we care about:
*how good is our selector at reaching the best possible choice, and when does
selecting even help?* It gives us (a) a clean way to frame our bounds, and
(b) a predictive threshold that tells us in advance whether our approach can
beat the simplest baseline.

---

## 2. The core variables (mapped to our setup)

The framework describes a team of candidate answers with two numbers, plus a
selector-quality parameter. Here is the direct mapping:

| Framework term | Meaning | Our equivalent |
|---|---|---|
| **M(T)** — team mean | quality of a *randomly picked* candidate | Accuracy of a **random judge as aggregator** = our **lower bound** |
| **O(T)** — team oracle | quality of the *best* candidate, picked perfectly | Accuracy of the **best judge as aggregator (Oracle-B)** = our **upper bound / true ceiling** |
| **Δ = O(T) − M(T)** — exploitable diversity | how much a perfect picker could gain | The **gap our D–S method tries to close** (distance between lower and upper bound) |
| **s ∈ [0,1]** — selector quality | how good the aggregator is at reaching the oracle | How good our **D–S reliability ranking** is at picking the true best aggregator judge |

Selector quality endpoints:
- **s = 0** → selector is no better than random (or pure synthesis) → lands at the mean M(T).
- **s = 1** → selector always picks the best → reaches the oracle O(T).

---

## 3. The output-quality equation

Final accuracy is modelled as a blend of the lower and upper bound, weighted by
selector quality:

```
Q(T, s) = s · O(T) + (1 − s) · M(T)
```

**In words:** a great selector (s → 1) gets us the *best* aggregator; a poor
one (s → 0) gets us the *average* aggregator. Our D–S selector's entire job is
to push **s** as high as possible — i.e. to reliably identify the judge that
Oracle-B would have chosen, but without using labels to do so.

---

## 4. The crossover threshold s\* (the useful diagnostic)

For a diverse team whose *mean* is below the best single model but whose *oracle*
is above it, there is a threshold **s\*** where selection flips from hurting to
helping:

```
s* = (μ_best − M(T)) / (O(T) − M(T))
```

where **μ_best** = accuracy of the single best individual judge (its own label
quality, *not* acting as aggregator).

**Decision rule:**

```
selection helps  ⟺  s > s*
```

- **Below s\*:** the diverse team's mean drags results down → diversity *hurts*.
  We'd be better off just trusting the best single judge. (This is the regime
  where naive synthesis / Self-MoA underperforms.)
- **Above s\*:** a good selector reaches the oracle → diversity *helps*. This is
  the regime we want our D–S selection to land in.

**Why this matters:** we can compute s\* from numbers our experiment already
produces (best-single-judge accuracy, random-aggregator accuracy,
oracle-aggregator accuracy). It tells us whether aggregator selection is
fundamentally worth it *before* we over-invest.

Reference data point from the paper: judge-selection sits well above s\*
(win-rate ~0.81), majority vote sits right at it (~0.50), and MoA synthesis
sits far below (~0.18). Since **our approach is judge-selection**, the framework
predicts we should land in the favourable regime — which is encouraging
motivation.

---

## 5. How to use this in our project

**(a) As framing / motivation.**
We adopt the selector-quality framework: our D–S reliability ranking is a
selector with quality *s*, and our goal is to push *s* above the crossover
threshold *s\** where selection beats the best single judge.

**(b) As a diagnostic.**
Compute s\* from our bounds. Compute our D–S selector's *effective* s (from
where D–S-selected accuracy lands between M and O). Report whether **s > s\***.
That single comparison tells us if the approach is worthwhile.

**(c) As an explanatory lens for results.**
If D–S selection does *not* beat the best single judge, this framework explains
*why* (our s fell below s\* — the selector wasn't discriminating enough). An
explained negative result is far more useful than an unexplained null.

---

## 6. Computing effective s and s\* from our four numbers

Our experiment produces four accuracies:

- `A_random`   = random judge as aggregator (**M(T)**, lower bound)
- `A_ds`       = D–S-selected judge as aggregator (**our method**)
- `A_oracleA`  = best-individual-judge as aggregator (proxy oracle)
- `A_oracleB`  = best-aggregator-by-trying-all (**O(T)**, true upper bound)

Plus `mu_best` = best individual judge's own label accuracy.

**Crossover threshold:**
```
s_star = (mu_best − A_random) / (A_oracleB − A_random)
```

**Effective selector quality of our D–S method:**
```
s_effective = (A_ds − A_random) / (A_oracleB − A_random)
```

**Interpretation:**
- `s_effective` close to 1 → D–S selection nearly reaches the oracle (excellent).
- `s_effective` close to 0 → D–S selection is barely better than random.
- `s_effective > s_star` → our aggregator selection **beats** just using the
  best single judge → the approach is justified.

---

## 7. Important caveats (do not oversell)

1. **It's a model, not a law.** The linear Q(T, s) = s·O + (1−s)·M is an
   idealization. It assumes selector quality maps linearly to output quality.
   Use it as a framing and prediction tool, not as ground truth to fit results to.

2. **Their selector picks an existing candidate; our aggregator re-decides.**
   In their model, s = 1 means "always pick the best existing answer." Our
   aggregator *synthesizes* — it reads reasoning and can output a verdict none
   of the judges gave, potentially better or worse than any single input. So
   the mapping is strong for the *selection* half, looser for the *synthesis*
   half. Their framework treats pure synthesis as s ≈ 0; our case is a hybrid
   (a *selected strong judge* doing synthesis) that they don't fully model.

3. **Our s is estimated, not observed.** We infer s after the fact from where
   D–S-selected lands between M and O. So s\* is a post-hoc check, not a pure
   a-priori gate — still useful, just not magic.

---

## 8. One-line summary

Our D–S reliability ranking is a *selector* with quality *s*; the goal is to
push *s* above the crossover threshold *s\** so that selecting an aggregator
judge beats simply trusting the best single judge — and we can measure both *s*
and *s\** directly from the four accuracies our experiment already produces.
