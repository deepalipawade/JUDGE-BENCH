# 5-Model Panel Analysis

**Anchors (fixed):** Qwen3.6-35B, gemma-4-26b-a4b-it-maas, MiniMax-M2.7, llama-3.3-70b-instruct-maas

**5th seat candidates:** GPT-OSS-120b (avg BAcc 76.0%), gemini-2.5-flash (75.8%), Apertus-8B (64.0%)

BAcc = balanced accuracy of panel majority vote vs gold label. Tied votes (2-2 in 4-anchor panel) excluded from BAcc.

4-anchor columns: (excl) = tied samples excluded; (best) = per-language best anchor breaks 2-2 ties (EN/ES/FR/HI=Qwen, DE=Gemma).

| Lang | n   | 7-model | 4-anchor (excl) | 4-anchor (best) | +GPT  | +Gemini | +Apertus |
|------|-----|---------|-----------------|-----------------|-------|---------|----------|
| EN   | 226 | 84.2%   | 88.8%           | 88.0%           | 87.2% | 85.5%   | 84.4%    |
| DE   | 272 | 78.0%   | 80.1%           | 78.9%           | 78.2% | 78.9%   | 78.5%    |
| ES   | 276 | 79.9%   | 82.5%           | 80.1%           | 78.8% | 79.8%   | 82.5%    |
| FR   | 370 | 77.7%   | 81.1%           | 80.4%           | 77.7% | 78.7%   | 80.0%    |
| HI   | 208 | 80.5%   | 84.9%           | 81.8%           | 80.5% | 79.9%   | 83.0%    |


## Delta vs 4-anchor (per-language best tiebreaker) baseline (pp = percentage points)

| Lang | +GPT    | +Gemini | +Apertus |
|------|---------|---------|----------|
| EN   | -0.8 pp | -2.5 pp | -3.6 pp  |
| DE   | -0.7 pp | +0.0 pp | -0.5 pp  |
| ES   | -1.3 pp | -0.3 pp | +2.4 pp  |
| FR   | -2.7 pp | -1.7 pp | -0.4 pp  |
| HI   | -1.3 pp | -1.9 pp | +1.3 pp  |


## Individual model BAcc (reference)

| Model                   | EN    | DE    | ES    | FR    | HI    | Avg   |
|-------------------------|-------|-------|-------|-------|-------|-------|
| Qwen3.6-35B             | 89.0% | 76.2% | 78.8% | 79.7% | 80.2% | 80.8% |
| gemma-4-26b-a4b-it-maas | 84.1% | 78.9% | 78.8% | 76.3% | 80.2% | 79.7% |
| MiniMax-M2.7            | 84.5% | 71.9% | 76.5% | 76.9% | 78.8% | 77.7% |
| llama-3.3-70b-instruct  | 81.3% | 77.4% | 76.7% | 71.0% | 75.8% | 76.4% |
| GPT-OSS-120b            | 82.1% | 73.6% | 73.9% | 72.9% | 77.9% | 76.1% |
| gemini-2.5-flash        | 78.0% | 75.5% | 75.3% | 76.8% | 73.5% | 75.8% |
| Apertus-8B              | 63.1% | 59.6% | 60.8% | 61.5% | 74.9% | 64.0% |