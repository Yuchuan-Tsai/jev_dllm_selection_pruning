# Week 4 Experiments Report

## Overview

Week 4 evaluated the frozen Jev pointwise selector in two settings: selecting from completed candidate pools and replaying early-draft pruning decisions. The selector scores each candidate independently and chooses the one with the highest `correct` probability; ties are resolved by original pool order. We also tested whether Jev's selection gains carry over to a stronger answer generator, Qwen3-14B.

## Stronger policy model: Qwen3-14B

Qwen3-14B generated 16 candidates for each of 200 development questions on GSM8K, MATH, and MBPP. We compared Jev pointwise with the first candidate and seeded-random selection.

| Development task | Jev pointwise | First candidate | Seeded random | Jev at N=8 |
|---|---:|---:|---:|---:|
| GSM8K  | **97.5%** | 95.5% | 97.0% | 97.5% |
| MATH  | **93.5%** | 87.5% | 89.0% | 94.5% |
| MBPP  | **84.0%** | 83.5% | 82.5% | 84.0% |

(Jev pointwise scores each answer separately and picks the one most likely to be correct. First candidate means the first answer generated for each question. Seeded random picks one answer at random using a fixed seed, so the choices can be reproduced. N=8 means Jev chooses from the first eight candidates instead of all 16.)

At N=16, Jev was 2.0 percentage points above first-answer and 0.5 points above seeded-random on GSM8K; the paired 95% bootstrap interval versus random was −1.5 to +2.5 points. On MATH, Jev was 6.0 points above first-answer and 4.5 points above random (interval versus random: 0 to +9.0 points). On MBPP, Jev was 0.5 points above first-answer and 1.5 points above random (interval versus random: 0 to +3.5 points). These are development results; the intervals touch or cross zero, so the gains are suggestive rather than conclusive.

## Partial-draft pruning replay

On 200 GSM8K development questions, we used Jev scores from saved intermediate drafts to retain 8 of the 16 candidates at each checkpoint.

| Checkpoint | Retained-pool oracle accuracy | Full-pool oracle accuracy | Top-ranked answer accuracy | Estimated compute saved |
|---:|---:|---:|---:|---:|
| 2 blocks (~64 tokens) | 94.5% | 95.5% | 84.0% | 30.7% GPU seconds; 32.3% NFE |
| 4 blocks (~128 tokens) | 95.0% | 95.5% | 82.5% | 16.5% GPU seconds; 18.7% NFE |
| 6 blocks (~192 tokens) | 95.5% | 95.5% | 85.5% | 7.5% GPU seconds; 9.0% NFE |

(A checkpoint is a saved point partway through generation. A block is one generation step, about 32 tokens in this experiment. The retained pool is the eight candidates kept from the original 16. Retained-pool oracle accuracy asks whether at least one kept answer was correct; full-pool oracle accuracy asks the same about all 16. An oracle assumes we know which answers are correct, so this is an upper bound. Top-ranked answer accuracy is the accuracy of the single highest-scored kept answer. NFE means the number of model function evaluations, a measure of generation work.)

This was a retrospective replay, not live stopping during generation. Retained-pool oracle accuracy is an upper bound because it measures whether any retained candidate was correct. Compute savings are estimates from completed generations and do not include live pruning or resume overhead.

## Held-out evaluations

| Dataset | Frozen Jev pointwise | Comparisons | Evaluation note |
|---|---:|---|---|
| GSM8K-500 | **88.8%** | First candidate 76.8%; seeded random 76.2%; majority vote 84.0% | Frozen selector applied to the held-out pool. |
| HumanEval | **56.7% (93/164)** | First candidate 35.4%; seeded random 36.6% | Uses the project's post-hoc code-extraction scoring policy; not an official HumanEval/EvalPlus result. |
| MATH500 | **57.0% (285/500)** | First candidate 37.8%; seeded random 40.8%; majority vote 47.2% | Eight candidates per question; pinned Math-Verify scoring policy. |

(Held-out questions are kept separate for final evaluation. Majority vote picks the answer in the largest agreement group. A paired bootstrap interval estimates uncertainty by repeatedly resampling the same questions and comparing the methods on each question.)


## Interpretation and limitations

The Qwen development results show Jev performing at or above the two comparison selectors on all three tasks, with the largest observed gain on MATH. The pruning replay suggests later checkpoints preserve more pool-oracle accuracy while saving less estimated compute. On held-out MATH500, Jev outperformed the first, random, and majority-vote baselines under the project's pinned scorer.
