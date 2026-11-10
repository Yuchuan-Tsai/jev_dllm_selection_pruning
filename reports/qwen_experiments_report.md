# Qwen Experiments Report

## Setup

Qwen3-14B generated 16 answers for each of 200 development questions per task. Jev pointwise selects the candidate with the highest correct probability; first-answer and seeded-random selection are comparison baselines.

## Results and status

| Task | Candidate generation | Jev/analysis status | Result |
|---|---|---|---|
| GSM8K | Complete: 3,200 candidates | Jev analysis complete | At N=16: Jev **97.5%**, first answer **95.5%**, seeded random **97.0%**. At N=8, Jev reached 97.5%. |
| MATH | Complete: 3,200 candidates | Pointwise recovery and analysis complete; oversized listwise requests were skipped | At N=16: Jev **93.5%**, first answer **87.5%**, seeded random **89.0%**. Jev reached 94.5% at N=8. |
| MBPP | Complete: 3,200 candidates | Jev and local analysis complete | At N=16: Jev **84.0%**, first answer **83.5%**, seeded random **82.5%**. Jev also reached 84.0% at N=8. |

At N=16, Jev is 2.0 percentage points above the first answer and 0.5 points above seeded random on GSM8K; the paired 95% bootstrap interval versus random is −1.5 to +2.5 points. On MATH, Jev is 6.0 points above first and 4.5 points above random; the paired interval versus random is 0 to +9.0 points. On MBPP, Jev is 0.5 points above first and 1.5 points above random; the paired interval versus random is 0 to +3.5 points. These are development results, and the GSM8K interval includes zero.

## Takeaway

Across all three development tasks, Jev pointwise selected answers at least as accurately as the first-answer and seeded-random baselines at N=16. The observed gain was largest on MATH and small on GSM8K and MBPP. The paired intervals touch or cross zero, so these results are suggestive rather than conclusive.
