# Ten-Week Roadmap

Based on pages 4–14 of the original Timeline in the project root. Work starts with Week 1; weeks are not automatically mapped to semester or calendar dates.

| Week | Objective and work | Deliverables / decision gate | Main locations |
| --- | --- | --- | --- |
| 1 | Environment, protocol, scoring; small official-sampler/v1 comparison; N=8 and performance diagnostics | Frozen generation protocol, verified scoring, pilot results, runtime profile, ProcessBench diagnostic | configs/week01, backends, data, evaluation |
| 2 | Shared development pools across three tasks, up to N=16; lightweight intermediate snapshots; pointwise/listwise interfaces | Candidate pools, trajectory logs, Jev interfaces, pool statistics | data, judges, artifacts |
| 3 | Fixed-pool comparison of random, majority, confidence, Jev pointwise/listwise/weighted voting, Qwen2.5-7B-Instruct judge, Qwen2.5-Math-PRM-7B, and oracle | Accuracy-versus-N curves, ranking diagnostics, selection costs | selection, evaluation |
| 4 | Failure analysis; choose prompts, aggregation, tie-breaking; predefine accuracy-loss tolerance | Decision A: is Jev final selection worth its cost? Frozen selector; Progress Report 1 | selection, docs, reports |
| 5 | M=8 development trajectories at blocks 2/4/6; raw masks versus fully predicted drafts; random/confidence comparisons | Decision B: do partial rankings and remaining computation justify pruning? Survival curves | pruning, judges, evaluation |
| 6 | One checkpoint, initially M=8 to k=2; independent RNG, state restoration, budget/failure/incomplete-candidate handling | No-pruning pause/resume validation, cost logger, preliminary system | backends, pruning, experiments |
| 7 | Complete sampling versus random/confidence/Jev pruning at comparable measured generation budgets, with a fixed final selector | Development accuracy–cost curves and core baseline table | experiments, reports |
| 8 | Online runs including intermediate/final Jev, draft preparation, orchestration, restoration; fixed concurrency, median/tail latency, targeted ablations | End-to-end curves, API costs, overhead breakdown; Progress Report 2; freeze primary settings | experiments, reports, configs |
| 9 | Frozen-system held-out GSM8K/MATH500/HumanEval evaluation; paired seeds, question-level confidence intervals, failure analysis | Final results, uncertainty estimates, reproducibility checks; optional Dream extension if resources remain | evaluation, reports |
| 10 | Integrate the research question, protocol, selection, pruning, budget comparisons, ablations, failures, and limitations | Final report, presentation, frozen configurations, scripts, result tables | reports, docs, configs |

## Decision rules and research constraints

- At A, retain the strongest development configuration and document limitations if results are mixed. With no useful signal, conduct only a limited partial-draft diagnostic before committing to expensive online experiments.
- B requires both useful partial-draft ranking and sufficient remaining computation to plausibly offset judging costs.
- The core comparison is complete sampling + Jev versus Jev pruning + the same Jev selector. The complete-sampling baseline may spend its comparable budget on more completed candidates; equal initial N is not required.
- Oracle is an offline diagnostic. Do not use code-string equality for code majority voting, or generate/rewrite answers during fixed-pool selector comparisons.
- Success means higher accuracy at comparable cost, or lower cost within the predefined accuracy-loss tolerance. A statistically nonsignificant difference does not establish equal accuracy.
- Correctness and reasoning consistency are different targets. Listwise choice probabilities are not automatically correctness probabilities.
- Multiple checkpoints, adaptive N, additional backends, autoregressive comparisons, and token/span control must not displace the primary experiments.
