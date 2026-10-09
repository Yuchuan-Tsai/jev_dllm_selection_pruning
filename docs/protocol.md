# Research Protocol (Draft; Not Frozen)

## Specified and unresolved settings

The Timeline specifies LLaDA-8B-Instruct as the primary model and Jev as the primary judge, with an initial comparison between the official sampler and Fast-dLLM v1. Start with generation length=256, block size=32, and generator BF16, preserving the chosen implementation's higher-precision sampling operations.

`configs/week01/protocol.json` is the machine-readable draft. A null value marks an unresolved model/tokenizer revision, backend commit or step count, temperature, fixed Jev version/API contract, data manifest, evaluator revision, hardware, or dependency lock. Microbatch=1, concurrency=1, seed=20261008, and timeout=60 seconds are scaffold defaults, not explicit requirements in the original plan. Structural validation does not establish experimental readiness.

The original plan does not specify a daily schedule, GPU model, Jev provider, or API format. Fill these fields using actual development validation rather than assumptions.

## Data and access

| Task | Intended held-out coverage | Scoring |
| --- | --- | --- |
| GSM8K | Fixed 500-question test subset | Final numeric-answer accuracy |
| MATH500 | Standard 500 problems | Fixed extraction and mathematical equivalence |
| HumanEval | Standard 164 problems | Unit-test pass rate |
| ProcessBench | Limited GSM8K/MATH diagnostic subsets | Official first-error-localization F1 |

Manifests must record source/revision, split, ID list, preprocessing/extractor versions, sampling seed, content hashes, and actual counts. Prefer separate development material. After ID and normalized-text checks, review source overlap and semantic near-duplicates manually; pay particular attention to the relationship between MATH development material and MATH500.

Jev and every deployable selector see only public question/candidate inputs. Reference answers, correctness labels, and hidden tests remain separate and are available only to offline evaluation. ProcessBench inputs contain the problem and steps, without label or final_answer_correct.

## Generation, selection, and state

Fix candidate IDs and seed schedules. Future N=2/4/8/16 comparisons use reproducible subsets of the same pool, with order and tie-breaking recorded. In addition to independent seeds, pause/resume requires per-trajectory token/mask, RNG, and implementation-specific cache state.

Week 5 checkpoints occur after blocks 2/4/6; do not infer fractions of total computation from these positions. Account for any additional forward passes needed to construct a fully predicted draft. Keep continuation rules identical across pruning methods.

## Measurement

Record the definition of trajectory-NFE, GPU time, generation wall time, and peak memory per trajectory/batch. Separately record draft preparation, judge requests, orchestration, restoration, end-to-end time, and API cost in USD. Distinguish throughput from per-question latency and document warmup, synchronization, and concurrency policies. Do not combine NFE and seconds into an undefined total cost. Remote judging is not free.

Real backends must record completion/truncation, retries, timeouts, failures, and incomplete-candidate policies. None/null means unmeasured. Smoke end-to-end seconds measure the local verification workflow and are not model inference performance.

## Freeze gates

Freeze the generation/scoring protocol in Week 1. Freeze the final selector and predefine the accuracy-loss tolerance in Week 4. Freeze primary prompts, configurations, and evaluation decisions at the end of Week 8. Record the date, content hash, and supporting evidence for each freeze. Week 1 does not require choosing the future partial-draft prompt or pruning settings.

In Week 9, evaluate only the frozen system and estimate paired uncertainty at the question level. Do not use held-out results to keep selecting prompts/configurations. Separate published reference scores from local measurements; Fast-dLLM MATH scores must not be relabeled as MATH500.
