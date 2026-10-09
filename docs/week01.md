# Week 1: Protocol, Setup, and Baseline Validation

Source: pages 4–6 of the original Timeline. Establish a reproducible generation and evaluation setup before collecting large candidate pools.

## Requirements from the plan

| Workstream | Requirements | Acceptance evidence |
| --- | --- | --- |
| Models and backends | LLaDA-8B-Instruct + Jev; small comparison of the official sampler and Fast-dLLM v1 | Single-sample accuracy, diversity, measured generation time on the same dev questions, and backend-selection rationale |
| Initial generation settings | Length=256, block=32, BF16; preserve higher-precision sampling operations | Truncation/completion rates and final settings validated beyond default values |
| Reproducibility | Model, tokenizer, code, prompts, sampling, cache, hardware, microbatch, Jev version | Pinned revisions/commits, configuration, and environment records |
| Data | Fixed 500-question GSM8K subset, MATH500 500, HumanEval 164; disjoint dev/held-out sets | Sources, versions, question IDs, hashes, and actual coverage |
| Scoring validation | Known correct/incorrect examples; answer extraction, mathematical equivalence, code execution | Positive, negative, malformed-output, code-failure, and timeout checks |
| Stochastic candidates | N=8 | Actual sampling seeds, distinct candidates, and duplicate rates; prewritten fixtures are insufficient |
| Performance | Memory, generation time, Jev request latency | Measured profiles on representative inputs; API cost reported separately |
| ProcessBench | Limited GSM8K/MATH diagnostic; official first-error-localization F1 | Fixed subsets, prompt, version, official scoring, and limitations |

Prefer separate development material to preserve intended benchmark coverage. If a benchmark is internally split, report the actual held-out size. Do not present cross-paper scores as local experimental results.

## Suggested execution order

This sequence reflects dependencies, not a daily schedule specified by the PDF. Calendar dates have not been assigned.

1. Identify GPU resources, storage, and the Jev interface; pin model/backend sources and record the environment.
2. Select development material, create dev/held-out manifests, and check IDs, normalized text, and known source overlap.
3. Integrate evaluators and validate correct/incorrect examples, extraction failures, code failures, and timeouts.
4. Integrate the official LLaDA sampler; run a small length=256, block=32 generation check and N=8 diversity check.
5. Integrate Fast-dLLM v1; run the same dev questions under comparable conditions and record differences in cache, steps, and sampling settings.
6. Measure Jev latency/cost and run a limited ProcessBench diagnostic; summarize performance.
7. Select the backend using development results, check truncation, freeze the generation protocol, and complete the Week 1 report.

## Current completion status

### Project initialization

- [x] Ten-week directory structure, module boundaries, and README.
- [x] Root AGENTS.md.
- [x] Draft Week 1 configuration and unresolved-field checks.
- [x] Basic numeric extraction/scoring, split-overlap checks, judge payload allowlists, seed/cost structures.
- [x] Synthetic workflow and automated checks; see `reports/weekly/setup-verification.md` for the verification record.

### Week 1 research deliverables

- [ ] GPU environment and compatible dependencies validated and locked.
- [ ] LLaDA model/tokenizer revisions and both backend commits pinned.
- [ ] Jev API contract, version, authentication, and pricing confirmed.
- [ ] Dev/held-out sources and IDs fixed, with overlap review completed.
- [ ] Real scoring pipelines for all three tasks validated against known correct/incorrect examples.
- [ ] Real N=8 stochastic generation and truncation/completion checks.
- [ ] Small official-sampler versus Fast-dLLM v1 comparison.
- [ ] Memory, generation time, Jev latency, and cost profiles.
- [ ] Limited official ProcessBench diagnostic.
- [ ] Backend selected from development evidence and generation protocol frozen.
- [ ] Week 1 report populated with real results.

## Five final outputs

1. Frozen generation protocol: configuration, model/implementation revisions, prompts, dataset manifests, and environment lock.
2. Verified scoring pipeline: real correct/incorrect checks for GSM8K, MATH500, and HumanEval.
3. Pilot accuracy/diversity: raw N=8 candidates and question-level summaries.
4. Runtime profile: memory and itemized generation, Jev, and end-to-end measurements.
5. ProcessBench diagnostic results: official F1, subset definition, and limitations.

Completing the project scaffold does not complete these five research outputs.
