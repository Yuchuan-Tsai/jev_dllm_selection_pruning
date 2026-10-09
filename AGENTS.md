# Jev × dLLM Agent Guidelines

## Reading order

1. This file applies throughout the project. Any future AGENTS.md in a subdirectory provides more specific guidance within that directory.
2. `README.md`: startup commands and current implementation status.
3. `docs/week01.md`: current tasks, acceptance criteria, and outstanding work.
4. `docs/protocol.md` and `configs/week01/protocol.json`: research constraints and the draft protocol.
5. `docs/roadmap.md`: dependencies, decision points, and deliverables across ten weeks.
6. The root-level `Jev × dLLM 10-Week Timeline.pdf` is the original plan. Week 1 appears on PDF pages 4–6.

Explicit user instructions in the conversation take precedence. If implementation differs from the original PDF, explain the difference and record the rationale in `docs/decisions.md`; do not silently modify the PDF. Confirm the actual project directory before working instead of relying on an old absolute path.

## Project language

Use English for all project documentation, code comments, docstrings, CLI messages, configuration descriptions, authored prompts, reports, and project summaries unless the user explicitly requests otherwise. Keep terminology consistent across documentation and code. This language convention supersedes the initial instruction to deliver summaries in Chinese.

## Research objective and scope

With LLaDA-8B-Instruct and the selected sampler held fixed, test whether Jev final-answer selection and partial-trajectory pruning improve the quality–cost trade-off. The main comparison is complete sampling + Jev final selection versus Jev pruning + the same final selector.

- The primary generator is `GSAI-ML/LLaDA-8B-Instruct`. Do not substitute LLaDA-Base, iLLaDA, LLaDA2, or Fast-dLLM v2 without an explicit scope change.
- Fast-dLLM v1 is a candidate inference backend, not a second primary model. Compare it with the official sampler in Week 1, then select a backend using development evidence.
- Jev is an external judge. It receives only the problem, candidate text, and necessary progress/budget metadata. Token/mask states, random states, and KV caches stay local.
- ProcessBench is a separate diagnostic. It is neither a prerequisite for success nor a substitute for selection/pruning experiments.
- Training, token/span control, new cache policies, adaptive N, autoregressive-model comparisons, and multiple pruning checkpoints are optional extensions after the primary evaluation.
- The current stage is Week 1 infrastructure and validation preparation. Interface definitions for future modules do not constitute implemented experimental systems.

## Data and scoring boundaries

- `schema.Question` and `schema.Candidate` contain only inference-visible fields. Store reference answers and hidden tests in `evaluation.Reference`, `data/private/`, or a run's `evaluation_private/` directory.
- Construct judge payloads through the explicit field allowlists in `judges/payloads.py`. Never send whole raw dataset rows, Reference records, correctness labels, hidden unit tests, or oracle outcomes to judges/selectors.
- Use stable question IDs with dataset context. Development and held-out questions must be disjoint; record sources, revisions, sampling rules, ID lists, and content hashes. ID/normalized-text checks do not establish the absence of all semantic near-duplicates.
- Planned evaluation coverage: a fixed 500-question GSM8K subset, the standard 500 MATH500 questions, and the standard 164 HumanEval problems. Report actual held-out counts if an internal split reduces coverage.
- Tune prompts, thresholds, backends, and parameters only on development data. Do not tune using held-out outcomes.
- `evaluation/answers.py` currently provides conservative numeric extraction/equality checks for basic validation. It is not a MATH500 equivalence evaluator. Symbolic expressions, sets, multiple answers, and units require a validated task-specific evaluator.
- Execute HumanEval candidate code only in an isolated evaluation worker with time and resource limits. Do not directly exec/eval generated code in the main process. Until the worker is connected, fail explicitly as unimplemented.
- Use the pinned official first-error-localization F1 definition for ProcessBench. Do not replace it with generic binary-classification F1.

## Reproducibility and experimental protocol

- Week 1 starting point: generation length=256, block size=32, generator BF16. Check truncation/completion rates before freezing these settings.
- Preserve the selected sampler's higher-precision sampling operations. Do not cast every tensor to BF16 for speed.
- For every real experiment, save model/tokenizer revisions, upstream code commits, project commit/working-tree difference hash, prompt text/hashes, complete configuration, dependency lock, hardware, drivers, cache settings, microbatch size, concurrency policy, dataset manifest, and seed schedule.
- Use `reproducibility.trajectory_seed` for independent question/candidate seeds. Real backends must also maintain independent RNG states; a seed function alone does not implement pause/resume.
- Make candidate order and tie-breaking reproducible. Use identical candidate pools when comparing final selectors and the same frozen final selector when comparing pruning methods.
- Blocks 2/4/6 are block-completion checkpoints, not known fractions of total computation. Pruning must not alter the sampler of retained trajectories.
- Pass a no-pruning pause/resume control in Week 6 before making online-pruning claims.
- Distinguish real results from smoke/fixture outputs. Never invent accuracy, GPU time, Jev cost, frozen status, or diagnostic conclusions. Use null for unmeasured values, not zero.
- Record trajectory-NFE, GPU time, generation wall time, draft preparation, judge request time, restoration overhead, end-to-end latency, peak memory, and API cost in USD separately. Do not add milliseconds to forward counts; under concurrency, do not treat the sum of request latencies as end-to-end time.
- Account for all judge calls, including final selection, retries, and failures. Preserve negative and mixed results.
- Use the question as the statistical unit. Multiple candidates for one question are not independent test questions. Report gap-closed as undefined/null when its denominator is zero.
- Predefine the accuracy-loss tolerance in Week 4, freeze primary settings at the end of Week 8, and perform held-out evaluation in Week 9.

## Code organization and implementation

- Place source code in `src/jev_dllm/`. Keep scripts as thin entry points and experiment logic in the package. Reuse core types; do not duplicate experimental logic in notebooks.
- `backends/` integrates samplers; `judges/` integrates Jev; `evaluation/` handles private scoring; `selection/` and `pruning/` implement decisions; `experiments/` orchestrates runs.
- The core scaffold uses only the Python 3.11+ standard library. Keep model dependencies in a separate environment and lock them after choosing the backend. Do not automatically download 8B weights locally or assume a Mac can run the reference CUDA workflow.
- Do not guess Jev endpoints, authentication, response schemas, or model versions, or assume OpenAI API compatibility. Continue independent work when information is missing and record specific gaps.
- Imports, tests, and smoke checks must not trigger model calls or paid API requests. Follow the user's authorized experimental scope without adding repetitive approval steps.
- Exclude `.env`, weights, datasets, trajectory states, and raw run outputs from Git. Version configuration templates, code, documentation, nonsensitive manifests, and reviewed summaries.
- Create new run directories rather than overwriting raw outputs. Assign each run a unique run_id and retain a configuration snapshot.
- Keep substantial third-party implementations in `third_party/` and record revisions. Do not copy them into project code without provenance.

## Validation and completion criteria

Run from the project root:

```sh
make setup
make doctor
make check
make test
make smoke
```

Alternatively, use `python3 scripts/jev.py ...` without installing the package. `check-config --require-resolved` should return a nonzero status while required protocol fields are unresolved. Structural validation does not establish model integration or protocol freeze.

When changing scoring, data isolation, seeds, cost accounting, or configuration, add tests that can reveal substantive errors and run the relevant checks. Documentation-only changes do not require broader testing. Basic checks must work without networking, a GPU, or an API key.

Deliver English summaries of changes, verification, and outstanding dependencies, with local file links. Mark tasks in `docs/week01.md` complete only when real artifacts and their verification records exist. Synchronize documentation when changing the protocol, data splits, or project status. Do not automatically publish remotely, commit, or push without a corresponding request.
