# Runbook

## Local framework

From the project root, run `make setup` to create `.venv`, followed by `make doctor`, `make check`, `make test`, and `make smoke`. The core requires no third-party packages or network access. Unresolved configuration fields do not fail ordinary `make check`; it validates structure and basic constraints.

To fail when required fields remain unresolved:

```sh
.venv/bin/python scripts/jev.py check-config --require-resolved
```

`check-config` does not verify that upstream revisions exist or that a GPU/API is usable. Add appropriate runtime checks when integrating them. Do not fill arbitrary strings merely to remove unresolved fields.

For existing dev/held-out JSON arrays:

```sh
.venv/bin/python scripts/jev.py check-splits --dev data/processed/dev.json --heldout data/processed/heldout.json
```

Each element must contain only `question_id`, `dataset`, `split` (dev or heldout), and `prompt`. Store reference answers separately. Maintain a separate manifest for complete provenance.

## Integrating real Week 1 experiments

1. Identify the CUDA/BF16 experiment machine, GPU/memory, drivers, and Python version. Use the local Mac for development and analysis; passing local checks does not establish GPU readiness.
2. Create a separate inference environment from the selected official LLaDA and Fast-dLLM **v1** commits, and lock compatible dependencies. Do not add arbitrary latest torch/transformers versions to the core environment.
3. Pin model/tokenizer snapshots and upstream sampler commits, verify sources/licenses, and populate the configuration. Automatic cloning and weight-download scripts are not currently provided.
4. Implement the `backends/base.py` interface while preserving upstream sampling precision and behavior. Handle GPU synchronization, warmup, microbatching, cache settings, and NFE correctly in measurements.
5. Prepare real data and evaluators. Use validated mathematical equivalence for MATH500, an isolated worker for HumanEval, and official scoring/prompt definitions for ProcessBench.
6. Implement `judges/base.py` from Jev documentation. Confirm the endpoint, authentication, model version, request/response schemas, rate limits, retries, and pricing. `.env.example` is a variable template; the CLI does not automatically load `.env`.
7. Validate each backend, N=8 diversity, truncation, and latency on a small development sample, then record the backend-selection rationale.
8. Complete `reports/weekly/week01.md` and freeze the Week 1 generation protocol. Leave future pruning prompts unfrozen.

## Outstanding information

- GPU/server access and available resources.
- Jev API documentation, access method, fixed model version, and pricing.
- Development material and dataset sources/revisions.
- Pinned backend/evaluator versions, sampling steps, temperature, task prompts, and few-shot rules.

These gaps do not prevent local scaffold checks, but they affect real experiments. Continue useful independent work while resolving them.
