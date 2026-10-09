# Architecture and Data Flow

## Planned research pipeline

```text
Public questions + frozen generation configuration
                         |
                         v
GeneratorBackend (official LLaDA / Fast-dLLM v1)
                         |
       +-----------------+------------------+
       |                                    |
       v                                    v
Complete all candidates           Local trajectory state
(selection-only baseline)         (token/mask/RNG/cache)
       |                                    |
       |                          Partial drafts -> Jev scores
       |                                    |
       |                          Local pruning controller
       |                                    |
       |                          Resume retained trajectories
       |                                    |
       +-----------------+------------------+
                         |
                         v
             Completed candidate pool
                         |
                         v
         Jev final judge + local final selector
                         |
                         v
              Selected candidate ID
                         |
                         v
 Offline evaluator <- private references / hidden tests
                         |
                         v
   Question-level results, pool metrics, cost reports
```

The pruning branch is planned for Weeks 5–8. Both branches use the same frozen final selector. References and hidden tests enter only the offline evaluator.

## Module responsibilities and status

| Module | Responsibility | Current implementation |
| --- | --- | --- |
| `backends/` | Run the fixed diffusion sampler | Generator protocol; real adapters pending |
| `judges/` | Construct public requests and obtain Jev judgments | Payload construction works; transport pending |
| `data/` | Validate development/held-out separation | ID and normalized-text overlap checks |
| `evaluation/` | Score candidates privately and summarize pools | Basic numeric scoring and pool metrics; full task evaluators pending |
| `selection/` | Choose a final candidate from a fixed pool | Extension point for Weeks 2–4 |
| `pruning/` | Retain promising trajectories and manage continuation | Extension point for Weeks 5–8 |
| `experiments/` | Coordinate components and save run artifacts | Synthetic offline smoke workflow |

`GeneratorBackend` and `Judge` are currently Python Protocol definitions without remote calls. The MATH500 and HumanEval evaluator placeholders in `evaluation.base` explicitly raise NotImplementedError until integrated. The framework does not silently replace real models with random scores or synthetic text.

## Existing records

- `Question`: question_id, dataset, split, prompt.
- `Candidate`: question_id, candidate_id, seed, text, completed, truncated, optional confidence.
- `PartialDraft`: candidate text, completed/total blocks, remaining budget and unit, presentation format.
- `CostRecord`: separate timings, NFE, memory, and API cost; unmeasured fields are null.
- `Reference`: private evaluation record containing the reference answer, hidden tests, and entry point.

These are foundation contracts, not a complete experimental schema. Real adapters must add verifiable backend metadata, failure records, and intermediate-state references while keeping local state out of judge inputs.

## Local smoke outputs

```text
artifacts/runs/<run_id>/
  manifest.json             Provenance, versions, seeds, hashes, non-benchmark flag
  config.snapshot.json      Configuration used for this run
  candidates.jsonl          Candidates without references or correctness labels
  judge_payloads.jsonl      Label-free inputs; no actual Jev call
  evaluation_private/
    labels.json             Private local labels and reference values
  metrics.json              Pool statistics for synthetic fixtures only
  costs.json                Measured local runtime; model/remote timings unmeasured
```

Real experiments must also record prompt snapshots, working-tree state hashes, complete dependencies/hardware, dataset manifests, full itemized costs, and errors. The smoke manifest is insufficient for a real experiment. Here, public means inference-visible fields; it does not imply permission to share candidate files externally.
