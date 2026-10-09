# Jev × dLLM

Study whether Jev candidate selection and intermediate trajectory pruning improve accuracy and inference cost for a fixed diffusion language model.

**Current status: the Week 1 local framework is in place; real model experiments have not run.** The original plan is `Jev × dLLM 10-Week Timeline.pdf` in the project root. Agents should read [AGENTS.md](AGENTS.md) first. All project-authored text uses English.

## Quick start

The core framework requires Python 3.11+. Basic checks need no GPU, network access, or API key:

```sh
make setup
make doctor
make check
make test
make smoke
```

`make setup` creates a project-local `.venv` without installing global dependencies. You can also run these commands without activating the environment:

```sh
python3 scripts/jev.py doctor
python3 scripts/jev.py check-config
python3 scripts/jev.py smoke
python3 -m unittest discover -s tests -v
```

The smoke check uses two synthetic arithmetic questions with eight prewritten candidates each to verify storage, judge-input isolation, numeric scoring, and summary metrics. It makes no LLaDA/Jev calls, does not demonstrate stochastic generation diversity, and produces no benchmark results. Outputs are saved under `artifacts/runs/smoke-*/` with `is_benchmark_result=false`.

## Framework overview

The planned research pipeline is: questions → fixed dLLM generation → optional Jev-guided pruning → completed candidates → final-answer selection → private offline evaluation → accuracy/cost reports. The baseline completes all generation without pruning and uses the same final selector.

```text
AGENTS.md                        Collaboration and research rules
configs/week01/protocol.json      Draft Week 1 protocol and unresolved versions
prompts/                         Generation/judging templates for dev validation
src/jev_dllm/
  backends/                      Official LLaDA / Fast-dLLM v1 interfaces
  judges/                        Jev interfaces and label-free payload construction
  data/                          Dataset splitting and overlap checks
  evaluation/                    Private scoring, extraction, candidate-pool metrics
  selection/                     Final-answer selection extension point, Weeks 2–4
  pruning/                       Trajectory pruning extension point, Weeks 5–8
  experiments/                   Run orchestration and the local smoke workflow
  schema.py                      Question, candidate, draft, and cost records
  reproducibility.py             Hashes and independent trajectory seed rules
scripts/                         Command entry points
tests/                           Scoring/boundary tests and synthetic fixtures
data/{raw,processed,private}/    Local data; excluded from Git by default
data/manifests/                  Reproducible source and split manifests
artifacts/{runs,checkpoints}/    Run outputs and local trajectory states
reports/{weekly,figures,tables}/ Weekly reports, figures, and tables
docs/                           Plans, protocol, architecture, and runbook
third_party/                    Future pinned upstream implementations
```

## Implementation status

| Capability | Current status |
| --- | --- |
| CLI, environment inspection, JSON configuration validation | Working |
| Dev/held-out ID and normalized-text overlap checks | Working; semantic deduplication still needs review |
| Explicit final-marker/boxed extraction and numeric equality | Working within a limited numeric scope |
| Judge payload allowlists, itemized cost records, trajectory seeds | Working foundation components |
| Synthetic candidates through run logging | Working; not a real experiment |
| Official LLaDA and Fast-dLLM v1 | Interfaces defined; real adapters pending |
| Jev API, fixed model version, request cost collection | Interfaces defined; service details pending |
| Real dataset downloads and fixed splits | Pending |
| MATH500 equivalence scoring and isolated HumanEval worker | Explicitly raise unimplemented errors |
| Official ProcessBench diagnostics | Integration pending |
| Selectors, online pruning, GPU profiling | Planned for their respective weeks |

## Documentation

- [Week 1 tasks and acceptance criteria](docs/week01.md)
- [Ten-week roadmap](docs/roadmap.md)
- [Research protocol](docs/protocol.md)
- [Architecture and data flow](docs/architecture.md)
- [Local operation and model environment setup](docs/runbook.md)
- [Decision log](docs/decisions.md)
- [Official external references](docs/references.md)
- [Week 1 report template](reports/weekly/week01.md)

Next, establish the GPU environment and Jev service details, integrate one real generator backend, validate scoring and small-sample generation, and compare the other backend. The configuration remains a draft; `frozen=false` is intentional.
