# Jev × dLLM

This project tests whether Jev can improve answers from a fixed LLaDA-8B-Instruct generator by choosing among completed samples. The server generates answers; the local computer prepares public-only Jev requests and privately evaluates the chosen IDs. Jev runs through the Jev AI decision API. It does not modify the selected answer.

```text
Public questions -> LLaDA / Fast-dLLM on a GPU -> saved candidate pools
                                                  |
                         first / random / majority / Jev selectors
                                                  |
                            private benchmark evaluation on the Mac
```

Eight candidates per question is `N=8`. Questions, candidate order and sampling seeds are recorded. The commands in this repository perform completed-answer selection.

All commands use the root `Makefile`: run `make help` for the overview, or
`make benchmarks-help`, `make humaneval-help` and `make math500-help` for task commands.
Recovery targets include the task name so HumanEval and MATH500 cannot shadow
one another. No `make -f` option is needed.

## Version compatibility

The consolidated commands use new `gsm8k500-freeze-v2.json` and
`generation-lock-v2.json` implementation records for **new runs**. Sampling
settings, source revisions, prompts, seeds and scoring rules are unchanged.
The original records remain intact as historical metadata; old archives and Jev
plans must be analyzed or resumed using the code revision recorded when they
were created. Do not refresh their checksums to make them pass under this version.
The previous public implementation is available at commit `872ecad`.
Use separate server directories for different versions, and upload the complete
matching version with `make remote-upload` before using task-specific transfers.

## Setup and local checks

Python 3.11+ is required for the dependency-free local core. The established GPU lock targets Linux/Python 3.11. A provisional AutoDL path is also provided for the PyTorch 2.5.1 / Python 3.12 / CUDA 12.4 image; it reuses the image's PyTorch and is not a frozen experiment environment until validated and captured on the A800. HumanEval code execution uses an isolated, pinned Docker image. Optional MATH500 scoring uses a separate Python 3.12 CPU environment with a wheel-hash lock.

The optional device resolver in `jev_dllm.backends.device` accepts `cpu`, `mps`, `cuda` or `auto`. CPU development requires no PyTorch install; requesting MPS or CUDA checks the installed PyTorch runtime.

```sh
python3.11 -m venv .venv
make doctor check test smoke
make week1-data
make fetch-fast-inputs
```

Data preparation uses pinned public sources and writes ignored local files. Keep private references and API credentials on the local computer. For SSH transfers, copy `configs/server/example.json` to ignored `configs/server/local.json` and supply your SSH alias, server project directory and local results directory. The remote checkout must have its GPU environment and cached model prepared before generation; `make help` lists setup commands. GPU/model setup is explicit and can download large packages or weights. Do not run it on the Mac.

## Benchmarks and generation

| Dataset | Current coverage | Workflow |
| --- | --- | --- |
| GSM8K | 20 development + 500 held-out questions | Generation, numeric scoring, baseline and Jev selection |
| MATH500 | 500 questions | Answer collection, raw-answer Jev selection and frozen offline mathematical-equivalence scoring |
| HumanEval | 164 programming tasks | Answer collection, isolated strict/extracted-code evaluation, Jev selection |
| MBPP | 464 train + validation tasks | Separate code development pool; IDs 511–974 only, with reference code/tests private |
| ProcessBench | 50 GSM8K + 50 MATH examples | Separate first-error diagnostic; not answer generation |

Public inference inputs live in `data/processed/`; reference answers and hidden tests live in `data/private/`. Neither is tracked. Source revisions, hashes and selected IDs are tracked in `data/manifests/`. Do not treat a diagnostic subset as the full benchmark.

MBPP uses the pinned Google Research source and its documented ID ranges: IDs 511–600 (validation) and 601–974 (train) are development data; IDs 1–10 (few-shot) and 11–510 (test) are excluded. The development questions are prepared as `data/processed/mbpp-dev.json`; reference code and tests are kept in `data/private/mbpp-dev-references.json`. No candidate-code execution is wired for MBPP yet.

On the local computer, after configuring SSH:

```sh
make remote-plan
make remote-upload
make benchmarks-check benchmarks-upload
```

`remote-upload` sends the GSM8K development input by default. It skips the frozen 500-question held-out input so ordinary code changes do not require changing the held-out protocol record. Set `"upload_frozen_inputs": true` in the ignored `configs/server/local.json` only when deploying that explicitly frozen workflow; its protocol checks must pass unchanged.

On the server, enter its project directory, set `JEV_SHARED_HUB_CACHE` to the existing model cache and inspect GPU availability. Supply one available A100 index to the chosen command:

```sh
nvidia-smi
make gpu-gsm8k500 GPU_ID=5
# Alternative tasks; choose an available GPU for each concurrent job:
make start-humaneval GPU_ID=0
make start-math500 GPU_ID=5
```

The `start-*` targets create separate tmux sessions and refuse a duplicate session name. Foreground alternatives are `gpu-humaneval` and `gpu-math500`. Generation creates a new uniquely named run each time. These example GPU indices are not allocations. The current BF16 profiles require native BF16 hardware; V100 needs a separately reviewed precision protocol.

### AutoDL A800 image

For the AutoDL **PyTorch 2.5.1 / Python 3.12 / CUDA 12.4** image, the existing Python 3.11 lock and `gpu-fast-check` are not applicable. Keep the image's CUDA-enabled PyTorch by creating a virtual environment that sees system packages, then install the pinned model-library requirements:

```sh
cd /root/autodl-tmp/jev_dllm_selection_pruning
mkdir -p /root/autodl-tmp/huggingface/hub
export JEV_SHARED_HUB_CACHE=/root/autodl-tmp/huggingface/hub
make gpu-autodl-setup AUTODL_PYTHON=python3.12
make gpu-autodl-install
make gpu-autodl-check GPU_ID=0
make gpu-fetch-model CONFIG=configs/week01/diversity.json GPU_ID=0
make gpu-diversity GPU_ID=0
make gpu-fast-diversity-autodl GPU_ID=0
```

`gpu-autodl-check` verifies the image package versions and CUDA/BF16 visibility. `gpu-fast-diversity-autodl` runs the Fast-dLLM synthetic compatibility check before generation. Both checks are prerequisites, not evidence that full LLaDA generation has completed. The AutoDL environment remains provisional until the real GPU runs succeed and their complete dependency captures are reviewed. The pre-existing Python 3.11 lock and frozen GSM8K protocol records are retained unchanged.

On the local computer, copy outputs and verify a terminal run before analysis:

```sh
make results-pull
read -r RUN_ID  # Paste the exact completed run directory name, then Enter.
make results-verify RUN_ID="$RUN_ID"
```

Use `make results-watch` for periodic copying. Copying does not delete server data. `make results-clean RUN_ID="$RUN_ID"` is an explicit deletion of one verified terminal server run; it never cleans the shared model cache.

## Resume interrupted collection

Use `make collection-recovery-plan` for a MATH500 or HumanEval collection stopped
by Ctrl+C. After copying and verifying the interrupted run as above, prepare and
upload its recovery plan locally:

```sh
make collection-recovery-plan RUN_DIR="artifacts/downloads/lab/$RUN_ID"
make collection-recovery-upload RECOVERY_PLAN="artifacts/collection-recovery-plans/$RUN_ID.json"
```

Then, in a server tmux session with the existing model cache configured:

```sh
make gpu-collection-resume GPU_ID=5 RECOVERY_PLAN="artifacts/collection-recovery-plans/$RUN_ID.json"
```

Set `RUN_ID` on both computers to the same exact interrupted directory name and
choose a currently available GPU. The plan verifies checksums, saved record order,
seeds and costs. Recovery copies every saved candidate into a new directory and
generates only missing IDs; the original directory stays intact. The unfinished
candidate restarts from its original seed. It does not restore token or cache
state, and GPU bitwise equivalence is not asserted. Each execution segment keeps
its own timing. No paid Jev requests occur. After another interruption, copy and
verify the **new** run and prepare its own plan; duplicate recovery launches from
the same source are blocked. Do not delete the launch claim to bypass this check.

After recovery finishes, copy and verify the new run, then use `collection-recovery-audit`
with its local `RUN_DIR` to check the retained prefix and provenance. The ordinary
`gpu-math500`/`gpu-humaneval` targets still start new collections from zero. GSM8K
uses `gsm8k500-recovery-plan`, `gsm8k500-recovery-upload` and `gpu-gsm8k500-resume`; see `make gsm8k500-recovery-help`.

## MATH500 Jev selection

Use the complete, verified MATH500 archive, including a verified recovered pool
when applicable. Run these commands on the local computer:

```sh
# Set RUN_ID to the exact complete archive directory name.
RUN_DIR="artifacts/downloads/lab/$RUN_ID"
PLAN_DIR="artifacts/runs/math500-jev-plan-$RUN_ID"
make math500-jev-prepare RUN_DIR="$RUN_DIR" PLAN_DIR="$PLAN_DIR"
make math500-jev-check PLAN_DIR="$PLAN_DIR"
# This command sends 500 PAID requests using JEV_API_KEY from the ignored .env:
make math500-jev-live PLAN_DIR="$PLAN_DIR" APPROVED_CALLS=500
make math500-jev-audit PLAN_DIR="$PLAN_DIR"
```

Each request contains one public question and its eight original answers in
seeded shuffled order. All IDs and raw text remain unchanged, including empty or
truncated answers. No extraction, filtering or repair occurs. References and
correctness labels stay local. Preparation/checking are offline; only `live`
contacts Jev AI. A repeated preparation at the same destination rechecks and
reuses the unchanged plan. A repeated paid launch is blocked, including through
a new plan directory for the same batch. Errors retain evidence and stop without
automatic retries for nontransient failures. The Jev client retries only
documented transient statuses with the same idempotency key; inspect other
failures before attempting a continuation.

The final audit verifies the raw responses and selected original IDs, saves
`selected-answers.jsonl`, and reports token usage and request time. It does not
calculate mathematical correctness: accuracy and dollar cost remain unknown.
Use the separate offline MATH500 scorer below for correctness comparisons; the existing GSM8K numeric scorer is not a substitute.

For a MATH500 Choice/probability mismatch after a valid saved prefix, `make math500-jev-continuation-prepare` prepares a one-retry amendment:

```sh
make math500-jev-continuation-prepare SOURCE_PLAN="$SOURCE_PLAN"
# Set PLAN_DIR to the new continuation directory printed above.
make math500-jev-continuation-check PLAN_DIR="$PLAN_DIR"
# APPROVED_CALLS must equal additional_paid_calls in the checked plan.
make math500-jev-continuation-live PLAN_DIR="$PLAN_DIR" APPROVED_CALLS="$APPROVED_CALLS"
make math500-jev-continuation-audit PLAN_DIR="$PLAN_DIR"
```

This explicitly retries the rejected request once, then processes only unattempted
questions. It preserves the original batch and accepted-response bytes and counts
the rejected attempt in token/time totals. It does not relax response validation;
any new failure stops the continuation. The original live command and its claim
remain unchanged. This amendment does not establish mathematical correctness.

If that continuation also stops on a Choice/probability mismatch, prepare a
further explicit continuation from the **most recent failed plan**:

```bash
make math500-jev-resume-prepare SOURCE_PLAN="$SOURCE_PLAN"
# Set PLAN_DIR to the new directory printed above.
make math500-jev-resume-check PLAN_DIR="$PLAN_DIR"
# Set APPROVED_CALLS to the exact additional_paid_calls value after reviewing it.
make math500-jev-resume-live PLAN_DIR="$PLAN_DIR" APPROVED_CALLS="$APPROVED_CALLS"
make math500-jev-resume-audit PLAN_DIR="$PLAN_DIR"
```

This extension validates every earlier segment, copies all accepted selections
unchanged and sends only the rejected question followed by the unattempted
suffix. Each new failure stops execution and requires a separately prepared,
explicitly approved plan. It preserves the original strict response validator,
prevents duplicate launches and includes all rejected responses in usage totals.
Interrupted requests with no validated response need separate investigation.
The final audit checks selection integrity and usage; mathematical correctness
remains a separate evaluation.

## MATH500 offline scoring

The optional CPU scorer uses pinned Math-Verify, latex2sympy2_extended and SymPy.
It validates independent positive/negative, format and voting fixtures before
freezing the policy, code, runtime, reference hash and exact source evidence.
The current entry point accepts a completed `math500-jev-resume` plan and audits
its full continuation ancestry. Other selection layouts need an explicit adapter;
they are not silently treated as equivalent.

```sh
# Requires an existing Python 3.12 interpreter; override its path if necessary.
make math500-scoring-setup MATH_BOOTSTRAP_PYTHON=python3.12
make math500-validate
# Set VALIDATION_RUN to the validation directory printed above.
# RUN_DIR is the verified generation archive; JEV_PLAN is the completed resume plan.
make math500-scoring-prepare \
    RUN_DIR="$RUN_DIR" JEV_PLAN="$JEV_PLAN" VALIDATION_RUN="$VALIDATION_RUN"
# Set SCORING_PLAN to the new scoring-plan directory printed above.
make math500-scoring-check SCORING_PLAN="$SCORING_PLAN"
make math500-evaluate SCORING_PLAN="$SCORING_PLAN"
```

These commands make no Jev requests and do not load generator weights. Setup
installs small CPU dependencies. Evaluation retains all original candidates,
compares first/random/majority/Jev on the same pool and keeps private labels in
`evaluation_private/`. Majority voting groups equivalent extracted answers
without references; the largest group wins, with ties resolved by original
candidate order. Invalid answers do not vote. Voting on code is not implemented.

The scorer is an automated answer check, with documented numeric, unit and
formatting conventions. It is frozen after collection/selection and before the
first mathematical correctness analysis; it does not retrospectively preregister
the whole study. Never edit a scored policy to improve held-out outcomes.
Version any later reanalysis separately. Independent fixtures and a pinned
runtime establish the exercised behavior, not flawless grading of arbitrary math.

## HumanEval extraction and Jev

Raw responses are preserved. Strict evaluation executes full saved source. The separately versioned extraction reanalysis accepts plain Python or the exact contents of one complete Python code fence. It does not repair code. Ambiguous/unclosed blocks remain failures in the denominator. This is a custom full-source test harness, not a claim of the unmodified official completion harness or EvalPlus.

```sh
make humaneval-image
make humaneval-validate
# Set RUN_DIR to your verified local generation archive:
make humaneval-score RUN_DIR="$RUN_DIR"
# Set STRICT_RUN to the completed evaluation printed above:
make humaneval-reanalyze RUN_DIR="$RUN_DIR" STRICT_RUN="$STRICT_RUN"
```

The following selection workflow reuses those answers and their saved extracted-code test results. It requires no GPU or new code execution. Put `JEV_API_KEY=your_key` in the ignored `.env` file, or set that environment variable. The file is read as data, never executed or printed. Live requests use `https://jev-ai.org/api/v1/systemone/` with model `jev-1.13`; transient HTTP failures retry with a stable idempotency key.

```sh
make humaneval-jev-prepare RUN_DIR="$RUN_DIR"
# Set PLAN_DIR to the directory printed by preparation.
# Executing the next command sends 164 PAID Jev AI requests:
make humaneval-jev-live PLAN_DIR="$PLAN_DIR" APPROVED_CALLS=164
# Set SCORE_RUN to the completed extracted-code evaluation for this same archive:
make humaneval-jev-evaluate PLAN_DIR="$PLAN_DIR" SCORE_RUN="$SCORE_RUN"
```

Each Jev request contains one public question and all eight answers in seeded shuffled order. Extractable answers use the exact code substring; extraction failures retain their original response. Hidden tests, canonical solutions and correctness labels never enter the requests. Private evaluation maps Jev's choice back to the original candidate ID and compares it with first-sample, seeded-random and oracle results. Scores after extraction are labeled reanalysis.

The live command stops on any error, retains partial evidence and prevents replaying the same batch—even through a newly prepared plan. Do not delete its claim to force a retry. A failed/uncertain billed request needs a reviewed continuation rather than rerunning the entire batch. Token usage and latency are recorded; unknown dollar cost stays null.

For a HumanEval batch stopped specifically by a Choice/probability mismatch,
use `make humaneval-jev-continuation-prepare`. The original
batch and its frozen files stay unchanged. It supports one explicitly approved
retry of the rejected request followed by the unattempted questions:

```sh
make humaneval-jev-continuation-prepare SOURCE_PLAN="$SOURCE_PLAN"
# Set PLAN_DIR to the printed continuation-plan directory.
make humaneval-jev-continuation-check PLAN_DIR="$PLAN_DIR"
# APPROVED_CALLS must equal additional_paid_calls in that checked plan.
make humaneval-jev-continuation-live PLAN_DIR="$PLAN_DIR" APPROVED_CALLS="$APPROVED_CALLS"
make humaneval-jev-continuation-evaluate PLAN_DIR="$PLAN_DIR" SCORE_RUN="$SCORE_RUN"
```

Executing the live command authorizes that exact additional paid batch. It keeps
the accepted prefix unchanged, resends no previously accepted question, retries
transient transport errors idempotently, and blocks duplicate launches across
plans for the same failed batch. The amended evaluation records the rejected request, total
attempts, token usage and latency separately from answer accuracy. It is not
reported as an uninterrupted no-retry run. Full-batch accuracy is unavailable
until the continuation is complete.

GSM8K has separate `gsm8k500-dry-run`, `gsm8k500-judge` and `gsm8k500-evaluate` targets. Use `RUN_DIR` for the verified archive and `SELECTIONS` for the saved Jev selections. Never reuse one task's scoring rule or judge prompt for another task.

## Repository and local files

- `src/jev_dllm/`: generators, public request adapters, selectors, private evaluation and operations.
- `Makefile`: the single command entry point for setup, generation, selection, scoring and recovery.
- `scripts/`: Python entry points called by the Makefile.
- `configs/`, `prompts/`, `requirements/`, `data/manifests/`: reproducible experiment settings and provenance.
- `tests/`: isolation, integrity, parser and workflow checks; mock requests incur no API charges.
- `docs/`, the research PDF, `reports/`, `artifacts/`, datasets and credentials: local only, ignored by Git.

Unused future interfaces and detailed module notes also stay local. Frozen profiles and scored implementations must retain their recorded hashes. A new experiment or scoring policy needs a separate version. Preserved comments in historical configurations are experiment metadata, not additional runnable project components. Core smoke tests do not establish model quality. Each completed analysis retains its scoring policy and protocol deviations; later research milestones need their own experiments.

This public snapshot includes experiment implementations and reproducibility inputs. Generated answers, evaluation results, research reports, model weights and credentials are excluded. Inspect tracked files with `git ls-files` and staged changes with `git diff --cached --stat` before publishing changes. Ignore rules do not remove files from earlier Git history.
