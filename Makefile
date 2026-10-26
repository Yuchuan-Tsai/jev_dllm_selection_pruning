PYTHON ?= .venv/bin/python
GPU_PYTHON ?= .venv-gpu/bin/python
RUNTIME_TEST_PYTHON ?= $(GPU_PYTHON)
SERVER_PYTHON ?= python3.11
CONFIG ?= configs/week01/pilot.json
QUESTIONS ?= data/processed/gsm8k-dev.json
REFERENCES ?= data/private/gsm8k-dev-references.json
SERVER_CONFIG ?= configs/server/local.json
RUN_DIR ?=
RUN_ID ?=
SELECTIONS ?=
SCORER ?= gsm8k_strict_v1
OFFICIAL_RUN_DIR ?=
FAST_RUN_DIR ?=
ENV_FILE ?= .env
DIAGNOSTIC_RUN_DIR ?=
GPU_ID ?=
N ?= 8
QUESTION_COUNT ?= 1164
RUNS ?= 10
RECOVERY_PLAN ?=
PLAN ?=
APPROVED_CALLS ?=
STRICT_RUN ?=
PLAN_DIR ?=
SCORE_RUN ?=
SOURCE_PLAN ?=
MATH_PYTHON ?= .venv-math/bin/python
MATH_BOOTSTRAP_PYTHON ?= python3.12
JEV_PLAN ?=
VALIDATION_RUN ?=
SCORING_PLAN ?=
export GPU_ID

.DEFAULT_GOAL := help
.PHONY: gsm8k500-check gpu-gsm8k500 gsm8k500-dry-run gsm8k500-judge gsm8k500-evaluate
.PHONY: test-fast-runtime scoring-check
.PHONY: fetch-fast-inputs gpu-lock-check gpu-fast-check gpu-fast-diversity gpu-install-locked compare-backends
.PHONY: git-audit smoke-keep smoke-clean processbench-dry-run processbench-live processbench-evaluate help setup doctor check test smoke week1-check fetch-inputs data-fetch data-prepare data-check week1-data check-splits storage gpu-disk gpu-setup gpu-install gpu-check gpu-status gpu-fetch-model gpu-pilot gpu-diversity judge-dry-run judge-live evaluate remote-plan remote-check remote-upload results-pull results-watch results-verify results-clean

help:
	@echo 'Single command entry point: make TARGET [VARIABLE=value]; see README.md.'
	@echo 'More commands: benchmarks-help | humaneval-help | math500-help | math500-scoring-help'
	@echo 'Generation recovery: gsm8k500-recovery-help | collection-recovery-help'
	@echo 'Jev recovery: jev-continuation-{check,live}; humaneval-jev-continuation-{prepare,check,live,evaluate}'
	@echo 'MATH500 Jev recovery: math500-jev-continuation-{prepare,check,live,audit}; math500-jev-resume-{prepare,check,live,audit}'
	@echo 'GSM8K-500: gsm8k500-check (Mac); remote-upload; gpu-gsm8k500 GPU_ID=... (server)'
	@echo 'Verified 500-question archive: gsm8k500-dry-run | gsm8k500-judge (500 paid calls) | gsm8k500-evaluate RUN_DIR=...'
	@echo 'Week 1 backend comparison: fetch-fast-inputs (Mac), gpu-fast-check, gpu-fast-diversity GPU_ID=... (server)'
	@echo 'Existing environment: gpu-lock-check; new Linux environments only: gpu-install-locked'
	@echo 'Git (read-only): git-audit; result files/reports remain local, and old commits require a separate history review.'
	@echo 'Mac: setup | week1-check (offline checks only) | week1-data | data-check | storage'
	@echo 'Mac scoring policy: scoring-check; evaluate SCORER=gsm8k_units_v3 RUN_DIR=... SELECTIONS=...'
	@echo 'Synthetic smoke: smoke (auto-cleans on success) | smoke-keep | smoke-clean'
	@echo 'Mac Jev diagnostic: processbench-dry-run | processbench-live | processbench-evaluate DIAGNOSTIC_RUN_DIR=...'
	@echo 'Server (manual): gpu-setup | gpu-install | gpu-check GPU_ID=... | gpu-fetch-model GPU_ID=...'
	@echo 'Server (manual): gpu-pilot GPU_ID=... | gpu-diversity GPU_ID=... | gpu-status | gpu-disk'
	@echo 'Mac after transfer: judge-dry-run RUN_DIR=... | judge-live RUN_DIR=... | evaluate RUN_DIR=...'
	@echo 'Mac SSH: remote-plan | remote-check | remote-upload | results-pull | results-watch'
	@echo 'Mac terminal runs: results-verify RUN_ID=... | results-clean RUN_ID=... (deletes verified remote run)'
	@echo 'Week 2 development pools and Jev interfaces: week2-help'
	@echo 'Set SERVER_CONFIG to your ignored configs/server/local.json; see README.md.'

setup:
	python3 -m venv .venv

doctor:
	$(PYTHON) scripts/jev.py doctor

check:
	$(PYTHON) scripts/jev.py check-config --config configs/week01/protocol.json
	$(PYTHON) scripts/jev.py check-config --config configs/week01/pilot.json
	$(PYTHON) scripts/jev.py check-config --config configs/week01/diversity.json
	$(PYTHON) scripts/jev.py check-config --config configs/week01/fast-diversity.json
	$(PYTHON) scripts/jev.py check-config --config configs/week01/gsm8k500.json --require-resolved

test:
	$(PYTHON) -m unittest discover -s tests -v

scoring-check:
	$(PYTHON) -m unittest discover -s tests -p 'test_scoring_policy.py' -v
	$(PYTHON) -m unittest discover -s tests -p 'test_week1_backend.py' -v

smoke:
	$(PYTHON) scripts/jev.py smoke

smoke-keep:
	$(PYTHON) scripts/jev.py smoke --keep-artifacts

smoke-clean:
	$(PYTHON) scripts/jev.py clean-smokes

week1-check: doctor check test smoke

fetch-inputs:
	$(PYTHON) scripts/fetch_pilot_inputs.py

fetch-fast-inputs:
	$(PYTHON) scripts/fetch_fast_inputs.py

data-fetch:
	$(PYTHON) scripts/fetch_pilot_inputs.py
	$(PYTHON) scripts/ops.py data-fetch

data-prepare:
	$(PYTHON) scripts/ops.py data-prepare

data-check:
	$(PYTHON) scripts/ops.py data-check

week1-data:
	$(MAKE) data-fetch
	$(MAKE) data-prepare
	$(MAKE) data-check check-splits

check-splits:
	$(PYTHON) scripts/jev.py check-splits --dev data/processed/gsm8k-dev.json --heldout data/processed/gsm8k-heldout.json

storage:
	$(PYTHON) scripts/ops.py storage --candidates "$(N)" --question-count "$(QUESTION_COUNT)" --runs "$(RUNS)"

gpu-disk:
	@test -n "$$JEV_SHARED_HUB_CACHE" || (echo 'Set JEV_SHARED_HUB_CACHE to your personal model-cache path first'; exit 2)
	du -sh . "$$JEV_SHARED_HUB_CACHE"
	df -h . "$$JEV_SHARED_HUB_CACHE"
	@echo 'Project total includes .venv-gpu, temporary files, and outputs. Cache is separate only if outside the project.'

gpu-setup:
	@umask 077; $(SERVER_PYTHON) -m venv .venv-gpu

gpu-install:
	$(GPU_PYTHON) -c "import platform, sys; assert platform.system() == 'Linux' and platform.machine() == 'x86_64' and sys.version_info[:2] == (3, 11), 'GPU install profile requires Linux x86_64 and Python 3.11'"
	@umask 077; mkdir -p .cache/pip-tmp
	TMPDIR="$(CURDIR)/.cache/pip-tmp" $(GPU_PYTHON) -m pip install --no-cache-dir --only-binary=:all: 'torch==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
	TMPDIR="$(CURDIR)/.cache/pip-tmp" $(GPU_PYTHON) -m pip install --no-cache-dir --only-binary=:all: -r requirements/gpu.in
	$(GPU_PYTHON) -m pip check

gpu-install-locked:
	$(GPU_PYTHON) -c "import platform, sys; assert platform.system() == 'Linux' and platform.machine() == 'x86_64' and sys.version_info[:2] == (3, 11), 'GPU lock requires Linux x86_64 and Python 3.11'"
	@umask 077; mkdir -p .cache/pip-tmp
	TMPDIR="$(CURDIR)/.cache/pip-tmp" $(GPU_PYTHON) -m pip install --no-cache-dir --only-binary=:all: --no-deps 'torch==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
	TMPDIR="$(CURDIR)/.cache/pip-tmp" $(GPU_PYTHON) -m pip install --no-cache-dir --only-binary=:all: -r requirements/gpu-linux-py311.lock
	$(MAKE) gpu-lock-check

gpu-lock-check:
	$(GPU_PYTHON) scripts/gpu_runtime.py lock
	$(GPU_PYTHON) -m pip check

test-fast-runtime:
	PYTHONDONTWRITEBYTECODE=1 $(RUNTIME_TEST_PYTHON) tests/integration/test_fast_runtime.py -v

gpu-fast-check: gpu-lock-check
	$(GPU_PYTHON) scripts/gpu_runtime.py fast

gpu-fast-diversity: gpu-fast-check
	$(GPU_PYTHON) scripts/ops.py gpu-generate --config configs/week01/fast-diversity.json --questions "$(QUESTIONS)"

gpu-check:
	$(GPU_PYTHON) scripts/ops.py gpu-check

gpu-status:
	nvidia-smi

gpu-fetch-model:
	$(GPU_PYTHON) scripts/ops.py gpu-fetch-model --config "$(CONFIG)"

gpu-pilot:
	$(GPU_PYTHON) scripts/ops.py gpu-generate --config "$(CONFIG)" --questions "$(QUESTIONS)"

gpu-diversity:
	$(GPU_PYTHON) scripts/ops.py gpu-generate --config configs/week01/diversity.json --questions "$(QUESTIONS)"

judge-dry-run:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to a local generation run directory'; exit 2)
	$(PYTHON) scripts/jev.py judge-pool --config "$(CONFIG)" --questions "$(QUESTIONS)" --candidates "$(RUN_DIR)/candidates.jsonl"

judge-live:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to a local generation run directory'; exit 2)
	$(PYTHON) scripts/jev.py judge-pool --config "$(CONFIG)" --questions "$(QUESTIONS)" --candidates "$(RUN_DIR)/candidates.jsonl" --live --env-file "$(ENV_FILE)"

evaluate:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to a local generation run directory'; exit 2)
	$(PYTHON) scripts/jev.py evaluate-pool --config "$(CONFIG)" --questions "$(QUESTIONS)" --candidates "$(RUN_DIR)/candidates.jsonl" --references "$(REFERENCES)" --scorer "$(SCORER)" $(if $(SELECTIONS),--selections "$(SELECTIONS)",)

compare-backends:
	@test -n "$(OFFICIAL_RUN_DIR)" -a -n "$(FAST_RUN_DIR)" || (echo 'Set OFFICIAL_RUN_DIR and FAST_RUN_DIR to the verified local archives'; exit 2)
	$(PYTHON) scripts/compare_backends.py --official "$(OFFICIAL_RUN_DIR)" --fast "$(FAST_RUN_DIR)" --questions "$(QUESTIONS)" --references "$(REFERENCES)"

remote-plan remote-check remote-upload results-pull results-watch:
	$(PYTHON) scripts/ops.py $@ --server-config "$(SERVER_CONFIG)"

results-verify results-clean:
	@test -n "$(RUN_ID)" || (echo 'Set RUN_ID to one exact terminal run directory name'; exit 2)
	$(PYTHON) scripts/ops.py $@ --server-config "$(SERVER_CONFIG)" --run-id "$(RUN_ID)"

processbench-dry-run:
	$(PYTHON) scripts/processbench.py judge

processbench-live:
	$(PYTHON) scripts/processbench.py judge --live --env-file "$(ENV_FILE)"

processbench-evaluate:
	@test -n "$(DIAGNOSTIC_RUN_DIR)" || (echo 'Set DIAGNOSTIC_RUN_DIR to the completed diagnostic run'; exit 2)
	$(PYTHON) scripts/processbench.py evaluate --run-dir "$(DIAGNOSTIC_RUN_DIR)"

git-audit:
	@echo 'Tracked files under artifacts/ and reports/ (expected empty):'
	@git ls-files -- artifacts reports
	@echo 'Tracked files matching ignore rules (expected empty):'
	@git ls-files -ci --exclude-standard
	@echo 'Current branch and working-tree state:'
	@git status --short --branch
	@echo 'Local commits ahead of the cached upstream reference (this does not contact GitHub):'
	@git log --oneline '@{upstream}..HEAD'

# Dedicated targets deliberately ignore the development CONFIG/QUESTIONS defaults.
gsm8k500-check:
	$(PYTHON) scripts/jev.py gsm8k500-check

gpu-gsm8k500:
	$(GPU_PYTHON) scripts/jev.py gsm8k500-check
	$(MAKE) gpu-fast-check
	$(GPU_PYTHON) scripts/ops.py gpu-generate --config configs/week01/gsm8k500.json --questions data/processed/gsm8k-heldout.json

gsm8k500-dry-run:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to the verified GSM8K-500 archive'; exit 2)
	$(PYTHON) scripts/jev.py judge-pool --config configs/week01/gsm8k500.json --questions data/processed/gsm8k-heldout.json --candidates "$(RUN_DIR)/candidates.jsonl"

gsm8k500-judge:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to the verified GSM8K-500 archive'; exit 2)
	$(PYTHON) scripts/jev.py judge-pool --config configs/week01/gsm8k500.json --questions data/processed/gsm8k-heldout.json --candidates "$(RUN_DIR)/candidates.jsonl" --live --env-file "$(ENV_FILE)"

gsm8k500-evaluate:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to the verified GSM8K-500 archive'; exit 2)
	$(PYTHON) scripts/jev.py evaluate-pool --config configs/week01/gsm8k500.json --questions data/processed/gsm8k-heldout.json --candidates "$(RUN_DIR)/candidates.jsonl" --references data/private/gsm8k-heldout-references.json --scorer gsm8k_units_v3 $(if $(SELECTIONS),--selections "$(SELECTIONS)",)

# MATH500 and HumanEval generation
.PHONY: benchmarks-help benchmarks-check benchmarks-upload collection-audit start-math500 start-humaneval gpu-math500 gpu-humaneval

benchmarks-help:
	@echo 'Mac: make benchmarks-check benchmarks-upload'
	@echo 'Server: make start-math500 GPU_ID=2'
	@echo 'Server: make start-humaneval GPU_ID=4'
	@echo 'Foreground alternatives: gpu-math500 | gpu-humaneval GPU_ID=...'
	@echo 'After results-pull/results-verify: collection-audit RUN_DIR=...'
	@echo 'Generation only. No automatic scoring, paid requests, model downloads, or code execution.'

benchmarks-check:
	$(PYTHON) scripts/collect_benchmarks.py check --task math500
	$(PYTHON) scripts/collect_benchmarks.py check --task humaneval

benchmarks-upload:
	$(PYTHON) scripts/collect_benchmarks.py upload --server-config "$(SERVER_CONFIG)"

start-math500:
	$(GPU_PYTHON) scripts/collect_benchmarks.py start --task math500

start-humaneval:
	$(GPU_PYTHON) scripts/collect_benchmarks.py start --task humaneval

gpu-math500:
	$(GPU_PYTHON) scripts/collect_benchmarks.py check --task math500
	$(MAKE) gpu-fast-check
	$(GPU_PYTHON) scripts/collect_benchmarks.py run --task math500

gpu-humaneval:
	$(GPU_PYTHON) scripts/collect_benchmarks.py check --task humaneval
	$(MAKE) gpu-fast-check
	$(GPU_PYTHON) scripts/collect_benchmarks.py run --task humaneval

collection-audit:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to a copied, SHA256-verified collection run'; exit 2)
	$(PYTHON) scripts/collect_benchmarks.py audit --run-dir "$(RUN_DIR)"


# GSM8K generation recovery
.PHONY: gsm8k500-recovery-help gsm8k500-recovery-plan gsm8k500-recovery-audit gsm8k500-recovery-upload gpu-gsm8k500-recovery-check gpu-gsm8k500-resume

gsm8k500-recovery-help:
	@echo 'Mac: gsm8k500-recovery-plan | gsm8k500-recovery-audit RUN_DIR=artifacts/downloads/lab/ACTUAL_RUN_ID'
	@echo 'Mac: gsm8k500-recovery-upload RECOVERY_PLAN=artifacts/recovery-plans/ACTUAL_RUN_ID.json'
	@echo 'Server: gpu-gsm8k500-resume GPU_ID=... RECOVERY_PLAN=artifacts/recovery-plans/ACTUAL_RUN_ID.json'

gsm8k500-recovery-plan:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to the verified interrupted local archive'; exit 2)
	$(PYTHON) scripts/recovery.py plan --run-dir "$(RUN_DIR)"

gsm8k500-recovery-audit:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to a verified local archive'; exit 2)
	$(PYTHON) scripts/recovery.py audit --run-dir "$(RUN_DIR)"

gsm8k500-recovery-upload:
	@test -n "$(RECOVERY_PLAN)" || (echo 'Set RECOVERY_PLAN to the prepared plan file'; exit 2)
	$(PYTHON) scripts/recovery.py upload --plan "$(RECOVERY_PLAN)" --server-config "$(SERVER_CONFIG)"

gpu-gsm8k500-recovery-check:
	@test -n "$(RECOVERY_PLAN)" || (echo 'Set RECOVERY_PLAN to the uploaded plan file'; exit 2)
	$(GPU_PYTHON) scripts/recovery.py source-check --plan "$(RECOVERY_PLAN)"

gpu-gsm8k500-resume: gpu-gsm8k500-recovery-check
	$(MAKE) gpu-fast-check
	$(GPU_PYTHON) scripts/recovery.py run --plan "$(RECOVERY_PLAN)"


# MATH500 and HumanEval generation recovery
.PHONY: collection-recovery-help collection-recovery-plan collection-recovery-audit collection-recovery-upload gpu-collection-recovery-check gpu-collection-resume

collection-recovery-help:
	@echo 'Mac: collection-recovery-plan | collection-recovery-audit RUN_DIR=artifacts/downloads/lab/ACTUAL_RUN_ID'
	@echo 'Mac: collection-recovery-upload RECOVERY_PLAN=artifacts/collection-recovery-plans/ACTUAL_RUN_ID.json'
	@echo 'Server: gpu-collection-resume GPU_ID=... RECOVERY_PLAN=artifacts/collection-recovery-plans/ACTUAL_RUN_ID.json'

collection-recovery-plan:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to the verified interrupted local archive'; exit 2)
	$(PYTHON) scripts/recover_collection.py plan --run-dir "$(RUN_DIR)"

collection-recovery-audit:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to a verified local archive'; exit 2)
	$(PYTHON) scripts/recover_collection.py audit --run-dir "$(RUN_DIR)"

collection-recovery-upload:
	@test -n "$(RECOVERY_PLAN)" || (echo 'Set RECOVERY_PLAN to the prepared plan file'; exit 2)
	$(PYTHON) scripts/recover_collection.py upload --plan "$(RECOVERY_PLAN)" --server-config "$(SERVER_CONFIG)"

gpu-collection-recovery-check:
	@test -n "$(RECOVERY_PLAN)" || (echo 'Set RECOVERY_PLAN to the uploaded plan file'; exit 2)
	$(GPU_PYTHON) scripts/recover_collection.py source-check --plan "$(RECOVERY_PLAN)"

gpu-collection-resume: gpu-collection-recovery-check
	$(MAKE) gpu-fast-check
	$(GPU_PYTHON) scripts/recover_collection.py run --plan "$(RECOVERY_PLAN)"


# GSM8K Jev continuation
.PHONY: jev-continuation-check jev-continuation-live

jev-continuation-check:
	@test -n "$(PLAN)" || (echo 'Set PLAN to the prepared continuation directory'; exit 2)
	$(PYTHON) scripts/continue_jev.py --plan "$(PLAN)"

jev-continuation-live:
	@test -n "$(PLAN)" -a -n "$(APPROVED_CALLS)" || (echo 'Set PLAN and the explicitly approved additional call count'; exit 2)
	$(PYTHON) scripts/continue_jev.py --plan "$(PLAN)" --live --approved-calls "$(APPROVED_CALLS)" --env-file "$(ENV_FILE)"


# HumanEval scoring and Jev selection
.PHONY: humaneval-help humaneval-image humaneval-validate humaneval-score

humaneval-help:
	@echo 'Mac only: humaneval-image (explicit image download); humaneval-validate (independent container fixtures)'
	@echo 'humaneval-score RUN_DIR=... (verified HumanEval archive; no API calls)'
	@echo 'Extraction: humaneval-extraction-check | humaneval-extraction-validate | humaneval-reanalyze RUN_DIR=... STRICT_RUN=...'

humaneval-image:
	docker pull python@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce

humaneval-validate:
	$(PYTHON) scripts/score_humaneval.py --validate-only

humaneval-score:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to a complete, verified HumanEval archive'; exit 2)
	$(PYTHON) scripts/score_humaneval.py --run-dir "$(RUN_DIR)"

.PHONY: humaneval-extraction-check humaneval-extraction-validate humaneval-reanalyze

humaneval-extraction-check:
	$(PYTHON) -m unittest discover -s tests -p 'test_humaneval_extraction.py' -v

humaneval-extraction-validate:
	$(PYTHON) scripts/reanalyze_humaneval.py --validate-only

humaneval-reanalyze:
	@test -n "$(RUN_DIR)" -a -n "$(STRICT_RUN)" || (echo 'Set RUN_DIR and its original STRICT_RUN evaluation'; exit 2)
	$(PYTHON) scripts/reanalyze_humaneval.py --run-dir "$(RUN_DIR)" --strict-run "$(STRICT_RUN)"

.PHONY: humaneval-jev-prepare humaneval-jev-live humaneval-jev-evaluate
humaneval-jev-prepare:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to the verified HumanEval archive'; exit 2)
	$(PYTHON) scripts/judge_humaneval.py prepare --run-dir "$(RUN_DIR)"

humaneval-jev-live:
	@test -n "$(PLAN_DIR)" -a -n "$(APPROVED_CALLS)" || (echo 'Set PLAN_DIR and APPROVED_CALLS to the planned paid request count'; exit 2)
	$(PYTHON) scripts/judge_humaneval.py live --plan-dir "$(PLAN_DIR)" --approved-calls "$(APPROVED_CALLS)" --env-file "$(ENV_FILE)"

humaneval-jev-evaluate:
	@test -n "$(PLAN_DIR)" -a -n "$(SCORE_RUN)" || (echo 'Set PLAN_DIR and SCORE_RUN to its completed extraction evaluation'; exit 2)
	$(PYTHON) scripts/judge_humaneval.py evaluate --plan-dir "$(PLAN_DIR)" --score-run "$(SCORE_RUN)"


# HumanEval Jev continuation
.PHONY: humaneval-jev-continuation-prepare humaneval-jev-continuation-check humaneval-jev-continuation-live humaneval-jev-continuation-evaluate

humaneval-jev-continuation-prepare:
	@test -n "$(SOURCE_PLAN)" || (echo 'Set SOURCE_PLAN to the original failed HumanEval Jev plan'; exit 2)
	$(PYTHON) scripts/continue_humaneval_jev.py prepare --source-plan "$(SOURCE_PLAN)"

humaneval-jev-continuation-check:
	@test -n "$(PLAN_DIR)" || (echo 'Set PLAN_DIR to the prepared continuation'; exit 2)
	$(PYTHON) scripts/continue_humaneval_jev.py check --plan "$(PLAN_DIR)"

humaneval-jev-continuation-live:
	@test -n "$(PLAN_DIR)" -a -n "$(APPROVED_CALLS)" || (echo 'Set PLAN_DIR and the explicitly approved additional request count'; exit 2)
	$(PYTHON) scripts/continue_humaneval_jev.py live --plan "$(PLAN_DIR)" --approved-calls "$(APPROVED_CALLS)" --env-file "$(ENV_FILE)"

humaneval-jev-continuation-evaluate:
	@test -n "$(PLAN_DIR)" -a -n "$(SCORE_RUN)" || (echo 'Set PLAN_DIR and its original extracted-code SCORE_RUN'; exit 2)
	$(PYTHON) scripts/continue_humaneval_jev.py evaluate --plan "$(PLAN_DIR)" --score-run "$(SCORE_RUN)"


# MATH500 Jev selection
.PHONY: math500-help math500-jev-prepare math500-jev-check math500-jev-live math500-jev-audit

math500-help:
	@echo 'Mac: math500-jev-prepare RUN_DIR=... PLAN_DIR=artifacts/runs/UNIQUE_PLAN_NAME'
	@echo 'Mac: math500-jev-check PLAN_DIR=... (offline public request verification)'
	@echo 'Mac: math500-jev-live PLAN_DIR=... APPROVED_CALLS=500 (500 PAID requests)'
	@echo 'Mac: math500-jev-audit PLAN_DIR=... (selection integrity/usage; correctness scoring separate)'

math500-jev-prepare:
	@test -n "$(RUN_DIR)" -a -n "$(PLAN_DIR)" || (echo 'Set RUN_DIR and PLAN_DIR'; exit 2)
	$(PYTHON) scripts/judge_math500.py prepare --run-dir "$(RUN_DIR)" --plan-dir "$(PLAN_DIR)"

math500-jev-check:
	@test -n "$(PLAN_DIR)" || (echo 'Set PLAN_DIR to the prepared request directory'; exit 2)
	$(PYTHON) scripts/judge_math500.py check --plan-dir "$(PLAN_DIR)"

math500-jev-live:
	@test -n "$(PLAN_DIR)" -a -n "$(APPROVED_CALLS)" || (echo 'Set PLAN_DIR and APPROVED_CALLS to the planned paid request count'; exit 2)
	$(PYTHON) scripts/judge_math500.py live --plan-dir "$(PLAN_DIR)" --approved-calls "$(APPROVED_CALLS)" --env-file "$(ENV_FILE)"

math500-jev-audit:
	@test -n "$(PLAN_DIR)" || (echo 'Set PLAN_DIR to the completed request directory'; exit 2)
	$(PYTHON) scripts/judge_math500.py audit --plan-dir "$(PLAN_DIR)"


# MATH500 Jev continuation
.PHONY: math500-jev-continuation-prepare math500-jev-continuation-check math500-jev-continuation-live math500-jev-continuation-audit

math500-jev-continuation-prepare:
	@test -n "$(SOURCE_PLAN)" || (echo 'Set SOURCE_PLAN to the original failed MATH500 Jev plan'; exit 2)
	$(PYTHON) scripts/continue_math500_jev.py prepare --source-plan "$(SOURCE_PLAN)"

math500-jev-continuation-check:
	@test -n "$(PLAN_DIR)" || (echo 'Set PLAN_DIR to the prepared continuation'; exit 2)
	$(PYTHON) scripts/continue_math500_jev.py check --plan "$(PLAN_DIR)"

math500-jev-continuation-live:
	@test -n "$(PLAN_DIR)" -a -n "$(APPROVED_CALLS)" || (echo 'Set PLAN_DIR and the explicitly approved additional request count'; exit 2)
	$(PYTHON) scripts/continue_math500_jev.py live --plan "$(PLAN_DIR)" --approved-calls "$(APPROVED_CALLS)" --env-file "$(ENV_FILE)"

math500-jev-continuation-audit:
	@test -n "$(PLAN_DIR)" || (echo 'Set PLAN_DIR to the completed continuation'; exit 2)
	$(PYTHON) scripts/continue_math500_jev.py audit --plan "$(PLAN_DIR)"


# MATH500 Jev continuation-chain resume
.PHONY: math500-jev-resume-prepare math500-jev-resume-check math500-jev-resume-live math500-jev-resume-audit

math500-jev-resume-prepare:
	@test -n "$(SOURCE_PLAN)" || (echo 'Set SOURCE_PLAN to the most recent failed MATH500 Jev continuation'; exit 2)
	$(PYTHON) scripts/resume_math500_jev.py prepare --source-plan "$(SOURCE_PLAN)"

math500-jev-resume-check:
	@test -n "$(PLAN_DIR)" || (echo 'Set PLAN_DIR to the prepared continuation'; exit 2)
	$(PYTHON) scripts/resume_math500_jev.py check --plan "$(PLAN_DIR)"

math500-jev-resume-live:
	@test -n "$(PLAN_DIR)" -a -n "$(APPROVED_CALLS)" || (echo 'Set PLAN_DIR and the explicitly approved additional request count'; exit 2)
	$(PYTHON) scripts/resume_math500_jev.py live --plan "$(PLAN_DIR)" --approved-calls "$(APPROVED_CALLS)" --env-file "$(ENV_FILE)"

math500-jev-resume-audit:
	@test -n "$(PLAN_DIR)" || (echo 'Set PLAN_DIR to the completed continuation'; exit 2)
	$(PYTHON) scripts/resume_math500_jev.py audit --plan "$(PLAN_DIR)"


# MATH500 offline scoring
.PHONY: math500-scoring-help math500-scoring-setup math500-validate math500-scoring-prepare math500-scoring-check math500-evaluate

math500-scoring-help:
	@echo 'Mac offline: math500-scoring-setup; math500-validate'
	@echo 'Freeze: math500-scoring-prepare RUN_DIR=... JEV_PLAN=... VALIDATION_RUN=...'
	@echo 'Score: math500-scoring-check SCORING_PLAN=...; math500-evaluate SCORING_PLAN=...'

math500-scoring-setup:
	$(MATH_BOOTSTRAP_PYTHON) -m venv .venv-math
	$(MATH_PYTHON) -m pip install --require-hashes --only-binary=:all: -r requirements/math500-cpu.lock
	$(MATH_PYTHON) -m pip check

math500-validate:
	$(PYTHON) scripts/score_math500.py validate --math-python "$(MATH_PYTHON)"

math500-scoring-prepare:
	@test -n "$(RUN_DIR)" -a -n "$(JEV_PLAN)" -a -n "$(VALIDATION_RUN)" || (echo 'Set RUN_DIR, JEV_PLAN and VALIDATION_RUN'; exit 2)
	$(PYTHON) scripts/score_math500.py prepare --math-python "$(MATH_PYTHON)" --run-dir "$(RUN_DIR)" --jev-plan "$(JEV_PLAN)" --validation "$(VALIDATION_RUN)"

math500-scoring-check:
	@test -n "$(SCORING_PLAN)" || (echo 'Set SCORING_PLAN'; exit 2)
	$(PYTHON) scripts/score_math500.py check --math-python "$(MATH_PYTHON)" --plan "$(SCORING_PLAN)"

math500-evaluate:
	@test -n "$(SCORING_PLAN)" || (echo 'Set SCORING_PLAN'; exit 2)
	$(PYTHON) scripts/score_math500.py evaluate --math-python "$(MATH_PYTHON)" --plan "$(SCORING_PLAN)"

# Week 2 development pools (N=16) and Jev pointwise/listwise interfaces
TASK ?=
REFERENCE ?=
RESUME_FROM ?=
JEV_FIRST ?= 3
.PHONY: week2-help week2-data week2-check week2-upload test-week2-runtime gpu-week2-equivalence gpu-week2-pool start-week2-pool week2-labels week2-jev-prepare week2-jev-check week2-jev-live

week2-help:
	@echo 'Local: week2-data | week2-check | remote-upload then week2-upload'
	@echo 'Server: gpu-week2-equivalence GPU_ID=... REFERENCE=artifacts/runs/<Week 1 fast-llada-dev run>'
	@echo 'Server: start-week2-pool TASK=gsm8k|math|mbpp GPU_ID=... [RESUME_FROM=artifacts/runs/<failed run>]'
	@echo 'Server, CPU only: test-week2-runtime (recording leaves sampling unchanged)'
	@echo 'Local after results-pull/results-verify: week2-labels RUN_DIR=... (MATH needs math500-scoring-setup; MBPP needs Docker)'
	@echo 'Local Jev: week2-jev-prepare RUN_DIR=... [JEV_FIRST=3] | week2-jev-check PLAN_DIR=... | week2-jev-live PLAN_DIR=... APPROVED_CALLS=...'
	@echo 'See the Week 2 section of README.md.'

week2-data:
	$(PYTHON) scripts/week2.py data-fetch
	$(PYTHON) scripts/week2.py data-prepare
	$(PYTHON) scripts/week2.py data-check

week2-check:
	$(PYTHON) scripts/week2.py data-check
	$(PYTHON) scripts/week2.py pool-check --task gsm8k
	$(PYTHON) scripts/week2.py pool-check --task math
	$(PYTHON) scripts/week2.py pool-check --task mbpp

week2-upload:
	$(PYTHON) scripts/week2.py upload --server-config "$(SERVER_CONFIG)"

test-week2-runtime:
	PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= $(RUNTIME_TEST_PYTHON) tests/integration/test_week2_instrumentation.py -v

gpu-week2-equivalence:
	@test -n "$(REFERENCE)" || (echo 'Set REFERENCE to a complete Week 1 fast-llada-dev run directory'; exit 2)
	$(MAKE) gpu-lock-check
	$(GPU_PYTHON) scripts/week2.py equivalence --reference "$(REFERENCE)"

gpu-week2-pool:
	@test -n "$(TASK)" || (echo 'Set TASK to gsm8k, math or mbpp'; exit 2)
	$(GPU_PYTHON) scripts/week2.py pool-run --task "$(TASK)" $(if $(RESUME_FROM),--resume-from "$(RESUME_FROM)",)

start-week2-pool:
	@test -n "$(TASK)" || (echo 'Set TASK to gsm8k, math or mbpp'; exit 2)
	$(GPU_PYTHON) scripts/week2.py pool-start --task "$(TASK)" $(if $(RESUME_FROM),--resume-from "$(RESUME_FROM)",)

week2-labels:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to a verified local Week 2 pool copy'; exit 2)
	$(PYTHON) scripts/week2.py labels --run-dir "$(RUN_DIR)" --math-python "$(MATH_PYTHON)"

week2-jev-prepare:
	@test -n "$(RUN_DIR)" || (echo 'Set RUN_DIR to a verified local Week 2 pool copy'; exit 2)
	$(PYTHON) scripts/week2.py jev-prepare --run-dir "$(RUN_DIR)" --first "$(JEV_FIRST)"

week2-jev-check:
	@test -n "$(PLAN_DIR)" || (echo 'Set PLAN_DIR to a prepared Week 2 Jev plan'; exit 2)
	$(PYTHON) scripts/week2.py jev-check --plan-dir "$(PLAN_DIR)"

week2-jev-live:
	@test -n "$(PLAN_DIR)" -a -n "$(APPROVED_CALLS)" || (echo 'Set PLAN_DIR and the explicitly approved paid request count'; exit 2)
	$(PYTHON) scripts/week2.py jev-live --plan-dir "$(PLAN_DIR)" --approved-calls "$(APPROVED_CALLS)" --env-file "$(ENV_FILE)"
