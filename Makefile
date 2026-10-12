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
export GPU_ID

.DEFAULT_GOAL := help
.PHONY: gsm8k500-check gpu-gsm8k500 gsm8k500-dry-run gsm8k500-judge gsm8k500-evaluate
.PHONY: test-fast-runtime scoring-check
.PHONY: fetch-fast-inputs gpu-lock-check gpu-fast-check gpu-fast-diversity gpu-install-locked compare-backends
.PHONY: git-audit smoke-keep smoke-clean processbench-dry-run processbench-live processbench-evaluate help setup doctor check test smoke week1-check fetch-inputs data-fetch data-prepare data-check week1-data check-splits storage gpu-disk gpu-setup gpu-install gpu-check gpu-status gpu-fetch-model gpu-pilot gpu-diversity judge-dry-run judge-live evaluate remote-plan remote-check remote-upload results-pull results-watch results-verify results-clean

help:
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
	@echo 'Set SERVER_CONFIG to your ignored configs/server/local.json; see docs/server-workflow.md.'

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
	@echo 'Tracked files under artifacts/ and reports/ (only their README.md files should appear):'
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
