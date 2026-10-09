# Local Setup Verification Record

Verification time: 2026-10-08T22:37:56-05:00 (America/Chicago).

- Actual root: `/Users/caiyuchuan/Documents/Jev + dLLM(Tong Zhang)`.
- The original PDF's SHA-256 is unchanged.
- A separate `.venv` was created; core code requires no third-party dependencies.
- Local Git was initialized, with no commits or remote configured at setup.
- `make doctor`: passed; real generator/Jev adapters reported pending.
- `make check`: structural validation passed; version/environment fields remained unresolved and frozen=false.
- `make test`: all 24 tests passed.
- `make smoke`: passed using two synthetic questions and eight prewritten candidates per question; not a benchmark.
- `check-config --require-resolved`: returned 1 as expected, preventing an unresolved protocol from being treated as ready.
- Smoke run: `artifacts/runs/smoke-20261009T033756Z-0341ac1c`.
- Labels were stored in `evaluation_private/` and excluded from judge payloads. No remote service was called.

These checks establish that the local scaffold runs. They do not establish integration of LLaDA inference, Jev, full mathematical equivalence, HumanEval, or ProcessBench. Real GPU experiments have not run; Week 1 research deliverables remain outstanding.

This document translates the original setup record; its timestamp and results refer to that setup verification.
