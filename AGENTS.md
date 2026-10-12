# Contributor Instructions

This repository implements completed-answer selection for a fixed diffusion
language model. LLaDA generates candidates; TypeSafe Jev selects an existing
candidate. README.md describes the supported experiment commands.

## Research and data boundaries

- Use English in code, documentation, prompts and CLI output.
- Keep reference answers, hidden tests and correctness labels out of generation,
  Jev requests and deployed selectors. Only private evaluation may read them.
- Compare selectors on identical candidate pools, preserving IDs, order and seeds.
- Retain failed/unparseable answers in evaluation denominators. Record failures
  and explicit retries; never silently replace candidates or responses.
- Use question-level paired statistics and report measured costs separately.
- Oracle selection is an evaluation diagnostic, never a deployed selector.

## Implementation and verification

- Keep all Make targets in the root Makefile; use task-prefixed names for recovery.
- New implementation locks must use new versioned records. Keep older records and
  archives unchanged, and use their original code revision to analyze or resume them.
- Keep reusable logic in src/jev_dllm and command entry points in scripts.
- The core uses the Python standard library. GPU, mathematical scoring and
  isolated code-execution runtimes have separate pinned dependencies.
- Preserve frozen file hashes. Change a scored policy through a separately
  versioned implementation and analysis, rather than modifying its evidence.
- Run make doctor check test smoke for framework changes. Add meaningful tests
  for scoring, response validation and isolation changes.
- Imports, tests and smoke checks must not call paid APIs or load model weights.
- Live API execution, model downloads and GPU jobs are explicit user actions.

## Public repository contents

Commit reusable code, README, required configs, prompts, dependency locks, public
source manifests and independent test fixtures. Keep generated answers, measured
results, reports, private references, credentials, environments and model weights
out of Git. Check both file contents and commit ancestry before publication.
