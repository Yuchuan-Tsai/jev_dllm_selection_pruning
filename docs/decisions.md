# Decision Log

## 2026-10-08: Initialization

- **Project root:** the actual directory is `Documents/Jev + dLLM(Tong Zhang)`. The session's original `Tong_Zhang's task` path no longer exists. The scaffold was added beside the unique matching Timeline, preserving the original PDF.
- **Project form:** a Python research project with a standard-library-only core. Local Mac validation is separate from the GPU inference environment.
- **Source authority:** the original PDF defines research scope. Upstream documentation identifies integration points; newer models/versions do not override the plan.
- **Backend selection:** pending development pilot results; Fast-dLLM is not presumed superior.
- **Jev integration:** awaiting service documentation; API details and model versions are not guessed.
- **Protocol status:** draft. The planned starting point is 256/32/BF16; unresolved values remain null.
- **Engineering defaults:** microbatch=1, request concurrency=1, master_seed=20261008, timeout=60 seconds. Record and validate these against the development environment.
- **Implementation boundaries:** real generation, Jev, full MATH500 equivalence, the HumanEval worker, ProcessBench, selectors, and pruning remain pending.
- **Initial scope:** infrastructure, collaboration documentation, and local verification. No remote publication or model experiment was performed.

## Language convention

The user specified that this is an English-led project. All project-authored text uses English, including documentation, comments, prompts, reports, and project summaries. This supersedes the initial Chinese delivery convention in AGENTS.md. The language update preserves experiment settings and implementation status.

For future protocol changes, record the date, old/new values, development evidence, affected runs, and whether a new freeze is required. Preserve the history of failed experiments.
