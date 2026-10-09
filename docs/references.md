# References

## Original project scope

`Jev × dLLM 10-Week Timeline.pdf` in the project root contains 16 pages. Week 1 is on pages 4–6; the full weekly plan continues through page 14.

SHA-256: `4046ae4460ba7633bcaa926edf27c986011af038004f11a2dba62142aeae9a49`.

## Official integration references checked during setup

Checked on 2026-10-08 (America/Chicago). These URLs are reference entry points, **not pinned experimental revisions**.

- [Official LLaDA implementation](https://github.com/ML-GSAI/LLaDA): use the specified LLaDA-8B-Instruct model; pin the commit containing `generate.py` and the model snapshot before experiments.
- [Official Fast-dLLM v1](https://github.com/NVlabs/Fast-dLLM/blob/main/v1/README.md): the candidate backend is in `v1/llada`; do not substitute v2.
- [Official ProcessBench documentation](https://github.com/QwenLM/ProcessBench/blob/main/README.md): diagnostic input/label structure and evaluation/prompt implementations in `code/`.
- [Official HumanEval implementation](https://github.com/openai/human-eval): source for the future isolated code-evaluation worker.

Published reference scores, third-party summaries, and other models' results are not reported as this project's results. Official Jev service documentation has not yet been provided.
