# Upstream Implementations

The prepared official sampler is `llada/generate.py`, from `ML-GSAI/LLaDA` commit `9182493720ed723ef8031210d85959364e51cbe0`, SHA256 `d4d2d3c015511e63fe9237cd3197346148fb0c795b6c89e0b5ef4846be25252e`. This source is ignored by Git and must be restored on another machine using the runbook. The adapter verifies its hash before importing it.

Model weights are not stored in this directory. Fast-dLLM v1 is the alternate backend required by the full Timeline. Do not substitute v2 or a different model without recording the scope change.

## Fast-dLLM v1 source

`make fetch-fast-inputs` downloads the exact source bundle specified by
`configs/week01/fast-source.json` under ignored `third_party/fast_dllm_v1/`.
The Apache-2.0 license is included. Each source file is hash-checked before
upload and import. No weights are downloaded by this target.

The pinned upstream revision uses `llada/`, before the repository moved v1 to
`v1/llada/` and added compiler changes. The prefix-cache adapter loads the same
pinned LLaDA checkpoint as the official backend. Its real server compatibility
and performance still need validation.
