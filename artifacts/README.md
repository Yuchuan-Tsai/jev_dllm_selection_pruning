# Run Artifacts

Create a new `runs/<run_id>/` for every run; never overwrite existing outputs. `checkpoints/` stores only local state needed for continuation. Both are excluded from Git by default.

Smoke outputs are explicitly marked synthetic and must not enter benchmark result tables. Future real runs must preserve traceable configuration, dataset, model, prompt, and environment metadata. Use null for unmeasured costs.
