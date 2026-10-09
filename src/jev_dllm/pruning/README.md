# Trajectory Pruning: Weeks 5–8

First evaluate partial-draft predictiveness and correct-candidate survival offline, then decide whether online pruning is justified. Initially consider M=8, k=2, and one checkpoint after block 2, 4, or 6; select the checkpoint using development results.

The real controller requires pause/resume, independent RNG, token/mask/cache state, budget handling, and failure policies. A no-pruning control must match uninterrupted generation. Account for cache rebuilding and extra draft-preparation computation. Existing schemas and payload builders do not implement these mechanisms.
