# Final-Answer Selection: Weeks 2–4

Planned methods: random selection, majority voting over comparable answers, dLLM confidence, Jev pointwise/listwise/weighted voting, and the specified Qwen judge/PRM baselines. Oracle is an offline diagnostic only.

Inputs are the same frozen candidate pool and label-free judge scores. Outputs include the selected candidate_id, ordering/tie-breaking information, and selection cost. Do not rewrite or generate candidates. Do not use code-string equality for code majority voting. Freeze the selector using development results in Week 4; every later pruning method uses that same selector.
