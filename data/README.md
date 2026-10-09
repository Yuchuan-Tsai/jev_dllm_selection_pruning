# Data Conventions

- `raw/`: original downloads, stored locally.
- `processed/`: inference-visible questions and candidates; separate development and held-out data.
- `private/`: reference answers, hidden tests, and correctness labels; never send these to judges/selectors.
- `manifests/`: sources, revisions, IDs, splits, sampling rules, hashes, and actual coverage; suitable for version control after review.

Only synthetic examples in `tests/fixtures/` are included; benchmark datasets have not been downloaded. `check-splits` checks IDs and normalized text, not semantic near-duplicates. Planned held-out coverage is GSM8K 500, MATH500 500, and HumanEval 164. Report actual sample counts after any splitting.
