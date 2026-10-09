# Configuration

`week01/protocol.json` is a readable development draft. Use null for unresolved values rather than guessed revisions. Ordinary validation lists missing fields; `--require-resolved` treats them as a failure. A configuration marked `frozen=true` is rejected if required fields remain unresolved.

Week 1 freezes generation/scoring settings and relevant service metadata. The `future` section and `jev_partial` prompt belong to later research stages; completing Week 1 does not freeze them. Add later-week configurations as needed and connect them to real outputs through content hashes and run IDs.
