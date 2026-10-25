# Project decisions

## Jev API provider integration

- **Decision:** New Jev requests use the Jev AI System One API at
  `https://jev-ai.org/api/v1/systemone/`, with the `jev-1.13` model alias and
  the `JEV_API_KEY` environment variable.
- **Rationale:** The user has Jev AI access. Its current API contract differs
  from the TypeSafe endpoint previously configured in this repository. The
  adapter now follows Jev AI's documented request and response fields, uses a
  deterministic idempotency key, and retries only documented transient HTTP
  statuses with bounded backoff.
- **Protocol impact:** This changes the judge service integration, not the
  generator, candidate pool, or research comparison. Existing archived runs
  and frozen checksum records remain unchanged. A new protocol/implementation
  lock must be reviewed before using the updated integration for a frozen
  held-out run.
- **Validation status:** Client code and configuration were updated offline.
  No live request was made, so credentials, account balance, and live response
  compatibility have not been verified.
- **Source:** [Jev AI API documentation](https://jev-ai.org/jev-api/).
