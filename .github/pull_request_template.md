## What changed

Describe the smallest behavior or documentation change.

## Why

Explain the user or agent problem this resolves.

## Verification

- [ ] `uv sync --frozen --group dev`
- [ ] `uv run pytest`
- [ ] `uv run ruff check .`
- [ ] `uv run ruff format --check .`
- [ ] `uv run isort --check-only fastmcp scripts`
- [ ] `uv run mypy`
- [ ] `uv run python scripts/check_source_safety.py`
- [ ] Docker smoke passes in `standalone` and `portal` modes
- [ ] Tool count, manifest, docs, and endpoint coverage remain synchronized
- [ ] No real credential, personal data, private path, ticket, agent memory, or runtime evidence is included

## Security boundary

State any change to authentication, BYOK headers, confirmation, retries, output bounds, scopes, or provider endpoints. Write `none` when unchanged.
