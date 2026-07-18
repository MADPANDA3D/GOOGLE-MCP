# Contributing

Thanks for improving MADPANDA3D Google MCP.

## Before coding

- Search existing issues before opening a new one.
- Discuss behavior-changing or large-surface work before implementation.
- Keep the exact 151-tool contract stable unless the proposal updates the
  ToolManifest, endpoint ledger, public documentation, and tests together.
- Prefer a curated tool over expanding the hidden raw-request surface.
- Never add real credentials, Google resource IDs, email addresses, production
  domains, host paths, tickets, agent memory, operator evidence, or private
  topology.

## Development

Python 3.12 or 3.13 and `uv` are required.

```sh
uv sync --frozen --group dev
uv run python -m compileall -q fastmcp scripts
uv run pytest -q -p no:cacheprovider
uv run ruff check .
uv run ruff format --check .
uv run isort --check-only fastmcp scripts
uv run mypy
uv run python scripts/check_source_safety.py
```

Tests must use synthetic credentials and mocked provider behavior. They must not
contact Google APIs, send email, create calendar events, mutate files, consume
Maps quota, or depend on a private deployment.

Use semantic commits:

```text
fix(auth): reject partial Google BYOK headers
docs(deploy): clarify standalone reverse proxy
test(manifest): preserve exact tier counts
```

## Tool-contract changes

Any tool addition, removal, rename, annotation change, or schema change must
update:

1. `fastmcp/tool_manifest.py`
2. the relevant implementation and provider-free tests
3. [the tool catalog](docs/tool-catalog.md)
4. [the endpoint ledger](docs/endpoint-coverage.md)
5. README counts and compatibility notes

Do not describe the provider as fully covered unless every stable official
operation has been re-reviewed and accounted for.

## Pull requests

A pull request should include:

- what changed and why
- the affected tool, API family, and risk level
- exact verification commands and results
- any configuration or documentation change
- confirmation that no live provider call or credential was used

Keep changes narrow. Preserve compatibility deliberately rather than carrying
unexplained aliases.

## Security reports

Do not open a public issue for a suspected vulnerability or exposed
credential. Follow [SECURITY.md](SECURITY.md).

By contributing, you agree that your contribution is licensed under the
[MIT License](LICENSE).
