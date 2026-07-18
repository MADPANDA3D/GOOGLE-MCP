# Support

## Public support

Use GitHub Issues for reproducible bugs, public-safe documentation corrections,
and focused feature proposals. Before opening an issue, search existing issues
and review:

- [README](README.md)
- [tool catalog](docs/tool-catalog.md)
- [endpoint coverage](docs/endpoint-coverage.md)
- [operator runbook](docs/operator-runbook.md)
- [security model](docs/security-model.md)

Include:

- release version or full source commit
- Python and Docker/Compose versions when relevant
- access mode: `standalone` or `portal`
- the tool name and a minimal synthetic input shape
- safe health fields and the exact sanitized error type
- the smallest provider-free reproduction available

## Never post

Do not include:

- OAuth client secrets, refresh tokens, access tokens, Maps keys, service
  tokens, Portal grants, or resolved request headers
- `.env`, OAuth JSON files, screenshots of secret stores, or copied CI
  environments
- email addresses, message content, calendar details, Drive or Workspace IDs,
  account/property/location identifiers, or raw provider responses
- production domains, IP addresses, filesystem paths, proxy configuration,
  internal tickets, agent memory, or private logs

Replace sensitive material with explicit placeholders such as
`<GOOGLE_REFRESH_TOKEN>` or `<RESOURCE_ID>`.

## Provider and account support

This repository cannot resolve Google Cloud billing, API enablement, OAuth
consent verification, account suspension, quota increases, Workspace
administration, or product-specific policy questions. Use the corresponding
official Google support channel for those matters.

## Security

Report vulnerabilities privately through [SECURITY.md](SECURITY.md), not a
public issue.

Support is best-effort and has no guaranteed response or resolution SLA.
