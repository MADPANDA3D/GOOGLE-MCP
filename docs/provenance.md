# Source and Release Provenance

This project uses separate identities for source, package, runtime contract,
and container artifacts. None is a substitute for the others.

## Identity chain

| Layer | Identity | Authority |
|---|---|---|
| Source | Full Git commit SHA | Exact release tag in the public repository |
| Source archive | SHA-256 of `git archive --format=tar <COMMIT>` | Release build record and runtime `source_fingerprint` |
| Python package | Version plus wheel/source-archive digests | Artifacts attached to the matching GitHub Release |
| MCP contract | Catalog version plus deterministic descriptor hash | Provider-owned ToolManifest from the exact build |
| Container | OCI registry name plus immutable `sha256` digest | Matching GitHub Release after the image is published |
| Runtime | Build SHA, source fingerprint, image reference, catalog, descriptor hash | Presence-only `/health` and local capability output |

A mutable branch, package name, image tag, or `latest` reference is not an
immutable release identity.

## Public source boundary

The publishable repository contains source, tests, generic deployment
templates, documentation, and community policy. It must not contain:

- OAuth credentials, access tokens, Maps keys, or service grants
- real `.env` files, token files, credential JSON, or secret-store exports
- production hostnames, IP addresses, filesystem paths, proxy configuration,
  container snapshots, or tenant identifiers
- private tickets, agent transcripts, operator reports, live-test evidence, or
  provider response data

Examples use inert placeholders. CI runs a dedicated public-source boundary
check and scans the complete publishable Git history for secrets before tests
or release publication.

## Dependency provenance

- Python runtime dependencies are hash-locked in `requirements.lock`.
- The development environment is locked in `uv.lock`.
- The container base is pinned by image digest in `Dockerfile`.
- GitHub Actions and scanning actions are pinned to exact commits.
- CI compiles source, runs provider-free tests, lints correctness failures,
  checks the public-source boundary, scans source, audits locked dependencies,
  builds package artifacts, validates Compose, scans the image, and runs both
  access modes without provider calls.

Review lockfile and base-image changes as supply-chain changes. Regenerate them
deliberately and require the full verification pipeline.

## Contract provenance

Tool descriptors are generated from the registered native tools and
`fastmcp/tool_manifest.py`. The descriptor hash covers the ordered public
contract, including names, schemas, annotations, categories, tiers, aliases,
and documentation metadata.

Release review must reconcile:

- registered native count
- `agent_ready`, `legacy`, and `hidden` tier counts
- category and risk counts
- catalog version and descriptor hash
- [complete tool catalog](tool-catalog.md)
- [endpoint coverage ledger](endpoint-coverage.md)
- README and changelog

A descriptor-hash change is a contract change even when the package version or
HTTP route did not change.

## Release procedure

The release workflow is tag-driven and should:

1. Require an exact stable `vMAJOR.MINOR.PATCH` tag whose target and package
   version agree.
2. Re-run source, test, lint, security, dependency, package, Compose, image, and
   provider-free runtime gates from that tag.
3. Produce wheel and source-archive artifacts and attest them.
4. Compute the exact-source fingerprint.
5. Build and scan the image from that same source.
6. Smoke both standalone and Portal modes without contacting Google.
7. Publish the image with SBOM and build provenance.
8. Scan and attest the immutable published digest.
9. Log out of GHCR and require an anonymous manifest read and pull of that
   exact digest.
10. Smoke the anonymously pulled digest in both access modes.
11. Record that digest and package artifacts in the matching GitHub Release.

The first container release has one owner-only bootstrap prerequisite. After
GHCR creates and links the `google-mcp-server` package, a package owner must set
its visibility to **Public** in Package Settings. GitHub treats public package
visibility as irreversible. The workflow does not claim to automate that owner
decision: it stops before the GitHub Release unless the exact digest is already
anonymously pullable, and the same tagged workflow must be rerun after the
visibility change.

No container digest is asserted in this documentation before those publication
steps produce it. Operators must obtain the digest from the actual release,
not copy a placeholder or infer it from a tag.

## Local verification

```sh
git rev-parse HEAD
git archive --format=tar HEAD | sha256sum
uv sync --frozen --group dev
uv run pytest -q -p no:cacheprovider
uv run ruff check .
uv run ruff format --check .
uv run isort --check-only fastmcp scripts
uv run mypy
uv run python scripts/check_source_safety.py
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.portal.yml config --quiet
```

Package and image attestations are release artifacts; verify them against the
exact subject digest using the tooling documented by the release platform.

## Upstream and trademark statement

The implementation uses public Google APIs and official Google client
libraries. Endpoint behavior, OAuth requirements, quotas, pricing, and terms
are controlled by Google and may change independently. The official references
used for the coverage ledger are linked in
[endpoint-coverage.md](endpoint-coverage.md).

Google and related product names are trademarks of their respective owners.
MADPANDA3D Google MCP is an independent open-source integration and is not an
official Google product, distribution, partnership, sponsorship, or
endorsement.
