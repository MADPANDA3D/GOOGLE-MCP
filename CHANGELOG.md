# Changelog

All notable public releases are documented here.

## Unreleased

- Updated the deterministic runtime lock from `cryptography` 49.0.0 to 50.0.0.
- Added a frozen 151-tool compatibility projection and corrected the documented
  descriptor hash for the current source contract.
- No native tool identity, canonical name, or compatibility alias changed.

## 1.0.1 — 2026-07-18

- Made first-release bootstrapping compatible with a clean source repository
  that remains private until its package and release gates pass.
- Added SHA-256 checksums for the wheel and source archive to every GitHub
  Release.
- Kept BuildKit SBOM and OCI provenance generation mandatory for every
  published image while enabling GitHub-hosted attestations only when the
  repository is public and that platform feature is available.
- Removed hard-coded package-version assertions from CI so the built wheel is
  always checked against the current `pyproject.toml` version.
- Standardized the GitHub Release title on the `MADPANDA3D` company name.
- Preserved `v1.0.0` as an immutable unsuccessful pre-publication candidate;
  it failed before any container image or GitHub Release was published.
- No MCP tool, risk, tier, authentication, or provider behavior changed.

## 1.0.0 — 2026-07-18

- Established a clean public source root for the Python 3.12/3.13 FastMCP
  server.
- Published the exact 151-tool contract: 144 agent-ready, five legacy, and two
  hidden tools under catalog `google-2026.07.18.2`.
- Added five standard local navigation tools for configuration, discovery,
  endpoint coverage, and exact tool-reference lookup.
- Added startup-selected `standalone` and `portal` access modes with no
  unauthenticated mode.
- Kept Google OAuth credentials request-scoped through three BYOK headers in
  both access modes.
- Added fingerprint-bound Gmail send-signature enforcement and a hidden,
  fingerprint-only preflight helper that never returns signature HTML.
- Fixed the advanced raw surface to bounded, non-retried GET/HEAD-only reads
  and enforced confirmation for all 44 state-destructive tools, five billable
  Maps reads, and six additional confirmation-bearing writes.
- Added bounded batch and provider-fan-out controls plus streamed, metadata-first
  Gmail attachment retrieval.
- Documented the 13 provider families, risk tiers, endpoint coverage,
  deployment, Portal integration, security model, provenance, support,
  contribution process, and trademark boundary.
- Added hardened container and provider-free verification guidance without
  publishing production topology, credentials, or pre-release image claims.
