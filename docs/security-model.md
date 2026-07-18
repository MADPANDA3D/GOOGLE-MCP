# Google MCP Security Model

This document defines the v1.0.0 trust boundary. It is an operating model, not
a guarantee that a deployment is secure in every environment.

## Trust boundaries

```text
MCP client
  |  service credential + request-scoped Google credentials
  v
trusted TLS / private proxy
  |  bounded HTTP request
  v
Google MCP
  |  OAuth access token or request-scoped Maps key
  v
allowlisted Google APIs
```

Three controls are independent:

1. **Network trust** determines who can reach the service.
2. **Service authentication** determines who can invoke MCP.
3. **Google authorization** determines which provider data that request may
   access or change.

Passing one boundary does not grant the others. OAuth scopes also do not
replace tool confirmation, dry-run, or provider-side permission checks.

## Startup modes

`MCP_MODE` must be exactly `standalone` or `portal`. Startup fails closed when
the selected mode has no configured service credential.

| Mode | Required service proof | Intended caller |
|---|---|---|
| `standalone` | `Authorization: Bearer <MCP_ACCESS_TOKEN>` | An independent MCP client or trusted local gateway |
| `portal` | Configurable grant header, default `X-MADPANDA-PORTAL-GRANT`, matching `MCP_PORTAL_GRANT_TOKEN` | A trusted MAD MCP Portal broker |

There is no public, anonymous, or request-switchable mode. A restart is
required to change modes.

The unauthenticated `GET /health` endpoint exposes only presence booleans and
release/catalog metadata. It must never return a credential value. `/mcp`
requires the selected mode's service proof.

## Request-scoped Google BYOK

Provider tools require an all-or-nothing OAuth tuple on the active MCP request:

- `X-Google-Client-Id`
- `X-Google-Client-Secret`
- `X-Google-Refresh-Token`

The header names are configurable, but clients and the server must change them
together. A partial tuple is rejected. Strict deployment defaults disable
server token-file and ambient credential fallback.

The runtime does not persist request credentials. Its optional client cache is
memory-only, bounded by size and TTL, and keyed by a one-way credential
fingerprint. The public Compose profile disables that cache. A process restart
clears it.

Google access tokens obtained with the tuple live in memory and inherit the
provider library's token lifetime. Operators must assume process memory is
sensitive and prevent core dumps, debug snapshots, and unsanitized tracing.

Maps tools additionally require `X-Google-Maps-Api-Key`. They require explicit
confirmation, do not automatically retry, and can consume billable quota even
though they perform reads.

## Request and response controls

The production-oriented defaults provide:

- explicit Host allowlisting
- browser-Origin rejection unless deliberately configured
- bounded request bodies, directly streamed provider HTTP bodies, tool output,
  and download sizes
- bounded list-shaped batch inputs and estimated provider-call fan-out per tool
- disabled request logging by default
- one non-root worker in a read-only, capability-dropped container
- no server-side provider credential in the image or Compose environment

These controls supplement, but do not replace, TLS, a trusted reverse proxy,
firewall policy, connection limits, rate limits, and host hardening.

`MCP_PROVIDER_RESPONSE_MAX_BYTES` applies before decoding to the direct streamed
HTTP paths, including raw reads, Maps requests, resumable-upload setup, and
Gmail attachment metadata/content. Metadata-only attachment reads never fetch
the content body. Curated tools implemented through Google's discovery client
use endpoint field, page, item, batch, and provider-fan-out limits plus the final
MCP tool-output bound; that client may buffer its provider response before the
final output limit is applied.

`MCP_MAX_BATCH_ITEMS` caps list-shaped tool input and
`MCP_MAX_PROVIDER_CALLS_PER_TOOL` caps known fan-out workflows. Both are
fail-closed runtime controls and should be lowered—not raised—when deployment
resources or provider quota are tighter than the public defaults.

Provider responses may contain email, documents, contacts, calendar details,
business data, location data, or identifiers. Treat all tool output as
sensitive application data. Do not feed it into logs, issue templates, model
training, or unrelated tools without an explicit data-handling decision.

## Mutation safety

Tool annotations communicate intended effects to clients, but are not an
authorization system.

- Write and destructive tools remain constrained by Google OAuth scopes and
  the provider account's permissions.
- All 49 manifest-destructive or billable tools and six additional
  non-destructive write tools require explicit `confirm=true`; deployment
  configuration cannot weaken that gate.
- Cleanup and batch workflows use planning or `dry_run=true` defaults where
  supported.
- Permanent deletion, broad sharing, ACL changes, sends, and bulk mutations
  deserve a separate human approval in production workflows.
- Idempotency is tool-specific. Do not automatically retry a write,
  destructive call, send, or billable Maps request unless its exact descriptor
  and operation semantics permit it.

## Hidden raw read surface

`google_raw_request` is hidden from default discovery and exists only for
advanced read coverage. In v1.0.0 it:

- permits only `GET` and `HEAD`
- requires HTTPS on the standard port and disables redirects
- rejects JSON request bodies and caller-supplied HTTP headers
- rejects credential-bearing query keys in both the URL and `params`
- restricts destinations to approved Google API hosts
- does not automatically retry the provider request
- bounds the provider response and does not expose provider response headers

The raw tool is not a substitute for a curated wrapper: endpoint schemas,
scopes, quota, billing, pagination, and response sensitivity remain the
caller's responsibility.

## Gmail send-signature binding

Gmail send, create-draft, update-draft, and send-draft paths bind the selected
configured send-as signature exactly once. Raw and draft send paths fail closed
when the active signature is absent, and a preflight fingerprint change aborts
before provider delivery.

The hidden `gmail_signature_preflight` helper exists for Portal binding. It
returns only the selected send-as alias plus SHA-256 fingerprints for the
signature and message. It never returns signature HTML. The fingerprints still
identify request state and should be kept in authorized workflow metadata, not
public logs.

## Credential and scope practices

- Create OAuth credentials in a Google Cloud project controlled by the
  deployer.
- Enable only required APIs and request the narrowest useful scopes.
- Use separate credentials and test accounts for development and production.
- Store service tokens and Google credentials in a secret manager or the MCP
  client's protected credential facility.
- Never put credentials in source, `.env.example`, Compose files, URLs,
  screenshots, support issues, or chat transcripts.
- Rotate a credential when there is evidence or a reasonable possibility that
  an unauthorized party obtained it. Deleting a file or making a repository
  private does not invalidate a credential already copied elsewhere.

## Operator verification order

Before a provider smoke test:

1. Confirm startup rejects an invalid or missing mode credential.
2. Confirm health contains no secret values.
3. Confirm unauthenticated and incorrectly authenticated `/mcp` calls fail.
4. Confirm the five local navigation tools work with service auth alone.
5. Confirm provider tools reject missing or partial BYOK tuples.
6. Confirm raw non-Google URLs, mutation methods, bodies, and sensitive query
   keys fail.
7. Confirm Maps calls reject missing confirmation and missing key.

Only then use a dedicated test account for the narrowest safe provider read.
Do not begin certification with a send, write, deletion, sharing change, or
billable Maps call.

## Reporting vulnerabilities

Follow [SECURITY.md](../SECURITY.md). Do not publish exploit details, tokens,
personal data, or deployment topology in a public issue.
