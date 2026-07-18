# Security policy

## Supported versions

Security fixes target the latest release and the current `main` branch.
Older releases may not receive backports.

## Report privately

Use **Security → Report a vulnerability** in this GitHub repository. Include:

- affected version or commit
- access mode: `standalone` or `portal`
- a minimal reproduction using synthetic credentials and synthetic resources
- expected and observed behavior
- practical impact and any suggested mitigation

If private vulnerability reporting is unavailable, open a minimal issue asking
the maintainers to provide a private contact channel. Do not include the
vulnerability, exploit, credential, Google resource identifier, response body,
or sensitive log material in that issue.

High-value reports include:

- service-authentication or mode-boundary bypass
- Google OAuth, Maps key, or Portal grant exposure
- cross-request credential or client-cache confusion
- an outbound host escape through `google_raw_request`
- a write or destructive tool incorrectly marked read-only
- confirmation or dry-run bypass
- Gmail send-signature binding, preflight, or fingerprint disclosure failure
- unbounded provider output, unsafe logging, or PII disclosure

## Safe research

- Use accounts, Cloud projects, and resources you own or are authorized to
  assess.
- Use synthetic data and the smallest practical request volume.
- Do not send email, invite attendees, alter sharing, incur billable usage, or
  delete provider data without explicit authorization.
- Do not test a public deployment or another user's credentials.
- Repository tests must remain provider-free.

If a real credential is exposed, revoke or rotate it through its issuer, then
report the source and affected version privately. Removing a file or repository
does not revoke a credential.

## Disclosure

Please allow maintainers time to reproduce, fix, verify, and release a
correction before public disclosure. Coordination depends on severity and fix
availability; no fixed response-time SLA is promised.
