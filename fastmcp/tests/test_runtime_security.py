import asyncio
import json
import stat
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

TEST_PORTAL_GRANT_TOKEN = "test-portal-grant-0000000000000000"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import google_auth_local as gal
import google_mcp_server as gm


def _ok_app():
    async def _app(scope, receive, send):
        body = b'{"ok":true}'
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return gm.build_hosted_mcp_http_wrapper(_app)


def _portal_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-MADPANDA-PORTAL-GRANT": TEST_PORTAL_GRANT_TOKEN,
    }


def test_runtime_configuration_requires_mode_token_and_explicit_hosts(monkeypatch):
    monkeypatch.setattr(gm, "MCP_MODE", "standalone")
    monkeypatch.setattr(gm, "MCP_ACCESS_TOKEN", "")
    with pytest.raises(gm.RuntimeConfigurationError, match="MCP_ACCESS_TOKEN"):
        gm._validate_runtime_configuration()

    standalone_token = "s" * 32
    monkeypatch.setattr(gm, "MCP_ACCESS_TOKEN", standalone_token)
    monkeypatch.setattr(gm, "MCP_ALLOWED_HOSTS", ("localhost", "*"))
    with pytest.raises(gm.RuntimeConfigurationError, match="without wildcards"):
        gm._validate_runtime_configuration()

    monkeypatch.setattr(gm, "MCP_ALLOWED_HOSTS", ("localhost",))
    gm._validate_runtime_configuration()

    monkeypatch.setattr(gm, "MCP_MODE", "portal")
    monkeypatch.setattr(gm, "MCP_PORTAL_GRANT_TOKEN", "too-short")
    with pytest.raises(gm.RuntimeConfigurationError, match="at least 32"):
        gm._validate_runtime_configuration()
    monkeypatch.setattr(gm, "MCP_PORTAL_GRANT_TOKEN", TEST_PORTAL_GRANT_TOKEN)
    gm._validate_runtime_configuration()

    monkeypatch.setattr(gm, "MCP_SOURCE_FINGERPRINT", "not-a-digest")
    with pytest.raises(gm.RuntimeConfigurationError, match="64-character"):
        gm._validate_runtime_configuration()

    monkeypatch.setattr(gm, "MCP_SOURCE_FINGERPRINT", "a" * 64)
    monkeypatch.setattr(gm, "MCP_IMAGE_REFERENCE", "google-mcp:mutable")
    with pytest.raises(gm.RuntimeConfigurationError, match="immutable"):
        gm._validate_runtime_configuration()


def test_health_reports_validated_non_secret_release_provenance(monkeypatch):
    source_fingerprint = "a" * 64
    image_reference = f"ghcr.io/madpanda3d/google-mcp@sha256:{'b' * 64}"
    monkeypatch.setattr(gm, "MCP_BUILD_SHA", "c" * 40)
    monkeypatch.setattr(gm, "MCP_SOURCE_FINGERPRINT", source_fingerprint)
    monkeypatch.setattr(gm, "MCP_IMAGE_REFERENCE", image_reference)
    client = TestClient(_ok_app())
    try:
        response = client.get("/health")
    finally:
        client.close()
    assert response.status_code == 200
    payload = response.json()
    assert payload["source_fingerprint"] == source_fingerprint
    assert payload["image_reference"] == image_reference
    assert payload["configuration"]["provenance_valid"] is True
    meta = gm._server_meta()
    assert meta["source_fingerprint"] == source_fingerprint
    assert meta["image_reference"] == image_reference


def test_standalone_bearer_auth_runs_before_body_parsing(monkeypatch):
    monkeypatch.setattr(gm, "MCP_MODE", "standalone")
    standalone_token = "s" * 32
    monkeypatch.setattr(gm, "MCP_ACCESS_TOKEN", standalone_token)
    client = TestClient(_ok_app())
    try:
        missing = client.post(
            "/mcp",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            content=b"not-json",
        )
        assert missing.status_code == 401
        assert "Bearer" in missing.json()["error"]["message"]

        authorized = client.post(
            "/mcp",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {standalone_token}",
            },
            content=b"not-json",
        )
        assert authorized.status_code == 400
        assert authorized.json()["error"]["code"] == -32600
    finally:
        client.close()


def test_portal_auth_runs_before_request_size_enforcement(monkeypatch):
    monkeypatch.setattr(gm, "MCP_REQUEST_BODY_MAX_BYTES", 1024)
    oversized = b"{" + (b"x" * 2048) + b"}"
    client = TestClient(_ok_app())
    try:
        unauthenticated = client.post(
            "/mcp",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            content=oversized,
        )
        assert unauthenticated.status_code == 401

        authenticated = client.post(
            "/mcp",
            headers=_portal_headers(),
            content=oversized,
        )
        assert authenticated.status_code == 413
        assert authenticated.json()["error"]["code"] == -32003
    finally:
        client.close()


def test_untrusted_hosts_and_origins_are_rejected_by_default():
    untrusted_client = TestClient(_ok_app(), base_url="http://evil.example")
    try:
        response = untrusted_client.get("/health")
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_host"
    finally:
        untrusted_client.close()

    client = TestClient(_ok_app())
    try:
        response = client.get("/health", headers={"Origin": "https://evil.example"})
        assert response.status_code == 403
        assert response.json()["error"] == "origin_not_allowed"
    finally:
        client.close()


def test_local_discovery_and_default_health_do_not_resolve_provider_credentials(
    monkeypatch,
):
    def _must_not_resolve(_headers):
        raise AssertionError("provider credential resolution must not run")

    monkeypatch.setattr(gm, "_resolve_request_client", _must_not_resolve)
    payloads = (
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "mcp_health_check", "arguments": {}},
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "google_mcp_welcome", "arguments": {}},
        },
    )
    client = TestClient(_ok_app())
    try:
        for payload in payloads:
            response = client.post("/mcp", headers=_portal_headers(), json=payload)
            assert response.status_code == 200
    finally:
        client.close()


def test_default_health_is_privacy_safe_and_does_not_load_credentials(monkeypatch):
    class _LocalOnlyClient:
        def _load_credentials(self):
            raise AssertionError("default health must not load provider credentials")

        def is_service_cached(self, _name, _version):
            return False

        def is_session_cached(self):
            return False

    monkeypatch.setattr(gm, "client", _LocalOnlyClient())
    payload = json.loads(asyncio.run(gm.mcp_health_check()))
    assert payload["ok"] is True
    data = payload["data"]
    assert data["provider_credentials_loaded"] is False
    serialized = json.dumps(data).lower()
    for private_key in (
        "user_email",
        "token_expiry",
        "storage_quota",
        "document",
        "spreadsheet",
        "presentation",
        "profile",
    ):
        assert private_key not in serialized


def test_byok_cache_and_default_file_fallback_are_disabled_by_default():
    assert gm.MCP_BYOK_CLIENT_CACHE_SIZE == 0
    assert gm.MCP_BYOK_CLIENT_CACHE_TTL_SECONDS == 0
    assert gm.MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK is True


def test_token_writers_force_owner_only_permissions(tmp_path):
    for index, writer in enumerate((gal._write_private_text, gm._write_private_text)):
        target = tmp_path / f"token-{index}.json"
        target.write_text("old")
        target.chmod(0o644)
        writer(str(target), '{"example":"synthetic"}')
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert target.read_text() == '{"example":"synthetic"}'
    assert not list(tmp_path.glob(".google-token-*"))


def test_download_limits_cannot_be_disabled_or_raised_by_callers():
    assert gm._effective_download_limit(4096) == 4096
    assert (
        gm._effective_download_limit(gm.DEFAULT_MAX_DOWNLOAD_BYTES + 1)
        == gm.DEFAULT_MAX_DOWNLOAD_BYTES
    )
    for invalid in (0, -1):
        with pytest.raises(ValueError, match="positive integer"):
            gm._effective_download_limit(invalid)


def test_tool_output_and_provider_errors_are_bounded_and_redacted(monkeypatch):
    monkeypatch.setattr(gm, "MCP_TOOL_OUTPUT_MAX_BYTES", 4096)
    large = json.loads(
        asyncio.run(
            gm.run_tool(
                "test",
                "large_output",
                lambda: {"value": "x" * 10000},
                allow_retry=False,
            )
        )
    )
    assert large["ok"] is True
    assert large["data"]["truncated"] is True
    assert len(json.dumps(large).encode("utf-8")) <= 4096

    secret = "provider-secret-value"

    def _provider_error():
        raise gm.GoogleProviderError(
            f"request failed refresh_token={secret}",
            details={"access_token": secret},
        )

    error_blob = asyncio.run(
        gm.run_tool("test", "provider_error", _provider_error, allow_retry=False)
    )
    assert secret not in error_blob
    error = json.loads(error_blob)["error"]
    assert "details" not in error
    assert "[REDACTED]" in error["message"]


def test_direct_provider_responses_are_bounded_before_json_decode(monkeypatch):
    monkeypatch.setattr(gm, "MCP_PROVIDER_RESPONSE_MAX_BYTES", 4096)

    class _DeclaredTooLarge:
        status_code = 200
        headers = {
            "content-type": "application/json",
            "content-length": "4097",
        }
        closed = False

        def close(self):
            self.closed = True

    declared = _DeclaredTooLarge()
    with pytest.raises(gm.GoogleProviderError) as declared_error:
        gm._bounded_provider_payload(declared, "Example provider")
    assert declared_error.value.error_type == "bounded_response"
    assert declared.closed is True

    class _ChunkedTooLarge:
        status_code = 200
        headers = {"content-type": "application/json"}
        closed = False

        def iter_content(self, *, chunk_size):
            assert chunk_size == 64 * 1024
            yield b"x" * 2048
            yield b"y" * 2049

        def close(self):
            self.closed = True

    chunked = _ChunkedTooLarge()
    with pytest.raises(gm.GoogleProviderError) as chunked_error:
        gm._bounded_provider_payload(chunked, "Example provider")
    assert chunked_error.value.error_type == "bounded_response"
    assert chunked.closed is True

    class _ValidChunked:
        status_code = 200
        headers = {"content-type": "application/json"}

        def iter_content(self, *, chunk_size):
            assert chunk_size == 64 * 1024
            yield b'{"items":'
            yield b"[]}"

        def close(self):
            pass

    assert gm._bounded_provider_payload(_ValidChunked(), "Example provider")["json"] == {
        "items": []
    }


class _StreamingJsonResponse:
    def __init__(self, payload: dict, *, status_code: int = 200):
        self.body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.status_code = status_code
        self.headers = {
            "content-type": "application/json",
            "content-length": str(len(self.body)),
        }
        self.ok = 200 <= status_code < 300
        self.closed = False

    def iter_content(self, *, chunk_size):
        assert chunk_size == 64 * 1024
        midpoint = max(1, len(self.body) // 2)
        yield self.body[:midpoint]
        yield self.body[midpoint:]

    def close(self):
        self.closed = True


class _AttachmentSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


class _AttachmentClient:
    def __init__(self, session):
        self.session = session

    def get_session(self):
        return self.session, False

    def is_session_cached(self):
        return False


def test_gmail_attachment_metadata_only_never_fetches_content(monkeypatch):
    metadata = _StreamingJsonResponse({"size": 7})
    session = _AttachmentSession(metadata)
    monkeypatch.setattr(gm, "client", _AttachmentClient(session))

    result = json.loads(
        asyncio.run(
            gm.gmail_get_attachment(
                message_id="message/with/slash",
                attachment_id="attachment?value",
                include_content=False,
            )
        )
    )

    assert result["ok"] is True
    assert result["data"] == {"attachment_id": "attachment?value", "size": 7}
    assert result["meta"]["provider_calls"] == 1
    assert len(session.calls) == 1
    method, url, kwargs = session.calls[0]
    assert method == "GET"
    assert "message%2Fwith%2Fslash" in url
    assert url.endswith("attachment%3Fvalue")
    assert kwargs["params"] == {"fields": "size"}
    assert kwargs["stream"] is True
    assert kwargs["allow_redirects"] is False
    assert metadata.closed is True


def test_gmail_attachment_declared_size_prevents_content_fetch(monkeypatch):
    metadata = _StreamingJsonResponse({"size": 4097})
    session = _AttachmentSession(metadata)
    monkeypatch.setattr(gm, "client", _AttachmentClient(session))

    result = json.loads(
        asyncio.run(
            gm.gmail_get_attachment(
                message_id="message",
                attachment_id="attachment",
                max_bytes=4096,
                include_content=True,
            )
        )
    )

    assert result["ok"] is True
    assert result["data"]["too_large"] is True
    assert result["data"]["max_bytes"] == 4096
    assert result["meta"]["provider_calls"] == 1
    assert len(session.calls) == 1
    assert metadata.closed is True


def test_gmail_attachment_content_is_streamed_bounded_and_closed(monkeypatch):
    metadata = _StreamingJsonResponse({"size": 3})
    content = _StreamingJsonResponse({"size": 3, "data": "YWJj"})
    session = _AttachmentSession(metadata, content)
    monkeypatch.setattr(gm, "client", _AttachmentClient(session))

    result = json.loads(
        asyncio.run(
            gm.gmail_get_attachment(
                message_id="message",
                attachment_id="attachment",
                max_bytes=4096,
                include_content=True,
            )
        )
    )

    assert result["ok"] is True
    assert result["data"] == {
        "attachment_id": "attachment",
        "size": 3,
        "data": "YWJj",
    }
    assert result["meta"]["provider_calls"] == 2
    assert [call[2]["params"] for call in session.calls] == [
        {"fields": "size"},
        {"fields": "data,size"},
    ]
    assert all(call[2]["stream"] is True for call in session.calls)
    assert metadata.closed is True
    assert content.closed is True


def test_raw_request_rejects_mutation_headers_credentials_and_token_host():
    with pytest.raises(ValueError, match="GET and HEAD"):
        asyncio.run(
            gm.google_raw_request(
                method="POST",
                url="https://www.googleapis.com/drive/v3/files",
            )
        )
    with pytest.raises(ValueError, match="caller-supplied HTTP headers"):
        asyncio.run(
            gm.google_raw_request(
                method="GET",
                url="https://www.googleapis.com/drive/v3/files",
                headers={"Authorization": "Bearer must-not-be-accepted"},
            )
        )
    with pytest.raises(ValueError, match="credential-bearing keys"):
        asyncio.run(
            gm.google_raw_request(
                method="GET",
                url="https://www.googleapis.com/drive/v3/files",
                params={"access_token": "must-not-be-accepted"},
            )
        )
    with pytest.raises(ValueError, match="host is not allowed"):
        gm._validate_raw_request_url("https://oauth2.googleapis.com/token")


def test_all_confirmed_tools_have_confirmation_and_transport_fails_closed():
    registry = gm._tool_registry()
    assert gm.DESTRUCTIVE_TOOLS
    for tool_name in gm.DESTRUCTIVE_TOOLS | gm.BILLABLE_MAPS_TOOLS:
        properties = registry[tool_name].parameters.get("properties", {})
        assert "confirm" in properties, tool_name

    client = TestClient(_ok_app())
    try:
        response = client.post(
            "/mcp",
            headers=_portal_headers(),
            json={
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "gmail_send_message",
                    "arguments": {
                        "to": "nobody@example.invalid",
                        "subject": "never sent",
                        "body": "never sent",
                    },
                },
            },
        )
    finally:
        client.close()
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32602


def test_confirmation_contract_covers_schema_and_transport_preflight():
    manifest = gm._current_tool_manifest()
    confirmed = [
        descriptor
        for descriptor in manifest["tools"]
        if "confirm" in descriptor["inputSchema"].get("properties", {})
    ]
    assert len(confirmed) == 55
    for descriptor in confirmed:
        assert descriptor["confirmation"]["required"] is True
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": descriptor["nativeToolName"], "arguments": {}},
        }
        with pytest.raises(ValueError, match="confirm=true"):
            gm._validate_tool_confirmation(payload)


def test_batch_and_provider_fanout_limits_fail_before_provider_resolution(monkeypatch):
    assert gm.MCP_MAX_BATCH_ITEMS == 1000
    assert gm.MCP_MAX_PROVIDER_CALLS_PER_TOOL == 128

    oversized_payload = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {
            "name": "drive_batch_get_metadata",
            "arguments": {"file_ids": [f"synthetic-{index}" for index in range(1001)]},
        },
    }
    with pytest.raises(ValueError, match="1000-item batch limit"):
        gm._validate_tool_execution_limits(oversized_payload)

    provider_fanout_payload = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "calendar_batch_get_events",
            "arguments": {"event_ids": [f"synthetic-{index}" for index in range(129)]},
        },
    }
    with pytest.raises(ValueError, match="128-item execution limit"):
        gm._validate_tool_execution_limits(provider_fanout_payload)

    scan_payload = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "gmail_sender_clusters",
            "arguments": {"max_messages": 1000, "page_size": 1},
        },
    }
    with pytest.raises(ValueError, match="provider-call limit"):
        gm._validate_tool_execution_limits(scan_payload)

    def _must_not_resolve(_headers):
        raise AssertionError("provider credentials must not resolve after a rejected batch")

    monkeypatch.setattr(gm, "_resolve_request_client", _must_not_resolve)
    client = TestClient(_ok_app())
    try:
        response = client.post("/mcp", headers=_portal_headers(), json=oversized_payload)
    finally:
        client.close()
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32602


def test_maps_calls_require_confirmation_and_disable_automatic_retry(monkeypatch):
    unconfirmed = json.loads(asyncio.run(gm.maps_geocode(address="example")))
    assert unconfirmed["ok"] is False
    assert "confirm=true" in unconfirmed["error"]["message"]

    captured: list[tuple[str, bool]] = []

    async def _capture_run_tool(api, action, func, *, allow_retry=True, **_kwargs):
        captured.append((action, allow_retry))
        return "{}"

    monkeypatch.setattr(gm, "run_tool", _capture_run_tool)
    asyncio.run(gm.maps_geocode(address="example", confirm=True))
    asyncio.run(gm.maps_reverse_geocode(lat=0.0, lng=0.0, confirm=True))
    asyncio.run(gm.maps_place_text_search(query="example", confirm=True))
    asyncio.run(gm.maps_place_details(place_id="place", confirm=True))
    asyncio.run(gm.maps_compute_routes(route_request={"origin": {}}, confirm=True))
    assert len(captured) == 5
    assert all(allow_retry is False for _, allow_retry in captured)


def test_raw_request_disables_automatic_retry(monkeypatch):
    captured: list[tuple[str, bool]] = []

    async def _capture_run_tool(api, action, func, *, allow_retry=True, **_kwargs):
        captured.append((action, allow_retry))
        return "{}"

    monkeypatch.setattr(gm, "run_tool", _capture_run_tool)
    asyncio.run(
        gm.google_raw_request(
            method="GET",
            url="https://www.googleapis.com/drive/v3/files",
        )
    )
    assert captured == [("google_raw_request", False)]


def test_public_console_entrypoint_is_synchronous_and_callable():
    assert callable(gm.main)
    assert not asyncio.iscoroutinefunction(gm.main)


def test_public_console_entrypoint_disables_uvicorn_access_logging(monkeypatch):
    captured = {}

    monkeypatch.setattr(gm, "_validate_runtime_configuration", lambda: None)
    monkeypatch.setattr(gm.uvicorn, "run", lambda app, **kwargs: captured.update(app=app, **kwargs))
    gm.main()

    assert captured["app"] is gm.build_app
    assert captured["factory"] is True
    assert captured["access_log"] is False
