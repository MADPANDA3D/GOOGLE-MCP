import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from starlette.testclient import TestClient

os.environ.setdefault("MCP_PORTAL_GRANT_TOKEN", "test-portal-grant")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import google_mcp_server as gm


def _minimal_ok_app():
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


def _mcp_headers(**overrides: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-MADPANDA-PORTAL-GRANT": "test-portal-grant",
        "X-Google-Client-Id": "cid",
        "X-Google-Client-Secret": "csecret",
        "X-Google-Refresh-Token": "rtok",
    }
    headers.update(overrides)
    return headers


def _post_json(app, path: str, *, headers: dict[str, str], payload: dict) -> tuple[int, dict]:
    client = TestClient(app)
    try:
        response = client.post(path, headers=headers, json=payload)
    finally:
        client.close()
    return response.status_code, response.json()


def _mcp_client():
    app = gm.build_hosted_mcp_http_wrapper(gm.mcp.streamable_http_app())
    return TestClient(app)


def test_missing_portal_grant_is_rejected_before_byok():
    app = _minimal_ok_app()
    headers = _mcp_headers()
    headers.pop("X-MADPANDA-PORTAL-GRANT")
    status_code, payload = _post_json(
        app,
        "/mcp",
        headers=headers,
        payload={"jsonrpc": "2.0", "id": 11, "method": "tools/list", "params": {}},
    )
    assert status_code == 401
    assert payload["error"]["code"] == -32001
    assert "x-madpanda-portal-grant" in payload["error"]["message"]
    assert "x-google-client-id" not in payload["error"]["message"]


def test_invalid_portal_grant_is_rejected():
    app = _minimal_ok_app()
    status_code, payload = _post_json(
        app,
        "/mcp",
        headers=_mcp_headers(**{"X-MADPANDA-PORTAL-GRANT": "wrong"}),
        payload={"jsonrpc": "2.0", "id": 12, "method": "tools/list", "params": {}},
    )
    assert status_code == 401
    assert payload["error"]["code"] == -32001
    assert payload["error"]["message"] == "Invalid portal grant token."


def test_missing_byok_headers_are_rejected():
    app = _minimal_ok_app()
    status_code, payload = _post_json(
        app,
        "/mcp",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-MADPANDA-PORTAL-GRANT": "test-portal-grant",
        },
        payload={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    assert status_code == 401
    assert payload["error"]["code"] == -32001
    assert "x-google-client-id" in payload["error"]["message"]
    assert "x-google-client-secret" in payload["error"]["message"]
    assert "x-google-refresh-token" in payload["error"]["message"]


def test_partial_byok_headers_are_rejected():
    app = _minimal_ok_app()
    status_code, payload = _post_json(
        app,
        "/mcp",
        headers=_mcp_headers(
            **{
                "X-Google-Client-Secret": "",
                "X-Google-Refresh-Token": "",
            }
        ),
        payload={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    assert status_code == 401
    assert payload["error"]["code"] == -32001
    assert "x-google-client-secret" in payload["error"]["message"]
    assert "x-google-refresh-token" in payload["error"]["message"]


def test_tools_list_succeeds_with_valid_byok_headers():
    with _mcp_client() as client:
        response = client.post(
            "/mcp",
            headers=_mcp_headers(),
            json={"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}},
        )

    payload = response.json()
    assert response.status_code == 200
    assert "result" in payload
    assert "tools" in payload["result"]
    assert any(tool["name"] == "mcp_health_check" for tool in payload["result"]["tools"])


def test_all_tools_have_openai_annotations_and_parameter_descriptions():
    tools = gm._tool_registry()
    assert len(tools) == gm.EXPECTED_TOOL_COUNT
    for tool in tools.values():
        assert tool.annotations is not None
        dumped = tool.annotations.model_dump()
        assert dumped["readOnlyHint"] is not None
        assert dumped["destructiveHint"] is not None
        assert dumped["openWorldHint"] is not None
        for schema in tool.parameters.get("properties", {}).values():
            assert "description" in schema


def test_common_camel_case_tool_arguments_are_normalized():
    payload = {
        "jsonrpc": "2.0",
        "id": 13,
        "method": "tools/call",
        "params": {
            "name": "calendar_list_events",
            "arguments": {
                "calendarId": "primary",
                "timeMin": "2026-04-15T00:00:00Z",
                "timeMax": "2026-04-16T00:00:00Z",
            },
        },
    }
    gm._normalize_tool_arguments(payload)
    args = payload["params"]["arguments"]
    assert args["calendar_id"] == "primary"
    assert args["time_min"] == "2026-04-15T00:00:00Z"
    assert args["time_max"] == "2026-04-16T00:00:00Z"
    assert "timeMin" not in args


def test_gmail_batch_metadata_camel_case_alias_is_normalized():
    payload = {
        "jsonrpc": "2.0",
        "id": 14,
        "method": "tools/call",
        "params": {
            "name": "gmail_batch_get_metadata",
            "arguments": {"messageIds": ["m1", "m2"], "metadataHeaders": ["Subject"]},
        },
    }
    gm._normalize_tool_arguments(payload)
    args = payload["params"]["arguments"]
    assert args["message_ids"] == ["m1", "m2"]
    assert args["metadata_headers"] == ["Subject"]


def test_health_check_succeeds_with_valid_byok_headers():
    class _DummyCreds:
        valid = True
        expiry = datetime(2026, 1, 1, tzinfo=timezone.utc)

    class _DummyClient:
        def _load_credentials(self):
            return _DummyCreds()

        def is_service_cached(self, api_name, api_version):
            return False

        def is_session_cached(self):
            return False

        def get_session(self):
            return SimpleNamespace(), False

    token = gm.ACTIVE_GOOGLE_CLIENT.set(_DummyClient())
    try:
        result_blob = asyncio.run(gm.mcp_health_check(run_checks=False, warm_all=False))
    finally:
        gm.ACTIVE_GOOGLE_CLIENT.reset(token)

    decoded = json.loads(result_blob)
    assert decoded["ok"] is True


def test_refresh_error_is_classified_as_auth_error():
    def _boom():
        raise gm.RefreshError("invalid_grant")

    payload = json.loads(asyncio.run(gm.run_tool("mcp", "health_check", _boom, allow_retry=False)))
    assert payload["ok"] is False
    assert payload["error"]["type"] == "auth_error"


def test_concurrent_requests_keep_client_context_isolated(monkeypatch):
    class _NamedClient:
        def __init__(self, name: str):
            self.name = name

    def _fake_resolve(header_items):
        headers = gm._normalize_header_map(header_items)
        name = headers.get("x-google-client-id", "missing")
        return _NamedClient(name), {"auth_mode": "byok", "byok_cache_hit": False}

    async def _echo_app(scope, receive, send):
        active = gm.ACTIVE_GOOGLE_CLIENT.get()
        body = json.dumps({"active_client": getattr(active, "name", None)}).encode("utf-8")
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

    async def _call(app, client_id: str) -> str:
        request_body = json.dumps(
            {"jsonrpc": "2.0", "id": 10, "method": "tools/list", "params": {}}
        ).encode("utf-8")
        sent = False
        messages = []

        async def _receive():
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": request_body, "more_body": False}

        async def _send(message):
            messages.append(message)

        headers = [
            (b"content-type", b"application/json"),
            (b"accept", b"application/json"),
            (b"x-madpanda-portal-grant", b"test-portal-grant"),
            (b"x-google-client-id", client_id.encode("utf-8")),
            (b"x-google-client-secret", b"secret"),
            (b"x-google-refresh-token", b"refresh"),
        ]
        scope = {"type": "http", "path": "/mcp", "method": "POST", "headers": headers}
        await app(scope, _receive, _send)
        raw_body = b"".join(m.get("body", b"") for m in messages if m.get("type") == "http.response.body")
        return json.loads(raw_body.decode("utf-8"))["active_client"]

    monkeypatch.setattr(gm, "_resolve_request_client", _fake_resolve)
    wrapped = gm.build_hosted_mcp_http_wrapper(_echo_app)

    async def _run():
        return await asyncio.gather(_call(wrapped, "alpha"), _call(wrapped, "beta"))

    first, second = asyncio.run(_run())
    assert first == "alpha"
    assert second == "beta"


def test_raw_request_blocks_non_google_hosts_in_strict_mode():
    assert gm.MCP_RAW_STRICT is True
    try:
        gm._validate_raw_request_url("https://example.com/drive/v3/files")
    except ValueError as exc:
        assert "host is not allowed" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected strict raw host validation error")
