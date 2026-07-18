import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

TEST_PORTAL_GRANT_TOKEN = "test-portal-grant-0000000000000000"

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
        "X-MADPANDA-PORTAL-GRANT": TEST_PORTAL_GRANT_TOKEN,
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


def test_missing_portal_grant_rejects_malformed_body_before_parsing():
    app = _minimal_ok_app()
    headers = _mcp_headers()
    headers.pop("X-MADPANDA-PORTAL-GRANT")
    client = TestClient(app)
    try:
        response = client.post(
            "/mcp",
            headers=headers,
            content=b"not-json-and-must-not-be-parsed",
        )
    finally:
        client.close()
    assert response.status_code == 401
    assert response.json()["error"]["code"] == -32001


def test_invalid_portal_grant_rejects_malformed_body_before_parsing():
    app = _minimal_ok_app()
    client = TestClient(app)
    try:
        response = client.post(
            "/mcp",
            headers=_mcp_headers(**{"X-MADPANDA-PORTAL-GRANT": "wrong"}),
            content=b"not-json-and-must-not-be-parsed",
        )
    finally:
        client.close()
    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Invalid portal grant token."


def test_missing_byok_headers_are_rejected():
    app = _minimal_ok_app()
    status_code, payload = _post_json(
        app,
        "/mcp",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-MADPANDA-PORTAL-GRANT": TEST_PORTAL_GRANT_TOKEN,
        },
        payload={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "drive_list_files", "arguments": {}},
        },
    )
    assert status_code == 401
    assert payload["error"]["code"] == -32001
    assert "x-google-client-id" in payload["error"]["message"]
    assert "x-google-client-secret" in payload["error"]["message"]
    assert "x-google-refresh-token" in payload["error"]["message"]


def test_local_discovery_tools_are_provider_free(monkeypatch):
    def _provider_resolution_must_not_run(_header_items):
        raise AssertionError("provider credential resolution must not run")

    monkeypatch.setattr(gm, "_resolve_request_client", _provider_resolution_must_not_run)
    calls = {
        "check_configuration": {},
        "list_capabilities": {"include_descriptors": False},
        "get_endpoint_coverage": {"limit": 1},
        "get_tool_usage": {"tool_name": "drive_list_files"},
        "find_tools": {"query": "drive search files"},
        "google_mcp_welcome": {},
        "google_mcp_list_capabilities": {},
        "google_mcp_get_endpoint_coverage": {},
        "google_mcp_get_tool_usage": {},
        "mcp_health_check": {"run_checks": False, "warm_all": False},
    }
    client = TestClient(_minimal_ok_app())
    try:
        for request_id, (name, arguments) in enumerate(calls.items(), start=30):
            response = client.post(
                "/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-MADPANDA-PORTAL-GRANT": TEST_PORTAL_GRANT_TOKEN,
                },
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
            )
            assert response.status_code == 200, (name, response.text)
            assert response.json()["ok"] is True, (name, response.text)
    finally:
        client.close()


def test_navigation_is_provider_free_but_provider_tools_require_byok():
    provider_free_payloads = [
        {
            "jsonrpc": "2.0",
            "id": 40,
            "method": "tools/call",
            "params": {
                "name": "google_mcp_list_capabilities",
                "arguments": {},
            },
        },
        {"jsonrpc": "2.0", "id": 42, "method": "tools/list", "params": {}},
    ]
    client = TestClient(_minimal_ok_app())
    try:
        for payload in provider_free_payloads:
            response = client.post(
                "/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-MADPANDA-PORTAL-GRANT": TEST_PORTAL_GRANT_TOKEN,
                },
                json=payload,
            )
            assert response.status_code == 200
            assert response.json()["ok"] is True

        for request_id, tool_name in enumerate(
            ("drive_list_files", "gmail_signature_preflight"),
            start=41,
        ):
            provider_response = client.post(
                "/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-MADPANDA-PORTAL-GRANT": TEST_PORTAL_GRANT_TOKEN,
                },
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": {}},
                },
            )
            assert provider_response.status_code == 401
            assert "x-google-client-id" in provider_response.json()["error"]["message"]
    finally:
        client.close()


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
        payload={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "drive_list_files", "arguments": {}},
        },
    )
    assert status_code == 401
    assert payload["error"]["code"] == -32001
    assert "x-google-client-secret" in payload["error"]["message"]
    assert "x-google-refresh-token" in payload["error"]["message"]


def test_tools_list_succeeds_with_valid_byok_headers():
    with _mcp_client() as client:
        grant_only_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-MADPANDA-PORTAL-GRANT": TEST_PORTAL_GRANT_TOKEN,
        }
        initialized = client.post(
            "/mcp",
            headers=grant_only_headers,
            json={
                "jsonrpc": "2.0",
                "id": 43,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
        )
        ready = client.post(
            "/mcp",
            headers=grant_only_headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        capabilities = client.post(
            "/mcp",
            headers=grant_only_headers,
            json={
                "jsonrpc": "2.0",
                "id": 44,
                "method": "tools/call",
                "params": {
                    "name": "list_capabilities",
                    "arguments": {"include_descriptors": False},
                },
            },
        )
        response = client.post(
            "/mcp",
            headers=_mcp_headers(),
            json={"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}},
        )

    assert initialized.status_code == 200
    assert initialized.json()["result"]["serverInfo"]["name"]
    assert ready.status_code in {200, 202}
    assert capabilities.status_code == 200
    capability_result = json.loads(capabilities.json()["result"]["content"][0]["text"])
    assert capability_result["serviceId"] == "google"
    assert capability_result["counts"]["raw"] == 151
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


def test_drive_upload_file_folder_aliases_are_normalized_to_parent_id():
    aliases = {
        "folder_id": "folder-1",
        "folderId": "folder-1",
        "parent_folder_id": "folder-1",
        "parentFolderId": "folder-1",
        "parents": ["folder-1"],
    }
    for alias, value in aliases.items():
        payload = {
            "jsonrpc": "2.0",
            "id": 15,
            "method": "tools/call",
            "params": {
                "name": "drive_upload_file",
                "arguments": {
                    "name": "tiny.txt",
                    "content": "hello",
                    alias: value,
                },
            },
        }
        gm._normalize_tool_arguments(payload)
        args = payload["params"]["arguments"]
        assert args["parent_id"] == "folder-1"
        assert alias not in args


def test_gmail_sender_key_prefers_list_id_then_domain():
    sender = gm._gmail_sender_key(
        {
            "from": "Example Sender <news@example.com>",
            "list-id": "Example List <list.example.com>",
        }
    )
    assert sender["key"] == "example list <list.example.com>"
    assert sender["sender_email"] == "news@example.com"
    assert sender["domain"] == "example.com"

    sender = gm._gmail_sender_key({"from": "Noise <promo@example.org>"})
    assert sender["key"] == "example.org"


def _gmail_signature_fixture() -> gm.GmailSendAsSignature:
    signature_html = (
        "<div>Example Regards,<br>Example Sender<br>"
        '<img src="https://cdn.example.com/logo.png" alt="Example Organization"></div>'
    )
    return gm.GmailSendAsSignature(
        alias="sender@example.com",
        html=signature_html,
        fingerprint=gm._gmail_signature_fingerprint(signature_html),
    )


def test_gmail_send_as_signature_selects_default_and_explicit_alias():
    class _Request:
        def execute(self):
            return {
                "sendAs": [
                    {
                        "sendAsEmail": "other@example.com",
                        "signature": "<div>Other</div>",
                        "isPrimary": True,
                    },
                    {
                        "sendAsEmail": "sender@example.com",
                        "signature": "<div>Example Sender</div>",
                        "isDefault": True,
                    },
                ]
            }

    class _Service:
        def users(self):
            return self

        def settings(self):
            return self

        def sendAs(self):
            return self

        def list(self, **kwargs):
            assert kwargs == {"userId": "me"}
            return _Request()

    default_signature = gm._resolve_gmail_send_as_signature(_Service())
    explicit_signature = gm._resolve_gmail_send_as_signature(
        _Service(),
        "Other Sender <other@example.com>",
    )

    assert default_signature.alias == "sender@example.com"
    assert default_signature.html == "<div>Example Sender</div>"
    assert explicit_signature.alias == "other@example.com"
    assert explicit_signature.html == "<div>Other</div>"


def test_gmail_send_as_signature_fails_when_selected_alias_is_unsigned():
    class _Request:
        def execute(self):
            return {
                "sendAs": [
                    {
                        "sendAsEmail": "sender@example.com",
                        "signature": "",
                        "isDefault": True,
                    }
                ]
            }

    class _Service:
        def users(self):
            return self

        def settings(self):
            return self

        def sendAs(self):
            return self

        def list(self, **kwargs):
            assert kwargs == {"userId": "me"}
            return _Request()

    with pytest.raises(ValueError, match="No Gmail signature is configured"):
        gm._resolve_gmail_send_as_signature(_Service())


def test_gmail_plain_message_adds_text_and_html_signature_once():
    signature = _gmail_signature_fixture()
    message = gm.build_email_message(
        to="recipient@example.com",
        subject="Signed message",
        body="Hello <team>\nSecond line",
        signature=signature,
    )

    plain_part = message.get_body(preferencelist=("plain",))
    html_part = message.get_body(preferencelist=("html",))

    assert plain_part is not None
    assert html_part is not None
    assert "Example Regards" in plain_part.get_content()
    assert "Hello &lt;team&gt;<br>Second line" in html_part.get_content()
    assert "https://cdn.example.com/logo.png" in html_part.get_content()
    assert html_part.get_content().count(gm.GMAIL_SIGNATURE_MARKER) == 1
    assert gm._gmail_message_contains_signature(message, signature) is True

    pre_signed_plain = gm.build_email_message(
        to="recipient@example.com",
        subject="Pre-signed plain body",
        body=(f"Hello\n\n{gm._gmail_signature_plain_text(signature.html)}"),
        signature=signature,
    )
    pre_signed_html = pre_signed_plain.get_body(preferencelist=("html",))
    assert pre_signed_html is not None
    assert pre_signed_html.get_content().count(signature.html) == 1
    assert pre_signed_html.get_content().count(gm.GMAIL_SIGNATURE_MARKER) == 1


def test_gmail_html_message_does_not_duplicate_existing_signature():
    signature = _gmail_signature_fixture()
    message = gm.build_email_message(
        to="recipient@example.com",
        subject="Already signed",
        body=f"<p>Hello</p>{signature.html}",
        is_html=True,
        signature=signature,
    )
    html_part = message.get_body(preferencelist=("html",))

    assert html_part is not None
    assert html_part.get_content().count(signature.html) == 1
    assert html_part.get_content().count(gm.GMAIL_SIGNATURE_MARKER) == 0
    assert gm._gmail_message_contains_signature(message, signature) is True

    quoted_history = gm.build_email_message(
        to="recipient@example.com",
        subject="Reply",
        body=(
            f"<blockquote>{signature.html}</blockquote><p>Current reply follows quoted history.</p>"
        ),
        is_html=True,
        signature=signature,
    )
    quoted_html = quoted_history.get_body(preferencelist=("html",))
    assert quoted_html is not None
    assert quoted_html.get_content().count(signature.html) == 2


def test_gmail_signature_detection_rejects_marker_only_and_mid_body_text():
    signature = _gmail_signature_fixture()
    marker_only = gm.build_email_message(
        to="recipient@example.com",
        subject="Forged marker",
        body=(
            f'<p>Hello</p><div {gm.GMAIL_SIGNATURE_MARKER}="{signature.fingerprint[:16]}"></div>'
        ),
        is_html=True,
    )
    short_html = "<div>Thanks</div>"
    short_signature = gm.GmailSendAsSignature(
        alias="sender@example.com",
        html=short_html,
        fingerprint=gm._gmail_signature_fingerprint(short_html),
    )
    mid_body_text = gm.build_email_message(
        to="recipient@example.com",
        subject="Body mention",
        body="Thanks for the update. More details follow.",
    )

    assert gm._gmail_message_contains_signature(marker_only, signature) is False
    assert gm._gmail_message_contains_signature(mid_body_text, short_signature) is False


def test_gmail_raw_message_signature_verification_fails_closed():
    signature = _gmail_signature_fixture()
    unsigned = gm.build_email_message(
        to="recipient@example.com",
        subject="Unsigned",
        body="No signature here",
    )
    signed = gm.build_email_message(
        to="recipient@example.com",
        subject="Signed",
        body="Signature follows",
        signature=signature,
    )

    assert gm._gmail_message_contains_signature(unsigned, signature) is False
    assert gm._gmail_message_contains_signature(signed, signature) is True
    assert (
        gm._gmail_message_contains_signature(
            gm._decode_email_message(gm.encode_email_message(signed)),
            signature,
        )
        is True
    )
    text_only_copy = gm.build_email_message(
        to="recipient@example.com",
        subject="Missing logo",
        body=gm._gmail_signature_plain_text(signature.html),
    )
    assert gm._gmail_message_contains_signature(text_only_copy, signature) is False


def test_gmail_signature_detection_ignores_attached_messages_and_quoted_history():
    signature = _gmail_signature_fixture()
    attached_signed_message = gm.build_email_message(
        to="recipient@example.com",
        subject="Attached signed message",
        body="Signed attachment",
        signature=signature,
    )
    unsigned_outer = gm.build_email_message(
        to="recipient@example.com",
        subject="Unsigned outer message",
        body="The attached message is signed, but this one is not.",
    )
    unsigned_outer.add_attachment(attached_signed_message)
    assert gm._gmail_message_contains_signature(unsigned_outer, signature) is False

    text_signature_html = "<div>Thanks,<br>Example Sender</div>"
    text_signature = gm.GmailSendAsSignature(
        alias="sender@example.com",
        html=text_signature_html,
        fingerprint=gm._gmail_signature_fingerprint(text_signature_html),
    )
    quoted_history = (
        "Current reply\n\n"
        "On Fri, Jul 17, 2026, Example Contact wrote:\n"
        "Older message\n\n-- \nThanks,\nExample Sender"
    )
    reply = gm.build_email_message(
        to="recipient@example.com",
        subject="Reply",
        body=quoted_history,
        signature=text_signature,
    )
    reply_html = reply.get_body(preferencelist=("html",))
    assert reply_html is not None
    assert gm.GMAIL_SIGNATURE_MARKER in reply_html.get_content()
    assert gm._gmail_message_contains_signature(reply, text_signature) is True


def test_gmail_send_message_uses_configured_signature_before_provider_send():
    captured = {}
    signature_html = _gmail_signature_fixture().html

    class _Request:
        def __init__(self, payload):
            self.payload = payload

        def execute(self):
            return self.payload

    class _Service:
        def users(self):
            return self

        def settings(self):
            return self

        def sendAs(self):
            return self

        def list(self, **kwargs):
            assert kwargs == {"userId": "me"}
            return _Request(
                {
                    "sendAs": [
                        {
                            "sendAsEmail": "sender@example.com",
                            "signature": signature_html,
                            "isDefault": True,
                        }
                    ]
                }
            )

        def messages(self):
            return self

        def send(self, **kwargs):
            captured.update(kwargs)
            return _Request({"id": "message-1"})

    class _Client:
        def get_service(self, api_name, api_version):
            assert (api_name, api_version) == ("gmail", "v1")
            return _Service(), False

        def is_session_cached(self):
            return False

    token = gm.ACTIVE_GOOGLE_CLIENT.set(_Client())
    try:
        result = json.loads(
            asyncio.run(
                gm.gmail_send_message(
                    to="recipient@example.com",
                    subject="Signed",
                    body="Hello",
                    confirm=True,
                )
            )
        )
    finally:
        gm.ACTIVE_GOOGLE_CLIENT.reset(token)

    sent_message = gm._decode_email_message(captured["body"]["raw"])
    html_part = sent_message.get_body(preferencelist=("html",))
    assert captured["userId"] == "me"
    assert html_part is not None
    assert signature_html in html_part.get_content()
    assert result["ok"] is True
    assert result["meta"]["signature_present"] is True
    assert result["meta"]["signature_alias"] == "sender@example.com"


def test_gmail_signature_preflight_returns_only_safe_binding_metadata():
    signature_html = _gmail_signature_fixture().html

    class _Request:
        def execute(self):
            return {
                "sendAs": [
                    {
                        "sendAsEmail": "sender@example.com",
                        "signature": signature_html,
                        "isDefault": True,
                    }
                ]
            }

    class _Service:
        def users(self):
            return self

        def settings(self):
            return self

        def sendAs(self):
            return self

        def list(self, **kwargs):
            assert kwargs == {"userId": "me"}
            return _Request()

    class _Client:
        def get_service(self, api_name, api_version):
            assert (api_name, api_version) == ("gmail", "v1")
            return _Service(), False

        def is_session_cached(self):
            return False

    token = gm.ACTIVE_GOOGLE_CLIENT.set(_Client())
    try:
        result = json.loads(
            asyncio.run(gm.gmail_signature_preflight(from_alias="sender@example.com"))
        )
    finally:
        gm.ACTIVE_GOOGLE_CLIENT.reset(token)

    serialized = json.dumps(result)
    assert result["ok"] is True
    assert result["data"]["signature_alias"] == "sender@example.com"
    assert len(result["data"]["signature_fingerprint"]) == 64
    assert result["data"]["message_fingerprint"] == ""
    assert signature_html not in serialized
    assert "cdn.example.com" not in serialized


def test_gmail_signature_fingerprint_binding_fails_when_signature_changes():
    signature = _gmail_signature_fixture()

    gm._verify_gmail_signature_fingerprint(signature, signature.fingerprint)
    with pytest.raises(ValueError, match="changed after preview"):
        gm._verify_gmail_signature_fingerprint(signature, "0" * 64)


def test_gmail_message_fingerprint_binding_fails_when_draft_changes():
    original = gm.build_email_message(
        to="recipient@example.com",
        subject="Original",
        body="Original body",
    )
    changed = gm.build_email_message(
        to="recipient@example.com",
        subject="Changed",
        body="Changed body",
    )
    original_raw = gm.encode_email_message(original)
    changed_raw = gm.encode_email_message(changed)
    fingerprint = gm._gmail_raw_message_fingerprint(original_raw)

    gm._verify_gmail_message_fingerprint(original_raw, fingerprint)
    with pytest.raises(ValueError, match="draft changed after preview"):
        gm._verify_gmail_message_fingerprint(changed_raw, fingerprint)


def test_gmail_mailbox_overview_caps_labels(monkeypatch):
    class _FakeRequest:
        def __init__(self, data):
            self.data = data

        def execute(self):
            return self.data

    class _FakeMessages:
        def list(self, **kwargs):
            return _FakeRequest(
                {
                    "resultSizeEstimate": 7,
                    "messages": [{"id": "m1"}],
                    "nextPageToken": "next",
                }
            )

    class _FakeLabels:
        def list(self, **kwargs):
            return _FakeRequest(
                {
                    "labels": [
                        {"id": "A", "name": "A", "type": "user"},
                        {"id": "B", "name": "B", "type": "user"},
                        {"id": "C", "name": "C", "type": "user"},
                    ]
                }
            )

    class _FakeUsers:
        def messages(self):
            return _FakeMessages()

        def labels(self):
            return _FakeLabels()

    class _FakeService:
        def users(self):
            return _FakeUsers()

    class _FakeClient:
        def get_service(self, api_name, api_version):
            assert (api_name, api_version) == ("gmail", "v1")
            return _FakeService(), False

        def is_session_cached(self):
            return False

    monkeypatch.setattr(gm, "client", _FakeClient())

    payload = json.loads(
        asyncio.run(
            gm.gmail_mailbox_overview(
                queries=["is:unread"],
                include_labels=True,
                max_labels=2,
            )
        )
    )

    assert payload["ok"] is True
    assert payload["data"]["labels_total"] == 3
    assert payload["data"]["labels_returned"] == 2
    assert [label["id"] for label in payload["data"]["labels"]] == ["A", "B"]


def test_gmail_sender_clusters_obeys_sample_limit(monkeypatch):
    class _FakeClient:
        def get_service(self, api_name, api_version):
            assert (api_name, api_version) == ("gmail", "v1")
            return object(), False

        def is_session_cached(self):
            return False

    metadata = [
        {
            "id": f"m{index}",
            "headers": {
                "from": "Example Sender <news@example.com>",
                "subject": f"Subject {index}",
            },
            "labelIds": ["UNREAD", "INBOX"],
        }
        for index in range(4)
    ]

    monkeypatch.setattr(gm, "client", _FakeClient())
    monkeypatch.setattr(
        gm,
        "_gmail_list_message_ids",
        lambda service, query, max_messages, page_size: (
            [{"id": item["id"]} for item in metadata],
            None,
            4,
            1,
        ),
    )
    monkeypatch.setattr(
        gm,
        "_gmail_get_metadata_batch",
        lambda service, message_ids, max_messages: metadata,
    )

    payload = json.loads(
        asyncio.run(
            gm.gmail_sender_clusters(
                query="is:unread",
                max_messages=4,
                top_n=1,
                sample_per_cluster=2,
            )
        )
    )

    cluster = payload["data"]["clusters"][0]
    assert payload["ok"] is True
    assert payload["data"]["sample_per_cluster"] == 2
    assert cluster["count"] == 4
    assert cluster["message_ids_sample"] == ["m0", "m1"]
    assert cluster["subjects_sample"] == ["Subject 0", "Subject 1"]


def test_gmail_metadata_fetch_uses_google_batch_requests():
    batch_sizes = []

    class _FakeRequest:
        def __init__(self, message_id):
            self.message_id = message_id

        def execute(self):  # pragma: no cover - batch path should not call this
            raise AssertionError("sequential get should not run")

    class _FakeBatch:
        def __init__(self, callback):
            self.callback = callback
            self.requests = []

        def add(self, request, request_id):
            self.requests.append((request_id, request))

        def execute(self):
            batch_sizes.append(len(self.requests))
            for request_id, request in self.requests:
                self.callback(
                    request_id,
                    {
                        "id": request.message_id,
                        "threadId": f"t-{request.message_id}",
                        "labelIds": ["UNREAD"],
                        "payload": {
                            "headers": [{"name": "From", "value": "News <news@example.com>"}]
                        },
                    },
                    None,
                )

    class _FakeMessages:
        def get(self, **kwargs):
            assert kwargs["format"] == "metadata"
            assert kwargs["fields"] == "id,threadId,labelIds,snippet,internalDate,payload/headers"
            return _FakeRequest(kwargs["id"])

    class _FakeUsers:
        def messages(self):
            return _FakeMessages()

    class _FakeService:
        def users(self):
            return _FakeUsers()

        def new_batch_http_request(self, callback):
            return _FakeBatch(callback)

    results = gm._gmail_get_metadata_batch(
        _FakeService(),
        [f"m{index}" for index in range(250)],
        max_messages=250,
    )

    assert batch_sizes == [5] * 50
    assert len(results) == 250
    assert results[0]["id"] == "m0"
    assert results[-1]["id"] == "m249"
    assert results[0]["headers"]["from"] == "News <news@example.com>"


def test_workspace_create_tools_request_compact_fields(monkeypatch):
    captured = {}

    class _FakeRequest:
        def __init__(self, data):
            self.data = data

        def execute(self):
            return self.data

    class _DocsDocuments:
        def create(self, **kwargs):
            captured["docs"] = kwargs
            return _FakeRequest({"documentId": "doc-1", "title": "Doc"})

    class _SheetsSpreadsheets:
        def create(self, **kwargs):
            captured["sheets"] = kwargs
            return _FakeRequest(
                {
                    "spreadsheetId": "sheet-1",
                    "spreadsheetUrl": "https://example.test/sheet",
                    "properties": {"title": "Sheet"},
                }
            )

    class _SlidesPresentations:
        def create(self, **kwargs):
            captured["slides"] = kwargs
            return _FakeRequest(
                {
                    "presentationId": "slide-1",
                    "title": "Slides",
                    "slides": [{"objectId": "p"}],
                }
            )

    class _DocsService:
        def documents(self):
            return _DocsDocuments()

    class _SheetsService:
        def spreadsheets(self):
            return _SheetsSpreadsheets()

    class _SlidesService:
        def presentations(self):
            return _SlidesPresentations()

    class _FakeClient:
        def get_service(self, api_name, api_version):
            services = {
                ("docs", "v1"): _DocsService(),
                ("sheets", "v4"): _SheetsService(),
                ("slides", "v1"): _SlidesService(),
            }
            return services[(api_name, api_version)], False

        def is_session_cached(self):
            return False

    monkeypatch.setattr(gm, "client", _FakeClient())

    docs_payload = json.loads(asyncio.run(gm.docs_create_document(title="Doc")))
    sheets_payload = json.loads(asyncio.run(gm.sheets_create_spreadsheet(title="Sheet")))
    slides_payload = json.loads(asyncio.run(gm.slides_create_presentation(title="Slides")))

    assert docs_payload["data"] == {"documentId": "doc-1", "title": "Doc"}
    assert captured["docs"]["fields"] == "documentId,title"
    assert "spreadsheetId" in captured["sheets"]["fields"]
    assert "sheets/properties" in captured["sheets"]["fields"]
    assert "presentationId" in captured["slides"]["fields"]
    assert "slides/objectId" in captured["slides"]["fields"]
    assert sheets_payload["data"]["spreadsheetId"] == "sheet-1"
    assert slides_payload["data"]["presentationId"] == "slide-1"


def test_drive_upload_file_direct_mode_uses_compact_fields(monkeypatch):
    captured = {}

    class _FakeRequest:
        def execute(self):
            return {"id": "file-1", "name": "tiny.txt", "mimeType": "text/plain"}

    class _FakeFiles:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeRequest()

    class _FakeDriveService:
        def files(self):
            return _FakeFiles()

    class _FakeClient:
        def get_service(self, api_name, api_version):
            assert (api_name, api_version) == ("drive", "v3")
            return _FakeDriveService(), False

        def is_session_cached(self):
            return False

    monkeypatch.setattr(gm, "client", _FakeClient())

    payload = json.loads(
        asyncio.run(
            gm.drive_upload_file(
                name="tiny.txt",
                content="hello",
                mime_type="text/plain",
            )
        )
    )

    assert payload["ok"] is True
    assert payload["data"]["id"] == "file-1"
    assert captured["body"] == {"name": "tiny.txt"}
    assert captured["fields"] == gm.DEFAULT_DRIVE_UPLOAD_FIELDS
    assert captured["media_body"].mimetype() == "text/plain"
    assert captured["media_body"].size() == 5


def test_drive_upload_file_resumable_mode_starts_metadata_only_session(monkeypatch):
    captured = {}

    class _FakeResponse:
        status_code = 200
        ok = True
        headers = {
            "Location": "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=session-1",
            "content-type": "application/json; charset=UTF-8",
        }
        text = ""
        content = b""

        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    class _FakeSession:
        def post(
            self,
            url,
            *,
            params=None,
            json=None,
            headers=None,
            stream=False,
            allow_redirects=True,
        ):
            assert stream is True
            assert allow_redirects is False
            captured.update(
                {
                    "url": url,
                    "params": params,
                    "json": json,
                    "headers": headers,
                }
            )
            return _FakeResponse()

    class _FakeClient:
        def get_session(self):
            return _FakeSession(), True

        def is_session_cached(self):
            return True

    monkeypatch.setattr(gm, "client", _FakeClient())

    payload = json.loads(
        asyncio.run(
            gm.drive_upload_file(
                name="large.pdf",
                content="",
                mime_type="application/pdf",
                parent_id="parent-1",
                upload_mode="resumable",
                file_size=23_600_000,
            )
        )
    )

    assert payload["ok"] is True
    assert captured["url"] == "https://www.googleapis.com/upload/drive/v3/files"
    assert captured["params"] == {
        "uploadType": "resumable",
        "fields": gm.DEFAULT_DRIVE_UPLOAD_FIELDS,
    }
    assert captured["json"] == {"name": "large.pdf", "parents": ["parent-1"]}
    assert captured["headers"]["Content-Type"] == "application/json; charset=UTF-8"
    assert captured["headers"]["X-Upload-Content-Type"] == "application/pdf"
    assert captured["headers"]["X-Upload-Content-Length"] == "23600000"

    data = payload["data"]
    assert data["upload_mode"] == "resumable"
    assert data["upload_method"] == "PUT"
    assert data["upload_url"].endswith("upload_id=session-1")
    assert data["upload_headers"] == {
        "Content-Type": "application/pdf",
        "Content-Length": 23_600_000,
    }
    assert data["chunk_size_multiple_bytes"] == gm.DRIVE_RESUMABLE_CHUNK_MULTIPLE


def test_chunked_uses_provider_sized_batches():
    chunks = gm._chunked(list(range(2501)), 1000)
    assert [len(chunk) for chunk in chunks] == [1000, 1000, 501]


def test_scope_list_normalizes_strings_and_sequences():
    assert gm._scope_list("scope.one, scope.two scope.three") == [
        "scope.one",
        "scope.two",
        "scope.three",
    ]
    assert gm._scope_list(("scope.one", "scope.two")) == ["scope.one", "scope.two"]
    assert gm._scope_list(None) == []


def test_maps_api_key_requires_header_or_env(monkeypatch):
    monkeypatch.setattr(gm, "GOOGLE_MAPS_API_KEY", "")
    token = gm.ACTIVE_REQUEST_HEADERS.set({})
    try:
        try:
            gm._require_maps_api_key()
        except ValueError as exc:
            assert "x-google-maps-api-key" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("expected missing Maps API key error")
    finally:
        gm.ACTIVE_REQUEST_HEADERS.reset(token)


def test_maps_environment_fallback_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(gm, "GOOGLE_MAPS_API_KEY", "ambient-maps-key")
    monkeypatch.setattr(gm, "MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK", True)
    token = gm.ACTIVE_REQUEST_HEADERS.set({})
    try:
        with pytest.raises(ValueError, match="x-google-maps-api-key"):
            gm._require_maps_api_key()
        configuration = json.loads(asyncio.run(gm.check_configuration()))
        assert configuration["data"]["mapsKeyReadyForCurrentRequest"] is False
    finally:
        gm.ACTIVE_REQUEST_HEADERS.reset(token)


def test_maps_legacy_requests_put_api_key_in_query(monkeypatch):
    captured = {}

    class _FakeResponse:
        status_code = 200
        ok = True
        headers = {"content-type": "application/json"}
        content = b'{"status":"OK","results":[]}'

        def json(self):
            return {"status": "OK", "results": []}

    class _FakeSession:
        def request(
            self,
            method,
            url,
            *,
            params=None,
            json=None,
            headers=None,
            stream=False,
            allow_redirects=True,
        ):
            assert stream is True
            assert allow_redirects is False
            captured.update(
                {
                    "method": method,
                    "url": url,
                    "params": params,
                    "headers": headers,
                }
            )
            return _FakeResponse()

    class _FakeClient:
        def get_session(self):
            return _FakeSession(), False

    monkeypatch.setattr(gm, "client", _FakeClient())
    monkeypatch.setattr(gm, "GOOGLE_MAPS_API_KEY", "")
    token = gm.ACTIVE_REQUEST_HEADERS.set({"x-google-maps-api-key": "maps-key"})
    try:
        result = gm._maps_request(
            "GET",
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": "New York, NY", "region": None},
            api_key_location="query",
        )
    finally:
        gm.ACTIVE_REQUEST_HEADERS.reset(token)

    assert result["json"]["status"] == "OK"
    assert captured["params"] == {"address": "New York, NY", "key": "maps-key"}
    assert "X-Goog-Api-Key" not in captured["headers"]


def test_maps_provider_status_errors_are_classified(monkeypatch):
    class _FakeResponse:
        status_code = 200
        ok = True
        headers = {"content-type": "application/json"}
        content = b'{"status":"REQUEST_DENIED","error_message":"bad key"}'

        def json(self):
            return {"status": "REQUEST_DENIED", "error_message": "bad key"}

    class _FakeSession:
        def request(
            self,
            method,
            url,
            *,
            params=None,
            json=None,
            headers=None,
            stream=False,
            allow_redirects=True,
        ):
            assert stream is True
            assert allow_redirects is False
            return _FakeResponse()

    class _FakeClient:
        def get_session(self):
            return _FakeSession(), False

    monkeypatch.setattr(gm, "client", _FakeClient())
    token = gm.ACTIVE_REQUEST_HEADERS.set({"x-google-maps-api-key": "maps-key"})
    try:
        try:
            gm._maps_request(
                "GET",
                "https://maps.googleapis.com/maps/api/geocode/json",
                api_key_location="query",
            )
        except gm.GoogleProviderError as exc:
            classified = gm._classify_error(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("expected maps provider error")
    finally:
        gm.ACTIVE_REQUEST_HEADERS.reset(token)

    assert classified["type"] == "auth_error"
    assert classified["status"] == 200
    assert "bad key" in classified["message"]


def test_analytics_metadata_is_compacted_by_default():
    metadata = {
        "name": "properties/123/metadata",
        "dimensions": [
            {
                "apiName": "city",
                "uiName": "City",
                "category": "Geography",
                "description": "City dimension " * 40,
            },
            {
                "apiName": "browser",
                "uiName": "Browser",
                "category": "Platform",
                "description": "Browser dimension",
            },
        ],
        "metrics": [
            {
                "apiName": "activeUsers",
                "uiName": "Active users",
                "category": "User",
                "type": "TYPE_INTEGER",
                "description": "Active users metric",
            }
        ],
    }

    compact = gm._compact_analytics_metadata(
        metadata,
        max_items_per_kind=1,
        query="city",
    )

    assert compact["name"] == "properties/123/metadata"
    assert compact["dimensions"]["total"] == 2
    assert compact["dimensions"]["matched"] == 1
    assert compact["dimensions"]["returned"] == 1
    assert compact["dimensions"]["items"][0]["apiName"] == "city"
    assert len(compact["dimensions"]["items"][0]["description"]) <= 220
    assert compact["metrics"]["matched"] == 0


def test_byok_refresh_uses_existing_grant_scopes(monkeypatch):
    calls = {}

    class _FakeCreds:
        valid = True

        @classmethod
        def from_authorized_user_info(cls, info, scopes=None):
            calls["scopes"] = scopes
            return cls()

    monkeypatch.setattr(gm, "Credentials", _FakeCreds)
    workspace = gm.GoogleWorkspaceClient(
        credentials_path=None,
        token_path=None,
        scopes=["https://www.googleapis.com/auth/youtube.readonly"],
        authorized_user_info={
            "client_id": "cid",
            "client_secret": "csecret",
            "refresh_token": "rtok",
            "token_uri": gm.GOOGLE_TOKEN_URI,
            "type": "authorized_user",
        },
        persist_token=False,
    )

    workspace._load_credentials()

    assert calls["scopes"] is None


def test_token_file_refresh_uses_existing_grant_scopes(monkeypatch, tmp_path):
    calls = {}
    token_path = tmp_path / "token.json"
    token_path.write_text(
        json.dumps(
            {
                "client_id": "cid",
                "client_secret": "csecret",
                "refresh_token": "rtok",
                "token_uri": gm.GOOGLE_TOKEN_URI,
                "type": "authorized_user",
                "scopes": ["https://www.googleapis.com/auth/gmail.modify"],
            }
        ),
        encoding="utf-8",
    )

    class _FakeCreds:
        valid = True

        @classmethod
        def from_authorized_user_file(cls, filename, scopes=None):
            calls["filename"] = filename
            calls["scopes"] = scopes
            return cls()

    monkeypatch.setattr(gm, "Credentials", _FakeCreds)
    workspace = gm.GoogleWorkspaceClient(
        credentials_path=None,
        token_path=str(token_path),
        scopes=["https://www.googleapis.com/auth/youtube.readonly"],
    )

    workspace._load_credentials()

    assert calls["filename"] == str(token_path)
    assert calls["scopes"] is None


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
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "drive_list_files", "arguments": {}},
            }
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
            (b"host", b"localhost"),
            (b"content-type", b"application/json"),
            (b"accept", b"application/json"),
            (b"x-madpanda-portal-grant", TEST_PORTAL_GRANT_TOKEN.encode("ascii")),
            (b"x-google-client-id", client_id.encode("utf-8")),
            (b"x-google-client-secret", b"secret"),
            (b"x-google-refresh-token", b"refresh"),
        ]
        scope = {"type": "http", "path": "/mcp", "method": "POST", "headers": headers}
        await app(scope, _receive, _send)
        raw_body = b"".join(
            m.get("body", b"") for m in messages if m.get("type") == "http.response.body"
        )
        return json.loads(raw_body.decode("utf-8"))["active_client"]

    monkeypatch.setattr(gm, "_resolve_request_client", _fake_resolve)
    wrapped = gm.build_hosted_mcp_http_wrapper(_echo_app)

    async def _run():
        return await asyncio.gather(_call(wrapped, "alpha"), _call(wrapped, "beta"))

    first, second = asyncio.run(_run())
    assert first == "alpha"
    assert second == "beta"


def test_raw_request_unconditionally_blocks_non_google_hosts():
    try:
        gm._validate_raw_request_url("https://example.com/drive/v3/files")
    except ValueError as exc:
        assert "host is not allowed" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected strict raw host validation error")


def test_raw_request_requires_https_and_rejects_embedded_credentials():
    with pytest.raises(ValueError, match="requires HTTPS"):
        gm._validate_raw_request_url("http://www.googleapis.com/drive/v3/files")
    with pytest.raises(ValueError, match="user information"):
        gm._validate_raw_request_url("https://user:password@www.googleapis.com/drive/v3/files")
    with pytest.raises(ValueError, match="standard HTTPS port"):
        gm._validate_raw_request_url("https://www.googleapis.com:8443/drive/v3/files")
    with pytest.raises(ValueError, match="credential-bearing keys"):
        gm._validate_raw_request_url(
            "https://www.googleapis.com/drive/v3/files?access%5Ftoken=must-not-pass"
        )
    with pytest.raises(ValueError, match="credential-bearing keys"):
        gm._validate_raw_request_url("/drive/v3/files?key=must-not-pass")


def test_raw_request_is_read_only_and_does_not_return_response_headers(monkeypatch):
    class _FakeResponse:
        status_code = 200
        ok = True
        headers = {
            "content-type": "application/json; charset=UTF-8",
            "set-cookie": "must-not-leak",
        }
        text = '{"files":[]}'
        content = b'{"files":[]}'

        def json(self):
            return {"files": []}

    class _FakeSession:
        def request(
            self,
            method,
            url,
            *,
            params=None,
            stream=False,
            allow_redirects=True,
        ):
            assert method == "GET"
            assert stream is True
            assert allow_redirects is False
            return _FakeResponse()

    class _FakeClient:
        def get_session(self):
            return _FakeSession(), True

        def is_session_cached(self):
            return True

    monkeypatch.setattr(gm, "client", _FakeClient())

    payload = json.loads(
        asyncio.run(
            gm.google_raw_request(
                method="GET",
                url="https://www.googleapis.com/drive/v3/files",
                params={"fields": "files(id,name)"},
            )
        )
    )

    assert payload["ok"] is True
    assert payload["data"]["status"] == 200
    assert payload["data"]["json"] == {"files": []}
    assert "headers" not in payload["data"]
