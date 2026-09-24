import asyncio
import json
import os

os.environ.setdefault("MCP_PORTAL_GRANT_TOKEN", "test-portal-grant")

import google_mcp_entrypoint as entrypoint
import google_mcp_server as gm


def test_gmail_reply_adds_provider_thread_linkage(monkeypatch):
    captured = {}

    class Request:
        def __init__(self, data):
            self.data = data

        def execute(self):
            return self.data

    class Service:
        def users(self):
            return self

        def settings(self):
            return self

        def sendAs(self):
            return self

        def list(self, **kwargs):
            return Request(
                {
                    "sendAs": [
                        {
                            "sendAsEmail": "sender@example.com",
                            "signature": "<p>Regards</p>",
                            "isDefault": True,
                        }
                    ]
                }
            )

        def threads(self):
            return self

        def get(self, **kwargs):
            assert kwargs == {
                "userId": "me",
                "id": "thread-1",
                "format": "metadata",
                "metadataHeaders": ["Subject", "Message-ID", "References"],
            }
            return Request(
                {
                    "messages": [
                        {
                            "payload": {
                                "headers": [
                                    {"name": "Subject", "value": "Original subject"},
                                    {"name": "Message-ID", "value": "<parent@example.com>"},
                                    {"name": "References", "value": "<root@example.com>"},
                                ]
                            }
                        }
                    ]
                }
            )

        def messages(self):
            return self

        def send(self, **kwargs):
            captured.update(kwargs)
            return Request({"id": "reply-1", "threadId": "thread-1"})

    monkeypatch.setattr(
        gm.GoogleWorkspaceClient,
        "get_service",
        lambda *_args: (Service(), False),
    )

    result = json.loads(
        asyncio.run(
            entrypoint.gmail_send_message(
                to="recipient@example.com",
                subject="Caller-provided subject",
                body="Reply body",
                thread_id="thread-1",
            )
        )
    )
    message = gm._decode_email_message(captured["body"]["raw"])

    assert captured["body"]["threadId"] == "thread-1"
    assert message["Subject"] == "Original subject"
    assert message["In-Reply-To"] == "<parent@example.com>"
    assert message["References"] == "<root@example.com> <parent@example.com>"
    assert result["ok"] is True


def test_gmail_non_reply_does_not_fetch_thread(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("thread metadata must not be fetched without thread_id")

    monkeypatch.setattr(entrypoint, "_reply_headers", fail)
    message = gm.build_email_message(
        to="recipient@example.com",
        subject="New message",
        body="Body",
    )
    assert "In-Reply-To" not in message
    assert "References" not in message


def test_install_replaces_registered_callable():
    original = gm._tool_registry()["gmail_send_message"].fn
    original_function = gm.gmail_send_message
    try:
        entrypoint.install_gmail_reply_repair()
        assert gm._tool_registry()["gmail_send_message"].fn is entrypoint.gmail_send_message
    finally:
        gm._tool_registry()["gmail_send_message"].fn = original
        gm.gmail_send_message = original_function
