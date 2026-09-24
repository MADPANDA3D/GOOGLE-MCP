"""Production entrypoint with isolated Gmail reply-thread repair."""

from __future__ import annotations

import os
from email.message import EmailMessage
from typing import Any

import uvicorn
from starlette.middleware.trustedhost import TrustedHostMiddleware

import google_mcp_server as server


def _header_map(message: dict[str, Any]) -> dict[str, str]:
    payload = message.get("payload")
    headers = payload.get("headers", []) if isinstance(payload, dict) else []
    return {
        str(header.get("name", "")).lower(): str(header.get("value", "")).strip()
        for header in headers
        if isinstance(header, dict) and header.get("name")
    }


def _reply_headers(service: Any, thread_id: str) -> tuple[str, str, str]:
    thread = (
        service.users()
        .threads()
        .get(
            userId="me",
            id=thread_id,
            format="metadata",
            metadataHeaders=["Subject", "Message-ID", "References"],
        )
        .execute()
    )
    messages = thread.get("messages", []) if isinstance(thread, dict) else []
    if not messages:
        raise ValueError("thread_id did not resolve to a Gmail thread with messages")

    headers = _header_map(messages[-1])
    subject = headers.get("subject", "")
    message_id = headers.get("message-id", "")
    if not subject or not message_id:
        raise ValueError(
            "thread_id did not resolve to a message with Subject and Message-ID headers"
        )
    references = " ".join(
        value for value in (headers.get("references", ""), message_id) if value
    )
    return subject, message_id, references


def _add_reply_headers(
    message: EmailMessage,
    *,
    subject: str,
    message_id: str,
    references: str,
) -> None:
    message.replace_header("Subject", subject)
    message["In-Reply-To"] = message_id
    message["References"] = references


async def gmail_send_message(
    to: str,
    subject: str,
    body: str,
    cc: str = "",
    bcc: str = "",
    reply_to: str = "",
    from_alias: str = "",
    thread_id: str = "",
    is_html: bool = False,
    expected_signature_fingerprint: str = "",
) -> str:
    """Send a Gmail message, including RFC 2822 linkage for thread replies."""

    def _send():
        if not to:
            raise ValueError("to cannot be empty")
        if not subject:
            raise ValueError("subject cannot be empty")
        if body is None:
            raise ValueError("body cannot be empty")

        service, cached = server.client.get_service("gmail", "v1")
        signature = server._resolve_gmail_send_as_signature(service, from_alias)
        server._verify_gmail_signature_fingerprint(
            signature,
            expected_signature_fingerprint,
        )
        message = server.build_email_message(
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
            reply_to=reply_to,
            from_alias=from_alias,
            is_html=is_html,
            signature=signature,
        )
        payload: dict[str, Any] = {}
        if thread_id:
            reply_subject, parent_message_id, references = _reply_headers(
                service, thread_id
            )
            _add_reply_headers(
                message,
                subject=reply_subject,
                message_id=parent_message_id,
                references=references,
            )
            payload["threadId"] = thread_id
        payload["raw"] = server.encode_email_message(message)
        request = service.users().messages().send(userId="me", body=payload)
        return request.execute(), {
            "cached_service": cached,
            "signature_present": bool(signature.html),
            "signature_alias": signature.alias,
            "signature_fingerprint": signature.fingerprint,
        }

    return await server.run_tool("gmail", "send_message", _send, allow_retry=False)


def install_gmail_reply_repair() -> None:
    tool = server._tool_registry().get("gmail_send_message")
    if tool is None or not hasattr(tool, "fn"):
        raise RuntimeError("gmail_send_message is missing from the FastMCP tool registry")
    tool.fn = gmail_send_message
    server.gmail_send_message = gmail_send_message


def build_app():
    install_gmail_reply_repair()
    app_factory = server.mcp.streamable_http_app
    app = app_factory() if callable(app_factory) else app_factory
    try:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    except Exception:
        pass
    return server.build_hosted_mcp_http_wrapper(app)


if __name__ == "__main__":
    os.environ.setdefault("HOST", server.MCP_BIND_ADDRESS)
    os.environ.setdefault("PORT", str(server.MCP_HTTP_PORT))
    uvicorn.run(
        build_app,
        host=server.MCP_BIND_ADDRESS,
        port=server.MCP_HTTP_PORT,
        factory=True,
        workers=server.MCP_WORKERS,
    )
