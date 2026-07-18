#!/usr/bin/env python3
"""Provider-free container smoke for both authenticated access modes."""

from __future__ import annotations

import http.client
import json
import os
from typing import Any

HOST = "127.0.0.1"
PORT = int(os.getenv("MCP_HTTP_PORT", "8086"))
MODE = os.environ["MCP_MODE"]
EXPECTED_TOOL_COUNT = int(os.getenv("MCP_EXPECTED_TOOL_COUNT", "151"))
EXPECTED_BUILD_SHA = os.environ["MCP_BUILD_SHA"]
EXPECTED_SOURCE_FINGERPRINT = os.environ["MCP_SOURCE_FINGERPRINT"]
EXPECTED_IMAGE_REFERENCE = os.environ["MCP_IMAGE_REFERENCE"]
ACCESS_TOKEN = os.getenv("MCP_ACCESS_TOKEN", "")
PORTAL_GRANT = os.getenv("MCP_PORTAL_GRANT_TOKEN", "")


def auth_headers(*, valid: bool = True) -> dict[str, str]:
    if MODE == "standalone":
        token = ACCESS_TOKEN if valid else "wrong-standalone-token-000000000000"
        return {"Authorization": f"Bearer {token}"}
    if MODE == "portal":
        token = PORTAL_GRANT if valid else "wrong-portal-grant-0000000000000000"
        return {"X-MADPANDA-PORTAL-GRANT": token}
    raise AssertionError(f"unexpected MCP_MODE={MODE!r}")


def request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], Any]:
    body: bytes | None
    if isinstance(payload, dict):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    else:
        body = payload
    merged = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if headers:
        merged.update(headers)
    connection = http.client.HTTPConnection(HOST, PORT, timeout=8)
    try:
        connection.request(method, path, body=body, headers=merged)
        response = connection.getresponse()
        raw = response.read()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
    finally:
        connection.close()
    try:
        decoded: Any = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        decoded = raw.decode("utf-8", errors="replace")
    return response.status, response_headers, decoded


def rpc(method: str, request_id: int, params: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    status, _, health = request("GET", "/health", headers={"Accept": "application/json"})
    require(status == 200, f"health status={status}")
    require(isinstance(health, dict), "health is not JSON")
    require(health.get("status") == "healthy", f"health={health}")
    require(health.get("tool_count") == EXPECTED_TOOL_COUNT, f"tool_count={health}")
    require(health.get("build_sha") == EXPECTED_BUILD_SHA, f"build_sha={health}")
    require(
        health.get("source_fingerprint") == EXPECTED_SOURCE_FINGERPRINT,
        f"source_fingerprint={health}",
    )
    require(
        health.get("image_reference") == EXPECTED_IMAGE_REFERENCE,
        f"image_reference={health}",
    )
    require(health.get("configuration", {}).get("mode") == MODE, f"mode={health}")
    require(health.get("configuration", {}).get("ready") is True, f"not ready: {health}")
    require(
        health.get("configuration", {}).get("provider_credentials_mode") == "per_request_byok",
        "BYOK mode missing",
    )

    status, _, denied = request(
        "POST",
        "/mcp",
        payload=b"malformed-before-auth",
        headers={},
    )
    require(status == 401, f"missing auth was not rejected first: {status} {denied}")
    require(denied.get("error", {}).get("code") == -32001, f"missing auth={denied}")

    status, _, denied = request(
        "POST",
        "/mcp",
        payload=b"malformed-before-auth",
        headers=auth_headers(valid=False),
    )
    require(status == 401, f"invalid auth was not rejected first: {status} {denied}")
    require(denied.get("error", {}).get("code") == -32001, f"invalid auth={denied}")

    origin_headers = auth_headers()
    origin_headers["Origin"] = "https://untrusted.invalid"
    status, _, denied = request(
        "POST",
        "/mcp",
        payload=rpc("tools/list", 2, {}),
        headers=origin_headers,
    )
    require(status == 403, f"browser Origin was not rejected: {status} {denied}")
    require(denied.get("error") == "origin_not_allowed", f"origin rejection={denied}")

    status, response_headers, initialized = request(
        "POST",
        "/mcp",
        payload=rpc(
            "initialize",
            3,
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "google-mcp-image-smoke", "version": "1"},
            },
        ),
        headers=auth_headers(),
    )
    require(status == 200, f"initialize failed: {status} {initialized}")
    require(isinstance(initialized, dict) and "result" in initialized, f"initialize={initialized}")

    discovery_headers = auth_headers()
    session_id = response_headers.get("mcp-session-id")
    if session_id:
        discovery_headers["Mcp-Session-Id"] = session_id
    status, _, tools = request(
        "POST",
        "/mcp",
        payload=rpc("tools/list", 4, {}),
        headers=discovery_headers,
    )
    require(status == 200, f"tools/list failed: {status} {tools}")
    listed = tools.get("result", {}).get("tools", []) if isinstance(tools, dict) else []
    require(len(listed) == EXPECTED_TOOL_COUNT, f"tools/list count={len(listed)}")
    names = {tool.get("name") for tool in listed if isinstance(tool, dict)}
    require("list_capabilities" in names, "standard navigation is missing")
    require("drive_list_files" in names, "Google provider tool is missing")

    status, _, capability = request(
        "POST",
        "/mcp",
        payload=rpc(
            "tools/call",
            5,
            {"name": "list_capabilities", "arguments": {"include_descriptors": False}},
        ),
        headers=discovery_headers,
    )
    require(status == 200, f"local navigation failed: {status} {capability}")
    require(not capability.get("result", {}).get("isError", False), f"navigation={capability}")

    status, _, provider_denied = request(
        "POST",
        "/mcp",
        payload=rpc("tools/call", 6, {"name": "drive_list_files", "arguments": {}}),
        headers=discovery_headers,
    )
    require(status == 401, f"missing BYOK status={status} payload={provider_denied}")
    require(
        provider_denied.get("error", {}).get("code") == -32001,
        f"missing BYOK error={provider_denied}",
    )
    required_headers = set(
        provider_denied.get("error", {}).get("data", {}).get("required_headers", [])
    )
    require("x-google-client-id" in required_headers, f"missing BYOK headers={provider_denied}")

    print(json.dumps({"ok": True, "mode": MODE, "tool_count": len(listed)}))


if __name__ == "__main__":
    main()
