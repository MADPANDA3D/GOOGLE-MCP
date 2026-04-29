#!/usr/bin/env python3
"""Live Google MCP certification harness.

Defaults are read-only. Disposable write checks run only with --include-writes.
Email send checks are never automatic; they require --send-test-to.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def load_env_file(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_google_oauth_values() -> tuple[str, str, str]:
    client_id = os.getenv("GOOGLE_CLIENT_ID") or os.getenv("X_GOOGLE_CLIENT_ID", "")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET") or os.getenv("X_GOOGLE_CLIENT_SECRET", "")
    refresh_token = os.getenv("GOOGLE_REFRESH_TOKEN") or os.getenv("X_GOOGLE_REFRESH_TOKEN", "")
    credentials_path = Path(os.getenv("GOOGLE_CREDENTIALS_PATH", "fastmcp/.google/credentials.json"))
    token_path = Path(os.getenv("GOOGLE_TOKEN_PATH", "fastmcp/.google/token.json"))
    if credentials_path.exists() and (not client_id or not client_secret):
        credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
        installed = credentials.get("installed") or credentials.get("web") or {}
        client_id = client_id or installed.get("client_id", "")
        client_secret = client_secret or installed.get("client_secret", "")
    if token_path.exists() and not refresh_token:
        token = json.loads(token_path.read_text(encoding="utf-8"))
        refresh_token = token.get("refresh_token", "")
        client_id = client_id or token.get("client_id", "")
        client_secret = client_secret or token.get("client_secret", "")
    return client_id, client_secret, refresh_token


class McpClient:
    def __init__(self, url: str, headers: dict[str, str]):
        self.url = url
        self.headers = headers
        self._next_id = 1

    def call(self, tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments or {}},
        }
        self._next_id += 1
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **self.headers,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            return {"transport_ok": False, "status": exc.code, "raw": raw}
        decoded = json.loads(raw)
        result_text = decoded.get("result", {}).get("content", [{}])[0].get("text")
        if result_text:
            try:
                decoded["tool_result"] = json.loads(result_text)
            except json.JSONDecodeError:
                decoded["tool_result"] = result_text
        return {"transport_ok": True, "status": 200, "raw": decoded}


def result_ok(response: dict[str, Any]) -> bool:
    if not response.get("transport_ok"):
        return False
    payload = response.get("raw", {})
    if "error" in payload:
        return False
    tool_result = payload.get("tool_result")
    if isinstance(tool_result, dict) and tool_result.get("ok") is False:
        return False
    return True


def record(results: list[dict[str, Any]], name: str, response: dict[str, Any], **extra: Any) -> None:
    results.append(
        {
            "name": name,
            "status": "passed" if result_ok(response) else "failed",
            **extra,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default="fastmcp/.env")
    parser.add_argument("--url", default=os.getenv("GOOGLE_MCP_URL", "http://127.0.0.1:8086/mcp"))
    parser.add_argument("--include-writes", action="store_true")
    parser.add_argument("--send-test-to", default="")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    load_env_file(args.env_file)
    grant = os.getenv("MCP_PORTAL_GRANT_TOKEN") or os.getenv("GOOGLE_MCP_PORTAL_GRANT", "")
    client_id, client_secret, refresh_token = load_google_oauth_values()
    if not all((grant, client_id, client_secret, refresh_token)):
        print(
            "Missing MCP_PORTAL_GRANT_TOKEN plus GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET/GOOGLE_REFRESH_TOKEN.",
            file=sys.stderr,
        )
        return 2

    client = McpClient(
        args.url,
        {
            "X-MADPANDA-PORTAL-GRANT": grant,
            "X-Google-Client-Id": client_id,
            "X-Google-Client-Secret": client_secret,
            "X-Google-Refresh-Token": refresh_token,
        },
    )
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    prefix = f"MAD_AUDIT_DELETE_ME_{stamp}_{int(time.time())}"
    results: list[dict[str, Any]] = []
    created_drive_ids: list[str] = []
    created_calendar_id = ""
    created_event_id = ""
    created_label_id = ""
    created_draft_id = ""

    for name, tool, arguments in [
        ("welcome", "google_mcp_welcome", {}),
        ("capabilities", "google_mcp_list_capabilities", {}),
        ("health", "mcp_health_check", {"run_checks": False, "warm_all": False}),
        ("gmail_labels", "gmail_list_labels", {}),
        ("gmail_overview", "gmail_mailbox_overview", {"queries": ["is:unread"], "include_labels": False}),
        ("gmail_clusters", "gmail_sender_clusters", {"query": "is:unread", "max_messages": 25, "top_n": 10}),
        ("calendar_list", "calendar_list_calendars", {}),
        ("calendar_events", "calendar_list_events", {"calendarId": "primary", "maxResults": 5}),
        ("drive_list", "drive_list_files", {"pageSize": 5}),
        ("raw_guardrail", "google_raw_request", {"method": "GET", "url": "https://example.com/blocked"}),
    ]:
        response = client.call(tool, arguments)
        if name == "raw_guardrail":
            passed = not result_ok(response)
            results.append({"name": name, "status": "passed" if passed else "failed"})
        else:
            record(results, name, response)

    if args.include_writes:
        doc = client.call("docs_create_document", {"title": f"{prefix}_doc"})
        record(results, "docs_create_document", doc)
        doc_id = doc.get("raw", {}).get("tool_result", {}).get("data", {}).get("documentId", "")
        if doc_id:
            created_drive_ids.append(doc_id)
            record(results, "docs_get_document", client.call("docs_get_document", {"documentId": doc_id}))

        sheet = client.call("sheets_create_spreadsheet", {"title": f"{prefix}_sheet"})
        record(results, "sheets_create_spreadsheet", sheet)
        sheet_id = sheet.get("raw", {}).get("tool_result", {}).get("data", {}).get("spreadsheetId", "")
        if sheet_id:
            created_drive_ids.append(sheet_id)
            record(
                results,
                "sheets_update_values",
                client.call(
                    "sheets_update_values",
                    {"spreadsheetId": sheet_id, "rangeA1": "Sheet1!A1:B2", "values": [["audit", "ok"]]},
                ),
            )

        slide = client.call("slides_create_presentation", {"title": f"{prefix}_slides"})
        record(results, "slides_create_presentation", slide)
        slide_id = slide.get("raw", {}).get("tool_result", {}).get("data", {}).get("presentationId", "")
        if slide_id:
            created_drive_ids.append(slide_id)
            record(results, "slides_get_presentation", client.call("slides_get_presentation", {"presentationId": slide_id}))

        label = client.call("gmail_create_label", {"name": f"{prefix}_label"})
        record(results, "gmail_create_label", label)
        created_label_id = label.get("raw", {}).get("tool_result", {}).get("data", {}).get("id", "")

        draft = client.call(
            "gmail_create_draft",
            {"to": args.send_test_to or "nobody@example.invalid", "subject": f"{prefix}_draft", "body": "audit draft"},
        )
        record(results, "gmail_create_draft", draft)
        created_draft_id = draft.get("raw", {}).get("tool_result", {}).get("data", {}).get("id", "")

        start = datetime.now(UTC) + timedelta(days=30)
        end = start + timedelta(minutes=15)
        cal = client.call("calendar_create_calendar", {"summary": f"{prefix}_calendar"})
        record(results, "calendar_create_calendar", cal)
        created_calendar_id = cal.get("raw", {}).get("tool_result", {}).get("data", {}).get("id", "")
        if created_calendar_id:
            event = client.call(
                "calendar_create_event",
                {
                    "calendarId": created_calendar_id,
                    "summary": f"{prefix}_event",
                    "startIso": start.isoformat(),
                    "endIso": end.isoformat(),
                    "timeZone": "UTC",
                },
            )
            record(results, "calendar_create_event", event)
            created_event_id = event.get("raw", {}).get("tool_result", {}).get("data", {}).get("id", "")

    if args.send_test_to and created_draft_id:
        record(results, "gmail_send_draft", client.call("gmail_send_draft", {"draftId": created_draft_id}))

    if created_event_id and created_calendar_id:
        record(
            results,
            "calendar_delete_event",
            client.call(
                "calendar_delete_event",
                {"calendarId": created_calendar_id, "eventId": created_event_id, "sendUpdates": "none", "confirm": True},
            ),
        )
    if created_calendar_id:
        record(results, "calendar_delete_calendar", client.call("calendar_delete_calendar", {"calendarId": created_calendar_id, "confirm": True}))
    if created_draft_id and not args.send_test_to:
        record(results, "gmail_delete_draft", client.call("gmail_delete_draft", {"draftId": created_draft_id, "confirm": True}))
    if created_label_id:
        record(results, "gmail_delete_label", client.call("gmail_delete_label", {"labelId": created_label_id, "confirm": True}))
    for file_id in created_drive_ids:
        record(results, "drive_delete_file", client.call("drive_delete_file", {"fileId": file_id, "mode": "permanent", "confirm": True}))

    report = {
        "ok": all(item["status"] == "passed" for item in results),
        "prefix": prefix,
        "include_writes": args.include_writes,
        "send_test": bool(args.send_test_to),
        "results": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
