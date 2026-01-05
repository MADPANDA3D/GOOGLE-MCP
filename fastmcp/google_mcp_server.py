import asyncio
import base64
import io
import json
import logging
import os
import random
import threading
import time
import urllib.parse
from email.message import EmailMessage
from typing import Any, Callable

from google.auth.transport.requests import AuthorizedSession, Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaInMemoryUpload
from mcp.server.fastmcp import FastMCP
import uvicorn
from starlette.middleware.trustedhost import TrustedHostMiddleware


DEFAULT_SCOPES = (
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/presentations",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
)
DEFAULT_DRIVE_FIELDS = "files(id,name,mimeType,modifiedTime,size),nextPageToken"
DEFAULT_DRIVE_GET_FIELDS = "id,name,mimeType,modifiedTime,size,parents"
DEFAULT_DOCS_FIELDS = "documentId,title"
DEFAULT_SHEETS_FIELDS = "spreadsheetId,properties.title,sheets.properties"
DEFAULT_SLIDES_FIELDS = "presentationId,title,slides(objectId)"
DEFAULT_CALENDAR_LIST_FIELDS = "items(id,summary,timeZone,accessRole),nextPageToken"
DEFAULT_CALENDAR_FIELDS = "id,summary,description,location,timeZone,accessRole"
DEFAULT_EVENT_FIELDS = "id,summary,description,location,start,end,status,updated"
DEFAULT_EVENT_LIST_FIELDS = "items(id,summary,start,end,status,updated),nextPageToken"
DEFAULT_GMAIL_METADATA_HEADERS = (
    "From",
    "To",
    "Cc",
    "Bcc",
    "Subject",
    "Date",
    "Message-ID",
)

MCP_HTTP_PORT = int(os.getenv("MCP_HTTP_PORT", "8086"))
MCP_BIND_ADDRESS = os.getenv("MCP_BIND_ADDRESS", "0.0.0.0")
GOOGLE_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_CREDENTIALS_PATH", "fastmcp/.google/credentials.json"
)
GOOGLE_TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", "fastmcp/.google/token.json")
GOOGLE_SCOPES_RAW = os.getenv("GOOGLE_SCOPES", " ".join(DEFAULT_SCOPES))
MCP_PRETTY_JSON = os.getenv("MCP_PRETTY_JSON", "").lower() in {"1", "true", "yes"}
MCP_RESPONSE_ENVELOPE = os.getenv("MCP_RESPONSE_ENVELOPE", "true").lower() in {
    "1",
    "true",
    "yes",
}
MCP_LOG_REQUESTS = os.getenv("MCP_LOG_REQUESTS", "").lower() in {"1", "true", "yes"}
MCP_LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "INFO")
MCP_RETRY_MAX = int(os.getenv("MCP_RETRY_MAX", "2"))
MCP_RETRY_BASE_SECONDS = float(os.getenv("MCP_RETRY_BASE_SECONDS", "0.5"))
MCP_RETRY_MAX_SECONDS = float(os.getenv("MCP_RETRY_MAX_SECONDS", "4.0"))
MCP_REQUIRE_CONFIRM = os.getenv("MCP_REQUIRE_CONFIRM", "").lower() in {
    "1",
    "true",
    "yes",
}
MCP_DRIVE_ALLOWLIST_PARENT_ID = os.getenv("MCP_DRIVE_ALLOWLIST_PARENT_ID", "")
DEFAULT_MAX_DOWNLOAD_BYTES = int(os.getenv("MCP_MAX_DOWNLOAD_BYTES", "5000000"))


mcp = FastMCP(
    name="google-mcp",
    stateless_http=True,
    json_response=True,
    host="0.0.0.0",
)

logger = logging.getLogger("google_mcp")
if MCP_LOG_REQUESTS:
    logging.basicConfig(
        level=getattr(logging, MCP_LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def parse_scopes(raw: str) -> list[str]:
    if not raw:
        return []
    cleaned = raw.replace(",", " ")
    return [scope.strip() for scope in cleaned.split() if scope.strip()]


class GoogleWorkspaceClient:
    def __init__(self, credentials_path: str, token_path: str, scopes: list[str]):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.scopes = scopes
        self._lock = threading.Lock()
        self._creds: Credentials | None = None
        self._service_cache: dict[tuple[str, str], Any] = {}
        self._session: AuthorizedSession | None = None

    def _save_token(self, creds: Credentials) -> None:
        token_dir = os.path.dirname(self.token_path)
        if token_dir:
            os.makedirs(token_dir, exist_ok=True)
        with open(self.token_path, "w", encoding="utf-8") as handle:
            handle.write(creds.to_json())

    def _load_credentials(self) -> Credentials:
        with self._lock:
            if self._creds is None:
                if not os.path.exists(self.token_path):
                    raise FileNotFoundError(
                        "Missing token.json. Run fastmcp/google_auth_local.py locally and copy it to the server."
                    )
                self._creds = Credentials.from_authorized_user_file(
                    self.token_path, self.scopes
                )
            creds = self._creds
            if not creds.valid:
                if creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                    self._save_token(creds)
                else:
                    raise RuntimeError(
                        "Token is invalid or expired without refresh token. Re-run the local auth flow."
                    )
            return creds

    def get_service(self, api_name: str, api_version: str) -> tuple[Any, bool]:
        creds = self._load_credentials()
        cache_key = (api_name, api_version, tuple(self.scopes))
        with self._lock:
            if cache_key in self._service_cache:
                return self._service_cache[cache_key], True
            service = build(api_name, api_version, credentials=creds, cache_discovery=False)
            self._service_cache[cache_key] = service
            return service, False

    def build_service(self, api_name: str, api_version: str):
        service, _ = self.get_service(api_name, api_version)
        return service

    def get_session(self) -> tuple[AuthorizedSession, bool]:
        creds = self._load_credentials()
        with self._lock:
            if self._session is None:
                self._session = AuthorizedSession(creds)
                return self._session, False
            return self._session, True

    def authed_session(self) -> AuthorizedSession:
        session, _ = self.get_session()
        return session


SCOPES = parse_scopes(GOOGLE_SCOPES_RAW)
if not SCOPES:
    raise RuntimeError("GOOGLE_SCOPES is not set")

client = GoogleWorkspaceClient(
    credentials_path=GOOGLE_CREDENTIALS_PATH,
    token_path=GOOGLE_TOKEN_PATH,
    scopes=SCOPES,
)


def normalize_url(url: str) -> str:
    if not url:
        raise ValueError("url cannot be empty")
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if not url.startswith("/"):
        url = "/" + url
    return f"https://www.googleapis.com{url}"


def json_dumps(data: Any) -> str:
    if MCP_PRETTY_JSON:
        return json.dumps(data, indent=2, sort_keys=True)
    return json.dumps(data, separators=(",", ":"), ensure_ascii=True)


def _retry_after_seconds(headers: dict[str, Any] | None) -> float | None:
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _classify_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return {
            "type": "invalid_params",
            "message": str(exc),
            "action": "Verify inputs and try again.",
        }
    if isinstance(exc, RefreshError):
        return {
            "type": "auth_error",
            "message": str(exc),
            "action": "Re-run the OAuth flow and refresh token.json.",
        }
    if isinstance(exc, HttpError):
        status = getattr(exc, "status_code", None)
        if status is None and getattr(exc, "resp", None) is not None:
            status = getattr(exc.resp, "status", None)
        headers = dict(getattr(exc, "resp", {}) or {})
        retry_after = _retry_after_seconds(headers)
        content = exc.content
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        error_type = "api_error"
        action = "Review the Google API error and adjust the request."
        if status in {401, 403}:
            error_type = "auth_error"
            action = "Verify OAuth scopes and credentials."
        elif status == 404:
            error_type = "not_found"
            action = "Confirm the resource ID exists and is accessible."
        elif status == 429:
            error_type = "rate_limited"
            action = "Retry later or reduce request rate."
        elif status and status >= 500:
            error_type = "upstream_error"
            action = "Retry after a short delay."
        return {
            "type": error_type,
            "message": str(exc),
            "status": status,
            "details": content,
            "retry_after": retry_after,
            "action": action,
        }
    return {
        "type": "unknown_error",
        "message": str(exc),
        "action": "Check server logs for details.",
    }


def _is_retryable(exc: Exception) -> bool:
    if not isinstance(exc, HttpError):
        return False
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "resp", None) is not None:
        status = getattr(exc.resp, "status", None)
    return status in {429, 500, 502, 503, 504}


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    if isinstance(exc, HttpError):
        headers = dict(getattr(exc, "resp", {}) or {})
        retry_after = _retry_after_seconds(headers)
        if retry_after is not None:
            return min(retry_after, MCP_RETRY_MAX_SECONDS)
    base = MCP_RETRY_BASE_SECONDS * (2 ** max(attempt - 1, 0))
    jitter = random.uniform(0, MCP_RETRY_BASE_SECONDS)
    return min(base + jitter, MCP_RETRY_MAX_SECONDS)


def _response_payload(
    ok: bool,
    data: Any,
    error: dict[str, Any] | None,
    meta: dict[str, Any],
) -> str:
    payload = {
        "ok": ok,
        "data": data if ok else None,
        "error": error if not ok else None,
        "meta": meta,
    }
    raw = json_dumps(payload)
    payload["meta"]["bytes_out"] = len(raw.encode("utf-8"))
    return json_dumps(payload)


async def run_tool(
    api: str,
    action: str,
    func: Callable[[], Any],
    *,
    allow_retry: bool = True,
    meta_extra: dict[str, Any] | None = None,
) -> str:
    start = time.perf_counter()
    retries = 0
    last_error: dict[str, Any] | None = None
    while True:
        try:
            result = await run_blocking(func)
            result_meta: dict[str, Any] = {}
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[1], dict)
            ):
                result, result_meta = result
            meta = {
                "api": api,
                "action": action,
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
                "retry_count": retries,
            }
            if meta_extra:
                meta.update(meta_extra)
            if result_meta:
                meta.update(result_meta)
            if MCP_LOG_REQUESTS:
                logger.info(
                    "tool_ok api=%s action=%s elapsed_ms=%s retries=%s",
                    api,
                    action,
                    meta["elapsed_ms"],
                    retries,
                )
            if MCP_RESPONSE_ENVELOPE:
                return _response_payload(True, result, None, meta)
            return json_dumps(result)
        except Exception as exc:
            last_error = _classify_error(exc)
            if allow_retry and retries < MCP_RETRY_MAX and _is_retryable(exc):
                delay = _retry_delay_seconds(exc, retries + 1)
                retries += 1
                await asyncio.sleep(delay)
                continue
            meta = {
                "api": api,
                "action": action,
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
                "retry_count": retries,
            }
            if meta_extra:
                meta.update(meta_extra)
            if MCP_LOG_REQUESTS:
                logger.warning(
                    "tool_error api=%s action=%s retries=%s error=%s",
                    api,
                    action,
                    retries,
                    last_error,
                )
            if MCP_RESPONSE_ENVELOPE:
                return _response_payload(False, None, last_error, meta)
            return json_dumps({"error": last_error, "meta": meta})


def _ensure_confirmed(action: str, confirm: bool) -> None:
    if MCP_REQUIRE_CONFIRM and not confirm:
        raise ValueError(f"confirm=true is required to {action}.")


def _enforce_drive_allowlist(parent_id: str, allow_any_parent: bool) -> str:
    if not MCP_DRIVE_ALLOWLIST_PARENT_ID:
        return parent_id
    if not parent_id:
        return MCP_DRIVE_ALLOWLIST_PARENT_ID
    if parent_id != MCP_DRIVE_ALLOWLIST_PARENT_ID and not allow_any_parent:
        raise ValueError("parent_id is outside the configured Drive allowlist.")
    return parent_id


CellValue = str | int | float | bool | None
Values = list[list[CellValue]]


def build_email_message(
    to: str,
    subject: str,
    body: str,
    cc: str = "",
    bcc: str = "",
    reply_to: str = "",
    from_alias: str = "",
    is_html: bool = False,
) -> EmailMessage:
    message = EmailMessage()
    message["To"] = to
    message["Subject"] = subject
    if cc:
        message["Cc"] = cc
    if bcc:
        message["Bcc"] = bcc
    if reply_to:
        message["Reply-To"] = reply_to
    if from_alias:
        message["From"] = from_alias
    if is_html:
        message.add_alternative(body, subtype="html")
    else:
        message.set_content(body)
    return message


def encode_email_message(message: EmailMessage) -> str:
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


def _decode_gmail_body(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", errors="replace")


def _extract_gmail_bodies(payload: dict[str, Any] | None) -> dict[str, str]:
    results: dict[str, str] = {}
    if not payload:
        return results

    def _walk(part: dict[str, Any]) -> None:
        mime_type = part.get("mimeType")
        body = part.get("body", {}) if isinstance(part, dict) else {}
        data = body.get("data")
        if data and mime_type in {"text/plain", "text/html"}:
            results[mime_type] = _decode_gmail_body(data)
        for sub in part.get("parts", []) or []:
            _walk(sub)

    _walk(payload)
    return results


async def run_blocking(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


@mcp.tool()
async def google_raw_request(
    method: str,
    url: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    """Send an arbitrary Google API request with OAuth credentials."""
    safe_methods = {"GET", "HEAD", "OPTIONS"}
    allow_retry = method.upper() in safe_methods

    def _request():
        session, cached = client.get_session()
        response = session.request(
            method.upper(),
            normalize_url(url),
            params=params,
            json=json_body,
            headers=headers,
        )
        content_type = response.headers.get("content-type", "")
        payload: dict[str, Any] = {
            "status": response.status_code,
            "headers": dict(response.headers),
        }
        if "application/json" in content_type:
            payload["json"] = response.json()
            return payload, {"cached_session": cached}
        try:
            text = response.text
            payload["text"] = text
        except UnicodeDecodeError:
            payload["content_base64"] = base64.b64encode(response.content).decode("ascii")
            payload["content_type"] = content_type
        return payload, {"cached_session": cached}

    return await run_tool(
        "raw",
        "google_raw_request",
        _request,
        allow_retry=allow_retry,
    )


@mcp.tool()
async def drive_list_files(
    query: str = "",
    page_size: int = 100,
    fields: str = "",
    order_by: str = "",
    page_token: str = "",
) -> str:
    """List files in Google Drive using the v3 API."""

    effective_fields = fields or DEFAULT_DRIVE_FIELDS

    def _list_files():
        service, cached = client.get_service("drive", "v3")
        request = service.files().list(
            q=query or None,
            pageSize=page_size,
            fields=effective_fields,
            orderBy=order_by or None,
            pageToken=page_token or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("drive", "list_files", _list_files, allow_retry=True)


@mcp.tool()
async def drive_search_files(
    query: str,
    page_size: int = 100,
    fields: str = "",
    order_by: str = "",
    page_token: str = "",
) -> str:
    """Search Drive files using a query string."""

    effective_fields = fields or DEFAULT_DRIVE_FIELDS

    def _search_files():
        if not query:
            raise ValueError("query cannot be empty")
        service, cached = client.get_service("drive", "v3")
        request = service.files().list(
            q=query,
            pageSize=page_size,
            fields=effective_fields,
            orderBy=order_by or None,
            pageToken=page_token or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("drive", "search_files", _search_files, allow_retry=True)


@mcp.tool()
async def drive_batch_get_metadata(
    file_ids: list[str],
    fields: str = "",
) -> str:
    """Fetch Drive file metadata for multiple file IDs."""

    effective_fields = fields or DEFAULT_DRIVE_GET_FIELDS

    def _batch_get():
        if not file_ids:
            raise ValueError("file_ids cannot be empty")
        service, cached = client.get_service("drive", "v3")
        files = []
        for file_id in file_ids:
            if not file_id:
                continue
            request = service.files().get(fileId=file_id, fields=effective_fields)
            files.append(request.execute())
        return {"files": files}, {"cached_service": cached}

    return await run_tool("drive", "batch_get_metadata", _batch_get, allow_retry=True)


@mcp.tool()
async def drive_get_file(
    file_id: str,
    fields: str = "",
) -> str:
    """Get Drive file metadata."""

    effective_fields = fields or DEFAULT_DRIVE_GET_FIELDS

    def _get_file():
        if not file_id:
            raise ValueError("file_id cannot be empty")
        service, cached = client.get_service("drive", "v3")
        request = service.files().get(fileId=file_id, fields=effective_fields)
        return request.execute(), {"cached_service": cached}

    return await run_tool("drive", "get_file", _get_file, allow_retry=True)


@mcp.tool()
async def drive_create_folder(
    name: str,
    parent_id: str = "",
    allow_any_parent: bool = False,
) -> str:
    """Create a Drive folder."""

    def _create_folder():
        if not name:
            raise ValueError("name cannot be empty")
        service, cached = client.get_service("drive", "v3")
        effective_parent_id = _enforce_drive_allowlist(parent_id, allow_any_parent)
        body: dict[str, Any] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if effective_parent_id:
            body["parents"] = [effective_parent_id]
        request = service.files().create(body=body, fields="id,name")
        return request.execute(), {"cached_service": cached}

    return await run_tool("drive", "create_folder", _create_folder, allow_retry=False)


@mcp.tool()
async def drive_upload_file(
    name: str,
    content: str,
    mime_type: str = "text/plain",
    parent_id: str = "",
    is_base64: bool = False,
    allow_any_parent: bool = False,
) -> str:
    """Upload a file to Drive from text or base64 content."""

    def _upload():
        if not name:
            raise ValueError("name cannot be empty")
        if content is None:
            raise ValueError("content cannot be empty")
        service, cached = client.get_service("drive", "v3")
        effective_parent_id = _enforce_drive_allowlist(parent_id, allow_any_parent)
        data = (
            base64.b64decode(content.encode("ascii")) if is_base64 else content.encode("utf-8")
        )
        media = MediaInMemoryUpload(data, mimetype=mime_type, resumable=False)
        body: dict[str, Any] = {"name": name}
        if effective_parent_id:
            body["parents"] = [effective_parent_id]
        request = service.files().create(
            body=body,
            media_body=media,
            fields="id,name,mimeType,parents",
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("drive", "upload_file", _upload, allow_retry=False)


@mcp.tool()
async def drive_download_file(
    file_id: str,
    export_mime_type: str = "",
    include_content: bool = False,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    range_start: int | None = None,
    range_end: int | None = None,
) -> str:
    """Download a file from Drive. For Google Docs/Sheets/Slides, use export."""

    def _download():
        if not file_id:
            raise ValueError("file_id cannot be empty")
        service, cached_service = client.get_service("drive", "v3")
        metadata = service.files().get(
            fileId=file_id, fields="id,name,mimeType,size,webViewLink,webContentLink"
        ).execute()
        mime_type = metadata.get("mimeType", "")

        download_mime = export_mime_type
        if mime_type.startswith("application/vnd.google-apps") and not export_mime_type:
            if mime_type.endswith("document"):
                download_mime = "text/plain"
            elif mime_type.endswith("spreadsheet"):
                download_mime = "text/csv"
            elif mime_type.endswith("presentation"):
                download_mime = "application/pdf"
            else:
                download_mime = "application/pdf"

        if mime_type.startswith("application/vnd.google-apps"):
            download_url = (
                "https://www.googleapis.com/drive/v3/files/"
                f"{file_id}/export?mimeType={urllib.parse.quote(download_mime)}"
            )
        else:
            download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
            download_mime = mime_type

        response_payload: dict[str, Any] = {
            "file": metadata,
            "download_mime_type": download_mime,
            "download_url": download_url,
        }
        if not include_content:
            return response_payload, {"cached_service": cached_service}

        size = metadata.get("size")
        if size is not None:
            try:
                size_int = int(size)
            except (TypeError, ValueError):
                size_int = None
            else:
                if max_bytes and size_int > max_bytes:
                    response_payload["too_large"] = True
                    response_payload["size"] = size_int
                    response_payload["max_bytes"] = max_bytes
                    return response_payload, {"cached_service": cached_service}

        session, cached_session = client.get_session()
        headers: dict[str, str] = {}
        if range_start is not None or range_end is not None:
            start = range_start or 0
            end = "" if range_end is None else str(range_end)
            headers["Range"] = f"bytes={start}-{end}"
        response = session.get(download_url, headers=headers, stream=True)
        if not response.ok:
            raise RuntimeError(
                f"Drive download failed with status {response.status_code}."
            )

        content_length = response.headers.get("content-length")
        if content_length and max_bytes:
            try:
                if int(content_length) > max_bytes:
                    response_payload["too_large"] = True
                    response_payload["size"] = int(content_length)
                    response_payload["max_bytes"] = max_bytes
                    response.close()
                    return response_payload, {
                        "cached_service": cached_service,
                        "cached_session": cached_session,
                    }
            except (TypeError, ValueError):
                pass

        buffer = io.BytesIO()
        total = 0
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            total += len(chunk)
            if max_bytes and total > max_bytes:
                response_payload["too_large"] = True
                response_payload["size"] = total
                response_payload["max_bytes"] = max_bytes
                response.close()
                return response_payload, {
                    "cached_service": cached_service,
                    "cached_session": cached_session,
                }
            buffer.write(chunk)

        content_bytes = buffer.getvalue()
        response_payload["content_base64"] = base64.b64encode(content_bytes).decode("ascii")
        response_payload["content_bytes"] = len(content_bytes)
        return response_payload, {
            "cached_service": cached_service,
            "cached_session": cached_session,
        }

    return await run_tool("drive", "download_file", _download, allow_retry=True)


@mcp.tool()
async def drive_empty_trash(confirm: bool = False) -> str:
    """Permanently delete all files in Drive trash."""

    def _empty_trash():
        _ensure_confirmed("empty Drive trash", confirm)
        service, cached = client.get_service("drive", "v3")
        request = service.files().emptyTrash()
        request.execute()
        return {"status": "ok"}, {"cached_service": cached}

    return await run_tool("drive", "empty_trash", _empty_trash, allow_retry=False)


@mcp.tool()
async def docs_create_document(title: str) -> str:
    """Create a Google Doc."""

    def _create_doc():
        if not title:
            raise ValueError("title cannot be empty")
        service, cached = client.get_service("docs", "v1")
        request = service.documents().create(body={"title": title})
        return request.execute(), {"cached_service": cached}

    return await run_tool("docs", "create_document", _create_doc, allow_retry=False)


@mcp.tool()
async def docs_get_document(document_id: str, fields: str = "") -> str:
    """Fetch a Google Doc document."""

    def _get_doc():
        if not document_id:
            raise ValueError("document_id cannot be empty")
        service, cached = client.get_service("docs", "v1")
        effective_fields = fields or DEFAULT_DOCS_FIELDS
        request = service.documents().get(documentId=document_id, fields=effective_fields)
        return request.execute(), {"cached_service": cached}

    return await run_tool("docs", "get_document", _get_doc, allow_retry=True)


@mcp.tool()
async def docs_insert_text(document_id: str, text: str, index: int = 1) -> str:
    """Insert text into a Google Doc at the given index."""

    def _insert():
        if not document_id:
            raise ValueError("document_id cannot be empty")
        if text is None:
            raise ValueError("text cannot be empty")
        service, cached = client.get_service("docs", "v1")
        body = {
            "requests": [
                {"insertText": {"location": {"index": index}, "text": text}}
            ]
        }
        request = service.documents().batchUpdate(documentId=document_id, body=body)
        return request.execute(), {"cached_service": cached}

    return await run_tool("docs", "insert_text", _insert, allow_retry=False)


@mcp.tool()
async def docs_replace_text(
    document_id: str,
    contains_text: str,
    replace_text: str,
    match_case: bool = False,
) -> str:
    """Replace text in a Google Doc."""

    def _replace():
        if not document_id:
            raise ValueError("document_id cannot be empty")
        if not contains_text:
            raise ValueError("contains_text cannot be empty")
        service, cached = client.get_service("docs", "v1")
        body = {
            "requests": [
                {
                    "replaceAllText": {
                        "containsText": {
                            "text": contains_text,
                            "matchCase": match_case,
                        },
                        "replaceText": replace_text,
                    }
                }
            ]
        }
        request = service.documents().batchUpdate(documentId=document_id, body=body)
        return request.execute(), {"cached_service": cached}

    return await run_tool("docs", "replace_text", _replace, allow_retry=False)


@mcp.tool()
async def sheets_create_spreadsheet(title: str) -> str:
    """Create a Google Sheet."""

    def _create_sheet():
        if not title:
            raise ValueError("title cannot be empty")
        service, cached = client.get_service("sheets", "v4")
        request = service.spreadsheets().create(body={"properties": {"title": title}})
        return request.execute(), {"cached_service": cached}

    return await run_tool("sheets", "create_spreadsheet", _create_sheet, allow_retry=False)


@mcp.tool()
async def sheets_get_spreadsheet(spreadsheet_id: str, fields: str = "") -> str:
    """Fetch a Google Sheet spreadsheet."""

    def _get_sheet():
        if not spreadsheet_id:
            raise ValueError("spreadsheet_id cannot be empty")
        service, cached = client.get_service("sheets", "v4")
        effective_fields = fields or DEFAULT_SHEETS_FIELDS
        request = service.spreadsheets().get(
            spreadsheetId=spreadsheet_id, fields=effective_fields
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("sheets", "get_spreadsheet", _get_sheet, allow_retry=True)


@mcp.tool()
async def sheets_get_values(spreadsheet_id: str, range_a1: str) -> str:
    """Read values from a Google Sheet range."""

    def _get_values():
        if not spreadsheet_id:
            raise ValueError("spreadsheet_id cannot be empty")
        if not range_a1:
            raise ValueError("range_a1 cannot be empty")
        service, cached = client.get_service("sheets", "v4")
        request = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_a1,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("sheets", "get_values", _get_values, allow_retry=True)


@mcp.tool()
async def sheets_update_values(
    spreadsheet_id: str,
    range_a1: str,
    values: Values,
    value_input_option: str = "RAW",
) -> str:
    """Write values to a Google Sheet range."""

    def _update_values():
        if not spreadsheet_id:
            raise ValueError("spreadsheet_id cannot be empty")
        if not range_a1:
            raise ValueError("range_a1 cannot be empty")
        if values is None:
            raise ValueError("values cannot be empty")
        service, cached = client.get_service("sheets", "v4")
        body = {"values": values}
        request = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_a1,
            valueInputOption=value_input_option,
            body=body,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("sheets", "update_values", _update_values, allow_retry=False)


@mcp.tool()
async def slides_create_presentation(title: str) -> str:
    """Create a Google Slides presentation."""

    def _create_presentation():
        if not title:
            raise ValueError("title cannot be empty")
        service, cached = client.get_service("slides", "v1")
        request = service.presentations().create(body={"title": title})
        return request.execute(), {"cached_service": cached}

    return await run_tool(
        "slides", "create_presentation", _create_presentation, allow_retry=False
    )


@mcp.tool()
async def slides_get_presentation(presentation_id: str, fields: str = "") -> str:
    """Fetch a Google Slides presentation."""

    def _get_presentation():
        if not presentation_id:
            raise ValueError("presentation_id cannot be empty")
        service, cached = client.get_service("slides", "v1")
        effective_fields = fields or DEFAULT_SLIDES_FIELDS
        request = service.presentations().get(
            presentationId=presentation_id, fields=effective_fields
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("slides", "get_presentation", _get_presentation, allow_retry=True)


@mcp.tool()
async def slides_replace_text(
    presentation_id: str,
    contains_text: str,
    replace_text: str,
    match_case: bool = False,
) -> str:
    """Replace text across a Slides presentation."""

    def _replace():
        if not presentation_id:
            raise ValueError("presentation_id cannot be empty")
        if not contains_text:
            raise ValueError("contains_text cannot be empty")
        service, cached = client.get_service("slides", "v1")
        body = {
            "requests": [
                {
                    "replaceAllText": {
                        "containsText": {
                            "text": contains_text,
                            "matchCase": match_case,
                        },
                        "replaceText": replace_text,
                    }
                }
            ]
        }
        request = service.presentations().batchUpdate(
            presentationId=presentation_id,
            body=body,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("slides", "replace_text", _replace, allow_retry=False)


@mcp.tool()
async def gmail_list_labels(fields: str = "", minimal: bool = False) -> str:
    """List Gmail labels for the authenticated user."""

    def _list_labels():
        service, cached = client.get_service("gmail", "v1")
        effective_fields = fields or ("labels(id,name)" if minimal else "")
        request = service.users().labels().list(
            userId="me",
            fields=effective_fields or None,
        )
        data = request.execute()
        if minimal:
            labels = [
                {"id": label.get("id"), "name": label.get("name")}
                for label in data.get("labels", []) or []
            ]
            return {"labels": labels}, {"cached_service": cached}
        return data, {"cached_service": cached}

    return await run_tool("gmail", "list_labels", _list_labels, allow_retry=True)


@mcp.tool()
async def gmail_create_label(
    name: str,
    label_list_visibility: str = "labelShow",
    message_list_visibility: str = "show",
) -> str:
    """Create a Gmail label."""

    def _create_label():
        if not name:
            raise ValueError("name cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        body = {
            "name": name,
            "labelListVisibility": label_list_visibility,
            "messageListVisibility": message_list_visibility,
        }
        request = service.users().labels().create(userId="me", body=body)
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "create_label", _create_label, allow_retry=False)


@mcp.tool()
async def gmail_delete_label(label_id: str, confirm: bool = False) -> str:
    """Delete a Gmail label."""

    def _delete_label():
        if not label_id:
            raise ValueError("label_id cannot be empty")
        _ensure_confirmed("delete a Gmail label", confirm)
        service, cached = client.get_service("gmail", "v1")
        request = service.users().labels().delete(userId="me", id=label_id)
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "delete_label", _delete_label, allow_retry=False)


@mcp.tool()
async def gmail_list_messages(
    query: str = "",
    label_ids: list[str] | None = None,
    max_results: int = 100,
    include_spam_trash: bool = False,
    page_token: str = "",
) -> str:
    """List Gmail messages matching a query or labels."""

    def _list_messages():
        service, cached = client.get_service("gmail", "v1")
        request = service.users().messages().list(
            userId="me",
            q=query or None,
            labelIds=label_ids or None,
            maxResults=max_results,
            includeSpamTrash=include_spam_trash,
            pageToken=page_token or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "list_messages", _list_messages, allow_retry=True)


@mcp.tool()
async def gmail_search_messages(
    query: str,
    label_ids: list[str] | None = None,
    max_results: int = 100,
    include_spam_trash: bool = False,
    page_token: str = "",
) -> str:
    """Search Gmail messages with a query string."""

    def _search_messages():
        if not query:
            raise ValueError("query cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        request = service.users().messages().list(
            userId="me",
            q=query,
            labelIds=label_ids or None,
            maxResults=max_results,
            includeSpamTrash=include_spam_trash,
            pageToken=page_token or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "search_messages", _search_messages, allow_retry=True)


@mcp.tool()
async def gmail_get_message(
    message_id: str,
    format: str = "metadata",
    metadata_headers: list[str] | None = None,
) -> str:
    """Get a Gmail message by ID."""

    def _get_message():
        if not message_id:
            raise ValueError("message_id cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        effective_headers = metadata_headers or list(DEFAULT_GMAIL_METADATA_HEADERS)
        request = service.users().messages().get(
            userId="me",
            id=message_id,
            format=format,
            metadataHeaders=effective_headers if format == "metadata" else None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "get_message", _get_message, allow_retry=True)


@mcp.tool()
async def gmail_get_message_headers(
    message_id: str, headers: list[str] | None = None
) -> str:
    """Fetch Gmail message headers only."""

    def _get_headers():
        if not message_id:
            raise ValueError("message_id cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        effective_headers = headers or list(DEFAULT_GMAIL_METADATA_HEADERS)
        request = service.users().messages().get(
            userId="me",
            id=message_id,
            format="metadata",
            metadataHeaders=effective_headers,
        )
        data = request.execute()
        header_list = data.get("payload", {}).get("headers", []) or []
        header_map = {
            entry.get("name"): entry.get("value")
            for entry in header_list
            if entry.get("name")
        }
        return (
            {
                "id": data.get("id"),
                "threadId": data.get("threadId"),
                "labelIds": data.get("labelIds", []),
                "snippet": data.get("snippet", ""),
                "headers": header_map,
            },
            {"cached_service": cached},
        )

    return await run_tool("gmail", "get_message_headers", _get_headers, allow_retry=True)


@mcp.tool()
async def gmail_get_message_body(message_id: str, prefer_html: bool = False) -> str:
    """Extract text/plain or text/html content from a Gmail message."""

    def _get_body():
        if not message_id:
            raise ValueError("message_id cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        request = service.users().messages().get(
            userId="me",
            id=message_id,
            format="full",
        )
        data = request.execute()
        bodies = _extract_gmail_bodies(data.get("payload"))
        text_plain = bodies.get("text/plain", "")
        text_html = bodies.get("text/html", "")
        selected = text_html if prefer_html and text_html else text_plain
        return (
            {
                "id": data.get("id"),
                "threadId": data.get("threadId"),
                "snippet": data.get("snippet", ""),
                "text_plain": text_plain,
                "text_html": text_html,
                "body": selected,
            },
            {"cached_service": cached},
        )

    return await run_tool("gmail", "get_message_body", _get_body, allow_retry=True)


@mcp.tool()
async def gmail_batch_get_metadata(
    message_ids: list[str],
    metadata_headers: list[str] | None = None,
) -> str:
    """Fetch Gmail message metadata for multiple message IDs."""

    def _batch_get():
        if not message_ids:
            raise ValueError("message_ids cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        effective_headers = metadata_headers or list(DEFAULT_GMAIL_METADATA_HEADERS)
        results = []
        for message_id in message_ids:
            if not message_id:
                continue
            request = service.users().messages().get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=effective_headers,
            )
            results.append(request.execute())
        return {"messages": results}, {"cached_service": cached}

    return await run_tool("gmail", "batch_get_metadata", _batch_get, allow_retry=True)


@mcp.tool()
async def gmail_list_threads(
    query: str = "",
    label_ids: list[str] | None = None,
    max_results: int = 50,
    page_token: str = "",
) -> str:
    """List Gmail threads."""

    def _list_threads():
        service, cached = client.get_service("gmail", "v1")
        request = service.users().threads().list(
            userId="me",
            q=query or None,
            labelIds=label_ids or None,
            maxResults=max_results,
            pageToken=page_token or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "list_threads", _list_threads, allow_retry=True)


@mcp.tool()
async def gmail_get_thread(
    thread_id: str,
    format: str = "metadata",
    metadata_headers: list[str] | None = None,
) -> str:
    """Get a Gmail thread by ID."""

    def _get_thread():
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        effective_headers = metadata_headers or list(DEFAULT_GMAIL_METADATA_HEADERS)
        request = service.users().threads().get(
            userId="me",
            id=thread_id,
            format=format,
            metadataHeaders=effective_headers if format == "metadata" else None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "get_thread", _get_thread, allow_retry=True)


@mcp.tool()
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
) -> str:
    """Send a Gmail message with basic headers."""

    def _send():
        if not to:
            raise ValueError("to cannot be empty")
        if not subject:
            raise ValueError("subject cannot be empty")
        if body is None:
            raise ValueError("body cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        message = build_email_message(
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
            reply_to=reply_to,
            from_alias=from_alias,
            is_html=is_html,
        )
        raw = encode_email_message(message)
        payload: dict[str, Any] = {"raw": raw}
        if thread_id:
            payload["threadId"] = thread_id
        request = service.users().messages().send(userId="me", body=payload)
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "send_message", _send, allow_retry=False)


@mcp.tool()
async def gmail_send_raw_message(raw_base64: str, thread_id: str = "") -> str:
    """Send a Gmail message using a base64url-encoded raw MIME message."""

    def _send_raw():
        if not raw_base64:
            raise ValueError("raw_base64 cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        payload: dict[str, Any] = {"raw": raw_base64}
        if thread_id:
            payload["threadId"] = thread_id
        request = service.users().messages().send(userId="me", body=payload)
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "send_raw_message", _send_raw, allow_retry=False)


@mcp.tool()
async def gmail_create_draft(
    to: str,
    subject: str,
    body: str,
    cc: str = "",
    bcc: str = "",
    reply_to: str = "",
    from_alias: str = "",
    is_html: bool = False,
) -> str:
    """Create a Gmail draft."""

    def _create_draft():
        if not to:
            raise ValueError("to cannot be empty")
        if not subject:
            raise ValueError("subject cannot be empty")
        if body is None:
            raise ValueError("body cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        message = build_email_message(
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
            reply_to=reply_to,
            from_alias=from_alias,
            is_html=is_html,
        )
        raw = encode_email_message(message)
        request = service.users().drafts().create(
            userId="me",
            body={"message": {"raw": raw}},
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "create_draft", _create_draft, allow_retry=False)


@mcp.tool()
async def gmail_send_draft(draft_id: str) -> str:
    """Send an existing Gmail draft."""

    def _send_draft():
        if not draft_id:
            raise ValueError("draft_id cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        request = service.users().drafts().send(userId="me", body={"id": draft_id})
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "send_draft", _send_draft, allow_retry=False)


@mcp.tool()
async def gmail_modify_message_labels(
    message_id: str,
    add_label_ids: list[str] | None = None,
    remove_label_ids: list[str] | None = None,
) -> str:
    """Add or remove labels on a Gmail message."""

    def _modify():
        if not message_id:
            raise ValueError("message_id cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        body = {
            "addLabelIds": add_label_ids or [],
            "removeLabelIds": remove_label_ids or [],
        }
        request = service.users().messages().modify(
            userId="me",
            id=message_id,
            body=body,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "modify_message_labels", _modify, allow_retry=False)


@mcp.tool()
async def gmail_trash_message(message_id: str, confirm: bool = False) -> str:
    """Move a Gmail message to trash."""

    def _trash():
        if not message_id:
            raise ValueError("message_id cannot be empty")
        _ensure_confirmed("trash a Gmail message", confirm)
        service, cached = client.get_service("gmail", "v1")
        request = service.users().messages().trash(userId="me", id=message_id)
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "trash_message", _trash, allow_retry=False)


@mcp.tool()
async def gmail_untrash_message(message_id: str, confirm: bool = False) -> str:
    """Restore a Gmail message from trash."""

    def _untrash():
        if not message_id:
            raise ValueError("message_id cannot be empty")
        _ensure_confirmed("untrash a Gmail message", confirm)
        service, cached = client.get_service("gmail", "v1")
        request = service.users().messages().untrash(userId="me", id=message_id)
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "untrash_message", _untrash, allow_retry=False)


@mcp.tool()
async def gmail_delete_message(message_id: str, confirm: bool = False) -> str:
    """Permanently delete a Gmail message."""

    def _delete():
        if not message_id:
            raise ValueError("message_id cannot be empty")
        _ensure_confirmed("delete a Gmail message", confirm)
        service, cached = client.get_service("gmail", "v1")
        request = service.users().messages().delete(userId="me", id=message_id)
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "delete_message", _delete, allow_retry=False)


@mcp.tool()
async def calendar_list_calendars(fields: str = "", page_token: str = "") -> str:
    """List calendars visible to the authenticated user."""

    def _list_calendars():
        service, cached = client.get_service("calendar", "v3")
        effective_fields = fields or DEFAULT_CALENDAR_LIST_FIELDS
        request = service.calendarList().list(
            fields=effective_fields,
            pageToken=page_token or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "list_calendars", _list_calendars, allow_retry=True)


@mcp.tool()
async def calendar_get_calendar(calendar_id: str, fields: str = "") -> str:
    """Get calendar metadata by ID."""

    def _get_calendar():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        service, cached = client.get_service("calendar", "v3")
        effective_fields = fields or DEFAULT_CALENDAR_FIELDS
        request = service.calendars().get(calendarId=calendar_id, fields=effective_fields)
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "get_calendar", _get_calendar, allow_retry=True)


@mcp.tool()
async def calendar_create_calendar(summary: str, description: str = "", time_zone: str = "") -> str:
    """Create a new calendar."""

    def _create_calendar():
        if not summary:
            raise ValueError("summary cannot be empty")
        service, cached = client.get_service("calendar", "v3")
        body: dict[str, Any] = {"summary": summary}
        if description:
            body["description"] = description
        if time_zone:
            body["timeZone"] = time_zone
        request = service.calendars().insert(body=body)
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "create_calendar", _create_calendar, allow_retry=False)


@mcp.tool()
async def calendar_delete_calendar(calendar_id: str, confirm: bool = False) -> str:
    """Delete a calendar."""

    def _delete_calendar():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        _ensure_confirmed("delete a calendar", confirm)
        service, cached = client.get_service("calendar", "v3")
        request = service.calendars().delete(calendarId=calendar_id)
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "delete_calendar", _delete_calendar, allow_retry=False)


@mcp.tool()
async def calendar_list_events(
    calendar_id: str = "primary",
    time_min: str = "",
    time_max: str = "",
    query: str = "",
    max_results: int = 100,
    single_events: bool = True,
    order_by: str = "startTime",
    fields: str = "",
    page_token: str = "",
) -> str:
    """List events in a calendar."""

    def _list_events():
        service, cached = client.get_service("calendar", "v3")
        effective_fields = fields or DEFAULT_EVENT_LIST_FIELDS
        request = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min or None,
            timeMax=time_max or None,
            q=query or None,
            maxResults=max_results,
            singleEvents=single_events,
            orderBy=order_by or None,
            fields=effective_fields,
            pageToken=page_token or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "list_events", _list_events, allow_retry=True)


@mcp.tool()
async def calendar_search_events(
    query: str,
    calendar_id: str = "primary",
    time_min: str = "",
    time_max: str = "",
    max_results: int = 100,
    single_events: bool = True,
    order_by: str = "startTime",
    fields: str = "",
    page_token: str = "",
) -> str:
    """Search events in a calendar with a query string."""

    def _search_events():
        if not query:
            raise ValueError("query cannot be empty")
        service, cached = client.get_service("calendar", "v3")
        effective_fields = fields or DEFAULT_EVENT_LIST_FIELDS
        request = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min or None,
            timeMax=time_max or None,
            q=query,
            maxResults=max_results,
            singleEvents=single_events,
            orderBy=order_by or None,
            fields=effective_fields,
            pageToken=page_token or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "search_events", _search_events, allow_retry=True)


@mcp.tool()
async def calendar_batch_get_events(
    calendar_id: str,
    event_ids: list[str],
    fields: str = "",
) -> str:
    """Fetch multiple calendar events by ID in a single tool call."""

    def _batch_get():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        if not event_ids:
            raise ValueError("event_ids cannot be empty")
        service, cached = client.get_service("calendar", "v3")
        effective_fields = fields or DEFAULT_EVENT_FIELDS
        events = []
        for event_id in event_ids:
            if not event_id:
                continue
            request = service.events().get(
                calendarId=calendar_id,
                eventId=event_id,
                fields=effective_fields,
            )
            events.append(request.execute())
        return {"events": events}, {"cached_service": cached}

    return await run_tool("calendar", "batch_get_events", _batch_get, allow_retry=True)


@mcp.tool()
async def calendar_get_event(calendar_id: str, event_id: str, fields: str = "") -> str:
    """Get a calendar event by ID."""

    def _get_event():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        if not event_id:
            raise ValueError("event_id cannot be empty")
        service, cached = client.get_service("calendar", "v3")
        effective_fields = fields or DEFAULT_EVENT_FIELDS
        request = service.events().get(
            calendarId=calendar_id,
            eventId=event_id,
            fields=effective_fields,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "get_event", _get_event, allow_retry=True)


@mcp.tool()
async def calendar_create_event(
    calendar_id: str,
    summary: str,
    start_iso: str,
    end_iso: str,
    time_zone: str = "UTC",
    description: str = "",
    location: str = "",
    attendees: list[str] | None = None,
    all_day: bool = False,
) -> str:
    """Create a calendar event."""

    def _create_event():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        if not summary:
            raise ValueError("summary cannot be empty")
        if not start_iso or not end_iso:
            raise ValueError("start_iso and end_iso are required")
        service, cached = client.get_service("calendar", "v3")
        event: dict[str, Any] = {"summary": summary}
        if description:
            event["description"] = description
        if location:
            event["location"] = location
        if all_day:
            event["start"] = {"date": start_iso}
            event["end"] = {"date": end_iso}
        else:
            event["start"] = {"dateTime": start_iso, "timeZone": time_zone}
            event["end"] = {"dateTime": end_iso, "timeZone": time_zone}
        if attendees:
            event["attendees"] = [{"email": email} for email in attendees]
        request = service.events().insert(calendarId=calendar_id, body=event)
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "create_event", _create_event, allow_retry=False)


@mcp.tool()
async def calendar_update_event(
    calendar_id: str,
    event_id: str,
    event_patch: dict[str, Any],
    send_updates: str = "all",
) -> str:
    """Patch a calendar event with a partial update."""

    def _update_event():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        if not event_id:
            raise ValueError("event_id cannot be empty")
        if event_patch is None:
            raise ValueError("event_patch cannot be empty")
        service, cached = client.get_service("calendar", "v3")
        request = service.events().patch(
            calendarId=calendar_id,
            eventId=event_id,
            body=event_patch,
            sendUpdates=send_updates or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "update_event", _update_event, allow_retry=False)


@mcp.tool()
async def calendar_delete_event(
    calendar_id: str,
    event_id: str,
    send_updates: str = "all",
    confirm: bool = False,
) -> str:
    """Delete a calendar event."""

    def _delete_event():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        if not event_id:
            raise ValueError("event_id cannot be empty")
        _ensure_confirmed("delete a calendar event", confirm)
        service, cached = client.get_service("calendar", "v3")
        request = service.events().delete(
            calendarId=calendar_id,
            eventId=event_id,
            sendUpdates=send_updates or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "delete_event", _delete_event, allow_retry=False)


@mcp.tool()
async def calendar_quick_add(calendar_id: str, text: str) -> str:
    """Create an event from a natural language text string."""

    def _quick_add():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        if not text:
            raise ValueError("text cannot be empty")
        service, cached = client.get_service("calendar", "v3")
        request = service.events().quickAdd(calendarId=calendar_id, text=text)
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "quick_add", _quick_add, allow_retry=False)


if __name__ == "__main__":
    os.environ.setdefault("HOST", MCP_BIND_ADDRESS)
    os.environ.setdefault("PORT", str(MCP_HTTP_PORT))
    app_factory = mcp.streamable_http_app

    def build_app():
        app = app_factory() if callable(app_factory) else app_factory
        try:
            app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
        except Exception:
            pass

        async def host_override(scope, receive, send):
            if scope["type"] == "http":
                headers = []
                for key, value in scope.get("headers", []):
                    if key.lower() == b"host":
                        continue
                    headers.append((key, value))
                headers.append((b"host", b"localhost"))
                scope = {**scope, "headers": headers}
            await app(scope, receive, send)

        return host_override

    uvicorn.run(build_app, host=MCP_BIND_ADDRESS, port=MCP_HTTP_PORT, factory=True)
