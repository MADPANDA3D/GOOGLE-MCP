import asyncio
import base64
import contextvars
import hashlib
import hmac
import io
import json
import logging
import os
import random
import re
import threading
import time
import urllib.parse
import uuid
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Any, Callable

from google.auth.transport.requests import AuthorizedSession, Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaInMemoryUpload
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
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
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/analytics.readonly",
    "https://www.googleapis.com/auth/webmasters.readonly",
    "https://www.googleapis.com/auth/business.manage",
    "https://www.googleapis.com/auth/content",
    "https://www.googleapis.com/auth/adsense.readonly",
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
GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
GOOGLE_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_CREDENTIALS_PATH", "fastmcp/.google/credentials.json"
)
GOOGLE_TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", "fastmcp/.google/token.json")
GOOGLE_SCOPES_RAW = os.getenv("GOOGLE_SCOPES", " ".join(DEFAULT_SCOPES))
MCP_WORKERS = int(os.getenv("MCP_WORKERS", "1"))
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
MCP_RAW_STRICT = os.getenv("MCP_RAW_STRICT", "true").lower() in {"1", "true", "yes"}
MCP_SERVER_VERSION = os.getenv("MCP_SERVER_VERSION", "unknown")
MCP_STRICT_PARAMS = os.getenv("MCP_STRICT_PARAMS", "").lower() in {"1", "true", "yes"}
MCP_ALLOW_REQUEST_OVERRIDES = os.getenv(
    "MCP_ALLOW_REQUEST_OVERRIDES", "true"
).lower() in {"1", "true", "yes"}
MCP_REQUIRE_REQUEST_GOOGLE_CLIENT_ID = os.getenv(
    "MCP_REQUIRE_REQUEST_GOOGLE_CLIENT_ID", "true"
).lower() in {"1", "true", "yes"}
MCP_REQUIRE_REQUEST_GOOGLE_CLIENT_SECRET = os.getenv(
    "MCP_REQUIRE_REQUEST_GOOGLE_CLIENT_SECRET", "true"
).lower() in {"1", "true", "yes"}
MCP_REQUIRE_REQUEST_GOOGLE_REFRESH_TOKEN = os.getenv(
    "MCP_REQUIRE_REQUEST_GOOGLE_REFRESH_TOKEN", "true"
).lower() in {"1", "true", "yes"}
MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK = os.getenv(
    "MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK", "true"
).lower() in {"1", "true", "yes"}
MCP_GOOGLE_CLIENT_ID_HEADER = os.getenv(
    "MCP_GOOGLE_CLIENT_ID_HEADER", "x-google-client-id"
).strip().lower()
MCP_GOOGLE_CLIENT_SECRET_HEADER = os.getenv(
    "MCP_GOOGLE_CLIENT_SECRET_HEADER", "x-google-client-secret"
).strip().lower()
MCP_GOOGLE_REFRESH_TOKEN_HEADER = os.getenv(
    "MCP_GOOGLE_REFRESH_TOKEN_HEADER", "x-google-refresh-token"
).strip().lower()
MCP_GOOGLE_MAPS_API_KEY_HEADER = os.getenv(
    "MCP_GOOGLE_MAPS_API_KEY_HEADER", "x-google-maps-api-key"
).strip().lower()
MCP_PORTAL_GRANT_HEADER = os.getenv(
    "MCP_PORTAL_GRANT_HEADER", "x-madpanda-portal-grant"
).strip().lower()
MCP_PORTAL_GRANT_TOKEN = os.getenv("MCP_PORTAL_GRANT_TOKEN", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
MCP_BYOK_CLIENT_CACHE_SIZE = max(int(os.getenv("MCP_BYOK_CLIENT_CACHE_SIZE", "256")), 0)
MCP_BYOK_CLIENT_CACHE_TTL_SECONDS = max(
    float(os.getenv("MCP_BYOK_CLIENT_CACHE_TTL_SECONDS", "900")),
    0.0,
)
EXPECTED_TOOL_COUNT = 145

SERVER_START_TIME = time.time()
SERVER_START_MONO = time.monotonic()
SERVER_INSTANCE_ID = f"{os.getpid()}-{int(SERVER_START_TIME)}"


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


def registered_tool_count() -> int:
    manager = getattr(mcp, "_tool_manager", None)
    for attr in ("_tools", "tools"):
        tools = getattr(manager, attr, None)
        if isinstance(tools, dict):
            return len(tools)
        if isinstance(tools, (list, tuple, set)):
            return len(tools)
    list_tools = getattr(manager, "list_tools", None)
    if callable(list_tools):
        try:
            tools = list_tools()
            if isinstance(tools, (list, tuple, set)):
                return len(tools)
        except Exception:
            pass
    return EXPECTED_TOOL_COUNT


def parse_scopes(raw: str) -> list[str]:
    if not raw:
        return []
    cleaned = raw.replace(",", " ")
    return [scope.strip() for scope in cleaned.split() if scope.strip()]


class GoogleWorkspaceClient:
    def __init__(
        self,
        credentials_path: str | None,
        token_path: str | None,
        scopes: list[str],
        *,
        authorized_user_info: dict[str, str] | None = None,
        persist_token: bool = True,
    ):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.scopes = scopes
        self.authorized_user_info = authorized_user_info
        self.persist_token = persist_token
        self._lock = threading.Lock()
        self._creds: Credentials | None = None
        self._service_cache: dict[tuple[str, str], Any] = {}
        self._session: AuthorizedSession | None = None

    def _save_token(self, creds: Credentials) -> None:
        if not self.persist_token or not self.token_path:
            return
        token_dir = os.path.dirname(self.token_path)
        if token_dir:
            os.makedirs(token_dir, exist_ok=True)
        with open(self.token_path, "w", encoding="utf-8") as handle:
            handle.write(creds.to_json())

    def _load_credentials(self) -> Credentials:
        with self._lock:
            if self._creds is None:
                if self.authorized_user_info:
                    self._creds = Credentials.from_authorized_user_info(
                        self.authorized_user_info,
                        self.scopes,
                    )
                else:
                    if not self.token_path or not os.path.exists(self.token_path):
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

    def is_service_cached(self, api_name: str, api_version: str) -> bool:
        cache_key = (api_name, api_version, tuple(self.scopes))
        with self._lock:
            return cache_key in self._service_cache

    def is_session_cached(self) -> bool:
        with self._lock:
            return self._session is not None


@dataclass(frozen=True)
class RequestGoogleOverrides:
    client_id: str
    client_secret: str
    refresh_token: str


class ByokClientCache:
    def __init__(self, max_size: int, ttl_seconds: float):
        self.max_size = max(0, max_size)
        self.ttl_seconds = max(0.0, ttl_seconds)
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[GoogleWorkspaceClient, float]] = OrderedDict()

    def _prune_locked(self, now_mono: float) -> None:
        expired = [
            cache_key
            for cache_key, (_, expires_at) in self._cache.items()
            if expires_at <= now_mono
        ]
        for cache_key in expired:
            self._cache.pop(cache_key, None)

    def get_or_create(
        self,
        cache_key: str,
        factory: Callable[[], GoogleWorkspaceClient],
    ) -> tuple[GoogleWorkspaceClient, bool]:
        if self.max_size <= 0 or self.ttl_seconds <= 0:
            return factory(), False

        now_mono = time.monotonic()
        with self._lock:
            self._prune_locked(now_mono)
            cached = self._cache.get(cache_key)
            if cached is not None:
                client_obj, expires_at = cached
                if expires_at > now_mono:
                    self._cache.move_to_end(cache_key)
                    return client_obj, True
                self._cache.pop(cache_key, None)

        client_obj = factory()
        now_mono = time.monotonic()
        with self._lock:
            self._prune_locked(now_mono)
            self._cache[cache_key] = (client_obj, now_mono + self.ttl_seconds)
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
        return client_obj, False


def _build_authorized_user_info(overrides: RequestGoogleOverrides) -> dict[str, str]:
    return {
        "client_id": overrides.client_id,
        "client_secret": overrides.client_secret,
        "refresh_token": overrides.refresh_token,
        "token_uri": GOOGLE_TOKEN_URI,
        "type": "authorized_user",
    }


def _fingerprint_request_overrides(overrides: RequestGoogleOverrides) -> str:
    digest_source = "\n".join(
        (
            overrides.client_id,
            overrides.client_secret,
            overrides.refresh_token,
            " ".join(SCOPES),
        )
    ).encode("utf-8")
    return hashlib.sha256(digest_source).hexdigest()


def _normalize_header_map(header_items: list[tuple[bytes, bytes]]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in header_items:
        try:
            key_text = key.decode("utf-8", errors="ignore").strip().lower()
            value_text = value.decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if key_text:
            normalized[key_text] = value_text
    return normalized


def _build_request_overrides(headers: dict[str, str]) -> RequestGoogleOverrides | None:
    if not MCP_ALLOW_REQUEST_OVERRIDES:
        return None

    client_id = headers.get(MCP_GOOGLE_CLIENT_ID_HEADER, "")
    client_secret = headers.get(MCP_GOOGLE_CLIENT_SECRET_HEADER, "")
    refresh_token = headers.get(MCP_GOOGLE_REFRESH_TOKEN_HEADER, "")
    missing_required: list[str] = []
    if MCP_REQUIRE_REQUEST_GOOGLE_CLIENT_ID and not client_id:
        missing_required.append(MCP_GOOGLE_CLIENT_ID_HEADER)
    if MCP_REQUIRE_REQUEST_GOOGLE_CLIENT_SECRET and not client_secret:
        missing_required.append(MCP_GOOGLE_CLIENT_SECRET_HEADER)
    if MCP_REQUIRE_REQUEST_GOOGLE_REFRESH_TOKEN and not refresh_token:
        missing_required.append(MCP_GOOGLE_REFRESH_TOKEN_HEADER)
    if missing_required:
        raise ValueError("Missing required header(s): " + ", ".join(missing_required) + ".")

    if not any((client_id, client_secret, refresh_token)):
        return None

    if not all((client_id, client_secret, refresh_token)):
        raise ValueError(
            "Partial Google BYOK headers supplied. Provide all of: "
            + ", ".join(
                [
                    MCP_GOOGLE_CLIENT_ID_HEADER,
                    MCP_GOOGLE_CLIENT_SECRET_HEADER,
                    MCP_GOOGLE_REFRESH_TOKEN_HEADER,
                ]
            )
            + "."
        )

    return RequestGoogleOverrides(
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=refresh_token,
    )


def _resolve_request_client(
    header_items: list[tuple[bytes, bytes]],
) -> tuple[GoogleWorkspaceClient | None, dict[str, Any]]:
    headers = _normalize_header_map(header_items)
    overrides = _build_request_overrides(headers)
    if overrides is None:
        if MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK:
            missing = [
                MCP_GOOGLE_CLIENT_ID_HEADER,
                MCP_GOOGLE_CLIENT_SECRET_HEADER,
                MCP_GOOGLE_REFRESH_TOKEN_HEADER,
            ]
            raise ValueError("Missing required header(s): " + ", ".join(missing) + ".")
        return None, {"auth_mode": "default_credentials"}

    cache_key = _fingerprint_request_overrides(overrides)
    byok_client, cache_hit = BYOK_CLIENT_CACHE.get_or_create(
        cache_key,
        lambda: GoogleWorkspaceClient(
            credentials_path=None,
            token_path=None,
            scopes=SCOPES,
            authorized_user_info=_build_authorized_user_info(overrides),
            persist_token=False,
        ),
    )
    return byok_client, {"auth_mode": "byok", "byok_cache_hit": cache_hit}


class ActiveClientProxy:
    def __init__(self, default_client: GoogleWorkspaceClient):
        self.default_client = default_client

    def _resolve(self) -> GoogleWorkspaceClient:
        request_client = ACTIVE_GOOGLE_CLIENT.get()
        return request_client or self.default_client

    def __getattr__(self, item: str):
        return getattr(self._resolve(), item)


SCOPES = parse_scopes(GOOGLE_SCOPES_RAW)
if not SCOPES:
    raise RuntimeError("GOOGLE_SCOPES is not set")

if MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK and not MCP_ALLOW_REQUEST_OVERRIDES:
    raise RuntimeError(
        "MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK=true requires MCP_ALLOW_REQUEST_OVERRIDES=true."
    )

DEFAULT_CLIENT = GoogleWorkspaceClient(
    credentials_path=GOOGLE_CREDENTIALS_PATH,
    token_path=GOOGLE_TOKEN_PATH,
    scopes=SCOPES,
)
BYOK_CLIENT_CACHE = ByokClientCache(
    max_size=MCP_BYOK_CLIENT_CACHE_SIZE,
    ttl_seconds=MCP_BYOK_CLIENT_CACHE_TTL_SECONDS,
)
ACTIVE_GOOGLE_CLIENT: contextvars.ContextVar[GoogleWorkspaceClient | None] = (
    contextvars.ContextVar("active_google_client", default=None)
)
ACTIVE_REQUEST_HEADERS: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "active_request_headers", default={}
)
client = ActiveClientProxy(DEFAULT_CLIENT)


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


def _estimate_bytes(data: Any) -> int:
    if data is None:
        return 0
    if isinstance(data, (bytes, bytearray)):
        return len(data)
    if isinstance(data, str):
        return len(data.encode("utf-8"))
    try:
        return len(json.dumps(data, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(data).encode("utf-8"))


def _validate_raw_request_url(url: str) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        raise ValueError(
            "raw_request url cannot be empty. Example: /drive/v3/files or https://www.googleapis.com/drive/v3/files"
        )
    if not MCP_RAW_STRICT:
        return normalize_url(cleaned)

    allowed_hosts = {
        "www.googleapis.com",
        "gmail.googleapis.com",
        "drive.googleapis.com",
        "sheets.googleapis.com",
        "docs.googleapis.com",
        "slides.googleapis.com",
        "calendar.googleapis.com",
        "oauth2.googleapis.com",
        "people.googleapis.com",
        "youtube.googleapis.com",
        "analyticsdata.googleapis.com",
        "searchconsole.googleapis.com",
        "mybusiness.googleapis.com",
        "mybusinessaccountmanagement.googleapis.com",
        "mybusinessbusinessinformation.googleapis.com",
        "mybusinessnotifications.googleapis.com",
        "mybusinessplaceactions.googleapis.com",
        "mybusinessqanda.googleapis.com",
        "mybusinessverifications.googleapis.com",
        "businessprofileperformance.googleapis.com",
        "shoppingcontent.googleapis.com",
        "merchantapi.googleapis.com",
        "adsense.googleapis.com",
        "geocode.googleapis.com",
        "places.googleapis.com",
        "routes.googleapis.com",
        "maps.googleapis.com",
    }
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        parsed = urllib.parse.urlparse(cleaned)
        host = (parsed.hostname or "").lower()
        if host not in allowed_hosts:
            raise ValueError(
                "raw_request url host is not allowed in strict mode. "
                "Use a Google API host or a relative path like /drive/v3/files."
            )
        return cleaned

    if not cleaned.startswith("/"):
        cleaned = "/" + cleaned
    return f"https://www.googleapis.com{cleaned}"


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
    payload["meta"].setdefault("bytes_out", 0)
    payload["meta"].setdefault("serialization_ms", 0.0)
    start = time.perf_counter()
    last_len = -1
    raw = ""
    for _ in range(3):
        raw = json_dumps(payload)
        raw_len = len(raw.encode("utf-8"))
        if raw_len == last_len:
            break
        payload["meta"]["bytes_out"] = raw_len
        last_len = raw_len
    payload["meta"]["serialization_ms"] = round(
        (time.perf_counter() - start) * 1000, 2
    )
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
    suggested_fields: str | None = None,
) -> str:
    request_id = uuid.uuid4().hex
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
                "bytes_in": _estimate_bytes(result),
                "request_id": request_id,
            }
            meta.update(_server_meta())
            if meta_extra:
                meta.update(meta_extra)
            if result_meta:
                meta.update(result_meta)
            if "cached_session" not in meta:
                meta["cached_session"] = client.is_session_cached()
            if MCP_LOG_REQUESTS:
                logger.info(
                    "tool_ok request_id=%s api=%s action=%s elapsed_ms=%s retries=%s",
                    request_id,
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
            if suggested_fields:
                last_error["suggested_fields"] = suggested_fields
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
                "bytes_in": 0,
                "request_id": request_id,
            }
            meta.update(_server_meta())
            if meta_extra:
                meta.update(meta_extra)
            if "cached_session" not in meta:
                meta["cached_session"] = client.is_session_cached()
            if MCP_LOG_REQUESTS:
                logger.warning(
                    "tool_error request_id=%s api=%s action=%s retries=%s error=%s",
                    request_id,
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


def _require_confirm(action: str, confirm: bool) -> None:
    if not confirm:
        raise ValueError(f"confirm=true is required to {action}.")


def _attach_page_meta(data: Any, cached: bool) -> tuple[Any, dict[str, Any]]:
    meta: dict[str, Any] = {"cached_service": cached}
    if isinstance(data, dict):
        next_token = data.get("nextPageToken")
        if next_token:
            meta["next_page_token"] = next_token
    return data, meta


def _server_meta() -> dict[str, Any]:
    return {
        "server_instance_id": SERVER_INSTANCE_ID,
        "server_uptime_ms": round((time.monotonic() - SERVER_START_MONO) * 1000, 2),
        "server_version": MCP_SERVER_VERSION,
    }


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


def _header_map(headers: list[dict[str, str]]) -> dict[str, str]:
    return {
        str(entry.get("name", "")).lower(): str(entry.get("value", ""))
        for entry in headers or []
        if entry.get("name")
    }


def _gmail_sender_key(headers: dict[str, str]) -> dict[str, str]:
    from_header = headers.get("from", "")
    sender_name, sender_email = parseaddr(from_header)
    sender_email = sender_email.lower()
    domain = sender_email.split("@", 1)[1] if "@" in sender_email else ""
    list_id = headers.get("list-id", "").strip()
    key = list_id.lower() or domain or sender_email or "unknown"
    return {
        "key": key,
        "sender_name": sender_name,
        "sender_email": sender_email,
        "domain": domain,
        "list_id": list_id,
    }


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(parsed, maximum))


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, size)
    return [items[index : index + size] for index in range(0, len(items), size)]


def _gmail_list_message_ids(
    service,
    *,
    query: str,
    label_ids: list[str] | None = None,
    include_spam_trash: bool = False,
    max_messages: int = 500,
    page_size: int = 500,
    page_token: str = "",
) -> tuple[list[dict[str, Any]], str, int | None, int]:
    remaining = _clamp_int(max_messages, minimum=1, maximum=5000)
    page_size = _clamp_int(page_size, minimum=1, maximum=500)
    next_token = page_token or ""
    messages: list[dict[str, Any]] = []
    estimate: int | None = None
    pages = 0
    while remaining > 0:
        request = service.users().messages().list(
            userId="me",
            q=query or None,
            labelIds=label_ids or None,
            includeSpamTrash=include_spam_trash,
            maxResults=min(page_size, remaining),
            pageToken=next_token or None,
        )
        data = request.execute()
        pages += 1
        if estimate is None and data.get("resultSizeEstimate") is not None:
            try:
                estimate = int(data.get("resultSizeEstimate"))
            except (TypeError, ValueError):
                estimate = None
        batch = data.get("messages", []) or []
        messages.extend(batch)
        remaining -= len(batch)
        next_token = data.get("nextPageToken", "") or ""
        if not next_token or not batch:
            break
    return messages, next_token, estimate, pages


def _gmail_get_metadata_batch(
    service,
    message_ids: list[str],
    metadata_headers: list[str] | None = None,
    *,
    max_messages: int = 500,
) -> list[dict[str, Any]]:
    headers = metadata_headers or [
        "From",
        "To",
        "Subject",
        "Date",
        "List-ID",
        "List-Unsubscribe",
    ]
    results: list[dict[str, Any]] = []
    for message_id in message_ids[: _clamp_int(max_messages, minimum=1, maximum=5000)]:
        if not message_id:
            continue
        data = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=headers,
            )
            .execute()
        )
        header_map = _header_map(data.get("payload", {}).get("headers", []) or [])
        results.append(
            {
                "id": data.get("id"),
                "threadId": data.get("threadId"),
                "labelIds": data.get("labelIds", []) or [],
                "snippet": data.get("snippet", ""),
                "internalDate": data.get("internalDate"),
                "headers": header_map,
            }
        )
    return results


def _require_maps_api_key() -> str:
    headers = ACTIVE_REQUEST_HEADERS.get() or {}
    api_key = headers.get(MCP_GOOGLE_MAPS_API_KEY_HEADER, "") or GOOGLE_MAPS_API_KEY
    if not api_key:
        raise ValueError(
            f"Missing Google Maps API key. Provide {MCP_GOOGLE_MAPS_API_KEY_HEADER} through the portal or GOOGLE_MAPS_API_KEY in the service environment."
        )
    return api_key


def _maps_request(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    session, _ = client.get_session()
    request_headers = dict(headers or {})
    request_headers.setdefault("X-Goog-Api-Key", _require_maps_api_key())
    response = session.request(method.upper(), url, params=params, json=json_body, headers=request_headers)
    payload: dict[str, Any] = {"status": response.status_code}
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload["json"] = response.json()
    else:
        payload["text"] = response.text[:20000]
    if not response.ok:
        raise RuntimeError(f"Maps request failed with status {response.status_code}: {payload}")
    return payload


def _google_json_request(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session, _ = client.get_session()
    response = session.request(method.upper(), url, params=params, json=json_body)
    payload: dict[str, Any] = {"status": response.status_code}
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload["json"] = response.json()
    else:
        payload["text"] = response.text[:20000]
    if not response.ok:
        raise RuntimeError(f"Google API request failed with status {response.status_code}: {payload}")
    return payload


async def run_blocking(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


READ_GOOGLE_TOOL = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)
WRITE_GOOGLE_TOOL = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)
DESTRUCTIVE_GOOGLE_TOOL = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=True,
)
NAVIGATION_TOOL = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

READ_ONLY_TOOLS = {
    "drive_list_files",
    "drive_search_files",
    "drive_batch_get_metadata",
    "drive_get_file",
    "drive_download_file",
    "docs_get_document",
    "sheets_get_spreadsheet",
    "sheets_get_values",
    "sheets_batch_get_values",
    "slides_get_presentation",
    "gmail_list_labels",
    "gmail_list_messages",
    "gmail_search_messages",
    "gmail_get_message",
    "gmail_get_message_headers",
    "gmail_get_message_body",
    "gmail_batch_get_metadata",
    "gmail_list_threads",
    "gmail_get_thread",
    "calendar_list_calendars",
    "calendar_get_calendar",
    "calendar_list_events",
    "calendar_search_events",
    "calendar_batch_get_events",
    "calendar_get_event",
    "mcp_health_check",
}
WRITE_TOOLS = {
    "drive_create_folder",
    "drive_upload_file",
    "docs_create_document",
    "docs_insert_text",
    "sheets_create_spreadsheet",
    "slides_create_presentation",
    "gmail_create_label",
    "gmail_create_draft",
    "gmail_untrash_message",
    "calendar_create_calendar",
    "calendar_create_event",
    "calendar_quick_add",
}
DESTRUCTIVE_TOOLS = {
    "google_raw_request",
    "drive_delete_file",
    "drive_empty_trash",
    "drive_purge_trash",
    "docs_replace_text",
    "sheets_update_values",
    "slides_replace_text",
    "gmail_delete_label",
    "gmail_send_message",
    "gmail_send_raw_message",
    "gmail_send_draft",
    "gmail_modify_message_labels",
    "gmail_trash_message",
    "gmail_delete_message",
    "calendar_delete_calendar",
    "calendar_update_event",
    "calendar_delete_event",
}
NAVIGATION_TOOLS = {
    "google_mcp_welcome",
    "google_mcp_list_capabilities",
    "google_mcp_get_endpoint_coverage",
    "google_mcp_get_tool_usage",
}

TOOL_DESCRIPTIONS = {
    "google_raw_request": (
        "Use this advanced Google API escape hatch only when a documented endpoint is "
        "not covered by a curated tool. It can read, write, or delete Google data "
        "depending on the HTTP method and URL, and returns a bounded HTTP response."
    ),
    "drive_list_files": (
        "Use this read-only Google Drive tool to list accessible files with optional "
        "query, ordering, pagination, and fields selection. Returns compact file "
        "metadata plus pagination metadata."
    ),
    "drive_search_files": (
        "Use this read-only Google Drive tool when you have a Drive query string and "
        "need matching files. Returns compact file metadata and the next page token."
    ),
    "drive_batch_get_metadata": (
        "Use this read-only Google Drive tool to fetch metadata for known file IDs in "
        "one call. Prefer fields selection to keep output small."
    ),
    "drive_get_file": (
        "Use this read-only Google Drive tool to fetch metadata for one known file ID. "
        "Returns ID, name, MIME type, parents, and other requested fields."
    ),
    "drive_create_folder": (
        "Use this Google Drive write tool to create a folder, optionally under a "
        "parent folder. It writes a new Drive folder and returns its ID and name."
    ),
    "drive_upload_file": (
        "Use this Google Drive write tool to upload text or base64 content as a new "
        "file. It creates a Drive file and returns its ID, name, MIME type, and parents."
    ),
    "drive_download_file": (
        "Use this read-only Google Drive tool to get a download/export URL or bounded "
        "base64 content for a file. Prefer return_mode='url' unless bytes are required."
    ),
    "drive_delete_file": (
        "Use this Google Drive destructive tool to trash a file or permanently delete "
        "one with confirm=true. Permanent deletion cannot be recovered through this MCP."
    ),
    "drive_empty_trash": (
        "Use this destructive Google Drive tool only to permanently delete all trashed "
        "files after confirm=true. It affects the authenticated user's trash."
    ),
    "drive_purge_trash": (
        "Compatibility alias for drive_empty_trash. Use only with confirm=true when the "
        "user explicitly wants Drive trash permanently emptied."
    ),
    "docs_create_document": (
        "Use this Google Docs write tool to create a blank document by title. It writes "
        "a new Doc and returns the provider response."
    ),
    "docs_get_document": (
        "Use this read-only Google Docs tool to fetch one document by ID. Use fields to "
        "avoid returning full document structure unless needed."
    ),
    "docs_insert_text": (
        "Use this Google Docs write tool to insert text at a document index. It changes "
        "the target document and returns the batchUpdate response."
    ),
    "docs_replace_text": (
        "Use this destructive Google Docs tool to replace matching document text. It can "
        "overwrite content across the document and returns the batchUpdate response."
    ),
    "sheets_create_spreadsheet": (
        "Use this Google Sheets write tool to create a blank spreadsheet by title. It "
        "writes a new Sheet and returns the provider response."
    ),
    "sheets_get_spreadsheet": (
        "Use this read-only Google Sheets tool to fetch spreadsheet metadata. Use fields "
        "to keep responses compact."
    ),
    "sheets_get_values": (
        "Use this read-only Google Sheets values tool to read one A1 range from a "
        "spreadsheet. Returns the values response from Google."
    ),
    "sheets_batch_get_values": (
        "Use this read-only Google Sheets values tool to read multiple A1 ranges in one "
        "call. Returns range values and render metadata."
    ),
    "sheets_update_values": (
        "Use this destructive Google Sheets values tool to overwrite cells in one A1 "
        "range. It writes values and returns update counts."
    ),
    "slides_create_presentation": (
        "Use this Google Slides write tool to create a blank presentation by title. It "
        "writes a new deck and returns the provider response."
    ),
    "slides_get_presentation": (
        "Use this read-only Google Slides tool to fetch presentation metadata and slide "
        "IDs. Use fields for compact output."
    ),
    "slides_replace_text": (
        "Use this destructive Google Slides tool to replace matching text throughout a "
        "presentation. It can overwrite slide content."
    ),
    "gmail_list_labels": (
        "Use this read-only Gmail tool to list labels for the authenticated mailbox. It "
        "defaults to compact label IDs and names."
    ),
    "gmail_create_label": (
        "Use this Gmail write tool to create a mailbox label. It writes a new label and "
        "returns the label resource."
    ),
    "gmail_delete_label": (
        "Use this destructive Gmail tool to delete a mailbox label. Set confirm=true "
        "when confirmation is required by runtime config."
    ),
    "gmail_list_messages": (
        "Use this read-only Gmail tool to list message IDs by query or label. It returns "
        "message stubs and pagination, not full message bodies."
    ),
    "gmail_search_messages": (
        "Use this read-only Gmail tool when you have a Gmail search query and need "
        "matching message IDs. Fetch bodies separately only when needed."
    ),
    "gmail_get_message": (
        "Use this read-only Gmail tool to fetch one message by ID. It defaults to "
        "metadata to avoid large or private body output."
    ),
    "gmail_get_message_headers": (
        "Use this read-only Gmail tool to fetch selected headers for one message. It "
        "returns a compact header map, snippet, labels, and thread ID."
    ),
    "gmail_get_message_body": (
        "Use this read-only Gmail tool to extract text/plain or text/html body content "
        "for one message when the user has asked to inspect email content."
    ),
    "gmail_batch_get_metadata": (
        "Use this read-only Gmail tool to fetch metadata for multiple message IDs. It "
        "accepts snake_case arguments and common camelCase aliases through the MCP wrapper."
    ),
    "gmail_list_threads": (
        "Use this read-only Gmail tool to list thread IDs by query or labels. Fetch a "
        "thread by ID for message-level details."
    ),
    "gmail_get_thread": (
        "Use this read-only Gmail tool to fetch one thread by ID. It defaults to "
        "metadata to reduce private content exposure."
    ),
    "gmail_send_message": (
        "Use this Gmail irreversible send tool only when the user explicitly approves "
        "sending. It sends a message from the authenticated mailbox."
    ),
    "gmail_send_raw_message": (
        "Use this Gmail irreversible send tool only for approved raw MIME sends. It "
        "sends the supplied base64url MIME payload."
    ),
    "gmail_create_draft": (
        "Use this Gmail write tool to create a draft instead of sending. Prefer drafts "
        "when user approval is still needed."
    ),
    "gmail_send_draft": (
        "Use this Gmail irreversible send tool only after explicit approval to send an "
        "existing draft."
    ),
    "gmail_modify_message_labels": (
        "Use this destructive Gmail tool to add or remove labels from a message. It "
        "changes mailbox organization for the target message."
    ),
    "gmail_trash_message": (
        "Use this destructive Gmail tool to move a message to trash. It changes mailbox "
        "state and may require confirm=true."
    ),
    "gmail_untrash_message": (
        "Use this Gmail write tool to restore a trashed message. It changes mailbox "
        "state but does not permanently delete data."
    ),
    "gmail_delete_message": (
        "Use this destructive Gmail tool only when the user explicitly requests permanent "
        "message deletion. It cannot be undone through this MCP."
    ),
    "calendar_list_calendars": (
        "Use this read-only Google Calendar tool to list calendars visible to the user. "
        "Returns compact calendar metadata and pagination."
    ),
    "calendar_get_calendar": (
        "Use this read-only Google Calendar tool to fetch metadata for one calendar ID. "
        "Returns summary, timezone, access role, and requested fields."
    ),
    "calendar_create_calendar": (
        "Use this Google Calendar write tool to create a secondary calendar. It writes "
        "a new calendar and returns the provider response."
    ),
    "calendar_delete_calendar": (
        "Use this destructive Google Calendar tool only when the user explicitly wants "
        "to delete a calendar. It may require confirm=true."
    ),
    "calendar_list_events": (
        "Use this read-only Google Calendar tool to list events, usually with time_min "
        "and time_max. The wrapper accepts common camelCase aliases such as timeMin and timeMax."
    ),
    "calendar_search_events": (
        "Use this read-only Google Calendar tool to search events with a query and "
        "optional time window. Returns compact event metadata and pagination."
    ),
    "calendar_batch_get_events": (
        "Use this read-only Google Calendar tool to fetch multiple known event IDs from "
        "one calendar. Returns compact event metadata."
    ),
    "calendar_get_event": (
        "Use this read-only Google Calendar tool to fetch one event by calendar ID and "
        "event ID. Use fields to keep output compact."
    ),
    "calendar_create_event": (
        "Use this Google Calendar write tool to create an event in a calendar. It can "
        "invite attendees and writes to the authenticated user's calendar."
    ),
    "calendar_update_event": (
        "Use this destructive Google Calendar tool to patch an existing event. It can "
        "overwrite event details and send attendee updates."
    ),
    "calendar_delete_event": (
        "Use this destructive Google Calendar tool to delete an event. It may notify "
        "attendees depending on send_updates and requires confirmed intent."
    ),
    "calendar_quick_add": (
        "Use this Google Calendar write tool to create an event from natural language "
        "text. Prefer explicit create_event when dates or attendees must be controlled."
    ),
    "mcp_health_check": (
        "Use this diagnostic tool to verify Google OAuth readiness, scopes, cache state, "
        "and optional safe API checks. It never returns credential values."
    ),
    "google_mcp_welcome": (
        "Use this read-only navigation tool first to understand Google MCP setup, tool "
        "groups, safety rules, and recommended next discovery calls."
    ),
    "google_mcp_list_capabilities": (
        "Use this read-only navigation tool to list Google provider categories, grouped "
        "tools, and read/write/destructive risk levels."
    ),
    "google_mcp_get_endpoint_coverage": (
        "Use this read-only navigation tool to inspect endpoint parity against official "
        "Google REST discovery resources by API, resource, or coverage status."
    ),
    "google_mcp_get_tool_usage": (
        "Use this read-only navigation tool to get usage, side effects, and related "
        "tools for a specific Google MCP tool."
    ),
}

COMMON_PARAMETER_DESCRIPTIONS = {
    "query": "Provider query string used to filter results.",
    "page_size": "Maximum number of items to request from Google for this page.",
    "max_results": "Maximum number of items to request from Google for this page.",
    "fields": "Optional Google partial-response fields selector for compact output.",
    "order_by": "Optional provider order expression.",
    "page_token": "Provider pagination token returned by a previous list call.",
    "file_id": "Google Drive file ID.",
    "file_ids": "List of Google Drive file IDs.",
    "name": "Provider-visible resource name.",
    "parent_id": "Optional Google Drive parent folder ID.",
    "allow_any_parent": "Set true only when intentionally bypassing the configured Drive parent allowlist.",
    "content": "Text content or base64 content to upload.",
    "mime_type": "MIME type for uploaded content.",
    "is_base64": "Set true when content is base64 encoded.",
    "export_mime_type": "MIME type to export Google-native files as.",
    "include_content": "Set true to include bounded base64 content in the response.",
    "return_mode": "Return mode for Drive downloads.",
    "max_bytes": "Maximum bytes to return when including file content.",
    "range_start": "Optional byte-range start for downloads.",
    "range_end": "Optional byte-range end for downloads.",
    "mode": "Deletion mode.",
    "confirm": "Required true for high-risk operations when confirmation is enforced.",
    "title": "Provider-visible title for the new resource.",
    "document_id": "Google Docs document ID.",
    "text": "Text to insert or quick-add.",
    "index": "Google Docs structural insertion index.",
    "contains_text": "Text to find and replace.",
    "replace_text": "Replacement text.",
    "match_case": "Whether text matching should be case-sensitive.",
    "spreadsheet_id": "Google Sheets spreadsheet ID.",
    "range_a1": "A1 notation range, for example Sheet1!A1:C20.",
    "ranges": "List of A1 notation ranges.",
    "value_render_option": "Google Sheets value render option.",
    "date_time_render_option": "Google Sheets date/time render option.",
    "major_dimension": "Google Sheets major dimension.",
    "values": "Two-dimensional array of cell values.",
    "value_input_option": "How Google Sheets should interpret written values.",
    "presentation_id": "Google Slides presentation ID.",
    "label_id": "Gmail label ID.",
    "label_list_visibility": "Gmail label list visibility value.",
    "message_list_visibility": "Gmail message list visibility value.",
    "label_ids": "Gmail label IDs used to filter results.",
    "include_spam_trash": "Whether to include spam and trash in Gmail list/search results.",
    "message_id": "Gmail message ID.",
    "message_ids": "List of Gmail message IDs.",
    "format": "Gmail message or thread format.",
    "metadata_headers": "Gmail metadata headers to return.",
    "headers": "HTTP headers for raw requests or Gmail header names, depending on tool.",
    "prefer_html": "Return HTML body when available instead of plain text.",
    "thread_id": "Gmail thread ID.",
    "to": "Recipient email address list accepted by Gmail.",
    "subject": "Email subject line.",
    "body": "Email body content.",
    "cc": "Optional CC recipients.",
    "bcc": "Optional BCC recipients.",
    "reply_to": "Optional Reply-To header.",
    "from_alias": "Optional configured Gmail send-as alias.",
    "is_html": "Set true when email body is HTML.",
    "raw_base64": "Base64url-encoded raw MIME message.",
    "draft_id": "Gmail draft ID.",
    "add_label_ids": "Gmail label IDs to add.",
    "remove_label_ids": "Gmail label IDs to remove.",
    "calendar_id": "Google Calendar calendar ID, often 'primary'.",
    "time_min": "Inclusive lower event time bound as RFC3339 timestamp.",
    "time_max": "Exclusive upper event time bound as RFC3339 timestamp.",
    "single_events": "Whether recurring events should be expanded into instances.",
    "event_ids": "List of Google Calendar event IDs.",
    "event_id": "Google Calendar event ID.",
    "summary": "Calendar summary or event title.",
    "description": "Optional provider-visible description.",
    "time_zone": "IANA time zone such as America/New_York or UTC.",
    "start_iso": "Event start as RFC3339 date-time or date for all-day events.",
    "end_iso": "Event end as RFC3339 date-time or date for all-day events.",
    "location": "Optional event location.",
    "attendees": "Optional attendee email addresses.",
    "all_day": "Set true to create all-day date events.",
    "event_patch": "Google Calendar events.patch request body.",
    "send_updates": "How Google Calendar should notify attendees.",
    "run_checks": "Whether to call safe Google API health checks.",
    "warm_all": "Whether to warm all supported Google service clients.",
    "doc_id": "Optional document ID for deeper Docs health check.",
    "sheet_id": "Optional spreadsheet ID for deeper Sheets health check.",
    "slide_id": "Optional presentation ID for deeper Slides health check.",
    "method": "HTTP method for google_raw_request.",
    "url": "Google API URL or relative path for google_raw_request.",
    "params": "Optional query parameters for google_raw_request.",
    "json_body": "Optional JSON request body for google_raw_request.",
    "category": "Optional capability category filter.",
    "api": "Optional Google API filter such as drive, docs, sheets, slides, gmail, or calendar.",
    "resource": "Optional provider resource filter such as files, users.messages, or events.",
    "status": "Optional coverage status filter.",
    "tool_name": "Google MCP tool name to describe.",
}

PARAMETER_ENUMS = {
    "method": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    "return_mode": ["", "url", "base64", "both"],
    "mode": ["trash", "permanent"],
    "format": ["minimal", "metadata", "full", "raw"],
    "value_input_option": ["RAW", "USER_ENTERED"],
    "label_list_visibility": ["labelShow", "labelShowIfUnread", "labelHide"],
    "message_list_visibility": ["show", "hide"],
    "send_updates": ["all", "externalOnly", "none"],
    "status": [
        "",
        "implemented",
        "partially_implemented",
        "missing",
        "intentionally_excluded",
        "blocked_scope",
    ],
    "api": ["", "drive", "docs", "sheets", "slides", "gmail", "calendar"],
}

CAPABILITY_GROUPS = [
    {
        "category": "Drive API v3 files",
        "read": [
            "drive_list_files",
            "drive_search_files",
            "drive_batch_get_metadata",
            "drive_get_file",
            "drive_download_file",
        ],
        "write": ["drive_create_folder", "drive_upload_file"],
        "destructive": ["drive_delete_file", "drive_empty_trash", "drive_purge_trash"],
    },
    {
        "category": "Docs API v1 documents",
        "read": ["docs_get_document"],
        "write": ["docs_create_document", "docs_insert_text"],
        "destructive": ["docs_replace_text"],
    },
    {
        "category": "Sheets API v4 spreadsheets and values",
        "read": [
            "sheets_get_spreadsheet",
            "sheets_get_values",
            "sheets_batch_get_values",
        ],
        "write": ["sheets_create_spreadsheet"],
        "destructive": ["sheets_update_values"],
    },
    {
        "category": "Slides API v1 presentations",
        "read": ["slides_get_presentation"],
        "write": ["slides_create_presentation"],
        "destructive": ["slides_replace_text"],
    },
    {
        "category": "Gmail API v1 labels, messages, threads, drafts",
        "read": [
            "gmail_list_labels",
            "gmail_list_messages",
            "gmail_search_messages",
            "gmail_get_message",
            "gmail_get_message_headers",
            "gmail_get_message_body",
            "gmail_batch_get_metadata",
            "gmail_list_threads",
            "gmail_get_thread",
        ],
        "write": ["gmail_create_label", "gmail_create_draft", "gmail_untrash_message"],
        "destructive": [
            "gmail_delete_label",
            "gmail_send_message",
            "gmail_send_raw_message",
            "gmail_send_draft",
            "gmail_modify_message_labels",
            "gmail_trash_message",
            "gmail_delete_message",
        ],
    },
    {
        "category": "Calendar API v3 calendars and events",
        "read": [
            "calendar_list_calendars",
            "calendar_get_calendar",
            "calendar_list_events",
            "calendar_search_events",
            "calendar_batch_get_events",
            "calendar_get_event",
        ],
        "write": [
            "calendar_create_calendar",
            "calendar_create_event",
            "calendar_quick_add",
        ],
        "destructive": [
            "calendar_delete_calendar",
            "calendar_update_event",
            "calendar_delete_event",
        ],
    },
    {
        "category": "MCP navigation and diagnostics",
        "read": [
            "google_mcp_welcome",
            "google_mcp_list_capabilities",
            "google_mcp_get_endpoint_coverage",
            "google_mcp_get_tool_usage",
            "mcp_health_check",
        ],
        "write": [],
        "destructive": ["google_raw_request"],
    },
]

ENDPOINT_COVERAGE = [
    {
        "api": "drive",
        "resource": "files",
        "status": "partially_implemented",
        "implemented": ["create", "delete", "emptyTrash", "export", "get", "list"],
        "missing": [
            "copy",
            "download",
            "generateCseToken",
            "generateIds",
            "listLabels",
            "modifyLabels",
            "update",
            "watch",
        ],
        "tool_refs": [
            "drive_list_files",
            "drive_search_files",
            "drive_get_file",
            "drive_batch_get_metadata",
            "drive_create_folder",
            "drive_upload_file",
            "drive_download_file",
            "drive_delete_file",
            "drive_empty_trash",
        ],
    },
    {
        "api": "drive",
        "resource": "about/apps/changes/comments/replies/permissions/drives/revisions/channels",
        "status": "missing",
        "implemented": [],
        "missing": [
            "about.get",
            "apps.get",
            "apps.list",
            "changes.getStartPageToken",
            "changes.list",
            "comments.*",
            "replies.*",
            "permissions.*",
            "drives.*",
            "revisions.*",
            "channels.stop",
        ],
        "tool_refs": ["google_raw_request"],
    },
    {
        "api": "docs",
        "resource": "documents",
        "status": "implemented",
        "implemented": ["create", "get", "batchUpdate"],
        "missing": [],
        "tool_refs": [
            "docs_create_document",
            "docs_get_document",
            "docs_insert_text",
            "docs_replace_text",
        ],
    },
    {
        "api": "sheets",
        "resource": "spreadsheets",
        "status": "partially_implemented",
        "implemented": ["create", "get"],
        "missing": ["batchUpdate", "getByDataFilter"],
        "tool_refs": ["sheets_create_spreadsheet", "sheets_get_spreadsheet"],
    },
    {
        "api": "sheets",
        "resource": "spreadsheets.values",
        "status": "partially_implemented",
        "implemented": ["batchGet", "get", "update"],
        "missing": [
            "append",
            "batchClear",
            "batchClearByDataFilter",
            "batchGetByDataFilter",
            "batchUpdate",
            "batchUpdateByDataFilter",
            "clear",
        ],
        "tool_refs": [
            "sheets_get_values",
            "sheets_batch_get_values",
            "sheets_update_values",
        ],
    },
    {
        "api": "sheets",
        "resource": "spreadsheets.developerMetadata/spreadsheets.sheets",
        "status": "missing",
        "implemented": [],
        "missing": ["developerMetadata.get", "developerMetadata.search", "sheets.copyTo"],
        "tool_refs": ["google_raw_request"],
    },
    {
        "api": "slides",
        "resource": "presentations",
        "status": "implemented",
        "implemented": ["batchUpdate", "create", "get"],
        "missing": [],
        "tool_refs": [
            "slides_create_presentation",
            "slides_get_presentation",
            "slides_replace_text",
        ],
    },
    {
        "api": "slides",
        "resource": "presentations.pages",
        "status": "missing",
        "implemented": [],
        "missing": ["get", "getThumbnail"],
        "tool_refs": ["google_raw_request"],
    },
    {
        "api": "gmail",
        "resource": "users.labels/users.messages/users.threads/users.drafts",
        "status": "partially_implemented",
        "implemented": [
            "labels.create",
            "labels.delete",
            "labels.list",
            "messages.delete",
            "messages.get",
            "messages.list",
            "messages.modify",
            "messages.send",
            "messages.trash",
            "messages.untrash",
            "threads.get",
            "threads.list",
            "drafts.create",
            "drafts.send",
        ],
        "missing": [
            "labels.get",
            "labels.patch",
            "labels.update",
            "messages.batchDelete",
            "messages.batchModify",
            "messages.import",
            "messages.insert",
            "messages.attachments.get",
            "threads.delete",
            "threads.modify",
            "threads.trash",
            "threads.untrash",
            "drafts.delete",
            "drafts.get",
            "drafts.list",
            "drafts.update",
        ],
        "tool_refs": [
            "gmail_list_labels",
            "gmail_create_label",
            "gmail_delete_label",
            "gmail_list_messages",
            "gmail_search_messages",
            "gmail_get_message",
            "gmail_get_message_headers",
            "gmail_get_message_body",
            "gmail_batch_get_metadata",
            "gmail_list_threads",
            "gmail_get_thread",
            "gmail_create_draft",
            "gmail_send_draft",
            "gmail_send_message",
            "gmail_send_raw_message",
            "gmail_modify_message_labels",
            "gmail_trash_message",
            "gmail_untrash_message",
            "gmail_delete_message",
        ],
    },
    {
        "api": "gmail",
        "resource": "users.history/users.settings/users.watch",
        "status": "blocked_scope",
        "implemented": [],
        "missing": ["history.list", "settings.*", "users.getProfile", "users.stop", "users.watch"],
        "tool_refs": ["mcp_health_check", "google_raw_request"],
    },
    {
        "api": "calendar",
        "resource": "calendars/calendarList/events",
        "status": "partially_implemented",
        "implemented": [
            "calendars.delete",
            "calendars.get",
            "calendars.insert",
            "calendarList.list",
            "events.delete",
            "events.get",
            "events.insert",
            "events.list",
            "events.patch",
            "events.quickAdd",
        ],
        "missing": [
            "calendars.clear",
            "calendars.patch",
            "calendars.update",
            "calendarList.delete",
            "calendarList.get",
            "calendarList.insert",
            "calendarList.patch",
            "calendarList.update",
            "calendarList.watch",
            "events.import",
            "events.instances",
            "events.move",
            "events.update",
            "events.watch",
        ],
        "tool_refs": [
            "calendar_list_calendars",
            "calendar_get_calendar",
            "calendar_create_calendar",
            "calendar_delete_calendar",
            "calendar_list_events",
            "calendar_search_events",
            "calendar_batch_get_events",
            "calendar_get_event",
            "calendar_create_event",
            "calendar_update_event",
            "calendar_delete_event",
            "calendar_quick_add",
        ],
    },
    {
        "api": "calendar",
        "resource": "acl/channels/colors/freebusy/settings",
        "status": "missing",
        "implemented": [],
        "missing": ["acl.*", "channels.stop", "colors.get", "freebusy.query", "settings.*"],
        "tool_refs": ["google_raw_request"],
    },
]


def _tool_registry() -> dict[str, Any]:
    manager = getattr(mcp, "_tool_manager", None)
    tools = getattr(manager, "_tools", None)
    if isinstance(tools, dict):
        return tools
    tools = getattr(manager, "tools", None)
    if isinstance(tools, dict):
        return tools
    return {}


def _annotation_for_tool(tool_name: str) -> ToolAnnotations:
    if tool_name in NAVIGATION_TOOLS:
        return NAVIGATION_TOOL
    if tool_name in READ_ONLY_TOOLS:
        return READ_GOOGLE_TOOL
    if tool_name in WRITE_TOOLS:
        return WRITE_GOOGLE_TOOL
    if tool_name in DESTRUCTIVE_TOOLS:
        return DESTRUCTIVE_GOOGLE_TOOL
    return WRITE_GOOGLE_TOOL


def _parameter_description(tool_name: str, parameter_name: str) -> str:
    if tool_name == "google_mcp_get_endpoint_coverage" and parameter_name == "status":
        return "Coverage status filter: implemented, missing, intentionally_excluded, or blocked_scope."
    if tool_name == "google_mcp_get_endpoint_coverage" and parameter_name == "resource":
        return "Provider resource filter, for example files, users.messages, or events."
    return COMMON_PARAMETER_DESCRIPTIONS.get(
        parameter_name,
        f"Argument for {tool_name}.",
    )


def _apply_tool_metadata() -> None:
    for name, tool in _tool_registry().items():
        tool.annotations = _annotation_for_tool(name)
        description = TOOL_DESCRIPTIONS.get(name)
        if description:
            tool.description = description
        properties = tool.parameters.get("properties", {})
        for parameter_name, schema in properties.items():
            if isinstance(schema, dict):
                schema.setdefault("description", _parameter_description(name, parameter_name))
                enum_values = PARAMETER_ENUMS.get(parameter_name)
                if enum_values and "enum" not in schema:
                    schema["enum"] = enum_values


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
            _validate_raw_request_url(url),
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
        data = request.execute()
        return _attach_page_meta(data, cached)

    return await run_tool(
        "drive",
        "list_files",
        _list_files,
        allow_retry=True,
        suggested_fields=DEFAULT_DRIVE_FIELDS,
    )


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
        data = request.execute()
        return _attach_page_meta(data, cached)

    return await run_tool(
        "drive",
        "search_files",
        _search_files,
        allow_retry=True,
        suggested_fields=DEFAULT_DRIVE_FIELDS,
    )


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

    return await run_tool(
        "drive",
        "batch_get_metadata",
        _batch_get,
        allow_retry=True,
        suggested_fields=DEFAULT_DRIVE_GET_FIELDS,
    )


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

    return await run_tool(
        "drive",
        "get_file",
        _get_file,
        allow_retry=True,
        suggested_fields=DEFAULT_DRIVE_GET_FIELDS,
    )


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
    return_mode: str = "",
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    range_start: int | None = None,
    range_end: int | None = None,
) -> str:
    """Download a file from Drive. For Google Docs/Sheets/Slides, use export."""

    def _download():
        if not file_id:
            raise ValueError("file_id cannot be empty")
        mode = (return_mode or "").strip().lower()
        if mode and mode not in {"url", "base64", "both"}:
            raise ValueError("return_mode must be one of: url, base64, both.")
        include_data = include_content
        if mode:
            include_data = mode in {"base64", "both"}
        include_url = mode in {"", "url", "both"}
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
        }
        if include_url:
            response_payload["download_url"] = download_url
        if not include_data:
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
async def drive_delete_file(
    file_id: str,
    mode: str = "trash",
    confirm: bool = False,
) -> str:
    """Trash or permanently delete a Drive file."""

    def _delete_file():
        if not file_id:
            raise ValueError("file_id cannot be empty")
        action = (mode or "trash").strip().lower()
        if action not in {"trash", "permanent"}:
            raise ValueError("mode must be one of: trash, permanent.")
        service, cached = client.get_service("drive", "v3")
        if action == "trash":
            request = service.files().update(
                fileId=file_id,
                body={"trashed": True},
                fields="id,trashed",
            )
            return request.execute(), {"cached_service": cached}
        _require_confirm("permanently delete a Drive file", confirm)
        service.files().delete(fileId=file_id).execute()
        return {"id": file_id, "deleted": True}, {"cached_service": cached}

    return await run_tool("drive", "delete_file", _delete_file, allow_retry=False)


@mcp.tool()
async def drive_empty_trash(confirm: bool = False) -> str:
    """Permanently delete all files in Drive trash."""

    def _empty_trash():
        _require_confirm("empty Drive trash", confirm)
        service, cached = client.get_service("drive", "v3")
        request = service.files().emptyTrash()
        request.execute()
        return {"status": "ok"}, {"cached_service": cached}

    return await run_tool("drive", "empty_trash", _empty_trash, allow_retry=False)


@mcp.tool()
async def drive_purge_trash(confirm: bool = False) -> str:
    """Alias for emptying Drive trash."""

    def _purge_trash():
        _require_confirm("purge Drive trash", confirm)
        service, cached = client.get_service("drive", "v3")
        request = service.files().emptyTrash()
        request.execute()
        return {"status": "ok"}, {"cached_service": cached}

    return await run_tool("drive", "purge_trash", _purge_trash, allow_retry=False)


@mcp.tool()
async def drive_about_get(fields: str = "user,storageQuota,importFormats,exportFormats") -> str:
    """Get Drive about/user/storage metadata."""

    def _about_get():
        service, cached = client.get_service("drive", "v3")
        return service.about().get(fields=fields).execute(), {"cached_service": cached}

    return await run_tool("drive", "about_get", _about_get, allow_retry=True)


@mcp.tool()
async def drive_list_permissions(file_id: str, fields: str = "permissions(id,type,role,emailAddress,domain),nextPageToken") -> str:
    """List Drive file permissions."""

    def _list_permissions():
        if not file_id:
            raise ValueError("file_id cannot be empty")
        service, cached = client.get_service("drive", "v3")
        request = service.permissions().list(fileId=file_id, fields=fields)
        data = request.execute()
        return _attach_page_meta(data, cached)

    return await run_tool("drive", "list_permissions", _list_permissions, allow_retry=True)


@mcp.tool()
async def drive_get_permission(file_id: str, permission_id: str, fields: str = "") -> str:
    """Get one Drive permission."""

    def _get_permission():
        if not file_id or not permission_id:
            raise ValueError("file_id and permission_id are required")
        service, cached = client.get_service("drive", "v3")
        request = service.permissions().get(
            fileId=file_id,
            permissionId=permission_id,
            fields=fields or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("drive", "get_permission", _get_permission, allow_retry=True)


@mcp.tool()
async def drive_create_permission(
    file_id: str,
    permission_body: dict[str, Any],
    send_notification_email: bool = False,
    confirm: bool = False,
) -> str:
    """Create a Drive sharing permission."""

    def _create_permission():
        if not file_id:
            raise ValueError("file_id cannot be empty")
        if not permission_body:
            raise ValueError("permission_body cannot be empty")
        _require_confirm("create a Drive sharing permission", confirm)
        service, cached = client.get_service("drive", "v3")
        request = service.permissions().create(
            fileId=file_id,
            body=permission_body,
            sendNotificationEmail=send_notification_email,
            fields="id,type,role,emailAddress,domain",
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("drive", "create_permission", _create_permission, allow_retry=False)


@mcp.tool()
async def drive_update_permission(
    file_id: str,
    permission_id: str,
    permission_body: dict[str, Any],
    confirm: bool = False,
) -> str:
    """Patch a Drive sharing permission."""

    def _update_permission():
        if not file_id or not permission_id:
            raise ValueError("file_id and permission_id are required")
        if not permission_body:
            raise ValueError("permission_body cannot be empty")
        _require_confirm("update a Drive sharing permission", confirm)
        service, cached = client.get_service("drive", "v3")
        request = service.permissions().update(
            fileId=file_id,
            permissionId=permission_id,
            body=permission_body,
            fields="id,type,role,emailAddress,domain",
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("drive", "update_permission", _update_permission, allow_retry=False)


@mcp.tool()
async def drive_delete_permission(file_id: str, permission_id: str, confirm: bool = False) -> str:
    """Delete a Drive sharing permission."""

    def _delete_permission():
        if not file_id or not permission_id:
            raise ValueError("file_id and permission_id are required")
        _require_confirm("delete a Drive sharing permission", confirm)
        service, cached = client.get_service("drive", "v3")
        service.permissions().delete(fileId=file_id, permissionId=permission_id).execute()
        return {"file_id": file_id, "permission_id": permission_id, "deleted": True}, {"cached_service": cached}

    return await run_tool("drive", "delete_permission", _delete_permission, allow_retry=False)


@mcp.tool()
async def drive_list_comments(file_id: str, fields: str = "comments(id,content,author,createdTime,modifiedTime,resolved),nextPageToken") -> str:
    """List Drive file comments."""

    def _list_comments():
        if not file_id:
            raise ValueError("file_id cannot be empty")
        service, cached = client.get_service("drive", "v3")
        data = service.comments().list(fileId=file_id, fields=fields).execute()
        return _attach_page_meta(data, cached)

    return await run_tool("drive", "list_comments", _list_comments, allow_retry=True)


@mcp.tool()
async def drive_create_comment(file_id: str, content: str, confirm: bool = False) -> str:
    """Create a Drive file comment."""

    def _create_comment():
        if not file_id:
            raise ValueError("file_id cannot be empty")
        if not content:
            raise ValueError("content cannot be empty")
        _ensure_confirmed("create a Drive comment", confirm)
        service, cached = client.get_service("drive", "v3")
        request = service.comments().create(fileId=file_id, body={"content": content}, fields="id,content")
        return request.execute(), {"cached_service": cached}

    return await run_tool("drive", "create_comment", _create_comment, allow_retry=False)


@mcp.tool()
async def drive_update_comment(file_id: str, comment_id: str, content: str, confirm: bool = False) -> str:
    """Update a Drive file comment."""

    def _update_comment():
        if not file_id or not comment_id:
            raise ValueError("file_id and comment_id are required")
        if not content:
            raise ValueError("content cannot be empty")
        _ensure_confirmed("update a Drive comment", confirm)
        service, cached = client.get_service("drive", "v3")
        request = service.comments().update(
            fileId=file_id,
            commentId=comment_id,
            body={"content": content},
            fields="id,content",
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("drive", "update_comment", _update_comment, allow_retry=False)


@mcp.tool()
async def drive_delete_comment(file_id: str, comment_id: str, confirm: bool = False) -> str:
    """Delete a Drive file comment."""

    def _delete_comment():
        if not file_id or not comment_id:
            raise ValueError("file_id and comment_id are required")
        _require_confirm("delete a Drive comment", confirm)
        service, cached = client.get_service("drive", "v3")
        service.comments().delete(fileId=file_id, commentId=comment_id).execute()
        return {"file_id": file_id, "comment_id": comment_id, "deleted": True}, {"cached_service": cached}

    return await run_tool("drive", "delete_comment", _delete_comment, allow_retry=False)


@mcp.tool()
async def drive_list_revisions(file_id: str, fields: str = "revisions(id,mimeType,modifiedTime,keepForever),nextPageToken") -> str:
    """List Drive file revisions."""

    def _list_revisions():
        if not file_id:
            raise ValueError("file_id cannot be empty")
        service, cached = client.get_service("drive", "v3")
        data = service.revisions().list(fileId=file_id, fields=fields).execute()
        return _attach_page_meta(data, cached)

    return await run_tool("drive", "list_revisions", _list_revisions, allow_retry=True)


@mcp.tool()
async def drive_list_shared_drives(page_size: int = 100, page_token: str = "") -> str:
    """List shared drives."""

    def _list_shared_drives():
        service, cached = client.get_service("drive", "v3")
        data = service.drives().list(
            pageSize=_clamp_int(page_size, minimum=1, maximum=100),
            pageToken=page_token or None,
        ).execute()
        return _attach_page_meta(data, cached)

    return await run_tool("drive", "list_shared_drives", _list_shared_drives, allow_retry=True)


@mcp.tool()
async def drive_copy_file(
    file_id: str,
    name: str = "",
    parent_id: str = "",
    allow_any_parent: bool = False,
    confirm: bool = False,
) -> str:
    """Copy a Drive file."""

    def _copy_file():
        if not file_id:
            raise ValueError("file_id cannot be empty")
        _ensure_confirmed("copy a Drive file", confirm)
        service, cached = client.get_service("drive", "v3")
        body: dict[str, Any] = {}
        if name:
            body["name"] = name
        effective_parent_id = _enforce_drive_allowlist(parent_id, allow_any_parent)
        if effective_parent_id:
            body["parents"] = [effective_parent_id]
        request = service.files().copy(fileId=file_id, body=body, fields=DEFAULT_DRIVE_GET_FIELDS)
        return request.execute(), {"cached_service": cached}

    return await run_tool("drive", "copy_file", _copy_file, allow_retry=False)


@mcp.tool()
async def drive_update_file_metadata(file_id: str, metadata: dict[str, Any], confirm: bool = False) -> str:
    """Patch Drive file metadata."""

    def _update_file_metadata():
        if not file_id:
            raise ValueError("file_id cannot be empty")
        if not metadata:
            raise ValueError("metadata cannot be empty")
        _ensure_confirmed("update Drive file metadata", confirm)
        service, cached = client.get_service("drive", "v3")
        request = service.files().update(fileId=file_id, body=metadata, fields=DEFAULT_DRIVE_GET_FIELDS)
        return request.execute(), {"cached_service": cached}

    return await run_tool("drive", "update_file_metadata", _update_file_metadata, allow_retry=False)


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

    return await run_tool(
        "docs",
        "get_document",
        _get_doc,
        allow_retry=True,
        suggested_fields=DEFAULT_DOCS_FIELDS,
    )


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
async def docs_batch_update(
    document_id: str,
    requests: list[dict[str, Any]],
    confirm: bool = False,
) -> str:
    """Run a controlled Google Docs batchUpdate request."""

    def _batch_update():
        if not document_id:
            raise ValueError("document_id cannot be empty")
        if not requests:
            raise ValueError("requests cannot be empty")
        _ensure_confirmed("run Docs batchUpdate", confirm)
        service, cached = client.get_service("docs", "v1")
        request = service.documents().batchUpdate(
            documentId=document_id,
            body={"requests": requests},
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("docs", "batch_update", _batch_update, allow_retry=False)


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

    return await run_tool(
        "sheets",
        "get_spreadsheet",
        _get_sheet,
        allow_retry=True,
        suggested_fields=DEFAULT_SHEETS_FIELDS,
    )


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

    return await run_tool(
        "sheets",
        "get_values",
        _get_values,
        allow_retry=True,
        suggested_fields="sheets.values.get(range=Sheet1!A1:C20)",
    )


@mcp.tool()
async def sheets_batch_get_values(
    spreadsheet_id: str,
    ranges: list[str],
    value_render_option: str = "",
    date_time_render_option: str = "",
    major_dimension: str = "",
) -> str:
    """Read values from multiple ranges in a Google Sheet."""

    def _batch_get_values():
        if not spreadsheet_id:
            raise ValueError("spreadsheet_id cannot be empty")
        if not ranges:
            raise ValueError("ranges cannot be empty")
        service, cached = client.get_service("sheets", "v4")
        request = service.spreadsheets().values().batchGet(
            spreadsheetId=spreadsheet_id,
            ranges=ranges,
            valueRenderOption=value_render_option or None,
            dateTimeRenderOption=date_time_render_option or None,
            majorDimension=major_dimension or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool(
        "sheets",
        "batch_get_values",
        _batch_get_values,
        allow_retry=True,
        suggested_fields="sheets.values.batchGet(ranges=[Sheet1!A1:C20])",
    )


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
async def sheets_append_values(
    spreadsheet_id: str,
    range_a1: str,
    values: Values,
    value_input_option: str = "RAW",
    insert_data_option: str = "INSERT_ROWS",
) -> str:
    """Append rows to a Google Sheet range."""

    def _append_values():
        if not spreadsheet_id or not range_a1:
            raise ValueError("spreadsheet_id and range_a1 are required")
        if values is None:
            raise ValueError("values cannot be empty")
        service, cached = client.get_service("sheets", "v4")
        request = service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=range_a1,
            valueInputOption=value_input_option,
            insertDataOption=insert_data_option,
            body={"values": values},
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("sheets", "append_values", _append_values, allow_retry=False)


@mcp.tool()
async def sheets_clear_values(
    spreadsheet_id: str,
    range_a1: str,
    confirm: bool = False,
) -> str:
    """Clear values from a Google Sheet range."""

    def _clear_values():
        if not spreadsheet_id or not range_a1:
            raise ValueError("spreadsheet_id and range_a1 are required")
        _require_confirm("clear Google Sheets values", confirm)
        service, cached = client.get_service("sheets", "v4")
        request = service.spreadsheets().values().clear(
            spreadsheetId=spreadsheet_id,
            range=range_a1,
            body={},
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("sheets", "clear_values", _clear_values, allow_retry=False)


@mcp.tool()
async def sheets_batch_update_values(
    spreadsheet_id: str,
    data: list[dict[str, Any]],
    value_input_option: str = "RAW",
    confirm: bool = False,
) -> str:
    """Write multiple Google Sheets ranges in one values batchUpdate."""

    def _batch_update_values():
        if not spreadsheet_id:
            raise ValueError("spreadsheet_id cannot be empty")
        if not data:
            raise ValueError("data cannot be empty")
        _ensure_confirmed("batch update Google Sheets values", confirm)
        service, cached = client.get_service("sheets", "v4")
        request = service.spreadsheets().values().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"valueInputOption": value_input_option, "data": data},
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool(
        "sheets", "batch_update_values", _batch_update_values, allow_retry=False
    )


@mcp.tool()
async def sheets_batch_clear_values(
    spreadsheet_id: str,
    ranges: list[str],
    confirm: bool = False,
) -> str:
    """Clear multiple Google Sheets ranges."""

    def _batch_clear_values():
        if not spreadsheet_id:
            raise ValueError("spreadsheet_id cannot be empty")
        if not ranges:
            raise ValueError("ranges cannot be empty")
        _require_confirm("batch clear Google Sheets values", confirm)
        service, cached = client.get_service("sheets", "v4")
        request = service.spreadsheets().values().batchClear(
            spreadsheetId=spreadsheet_id,
            body={"ranges": ranges},
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool(
        "sheets", "batch_clear_values", _batch_clear_values, allow_retry=False
    )


@mcp.tool()
async def sheets_batch_update(
    spreadsheet_id: str,
    requests: list[dict[str, Any]],
    confirm: bool = False,
) -> str:
    """Run a controlled Google Sheets spreadsheet batchUpdate."""

    def _batch_update():
        if not spreadsheet_id:
            raise ValueError("spreadsheet_id cannot be empty")
        if not requests:
            raise ValueError("requests cannot be empty")
        _ensure_confirmed("run Sheets batchUpdate", confirm)
        service, cached = client.get_service("sheets", "v4")
        request = service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": requests},
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("sheets", "batch_update", _batch_update, allow_retry=False)


@mcp.tool()
async def sheets_get_by_data_filter(spreadsheet_id: str, data_filters: list[dict[str, Any]]) -> str:
    """Fetch spreadsheet data by Google Sheets data filters."""

    def _get_by_data_filter():
        if not spreadsheet_id:
            raise ValueError("spreadsheet_id cannot be empty")
        if not data_filters:
            raise ValueError("data_filters cannot be empty")
        service, cached = client.get_service("sheets", "v4")
        request = service.spreadsheets().getByDataFilter(
            spreadsheetId=spreadsheet_id,
            body={"dataFilters": data_filters},
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool(
        "sheets", "get_by_data_filter", _get_by_data_filter, allow_retry=True
    )


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

    return await run_tool(
        "slides",
        "get_presentation",
        _get_presentation,
        allow_retry=True,
        suggested_fields=DEFAULT_SLIDES_FIELDS,
    )


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
async def slides_batch_update(
    presentation_id: str,
    requests: list[dict[str, Any]],
    confirm: bool = False,
) -> str:
    """Run a controlled Google Slides batchUpdate request."""

    def _batch_update():
        if not presentation_id:
            raise ValueError("presentation_id cannot be empty")
        if not requests:
            raise ValueError("requests cannot be empty")
        _ensure_confirmed("run Slides batchUpdate", confirm)
        service, cached = client.get_service("slides", "v1")
        request = service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={"requests": requests},
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("slides", "batch_update", _batch_update, allow_retry=False)


@mcp.tool()
async def slides_get_page(presentation_id: str, page_object_id: str) -> str:
    """Get one Google Slides page."""

    def _get_page():
        if not presentation_id or not page_object_id:
            raise ValueError("presentation_id and page_object_id are required")
        service, cached = client.get_service("slides", "v1")
        request = service.presentations().pages().get(
            presentationId=presentation_id,
            pageObjectId=page_object_id,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("slides", "get_page", _get_page, allow_retry=True)


@mcp.tool()
async def slides_get_page_thumbnail(
    presentation_id: str,
    page_object_id: str,
    thumbnail_size: str = "MEDIUM",
    mime_type: str = "PNG",
) -> str:
    """Get a thumbnail URL for a Google Slides page."""

    def _get_page_thumbnail():
        if not presentation_id or not page_object_id:
            raise ValueError("presentation_id and page_object_id are required")
        service, cached = client.get_service("slides", "v1")
        request = service.presentations().pages().getThumbnail(
            presentationId=presentation_id,
            pageObjectId=page_object_id,
            thumbnailProperties_thumbnailSize=thumbnail_size,
            thumbnailProperties_mimeType=mime_type,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool(
        "slides", "get_page_thumbnail", _get_page_thumbnail, allow_retry=True
    )


@mcp.tool()
async def gmail_list_labels(
    fields: str | dict | None = "",
    minimal: bool = True,
    include_visibility: bool = False,
) -> str:
    """List Gmail labels for the authenticated user."""

    def _list_labels():
        service, cached = client.get_service("gmail", "v1")
        warnings: list[str] = []
        if fields is not None and not isinstance(fields, str):
            if MCP_STRICT_PARAMS:
                raise ValueError("fields must be a string.")
            warnings.append("fields must be a string; ignored")
            field_text = ""
        else:
            field_text = fields.strip() if isinstance(fields, str) else ""
        if field_text:
            effective_fields = field_text
        elif minimal:
            if include_visibility:
                effective_fields = "labels(id,name,labelListVisibility,messageListVisibility)"
            else:
                effective_fields = "labels(id,name)"
        else:
            effective_fields = ""
        request = service.users().labels().list(
            userId="me",
            fields=effective_fields or None,
        )
        data = request.execute()
        if minimal and not field_text:
            labels = [
                {
                    "id": label.get("id"),
                    "name": label.get("name"),
                    **(
                        {
                            "labelListVisibility": label.get("labelListVisibility"),
                            "messageListVisibility": label.get("messageListVisibility"),
                        }
                        if include_visibility
                        else {}
                    ),
                }
                for label in data.get("labels", []) or []
            ]
            meta = {"cached_service": cached}
            if warnings:
                meta["warnings"] = warnings
            return {"labels": labels}, meta
        meta = {"cached_service": cached}
        if warnings:
            meta["warnings"] = warnings
        return data, meta

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
        data = request.execute()
        return _attach_page_meta(data, cached)

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
        data = request.execute()
        return _attach_page_meta(data, cached)

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

    return await run_tool(
        "gmail",
        "get_message",
        _get_message,
        allow_retry=True,
        suggested_fields="format=metadata, metadata_headers=From,To,Subject,Date",
    )


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
        data = request.execute()
        return _attach_page_meta(data, cached)

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

    return await run_tool(
        "gmail",
        "get_thread",
        _get_thread,
        allow_retry=True,
        suggested_fields="format=metadata, metadata_headers=From,To,Subject,Date",
    )


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
async def gmail_get_label(label_id: str) -> str:
    """Get one Gmail label."""

    def _get_label():
        if not label_id:
            raise ValueError("label_id cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        request = service.users().labels().get(userId="me", id=label_id)
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "get_label", _get_label, allow_retry=True)


@mcp.tool()
async def gmail_update_label(
    label_id: str,
    name: str = "",
    label_list_visibility: str = "",
    message_list_visibility: str = "",
) -> str:
    """Update a Gmail label."""

    def _update_label():
        if not label_id:
            raise ValueError("label_id cannot be empty")
        body: dict[str, Any] = {}
        if name:
            body["name"] = name
        if label_list_visibility:
            body["labelListVisibility"] = label_list_visibility
        if message_list_visibility:
            body["messageListVisibility"] = message_list_visibility
        if not body:
            raise ValueError("Provide at least one label field to update.")
        service, cached = client.get_service("gmail", "v1")
        request = service.users().labels().patch(userId="me", id=label_id, body=body)
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "update_label", _update_label, allow_retry=False)


@mcp.tool()
async def gmail_list_drafts(max_results: int = 100, page_token: str = "") -> str:
    """List Gmail drafts."""

    def _list_drafts():
        service, cached = client.get_service("gmail", "v1")
        request = service.users().drafts().list(
            userId="me",
            maxResults=_clamp_int(max_results, minimum=1, maximum=500),
            pageToken=page_token or None,
        )
        data = request.execute()
        return _attach_page_meta(data, cached)

    return await run_tool("gmail", "list_drafts", _list_drafts, allow_retry=True)


@mcp.tool()
async def gmail_get_draft(
    draft_id: str,
    format: str = "metadata",
    metadata_headers: list[str] | None = None,
) -> str:
    """Get one Gmail draft."""

    def _get_draft():
        if not draft_id:
            raise ValueError("draft_id cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        effective_headers = metadata_headers or list(DEFAULT_GMAIL_METADATA_HEADERS)
        request = service.users().drafts().get(
            userId="me",
            id=draft_id,
            format=format,
            metadataHeaders=effective_headers if format == "metadata" else None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "get_draft", _get_draft, allow_retry=True)


@mcp.tool()
async def gmail_update_draft(
    draft_id: str,
    to: str,
    subject: str,
    body: str,
    cc: str = "",
    bcc: str = "",
    reply_to: str = "",
    from_alias: str = "",
    is_html: bool = False,
) -> str:
    """Replace an existing Gmail draft message."""

    def _update_draft():
        if not draft_id:
            raise ValueError("draft_id cannot be empty")
        if not to:
            raise ValueError("to cannot be empty")
        if not subject:
            raise ValueError("subject cannot be empty")
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
        request = service.users().drafts().update(
            userId="me",
            id=draft_id,
            body={"id": draft_id, "message": {"raw": encode_email_message(message)}},
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "update_draft", _update_draft, allow_retry=False)


@mcp.tool()
async def gmail_delete_draft(draft_id: str, confirm: bool = False) -> str:
    """Delete a Gmail draft."""

    def _delete_draft():
        if not draft_id:
            raise ValueError("draft_id cannot be empty")
        _ensure_confirmed("delete a Gmail draft", confirm)
        service, cached = client.get_service("gmail", "v1")
        service.users().drafts().delete(userId="me", id=draft_id).execute()
        return {"id": draft_id, "deleted": True}, {"cached_service": cached}

    return await run_tool("gmail", "delete_draft", _delete_draft, allow_retry=False)


@mcp.tool()
async def gmail_get_attachment(
    message_id: str,
    attachment_id: str,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    include_content: bool = False,
) -> str:
    """Fetch Gmail attachment metadata or bounded base64 content."""

    def _get_attachment():
        if not message_id:
            raise ValueError("message_id cannot be empty")
        if not attachment_id:
            raise ValueError("attachment_id cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        data = (
            service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=message_id, id=attachment_id)
            .execute()
        )
        size = int(data.get("size") or 0)
        payload: dict[str, Any] = {"attachment_id": attachment_id, "size": size}
        if include_content:
            if max_bytes and size > max_bytes:
                payload.update({"too_large": True, "max_bytes": max_bytes})
            else:
                payload["data"] = data.get("data", "")
        return payload, {"cached_service": cached}

    return await run_tool("gmail", "get_attachment", _get_attachment, allow_retry=True)


@mcp.tool()
async def gmail_batch_modify_messages(
    message_ids: list[str],
    add_label_ids: list[str] | None = None,
    remove_label_ids: list[str] | None = None,
    dry_run: bool = True,
    confirm: bool = False,
) -> str:
    """Batch-add or remove Gmail labels for up to 1000 messages per provider call."""

    def _batch_modify():
        ids = [message_id for message_id in (message_ids or []) if message_id]
        if not ids:
            raise ValueError("message_ids cannot be empty")
        if not add_label_ids and not remove_label_ids:
            raise ValueError("Provide add_label_ids or remove_label_ids.")
        chunks = _chunked(ids, 1000)
        plan = {
            "dry_run": dry_run,
            "message_count": len(ids),
            "chunk_count": len(chunks),
            "add_label_ids": add_label_ids or [],
            "remove_label_ids": remove_label_ids or [],
        }
        if dry_run:
            return plan
        _require_confirm("batch modify Gmail messages", confirm)
        service, cached = client.get_service("gmail", "v1")
        for chunk in chunks:
            service.users().messages().batchModify(
                userId="me",
                body={
                    "ids": chunk,
                    "addLabelIds": add_label_ids or [],
                    "removeLabelIds": remove_label_ids or [],
                },
            ).execute()
        return {**plan, "applied": True}, {"cached_service": cached}

    return await run_tool("gmail", "batch_modify_messages", _batch_modify, allow_retry=False)


@mcp.tool()
async def gmail_batch_delete_messages(
    message_ids: list[str],
    dry_run: bool = True,
    confirm: bool = False,
) -> str:
    """Permanently delete Gmail messages in provider-supported batches."""

    def _batch_delete():
        ids = [message_id for message_id in (message_ids or []) if message_id]
        if not ids:
            raise ValueError("message_ids cannot be empty")
        chunks = _chunked(ids, 1000)
        plan = {"dry_run": dry_run, "message_count": len(ids), "chunk_count": len(chunks)}
        if dry_run:
            return plan
        _require_confirm("permanently batch delete Gmail messages", confirm)
        service, cached = client.get_service("gmail", "v1")
        for chunk in chunks:
            service.users().messages().batchDelete(
                userId="me",
                body={"ids": chunk},
            ).execute()
        return {**plan, "deleted": True}, {"cached_service": cached}

    return await run_tool("gmail", "batch_delete_messages", _batch_delete, allow_retry=False)


@mcp.tool()
async def gmail_modify_thread_labels(
    thread_id: str,
    add_label_ids: list[str] | None = None,
    remove_label_ids: list[str] | None = None,
    dry_run: bool = True,
    confirm: bool = False,
) -> str:
    """Add or remove labels on a Gmail thread."""

    def _modify_thread():
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        if not add_label_ids and not remove_label_ids:
            raise ValueError("Provide add_label_ids or remove_label_ids.")
        plan = {
            "thread_id": thread_id,
            "dry_run": dry_run,
            "add_label_ids": add_label_ids or [],
            "remove_label_ids": remove_label_ids or [],
        }
        if dry_run:
            return plan
        _require_confirm("modify Gmail thread labels", confirm)
        service, cached = client.get_service("gmail", "v1")
        request = service.users().threads().modify(
            userId="me",
            id=thread_id,
            body={"addLabelIds": add_label_ids or [], "removeLabelIds": remove_label_ids or []},
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "modify_thread_labels", _modify_thread, allow_retry=False)


@mcp.tool()
async def gmail_trash_thread(thread_id: str, confirm: bool = False) -> str:
    """Move a Gmail thread to trash."""

    def _trash_thread():
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        _ensure_confirmed("trash a Gmail thread", confirm)
        service, cached = client.get_service("gmail", "v1")
        request = service.users().threads().trash(userId="me", id=thread_id)
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "trash_thread", _trash_thread, allow_retry=False)


@mcp.tool()
async def gmail_untrash_thread(thread_id: str, confirm: bool = False) -> str:
    """Restore a Gmail thread from trash."""

    def _untrash_thread():
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        _ensure_confirmed("untrash a Gmail thread", confirm)
        service, cached = client.get_service("gmail", "v1")
        request = service.users().threads().untrash(userId="me", id=thread_id)
        return request.execute(), {"cached_service": cached}

    return await run_tool("gmail", "untrash_thread", _untrash_thread, allow_retry=False)


@mcp.tool()
async def gmail_delete_thread(thread_id: str, confirm: bool = False) -> str:
    """Permanently delete a Gmail thread."""

    def _delete_thread():
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        _require_confirm("permanently delete a Gmail thread", confirm)
        service, cached = client.get_service("gmail", "v1")
        service.users().threads().delete(userId="me", id=thread_id).execute()
        return {"id": thread_id, "deleted": True}, {"cached_service": cached}

    return await run_tool("gmail", "delete_thread", _delete_thread, allow_retry=False)


@mcp.tool()
async def gmail_list_history(
    start_history_id: str,
    history_types: list[str] | None = None,
    label_id: str = "",
    max_results: int = 100,
    page_token: str = "",
) -> str:
    """List Gmail mailbox history changes from a start history ID."""

    def _list_history():
        if not start_history_id:
            raise ValueError("start_history_id cannot be empty")
        service, cached = client.get_service("gmail", "v1")
        request = service.users().history().list(
            userId="me",
            startHistoryId=start_history_id,
            historyTypes=history_types or None,
            labelId=label_id or None,
            maxResults=_clamp_int(max_results, minimum=1, maximum=500),
            pageToken=page_token or None,
        )
        data = request.execute()
        return _attach_page_meta(data, cached)

    return await run_tool("gmail", "list_history", _list_history, allow_retry=True)


@mcp.tool()
async def gmail_mailbox_overview(
    queries: list[str] | None = None,
    include_labels: bool = True,
) -> str:
    """Return compact Gmail mailbox counts for inbox-cleanup planning."""

    def _overview():
        service, cached = client.get_service("gmail", "v1")
        query_list = queries or [
            "is:unread",
            "in:inbox is:unread",
            "category:promotions is:unread",
            "category:social is:unread",
            "older_than:90d is:unread",
            "has:unsubscribe is:unread",
        ]
        counts = []
        for query in query_list[:20]:
            data = service.users().messages().list(
                userId="me",
                q=query,
                maxResults=1,
                fields="resultSizeEstimate,messages/id,nextPageToken",
            ).execute()
            counts.append(
                {
                    "query": query,
                    "result_size_estimate": data.get("resultSizeEstimate", 0),
                    "has_results": bool(data.get("messages")),
                    "has_more": bool(data.get("nextPageToken")),
                }
            )
        labels = []
        if include_labels:
            label_data = service.users().labels().list(
                userId="me",
                fields="labels(id,name,type)",
            ).execute()
            labels = label_data.get("labels", []) or []
        return {"counts": counts, "labels": labels}, {"cached_service": cached}

    return await run_tool("gmail", "mailbox_overview", _overview, allow_retry=True)


@mcp.tool()
async def gmail_sender_clusters(
    query: str = "is:unread",
    max_messages: int = 500,
    page_size: int = 100,
    top_n: int = 25,
) -> str:
    """Cluster Gmail messages by sender/domain/List-ID without fetching bodies."""

    def _sender_clusters():
        service, cached = client.get_service("gmail", "v1")
        stubs, next_token, estimate, pages = _gmail_list_message_ids(
            service,
            query=query,
            max_messages=max_messages,
            page_size=page_size,
        )
        metadata = _gmail_get_metadata_batch(
            service,
            [stub.get("id", "") for stub in stubs],
            max_messages=max_messages,
        )
        clusters: dict[str, dict[str, Any]] = {}
        for message in metadata:
            headers = message.get("headers", {})
            sender = _gmail_sender_key(headers)
            cluster = clusters.setdefault(
                sender["key"],
                {
                    **sender,
                    "count": 0,
                    "message_ids_sample": [],
                    "subjects_sample": [],
                    "label_counts": Counter(),
                },
            )
            cluster["count"] += 1
            if len(cluster["message_ids_sample"]) < 25:
                cluster["message_ids_sample"].append(message.get("id"))
            subject = headers.get("subject", "")
            if subject and len(cluster["subjects_sample"]) < 5:
                cluster["subjects_sample"].append(subject[:160])
            cluster["label_counts"].update(message.get("labelIds", []))
        rows = sorted(clusters.values(), key=lambda item: item["count"], reverse=True)
        for row in rows:
            row["label_counts"] = dict(row["label_counts"].most_common(10))
        return {
            "query": query,
            "sampled_messages": len(metadata),
            "result_size_estimate": estimate,
            "pages_scanned": pages,
            "next_page_token": next_token or None,
            "clusters": rows[: _clamp_int(top_n, minimum=1, maximum=100)],
        }, {"cached_service": cached}

    return await run_tool("gmail", "sender_clusters", _sender_clusters, allow_retry=True)


@mcp.tool()
async def gmail_cleanup_plan(
    query: str = "is:unread",
    max_messages: int = 500,
    page_size: int = 100,
    top_n: int = 25,
    proposed_action: str = "label",
    target_label_id: str = "",
) -> str:
    """Build a dry-run Gmail cleanup plan grouped by sender/domain/List-ID."""

    def _cleanup_plan():
        service, cached = client.get_service("gmail", "v1")
        stubs, next_token, estimate, pages = _gmail_list_message_ids(
            service,
            query=query,
            max_messages=max_messages,
            page_size=page_size,
        )
        metadata = _gmail_get_metadata_batch(
            service,
            [stub.get("id", "") for stub in stubs],
            max_messages=max_messages,
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        senders: dict[str, dict[str, str]] = {}
        for message in metadata:
            sender = _gmail_sender_key(message.get("headers", {}))
            grouped[sender["key"]].append(message)
            senders[sender["key"]] = sender
        batches = []
        for index, (key, messages) in enumerate(
            sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True), start=1
        ):
            if len(batches) >= _clamp_int(top_n, minimum=1, maximum=100):
                break
            sender = senders[key]
            message_ids = [message.get("id") for message in messages if message.get("id")]
            sample_subjects = [
                message.get("headers", {}).get("subject", "")[:160]
                for message in messages[:5]
                if message.get("headers", {}).get("subject")
            ]
            batches.append(
                {
                    "batch_id": f"gmail-cleanup-{index:03d}",
                    "query": query,
                    "sender_key": key,
                    "sender": sender,
                    "message_count": len(message_ids),
                    "message_ids_sample": message_ids[:25],
                    "sample_subjects": sample_subjects,
                    "proposed_action": proposed_action,
                    "target_label_id": target_label_id or None,
                    "apply_requires": "Call gmail_apply_cleanup_plan with dry_run=false and confirm=true.",
                }
            )
        return {
            "dry_run": True,
            "query": query,
            "sampled_messages": len(metadata),
            "result_size_estimate": estimate,
            "pages_scanned": pages,
            "next_page_token": next_token or None,
            "batches": batches,
        }, {"cached_service": cached}

    return await run_tool("gmail", "cleanup_plan", _cleanup_plan, allow_retry=True)


@mcp.tool()
async def gmail_apply_cleanup_plan(
    message_ids: list[str],
    action: str,
    label_id: str = "",
    dry_run: bool = True,
    confirm: bool = False,
) -> str:
    """Apply an approved Gmail cleanup batch to explicit message IDs."""

    def _apply_plan():
        ids = [message_id for message_id in (message_ids or []) if message_id]
        if not ids:
            raise ValueError("message_ids cannot be empty")
        normalized_action = (action or "").strip().lower()
        allowed = {"label", "archive", "mark_read", "trash"}
        if normalized_action not in allowed:
            raise ValueError(f"action must be one of: {', '.join(sorted(allowed))}.")
        if normalized_action == "label" and not label_id:
            raise ValueError("label_id is required for action=label.")
        add_labels: list[str] = []
        remove_labels: list[str] = []
        if normalized_action == "label":
            add_labels = [label_id]
        elif normalized_action == "archive":
            remove_labels = ["INBOX"]
        elif normalized_action == "mark_read":
            remove_labels = ["UNREAD"]
        plan = {
            "dry_run": dry_run,
            "action": normalized_action,
            "message_count": len(ids),
            "chunk_count": len(_chunked(ids, 1000)),
            "add_label_ids": add_labels,
            "remove_label_ids": remove_labels,
        }
        if dry_run:
            return plan
        _require_confirm("apply Gmail cleanup plan", confirm)
        service, cached = client.get_service("gmail", "v1")
        if normalized_action == "trash":
            for message_id in ids:
                service.users().messages().trash(userId="me", id=message_id).execute()
        else:
            for chunk in _chunked(ids, 1000):
                service.users().messages().batchModify(
                    userId="me",
                    body={
                        "ids": chunk,
                        "addLabelIds": add_labels,
                        "removeLabelIds": remove_labels,
                    },
                ).execute()
        return {**plan, "applied": True}, {"cached_service": cached}

    return await run_tool("gmail", "apply_cleanup_plan", _apply_plan, allow_retry=False)


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
        data = request.execute()
        return _attach_page_meta(data, cached)

    return await run_tool(
        "calendar",
        "list_calendars",
        _list_calendars,
        allow_retry=True,
        suggested_fields=DEFAULT_CALENDAR_LIST_FIELDS,
    )


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

    return await run_tool(
        "calendar",
        "get_calendar",
        _get_calendar,
        allow_retry=True,
        suggested_fields=DEFAULT_CALENDAR_FIELDS,
    )


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
        data = request.execute()
        return _attach_page_meta(data, cached)

    return await run_tool(
        "calendar",
        "list_events",
        _list_events,
        allow_retry=True,
        suggested_fields=DEFAULT_EVENT_LIST_FIELDS,
    )


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
        data = request.execute()
        return _attach_page_meta(data, cached)

    return await run_tool(
        "calendar",
        "search_events",
        _search_events,
        allow_retry=True,
        suggested_fields=DEFAULT_EVENT_LIST_FIELDS,
    )


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

    return await run_tool(
        "calendar",
        "batch_get_events",
        _batch_get,
        allow_retry=True,
        suggested_fields=DEFAULT_EVENT_FIELDS,
    )


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

    return await run_tool(
        "calendar",
        "get_event",
        _get_event,
        allow_retry=True,
        suggested_fields=DEFAULT_EVENT_FIELDS,
    )


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


@mcp.tool()
async def calendar_freebusy_query(
    items: list[dict[str, str]],
    time_min: str,
    time_max: str,
    time_zone: str = "UTC",
) -> str:
    """Query free/busy blocks for calendars."""

    def _freebusy():
        if not items:
            raise ValueError("items cannot be empty")
        if not time_min or not time_max:
            raise ValueError("time_min and time_max are required")
        service, cached = client.get_service("calendar", "v3")
        body = {
            "timeMin": time_min,
            "timeMax": time_max,
            "timeZone": time_zone,
            "items": items,
        }
        request = service.freebusy().query(body=body)
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "freebusy_query", _freebusy, allow_retry=True)


@mcp.tool()
async def calendar_get_colors() -> str:
    """Get Google Calendar color definitions."""

    def _get_colors():
        service, cached = client.get_service("calendar", "v3")
        return service.colors().get().execute(), {"cached_service": cached}

    return await run_tool("calendar", "get_colors", _get_colors, allow_retry=True)


@mcp.tool()
async def calendar_list_settings(page_token: str = "") -> str:
    """List Google Calendar user settings."""

    def _list_settings():
        service, cached = client.get_service("calendar", "v3")
        data = service.settings().list(pageToken=page_token or None).execute()
        return _attach_page_meta(data, cached)

    return await run_tool("calendar", "list_settings", _list_settings, allow_retry=True)


@mcp.tool()
async def calendar_get_setting(setting_id: str) -> str:
    """Get one Google Calendar user setting."""

    def _get_setting():
        if not setting_id:
            raise ValueError("setting_id cannot be empty")
        service, cached = client.get_service("calendar", "v3")
        return service.settings().get(setting=setting_id).execute(), {"cached_service": cached}

    return await run_tool("calendar", "get_setting", _get_setting, allow_retry=True)


@mcp.tool()
async def calendar_get_calendar_list_entry(calendar_id: str) -> str:
    """Get one CalendarList entry."""

    def _get_entry():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        service, cached = client.get_service("calendar", "v3")
        return service.calendarList().get(calendarId=calendar_id).execute(), {"cached_service": cached}

    return await run_tool("calendar", "get_calendar_list_entry", _get_entry, allow_retry=True)


@mcp.tool()
async def calendar_update_calendar_list_entry(
    calendar_id: str,
    entry_patch: dict[str, Any],
    confirm: bool = False,
) -> str:
    """Patch a CalendarList entry such as color or hidden/selected state."""

    def _update_entry():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        if not entry_patch:
            raise ValueError("entry_patch cannot be empty")
        _ensure_confirmed("update a CalendarList entry", confirm)
        service, cached = client.get_service("calendar", "v3")
        request = service.calendarList().patch(calendarId=calendar_id, body=entry_patch)
        return request.execute(), {"cached_service": cached}

    return await run_tool(
        "calendar", "update_calendar_list_entry", _update_entry, allow_retry=False
    )


@mcp.tool()
async def calendar_delete_calendar_list_entry(calendar_id: str, confirm: bool = False) -> str:
    """Remove a calendar from the authenticated user's CalendarList."""

    def _delete_entry():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        _require_confirm("remove a calendar from CalendarList", confirm)
        service, cached = client.get_service("calendar", "v3")
        service.calendarList().delete(calendarId=calendar_id).execute()
        return {"calendar_id": calendar_id, "removed": True}, {"cached_service": cached}

    return await run_tool(
        "calendar", "delete_calendar_list_entry", _delete_entry, allow_retry=False
    )


@mcp.tool()
async def calendar_update_calendar(
    calendar_id: str,
    calendar_patch: dict[str, Any],
    confirm: bool = False,
) -> str:
    """Patch calendar metadata."""

    def _update_calendar():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        if not calendar_patch:
            raise ValueError("calendar_patch cannot be empty")
        _ensure_confirmed("update calendar metadata", confirm)
        service, cached = client.get_service("calendar", "v3")
        request = service.calendars().patch(calendarId=calendar_id, body=calendar_patch)
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "update_calendar", _update_calendar, allow_retry=False)


@mcp.tool()
async def calendar_clear_calendar(calendar_id: str, confirm: bool = False) -> str:
    """Clear all events from a primary calendar."""

    def _clear_calendar():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        _require_confirm("clear all events from a calendar", confirm)
        service, cached = client.get_service("calendar", "v3")
        service.calendars().clear(calendarId=calendar_id).execute()
        return {"calendar_id": calendar_id, "cleared": True}, {"cached_service": cached}

    return await run_tool("calendar", "clear_calendar", _clear_calendar, allow_retry=False)


@mcp.tool()
async def calendar_list_event_instances(
    calendar_id: str,
    event_id: str,
    time_min: str = "",
    time_max: str = "",
    max_results: int = 100,
    page_token: str = "",
) -> str:
    """List instances of a recurring calendar event."""

    def _list_instances():
        if not calendar_id or not event_id:
            raise ValueError("calendar_id and event_id are required")
        service, cached = client.get_service("calendar", "v3")
        data = service.events().instances(
            calendarId=calendar_id,
            eventId=event_id,
            timeMin=time_min or None,
            timeMax=time_max or None,
            maxResults=_clamp_int(max_results, minimum=1, maximum=2500),
            pageToken=page_token or None,
        ).execute()
        return _attach_page_meta(data, cached)

    return await run_tool("calendar", "list_event_instances", _list_instances, allow_retry=True)


@mcp.tool()
async def calendar_move_event(
    source_calendar_id: str,
    destination_calendar_id: str,
    event_id: str,
    send_updates: str = "all",
    confirm: bool = False,
) -> str:
    """Move an event to another calendar."""

    def _move_event():
        if not source_calendar_id or not destination_calendar_id or not event_id:
            raise ValueError("source_calendar_id, destination_calendar_id, and event_id are required")
        _require_confirm("move a calendar event", confirm)
        service, cached = client.get_service("calendar", "v3")
        request = service.events().move(
            calendarId=source_calendar_id,
            eventId=event_id,
            destination=destination_calendar_id,
            sendUpdates=send_updates or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "move_event", _move_event, allow_retry=False)


@mcp.tool()
async def calendar_import_event(calendar_id: str, event_body: dict[str, Any], confirm: bool = False) -> str:
    """Import an event without sending notifications."""

    def _import_event():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        if not event_body:
            raise ValueError("event_body cannot be empty")
        _require_confirm("import a calendar event", confirm)
        service, cached = client.get_service("calendar", "v3")
        request = service.events().import_(calendarId=calendar_id, body=event_body)
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "import_event", _import_event, allow_retry=False)


@mcp.tool()
async def calendar_replace_event(
    calendar_id: str,
    event_id: str,
    event_body: dict[str, Any],
    send_updates: str = "all",
    confirm: bool = False,
) -> str:
    """Replace a calendar event with a full event body."""

    def _replace_event():
        if not calendar_id or not event_id:
            raise ValueError("calendar_id and event_id are required")
        if not event_body:
            raise ValueError("event_body cannot be empty")
        _require_confirm("replace a calendar event", confirm)
        service, cached = client.get_service("calendar", "v3")
        request = service.events().update(
            calendarId=calendar_id,
            eventId=event_id,
            body=event_body,
            sendUpdates=send_updates or None,
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "replace_event", _replace_event, allow_retry=False)


@mcp.tool()
async def calendar_list_acl(calendar_id: str, page_token: str = "") -> str:
    """List Calendar ACL rules."""

    def _list_acl():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        service, cached = client.get_service("calendar", "v3")
        data = service.acl().list(calendarId=calendar_id, pageToken=page_token or None).execute()
        return _attach_page_meta(data, cached)

    return await run_tool("calendar", "list_acl", _list_acl, allow_retry=True)


@mcp.tool()
async def calendar_get_acl(calendar_id: str, rule_id: str) -> str:
    """Get one Calendar ACL rule."""

    def _get_acl():
        if not calendar_id or not rule_id:
            raise ValueError("calendar_id and rule_id are required")
        service, cached = client.get_service("calendar", "v3")
        return service.acl().get(calendarId=calendar_id, ruleId=rule_id).execute(), {"cached_service": cached}

    return await run_tool("calendar", "get_acl", _get_acl, allow_retry=True)


@mcp.tool()
async def calendar_upsert_acl(
    calendar_id: str,
    rule_body: dict[str, Any],
    rule_id: str = "",
    confirm: bool = False,
) -> str:
    """Create or patch a Calendar ACL rule."""

    def _upsert_acl():
        if not calendar_id:
            raise ValueError("calendar_id cannot be empty")
        if not rule_body:
            raise ValueError("rule_body cannot be empty")
        _require_confirm("change Calendar sharing ACL", confirm)
        service, cached = client.get_service("calendar", "v3")
        if rule_id:
            request = service.acl().patch(calendarId=calendar_id, ruleId=rule_id, body=rule_body)
        else:
            request = service.acl().insert(calendarId=calendar_id, body=rule_body)
        return request.execute(), {"cached_service": cached}

    return await run_tool("calendar", "upsert_acl", _upsert_acl, allow_retry=False)


@mcp.tool()
async def calendar_delete_acl(calendar_id: str, rule_id: str, confirm: bool = False) -> str:
    """Delete a Calendar ACL rule."""

    def _delete_acl():
        if not calendar_id or not rule_id:
            raise ValueError("calendar_id and rule_id are required")
        _require_confirm("delete Calendar sharing ACL", confirm)
        service, cached = client.get_service("calendar", "v3")
        service.acl().delete(calendarId=calendar_id, ruleId=rule_id).execute()
        return {"calendar_id": calendar_id, "rule_id": rule_id, "deleted": True}, {"cached_service": cached}

    return await run_tool("calendar", "delete_acl", _delete_acl, allow_retry=False)


@mcp.tool()
async def youtube_search(
    query: str,
    part: str = "snippet",
    max_results: int = 10,
    order: str = "relevance",
    type: str = "",
    page_token: str = "",
) -> str:
    """Search YouTube videos, channels, or playlists."""

    def _search():
        if not query:
            raise ValueError("query cannot be empty")
        service, cached = client.get_service("youtube", "v3")
        data = service.search().list(
            q=query,
            part=part,
            maxResults=_clamp_int(max_results, minimum=1, maximum=50),
            order=order or None,
            type=type or None,
            pageToken=page_token or None,
        ).execute()
        return _attach_page_meta(data, cached)

    return await run_tool("youtube", "search", _search, allow_retry=True)


@mcp.tool()
async def youtube_list_channels(
    part: str = "snippet,statistics,contentDetails",
    mine: bool = True,
    channel_ids: list[str] | None = None,
    for_username: str = "",
    max_results: int = 10,
    page_token: str = "",
) -> str:
    """List YouTube channels by authenticated user, ID, or username."""

    def _list_channels():
        service, cached = client.get_service("youtube", "v3")
        data = service.channels().list(
            part=part,
            mine=mine if not channel_ids and not for_username else None,
            id=",".join(channel_ids or []) or None,
            forUsername=for_username or None,
            maxResults=_clamp_int(max_results, minimum=1, maximum=50),
            pageToken=page_token or None,
        ).execute()
        return _attach_page_meta(data, cached)

    return await run_tool("youtube", "list_channels", _list_channels, allow_retry=True)


@mcp.tool()
async def youtube_list_videos(
    video_ids: list[str],
    part: str = "snippet,statistics,contentDetails,status",
    max_results: int = 50,
) -> str:
    """List YouTube video metadata by video IDs."""

    def _list_videos():
        ids = [video_id for video_id in (video_ids or []) if video_id]
        if not ids:
            raise ValueError("video_ids cannot be empty")
        service, cached = client.get_service("youtube", "v3")
        data = service.videos().list(
            part=part,
            id=",".join(ids[:50]),
            maxResults=_clamp_int(max_results, minimum=1, maximum=50),
        ).execute()
        return data, {"cached_service": cached}

    return await run_tool("youtube", "list_videos", _list_videos, allow_retry=True)


@mcp.tool()
async def youtube_list_playlists(
    part: str = "snippet,contentDetails,status",
    mine: bool = True,
    channel_id: str = "",
    max_results: int = 25,
    page_token: str = "",
) -> str:
    """List YouTube playlists."""

    def _list_playlists():
        service, cached = client.get_service("youtube", "v3")
        data = service.playlists().list(
            part=part,
            mine=mine if not channel_id else None,
            channelId=channel_id or None,
            maxResults=_clamp_int(max_results, minimum=1, maximum=50),
            pageToken=page_token or None,
        ).execute()
        return _attach_page_meta(data, cached)

    return await run_tool("youtube", "list_playlists", _list_playlists, allow_retry=True)


@mcp.tool()
async def youtube_list_playlist_items(
    playlist_id: str,
    part: str = "snippet,contentDetails,status",
    max_results: int = 25,
    page_token: str = "",
) -> str:
    """List items in a YouTube playlist."""

    def _list_playlist_items():
        if not playlist_id:
            raise ValueError("playlist_id cannot be empty")
        service, cached = client.get_service("youtube", "v3")
        data = service.playlistItems().list(
            playlistId=playlist_id,
            part=part,
            maxResults=_clamp_int(max_results, minimum=1, maximum=50),
            pageToken=page_token or None,
        ).execute()
        return _attach_page_meta(data, cached)

    return await run_tool(
        "youtube", "list_playlist_items", _list_playlist_items, allow_retry=True
    )


@mcp.tool()
async def youtube_list_comment_threads(
    part: str = "snippet,replies",
    video_id: str = "",
    channel_id: str = "",
    max_results: int = 20,
    page_token: str = "",
) -> str:
    """List YouTube comment threads for a video or channel."""

    def _list_comment_threads():
        if not video_id and not channel_id:
            raise ValueError("video_id or channel_id is required")
        service, cached = client.get_service("youtube", "v3")
        data = service.commentThreads().list(
            part=part,
            videoId=video_id or None,
            channelId=channel_id or None,
            maxResults=_clamp_int(max_results, minimum=1, maximum=100),
            pageToken=page_token or None,
        ).execute()
        return _attach_page_meta(data, cached)

    return await run_tool(
        "youtube", "list_comment_threads", _list_comment_threads, allow_retry=True
    )


@mcp.tool()
async def analytics_run_report(property_id: str, report_request: dict[str, Any]) -> str:
    """Run a Google Analytics Data API report."""

    def _run_report():
        if not property_id:
            raise ValueError("property_id cannot be empty")
        if not report_request:
            raise ValueError("report_request cannot be empty")
        url = f"https://analyticsdata.googleapis.com/v1beta/properties/{property_id}:runReport"
        return _google_json_request("POST", url, json_body=report_request)

    return await run_tool("analytics", "run_report", _run_report, allow_retry=True)


@mcp.tool()
async def analytics_batch_run_reports(property_id: str, requests: list[dict[str, Any]]) -> str:
    """Run multiple Google Analytics Data API reports."""

    def _batch_run_reports():
        if not property_id:
            raise ValueError("property_id cannot be empty")
        if not requests:
            raise ValueError("requests cannot be empty")
        url = f"https://analyticsdata.googleapis.com/v1beta/properties/{property_id}:batchRunReports"
        return _google_json_request("POST", url, json_body={"requests": requests})

    return await run_tool("analytics", "batch_run_reports", _batch_run_reports, allow_retry=True)


@mcp.tool()
async def analytics_run_realtime_report(property_id: str, report_request: dict[str, Any]) -> str:
    """Run a Google Analytics realtime report."""

    def _run_realtime_report():
        if not property_id:
            raise ValueError("property_id cannot be empty")
        if not report_request:
            raise ValueError("report_request cannot be empty")
        url = f"https://analyticsdata.googleapis.com/v1beta/properties/{property_id}:runRealtimeReport"
        return _google_json_request("POST", url, json_body=report_request)

    return await run_tool(
        "analytics", "run_realtime_report", _run_realtime_report, allow_retry=True
    )


@mcp.tool()
async def analytics_get_metadata(property_id: str) -> str:
    """Get Google Analytics Data API metadata for a property."""

    def _get_metadata():
        if not property_id:
            raise ValueError("property_id cannot be empty")
        url = f"https://analyticsdata.googleapis.com/v1beta/properties/{property_id}/metadata"
        return _google_json_request("GET", url)

    return await run_tool("analytics", "get_metadata", _get_metadata, allow_retry=True)


@mcp.tool()
async def searchconsole_list_sites() -> str:
    """List Search Console sites available to the authenticated user."""

    def _list_sites():
        service, cached = client.get_service("searchconsole", "v1")
        return service.sites().list().execute(), {"cached_service": cached}

    return await run_tool("searchconsole", "list_sites", _list_sites, allow_retry=True)


@mcp.tool()
async def searchconsole_query_search_analytics(site_url: str, request_body: dict[str, Any]) -> str:
    """Query Search Console search analytics."""

    def _query_search_analytics():
        if not site_url:
            raise ValueError("site_url cannot be empty")
        if not request_body:
            raise ValueError("request_body cannot be empty")
        service, cached = client.get_service("searchconsole", "v1")
        request = service.searchanalytics().query(siteUrl=site_url, body=request_body)
        return request.execute(), {"cached_service": cached}

    return await run_tool(
        "searchconsole", "query_search_analytics", _query_search_analytics, allow_retry=True
    )


@mcp.tool()
async def searchconsole_inspect_url(site_url: str, inspection_url: str, language_code: str = "en-US") -> str:
    """Inspect one URL with Search Console URL Inspection API."""

    def _inspect_url():
        if not site_url or not inspection_url:
            raise ValueError("site_url and inspection_url are required")
        service, cached = client.get_service("searchconsole", "v1")
        request = service.urlInspection().index().inspect(
            body={
                "siteUrl": site_url,
                "inspectionUrl": inspection_url,
                "languageCode": language_code,
            }
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool("searchconsole", "inspect_url", _inspect_url, allow_retry=True)


@mcp.tool()
async def searchconsole_list_sitemaps(site_url: str) -> str:
    """List Search Console sitemaps for a site."""

    def _list_sitemaps():
        if not site_url:
            raise ValueError("site_url cannot be empty")
        service, cached = client.get_service("searchconsole", "v1")
        return service.sitemaps().list(siteUrl=site_url).execute(), {"cached_service": cached}

    return await run_tool("searchconsole", "list_sitemaps", _list_sitemaps, allow_retry=True)


@mcp.tool()
async def business_profile_list_accounts() -> str:
    """List Google Business Profile accounts."""

    def _list_accounts():
        service, cached = client.get_service("mybusinessaccountmanagement", "v1")
        return service.accounts().list().execute(), {"cached_service": cached}

    return await run_tool("business_profile", "list_accounts", _list_accounts, allow_retry=True)


@mcp.tool()
async def business_profile_list_locations(
    account_name: str,
    read_mask: str = "name,title,storefrontAddress,phoneNumbers,websiteUri,metadata,categories",
    page_size: int = 50,
    page_token: str = "",
) -> str:
    """List Google Business Profile locations for an account."""

    def _list_locations():
        if not account_name:
            raise ValueError("account_name cannot be empty, for example accounts/123")
        service, cached = client.get_service("mybusinessbusinessinformation", "v1")
        data = service.accounts().locations().list(
            parent=account_name,
            readMask=read_mask,
            pageSize=_clamp_int(page_size, minimum=1, maximum=100),
            pageToken=page_token or None,
        ).execute()
        return _attach_page_meta(data, cached)

    return await run_tool(
        "business_profile", "list_locations", _list_locations, allow_retry=True
    )


@mcp.tool()
async def business_profile_get_location(
    location_name: str,
    read_mask: str = "name,title,storefrontAddress,phoneNumbers,websiteUri,metadata,categories",
) -> str:
    """Get one Google Business Profile location."""

    def _get_location():
        if not location_name:
            raise ValueError("location_name cannot be empty, for example locations/123")
        service, cached = client.get_service("mybusinessbusinessinformation", "v1")
        data = service.locations().get(name=location_name, readMask=read_mask).execute()
        return data, {"cached_service": cached}

    return await run_tool("business_profile", "get_location", _get_location, allow_retry=True)


@mcp.tool()
async def business_profile_fetch_performance(
    location_name: str,
    daily_metrics: list[str],
    start_date: dict[str, int],
    end_date: dict[str, int],
) -> str:
    """Fetch Google Business Profile daily performance metrics."""

    def _fetch_performance():
        if not location_name:
            raise ValueError("location_name cannot be empty, for example locations/123")
        if not daily_metrics:
            raise ValueError("daily_metrics cannot be empty")
        service, cached = client.get_service("businessprofileperformance", "v1")
        request = service.locations().fetchMultiDailyMetricsTimeSeries(
            location=location_name,
            dailyMetrics=daily_metrics,
            dailyRange_startDate_year=start_date.get("year"),
            dailyRange_startDate_month=start_date.get("month"),
            dailyRange_startDate_day=start_date.get("day"),
            dailyRange_endDate_year=end_date.get("year"),
            dailyRange_endDate_month=end_date.get("month"),
            dailyRange_endDate_day=end_date.get("day"),
        )
        return request.execute(), {"cached_service": cached}

    return await run_tool(
        "business_profile", "fetch_performance", _fetch_performance, allow_retry=True
    )


@mcp.tool()
async def maps_geocode(address: str, language: str = "en", region: str = "") -> str:
    """Geocode an address with Google Maps Geocoding API."""

    def _geocode():
        if not address:
            raise ValueError("address cannot be empty")
        return _maps_request(
            "GET",
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address, "language": language, "region": region or None},
        )

    return await run_tool("maps", "geocode", _geocode, allow_retry=True)


@mcp.tool()
async def maps_reverse_geocode(lat: float, lng: float, language: str = "en") -> str:
    """Reverse geocode latitude/longitude with Google Maps."""

    def _reverse_geocode():
        return _maps_request(
            "GET",
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"latlng": f"{lat},{lng}", "language": language},
        )

    return await run_tool("maps", "reverse_geocode", _reverse_geocode, allow_retry=True)


@mcp.tool()
async def maps_place_text_search(query: str, included_type: str = "", max_result_count: int = 10) -> str:
    """Search Google Places with text search."""

    def _place_text_search():
        if not query:
            raise ValueError("query cannot be empty")
        body: dict[str, Any] = {
            "textQuery": query,
            "maxResultCount": _clamp_int(max_result_count, minimum=1, maximum=20),
        }
        if included_type:
            body["includedType"] = included_type
        return _maps_request(
            "POST",
            "https://places.googleapis.com/v1/places:searchText",
            json_body=body,
            headers={
                "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount"
            },
        )

    return await run_tool("maps", "place_text_search", _place_text_search, allow_retry=True)


@mcp.tool()
async def maps_place_details(place_id: str, field_mask: str = "id,displayName,formattedAddress,location,rating,userRatingCount,websiteUri,nationalPhoneNumber") -> str:
    """Get Google Places details."""

    def _place_details():
        if not place_id:
            raise ValueError("place_id cannot be empty")
        return _maps_request(
            "GET",
            f"https://places.googleapis.com/v1/places/{place_id}",
            headers={"X-Goog-FieldMask": field_mask},
        )

    return await run_tool("maps", "place_details", _place_details, allow_retry=True)


@mcp.tool()
async def maps_compute_routes(route_request: dict[str, Any]) -> str:
    """Compute routes with Google Maps Routes API."""

    def _compute_routes():
        if not route_request:
            raise ValueError("route_request cannot be empty")
        return _maps_request(
            "POST",
            "https://routes.googleapis.com/directions/v2:computeRoutes",
            json_body=route_request,
            headers={
                "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline,routes.legs"
            },
        )

    return await run_tool("maps", "compute_routes", _compute_routes, allow_retry=True)


@mcp.tool()
async def merchant_list_products(merchant_id: str, max_results: int = 50, page_token: str = "") -> str:
    """List Merchant Center products with Content API for Shopping."""

    def _list_products():
        if not merchant_id:
            raise ValueError("merchant_id cannot be empty")
        service, cached = client.get_service("content", "v2.1")
        data = service.products().list(
            merchantId=merchant_id,
            maxResults=_clamp_int(max_results, minimum=1, maximum=250),
            pageToken=page_token or None,
        ).execute()
        return _attach_page_meta(data, cached)

    return await run_tool("merchant", "list_products", _list_products, allow_retry=True)


@mcp.tool()
async def merchant_get_product(merchant_id: str, product_id: str) -> str:
    """Get one Merchant Center product."""

    def _get_product():
        if not merchant_id or not product_id:
            raise ValueError("merchant_id and product_id are required")
        service, cached = client.get_service("content", "v2.1")
        data = service.products().get(merchantId=merchant_id, productId=product_id).execute()
        return data, {"cached_service": cached}

    return await run_tool("merchant", "get_product", _get_product, allow_retry=True)


@mcp.tool()
async def adsense_list_accounts(page_size: int = 50, page_token: str = "") -> str:
    """List Google AdSense accounts."""

    def _list_accounts():
        service, cached = client.get_service("adsense", "v2")
        data = service.accounts().list(
            pageSize=_clamp_int(page_size, minimum=1, maximum=100),
            pageToken=page_token or None,
        ).execute()
        return _attach_page_meta(data, cached)

    return await run_tool("adsense", "list_accounts", _list_accounts, allow_retry=True)


@mcp.tool()
async def adsense_generate_report(account_name: str, report_request: dict[str, Any]) -> str:
    """Generate an AdSense report."""

    def _generate_report():
        if not account_name:
            raise ValueError("account_name cannot be empty, for example accounts/pub-123")
        if not report_request:
            raise ValueError("report_request cannot be empty")
        service, cached = client.get_service("adsense", "v2")
        data = service.accounts().reports().generate(parent=account_name, **report_request).execute()
        return data, {"cached_service": cached}

    return await run_tool("adsense", "generate_report", _generate_report, allow_retry=True)


@mcp.tool()
async def google_mcp_welcome() -> str:
    """Show Google MCP navigation, setup requirements, and safe starting points."""

    def _welcome():
        return {
            "service": "google-mcp",
            "classification": "live_mcp",
            "framework": "FastMCP",
            "transport": "streamable_http",
            "tool_count": registered_tool_count(),
            "portal": {
                "intended_access_path": "MAD MCP Portal",
                "grant_header": MCP_PORTAL_GRANT_HEADER,
                "grant_configured": bool(MCP_PORTAL_GRANT_TOKEN),
            },
            "required_headers": [
                MCP_PORTAL_GRANT_HEADER,
                MCP_GOOGLE_CLIENT_ID_HEADER,
                MCP_GOOGLE_CLIENT_SECRET_HEADER,
                MCP_GOOGLE_REFRESH_TOKEN_HEADER,
            ],
            "optional_headers": [
                MCP_GOOGLE_MAPS_API_KEY_HEADER,
            ],
            "recommended_start": [
                "google_mcp_list_capabilities",
                "google_mcp_get_endpoint_coverage",
                "mcp_health_check",
            ],
            "safety": {
                "read_tools": sorted(READ_ONLY_TOOLS | NAVIGATION_TOOLS),
                "write_tools": sorted(WRITE_TOOLS),
                "destructive_tools": sorted(DESTRUCTIVE_TOOLS),
                "send_tools_need_explicit_user_approval": [
                    "gmail_send_message",
                    "gmail_send_raw_message",
                    "gmail_send_draft",
                ],
            },
        }

    return await run_tool("mcp", "welcome", _welcome, allow_retry=False)


@mcp.tool()
async def google_mcp_list_capabilities(category: str = "") -> str:
    """List provider-native Google MCP categories and tool risk groups."""

    def _list_capabilities():
        category_filter = category.strip().lower()
        groups = []
        for group in CAPABILITY_GROUPS:
            if category_filter and category_filter not in group["category"].lower():
                continue
            groups.append(
                {
                    "category": group["category"],
                    "read": group["read"],
                    "write": group["write"],
                    "destructive": group["destructive"],
                }
            )
        return {
            "groups": groups,
            "tool_count": registered_tool_count(),
            "category_filter": category or None,
            "raw_catalog_escape_hatch": "tools/list through MCP, or google_mcp_get_tool_usage for one tool.",
        }

    return await run_tool("mcp", "list_capabilities", _list_capabilities, allow_retry=False)


@mcp.tool()
async def google_mcp_get_endpoint_coverage(
    api: str = "",
    resource: str = "",
    status: str = "",
) -> str:
    """Return the compact Google REST endpoint coverage matrix."""

    def _coverage():
        api_filter = api.strip().lower()
        resource_filter = resource.strip().lower()
        status_filter = status.strip().lower()
        rows = []
        for row in ENDPOINT_COVERAGE:
            row_status = row["status"]
            normalized_status = (
                "missing" if row_status == "partially_implemented" else row_status
            )
            if api_filter and api_filter != row["api"]:
                continue
            if resource_filter and resource_filter not in row["resource"].lower():
                continue
            if status_filter and status_filter not in {row_status, normalized_status}:
                continue
            rows.append(row)
        return {
            "source": "Official Google REST discovery documents; see docs/endpoint-coverage.md.",
            "retrieved": "2026-04-29",
            "filters": {
                "api": api or None,
                "resource": resource or None,
                "status": status or None,
            },
            "rows": rows,
        }

    return await run_tool("mcp", "get_endpoint_coverage", _coverage, allow_retry=False)


@mcp.tool()
async def google_mcp_get_tool_usage(tool_name: str = "") -> str:
    """Describe usage, side effects, and related tools for one Google MCP tool."""

    def _usage():
        name = tool_name.strip()
        if not name:
            return {
                "message": "Provide tool_name for detailed usage.",
                "available_tools": sorted(_tool_registry().keys()),
            }
        if name not in _tool_registry():
            raise ValueError(f"Unknown tool_name: {name}")

        group_matches = [
            group["category"]
            for group in CAPABILITY_GROUPS
            if name in group["read"] or name in group["write"] or name in group["destructive"]
        ]
        if name in READ_ONLY_TOOLS or name in NAVIGATION_TOOLS:
            risk = "read"
        elif name in DESTRUCTIVE_TOOLS:
            risk = "destructive"
        else:
            risk = "write"

        return {
            "tool_name": name,
            "description": TOOL_DESCRIPTIONS.get(name, ""),
            "risk": risk,
            "annotations": _annotation_for_tool(name).model_dump(exclude_none=True),
            "categories": group_matches,
            "parameters": _tool_registry()[name].parameters.get("properties", {}),
            "related_tools": [
                tool
                for group in CAPABILITY_GROUPS
                if group["category"] in group_matches
                for tool in group["read"] + group["write"] + group["destructive"]
                if tool != name
            ],
        }

    return await run_tool("mcp", "get_tool_usage", _usage, allow_retry=False)


@mcp.tool()
async def mcp_health_check(
    run_checks: bool = True,
    warm_all: bool = False,
    doc_id: str = "",
    sheet_id: str = "",
    slide_id: str = "",
) -> str:
    """Report auth/scopes plus optional API health checks."""

    def _health_check():
        creds = client._load_credentials()
        cache_before = {
            "drive": client.is_service_cached("drive", "v3"),
            "docs": client.is_service_cached("docs", "v1"),
            "sheets": client.is_service_cached("sheets", "v4"),
            "slides": client.is_service_cached("slides", "v1"),
            "gmail": client.is_service_cached("gmail", "v1"),
            "calendar": client.is_service_cached("calendar", "v3"),
            "session": client.is_session_cached(),
        }
        checks: dict[str, Any] = {}
        user_email = ""

        def has_scope(fragment: str) -> bool:
            return any(fragment in scope for scope in SCOPES)

        def record_check(name: str, func, skip_reason: str | None = None) -> None:
            nonlocal user_email
            if skip_reason:
                checks[name] = {"ok": None, "skipped": True, "reason": skip_reason}
                return
            try:
                result = func()
                if isinstance(result, dict):
                    maybe_email = result.get("user_email")
                    if maybe_email and not user_email:
                        user_email = maybe_email
                checks[name] = {"ok": True, **result}
            except Exception as exc:
                checks[name] = {"ok": False, "error": _classify_error(exc)}

        warmup: dict[str, Any] = {}
        if run_checks or warm_all:
            _, cached_session = client.get_session()
            warmup["session"] = {
                "was_cached": cached_session,
                "cached_session": True,
                "warmed_now": not cached_session,
            }
        if warm_all:
            for api_name, api_version in (
                ("drive", "v3"),
                ("docs", "v1"),
                ("sheets", "v4"),
                ("slides", "v1"),
                ("gmail", "v1"),
                ("calendar", "v3"),
            ):
                _, cached = client.get_service(api_name, api_version)
                warmup[api_name] = {
                    "was_cached": cached,
                    "cached_service": True,
                    "warmed_now": not cached,
                }

        if run_checks and not has_scope("drive"):
            record_check("drive", lambda: {}, "missing_scope")
        elif run_checks:
            def _drive_check():
                service, cached = client.get_service("drive", "v3")
                data = service.about().get(fields="user,storageQuota").execute()
                user = data.get("user", {}) or {}
                return {
                    "cached_service": cached,
                    "user_email": user.get("emailAddress", ""),
                    "user": user,
                    "storage_quota": data.get("storageQuota", {}),
                }
            record_check("drive", _drive_check)
        else:
            record_check("drive", lambda: {}, "disabled")

        if run_checks and not has_scope("gmail"):
            record_check("gmail", lambda: {}, "missing_scope")
        elif run_checks:
            def _gmail_check():
                service, cached = client.get_service("gmail", "v1")
                data = service.users().getProfile(userId="me").execute()
                return {
                    "cached_service": cached,
                    "user_email": data.get("emailAddress", ""),
                    "profile": data,
                }
            record_check("gmail", _gmail_check)
        else:
            record_check("gmail", lambda: {}, "disabled")

        if run_checks and not has_scope("calendar"):
            record_check("calendar", lambda: {}, "missing_scope")
        elif run_checks:
            def _calendar_check():
                service, cached = client.get_service("calendar", "v3")
                data = service.calendarList().list(
                    maxResults=1,
                    fields="items(id,summary,primary),nextPageToken",
                ).execute()
                return {
                    "cached_service": cached,
                    "calendar_count": len(data.get("items", []) or []),
                }
            record_check("calendar", _calendar_check)
        else:
            record_check("calendar", lambda: {}, "disabled")

        if run_checks and not has_scope("documents"):
            record_check("docs", lambda: {}, "missing_scope")
        elif run_checks and not doc_id:
            if warm_all:
                record_check("docs", lambda: {}, "warmed_without_id")
            else:
                record_check("docs", lambda: {}, "missing_doc_id")
        elif run_checks:
            def _docs_check():
                service, cached = client.get_service("docs", "v1")
                data = service.documents().get(
                    documentId=doc_id,
                    fields=DEFAULT_DOCS_FIELDS,
                ).execute()
                return {"cached_service": cached, "document": data}
            record_check("docs", _docs_check)
        else:
            record_check("docs", lambda: {}, "disabled")

        if run_checks and not has_scope("spreadsheets"):
            record_check("sheets", lambda: {}, "missing_scope")
        elif run_checks and not sheet_id:
            if warm_all:
                record_check("sheets", lambda: {}, "warmed_without_id")
            else:
                record_check("sheets", lambda: {}, "missing_sheet_id")
        elif run_checks:
            def _sheets_check():
                service, cached = client.get_service("sheets", "v4")
                data = service.spreadsheets().get(
                    spreadsheetId=sheet_id,
                    fields=DEFAULT_SHEETS_FIELDS,
                ).execute()
                return {"cached_service": cached, "spreadsheet": data}
            record_check("sheets", _sheets_check)
        else:
            record_check("sheets", lambda: {}, "disabled")

        if run_checks and not has_scope("presentations"):
            record_check("slides", lambda: {}, "missing_scope")
        elif run_checks and not slide_id:
            if warm_all:
                record_check("slides", lambda: {}, "warmed_without_id")
            else:
                record_check("slides", lambda: {}, "missing_slide_id")
        elif run_checks:
            def _slides_check():
                service, cached = client.get_service("slides", "v1")
                data = service.presentations().get(
                    presentationId=slide_id,
                    fields=DEFAULT_SLIDES_FIELDS,
                ).execute()
                return {"cached_service": cached, "presentation": data}
            record_check("slides", _slides_check)
        else:
            record_check("slides", lambda: {}, "disabled")

        cache_after = {
            "drive": client.is_service_cached("drive", "v3"),
            "docs": client.is_service_cached("docs", "v1"),
            "sheets": client.is_service_cached("sheets", "v4"),
            "slides": client.is_service_cached("slides", "v1"),
            "gmail": client.is_service_cached("gmail", "v1"),
            "calendar": client.is_service_cached("calendar", "v3"),
            "session": client.is_session_cached(),
        }

        return {
            "server_instance_id": SERVER_INSTANCE_ID,
            "server_uptime_ms": round((time.monotonic() - SERVER_START_MONO) * 1000, 2),
            "server_version": MCP_SERVER_VERSION,
            "user_email": user_email,
            "token_valid": creds.valid,
            "token_expiry": creds.expiry.isoformat() if creds.expiry else None,
            "scopes": SCOPES,
            "cache_before": cache_before,
            "cache_after": cache_after,
            "warmup": warmup,
            "checks": checks,
        }

    return await run_tool("mcp", "health_check", _health_check, allow_retry=False)


EXPANDED_READ_TOOLS = {
    "drive_about_get",
    "drive_list_permissions",
    "drive_get_permission",
    "drive_list_comments",
    "drive_list_revisions",
    "drive_list_shared_drives",
    "sheets_get_by_data_filter",
    "slides_get_page",
    "slides_get_page_thumbnail",
    "gmail_get_label",
    "gmail_list_drafts",
    "gmail_get_draft",
    "gmail_get_attachment",
    "gmail_list_history",
    "gmail_mailbox_overview",
    "gmail_sender_clusters",
    "gmail_cleanup_plan",
    "calendar_freebusy_query",
    "calendar_get_colors",
    "calendar_list_settings",
    "calendar_get_setting",
    "calendar_get_calendar_list_entry",
    "calendar_list_event_instances",
    "calendar_list_acl",
    "calendar_get_acl",
    "youtube_search",
    "youtube_list_channels",
    "youtube_list_videos",
    "youtube_list_playlists",
    "youtube_list_playlist_items",
    "youtube_list_comment_threads",
    "analytics_run_report",
    "analytics_batch_run_reports",
    "analytics_run_realtime_report",
    "analytics_get_metadata",
    "searchconsole_list_sites",
    "searchconsole_query_search_analytics",
    "searchconsole_inspect_url",
    "searchconsole_list_sitemaps",
    "business_profile_list_accounts",
    "business_profile_list_locations",
    "business_profile_get_location",
    "business_profile_fetch_performance",
    "maps_geocode",
    "maps_reverse_geocode",
    "maps_place_text_search",
    "maps_place_details",
    "maps_compute_routes",
    "merchant_list_products",
    "merchant_get_product",
    "adsense_list_accounts",
    "adsense_generate_report",
}

EXPANDED_WRITE_TOOLS = {
    "drive_copy_file",
    "drive_create_comment",
    "gmail_update_label",
    "gmail_update_draft",
    "gmail_modify_thread_labels",
    "calendar_update_calendar_list_entry",
    "calendar_update_calendar",
}

EXPANDED_DESTRUCTIVE_TOOLS = {
    "drive_create_permission",
    "drive_update_permission",
    "drive_delete_permission",
    "drive_update_comment",
    "drive_delete_comment",
    "drive_update_file_metadata",
    "docs_batch_update",
    "sheets_append_values",
    "sheets_clear_values",
    "sheets_batch_update_values",
    "sheets_batch_clear_values",
    "sheets_batch_update",
    "slides_batch_update",
    "gmail_batch_modify_messages",
    "gmail_batch_delete_messages",
    "gmail_delete_draft",
    "gmail_trash_thread",
    "gmail_untrash_thread",
    "gmail_delete_thread",
    "gmail_apply_cleanup_plan",
    "calendar_delete_calendar_list_entry",
    "calendar_clear_calendar",
    "calendar_move_event",
    "calendar_import_event",
    "calendar_replace_event",
    "calendar_upsert_acl",
    "calendar_delete_acl",
}

EXPANDED_TOOL_DESCRIPTIONS = {
    "gmail_mailbox_overview": "Use this read-only Gmail workflow tool to get compact counts for unread/inbox cleanup planning without fetching message bodies.",
    "gmail_sender_clusters": "Use this read-only Gmail workflow tool to group large unread/search result sets by sender, domain, and List-ID with capped samples.",
    "gmail_cleanup_plan": "Use this read-only Gmail workflow tool to propose cleanup batches that must be approved before mutation.",
    "gmail_apply_cleanup_plan": "Use this destructive Gmail workflow tool only after approving an explicit cleanup batch; dry_run defaults to true.",
    "gmail_batch_modify_messages": "Use this destructive Gmail tool to apply label changes to explicit message IDs in provider-sized chunks; dry_run defaults to true.",
    "gmail_batch_delete_messages": "Use this destructive Gmail tool only for approved permanent batch deletion of explicit message IDs; dry_run defaults to true.",
    "youtube_search": "Use this read-only YouTube Data API tool to search videos, channels, or playlists with compact pagination.",
    "analytics_run_report": "Use this read-only Google Analytics Data API tool to run GA4 reports for an accessible property.",
    "searchconsole_query_search_analytics": "Use this read-only Search Console tool to query website search performance data.",
    "business_profile_fetch_performance": "Use this read-only Google Business Profile tool to fetch daily location performance metrics.",
    "maps_place_text_search": "Use this read-only Google Maps Places tool for business/place discovery; requires Maps API key configuration.",
    "maps_compute_routes": "Use this read-only Google Maps Routes tool for route estimates; requires Maps API key and can incur Maps Platform usage.",
}


def _extend_expanded_google_surface_metadata() -> None:
    READ_ONLY_TOOLS.update(EXPANDED_READ_TOOLS)
    WRITE_TOOLS.update(EXPANDED_WRITE_TOOLS)
    DESTRUCTIVE_TOOLS.update(EXPANDED_DESTRUCTIVE_TOOLS)
    TOOL_DESCRIPTIONS.update(EXPANDED_TOOL_DESCRIPTIONS)
    COMMON_PARAMETER_DESCRIPTIONS.update(
        {
            "dry_run": "When true, return the planned operation without mutating provider data.",
            "action": "Cleanup or mutation action to perform after approval.",
            "target_label_id": "Gmail label ID proposed for cleanup batches.",
            "start_history_id": "Gmail history ID to begin incremental history listing.",
            "history_types": "Gmail history event type filters.",
            "attachment_id": "Gmail attachment ID from a message payload part.",
            "permission_id": "Google Drive permission ID.",
            "permission_body": "Google Drive permission request body.",
            "comment_id": "Google Drive comment ID.",
            "metadata": "Provider metadata patch body.",
            "requests": "Provider batchUpdate request objects.",
            "data": "Google Sheets values batch update data entries.",
            "data_filters": "Google Sheets data filter objects.",
            "page_object_id": "Google Slides page object ID.",
            "thumbnail_size": "Google Slides thumbnail size.",
            "rule_id": "Google Calendar ACL rule ID.",
            "rule_body": "Google Calendar ACL rule request body.",
            "items": "Google Calendar freebusy items, each with an id field.",
            "source_calendar_id": "Source Google Calendar ID.",
            "destination_calendar_id": "Destination Google Calendar ID.",
            "event_body": "Google Calendar event request body.",
            "part": "Google API partial resource selector such as snippet or statistics.",
            "channel_ids": "YouTube channel IDs.",
            "for_username": "Legacy YouTube username lookup.",
            "video_ids": "YouTube video IDs.",
            "video_id": "YouTube video ID.",
            "playlist_id": "YouTube playlist ID.",
            "property_id": "Google Analytics property ID without the properties/ prefix.",
            "report_request": "Google Analytics report request body.",
            "site_url": "Search Console site URL or sc-domain property.",
            "inspection_url": "URL to inspect in Search Console.",
            "language_code": "BCP-47 language code for Search Console inspection output.",
            "account_name": "Google Business Profile or AdSense account resource name.",
            "location_name": "Google Business Profile location resource name.",
            "read_mask": "Google field mask selecting returned Business Profile fields.",
            "daily_metrics": "Business Profile performance daily metric names.",
            "start_date": "Date object with year, month, and day.",
            "end_date": "Date object with year, month, and day.",
            "lat": "Latitude.",
            "lng": "Longitude.",
            "included_type": "Optional Google Places included type filter.",
            "max_result_count": "Maximum number of Maps/Places results.",
            "place_id": "Google Maps Place ID.",
            "field_mask": "Google field mask selecting returned fields.",
            "route_request": "Google Routes API computeRoutes request body.",
            "merchant_id": "Google Merchant Center merchant ID.",
            "product_id": "Google Merchant Center product ID.",
        }
    )
    PARAMETER_ENUMS.update(
        {
            "action": ["label", "archive", "mark_read", "trash"],
            "proposed_action": ["label", "archive", "mark_read", "trash"],
            "thumbnail_size": ["THUMBNAIL_SIZE_UNSPECIFIED", "LARGE", "MEDIUM", "SMALL"],
            "mime_type": ["PNG", "JPEG"],
            "insert_data_option": ["OVERWRITE", "INSERT_ROWS"],
            "type": ["", "video", "channel", "playlist"],
            "order": ["date", "rating", "relevance", "title", "videoCount", "viewCount"],
        }
    )
    CAPABILITY_GROUPS.extend(
        [
            {
                "category": "Gmail API v1 inbox scale workflows",
                "read": [
                    "gmail_mailbox_overview",
                    "gmail_sender_clusters",
                    "gmail_cleanup_plan",
                ],
                "write": [],
                "destructive": [
                    "gmail_apply_cleanup_plan",
                    "gmail_batch_modify_messages",
                    "gmail_batch_delete_messages",
                ],
            },
            {
                "category": "Workspace depth tools",
                "read": sorted(
                    {
                        "drive_about_get",
                        "drive_list_permissions",
                        "drive_get_permission",
                        "drive_list_comments",
                        "drive_list_revisions",
                        "drive_list_shared_drives",
                        "sheets_get_by_data_filter",
                        "slides_get_page",
                        "slides_get_page_thumbnail",
                        "calendar_freebusy_query",
                        "calendar_get_colors",
                        "calendar_list_settings",
                        "calendar_get_setting",
                        "calendar_get_calendar_list_entry",
                        "calendar_list_event_instances",
                        "calendar_list_acl",
                        "calendar_get_acl",
                    }
                ),
                "write": sorted(EXPANDED_WRITE_TOOLS),
                "destructive": sorted(
                    EXPANDED_DESTRUCTIVE_TOOLS
                    - {
                        "gmail_apply_cleanup_plan",
                        "gmail_batch_modify_messages",
                        "gmail_batch_delete_messages",
                    }
                ),
            },
            {
                "category": "YouTube Data API v3",
                "read": [
                    "youtube_search",
                    "youtube_list_channels",
                    "youtube_list_videos",
                    "youtube_list_playlists",
                    "youtube_list_playlist_items",
                    "youtube_list_comment_threads",
                ],
                "write": [],
                "destructive": [],
            },
            {
                "category": "Business analytics and web presence",
                "read": [
                    "analytics_run_report",
                    "analytics_batch_run_reports",
                    "analytics_run_realtime_report",
                    "analytics_get_metadata",
                    "searchconsole_list_sites",
                    "searchconsole_query_search_analytics",
                    "searchconsole_inspect_url",
                    "searchconsole_list_sitemaps",
                    "business_profile_list_accounts",
                    "business_profile_list_locations",
                    "business_profile_get_location",
                    "business_profile_fetch_performance",
                    "merchant_list_products",
                    "merchant_get_product",
                    "adsense_list_accounts",
                    "adsense_generate_report",
                ],
                "write": [],
                "destructive": [],
            },
            {
                "category": "Google Maps Platform",
                "read": [
                    "maps_geocode",
                    "maps_reverse_geocode",
                    "maps_place_text_search",
                    "maps_place_details",
                    "maps_compute_routes",
                ],
                "write": [],
                "destructive": [],
            },
        ]
    )
    ENDPOINT_COVERAGE.extend(
        [
            {
                "api": "gmail",
                "resource": "users.messages/users.threads/users.drafts/users.history",
                "status": "partially_implemented",
                "implemented": [
                    "messages.batchDelete",
                    "messages.batchModify",
                    "threads.modify",
                    "threads.trash",
                    "threads.untrash",
                    "threads.delete",
                    "drafts.delete",
                    "drafts.get",
                    "drafts.list",
                    "drafts.update",
                    "history.list",
                    "messages.attachments.get",
                ],
                "missing": ["messages.import", "messages.insert", "users.watch", "users.stop"],
                "tool_refs": sorted(EXPANDED_READ_TOOLS | EXPANDED_DESTRUCTIVE_TOOLS),
            },
            {
                "api": "youtube",
                "resource": "channels/videos/playlists/playlistItems/search/commentThreads",
                "status": "partially_implemented",
                "implemented": ["channels.list", "videos.list", "playlists.list", "playlistItems.list", "search.list", "commentThreads.list"],
                "missing": ["captions.*", "comments.*", "subscriptions.*", "members.*", "uploads and mutations"],
                "tool_refs": ["youtube_search", "youtube_list_channels", "youtube_list_videos", "youtube_list_playlists", "youtube_list_playlist_items", "youtube_list_comment_threads"],
            },
            {
                "api": "analytics",
                "resource": "properties",
                "status": "partially_implemented",
                "implemented": ["runReport", "batchRunReports", "runRealtimeReport", "getMetadata"],
                "missing": ["runPivotReport", "batchRunPivotReports", "checkCompatibility", "audienceExports.*"],
                "tool_refs": ["analytics_run_report", "analytics_batch_run_reports", "analytics_run_realtime_report", "analytics_get_metadata"],
            },
            {
                "api": "searchconsole",
                "resource": "sites/sitemaps/searchanalytics/urlInspection",
                "status": "partially_implemented",
                "implemented": ["sites.list", "sitemaps.list", "searchanalytics.query", "urlInspection.index.inspect"],
                "missing": ["sites.add", "sites.delete", "sitemaps.submit", "sitemaps.delete"],
                "tool_refs": ["searchconsole_list_sites", "searchconsole_query_search_analytics", "searchconsole_inspect_url", "searchconsole_list_sitemaps"],
            },
            {
                "api": "business_profile",
                "resource": "accounts/locations/performance",
                "status": "partially_implemented",
                "implemented": ["accounts.list", "accounts.locations.list", "locations.get", "locations.fetchMultiDailyMetricsTimeSeries"],
                "missing": ["location mutation", "verifications", "notifications", "qanda", "place actions"],
                "tool_refs": ["business_profile_list_accounts", "business_profile_list_locations", "business_profile_get_location", "business_profile_fetch_performance"],
            },
            {
                "api": "maps",
                "resource": "geocoding/places/routes",
                "status": "partially_implemented",
                "implemented": ["geocode", "reverse_geocode", "places.searchText", "places.get", "routes.computeRoutes"],
                "missing": ["route optimization", "roads", "static maps", "distance matrix migration variants"],
                "tool_refs": ["maps_geocode", "maps_reverse_geocode", "maps_place_text_search", "maps_place_details", "maps_compute_routes"],
            },
            {
                "api": "merchant_ads_adsense",
                "resource": "merchant/content/adsense/google-ads",
                "status": "blocked_scope",
                "implemented": ["content.products.list", "content.products.get", "adsense.accounts.list", "adsense.reports.generate"],
                "missing": ["Merchant mutations", "Google Ads API", "Merchant API v1 productInputs"],
                "tool_refs": ["merchant_list_products", "merchant_get_product", "adsense_list_accounts", "adsense_generate_report", "google_raw_request"],
            },
        ]
    )


_extend_expanded_google_surface_metadata()
_apply_tool_metadata()


def _extract_header_from_scope(scope: dict[str, Any], name: str) -> str:
    target = name.lower().encode("utf-8")
    for key, value in scope.get("headers", []):
        if key.lower() == target:
            return value.decode("utf-8", errors="ignore")
    return ""


def _safe_parse_json(body: bytes) -> dict[str, Any] | None:
    if not body:
        return {}
    try:
        payload = json.loads(body)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _extract_jsonrpc_id(payload: dict[str, Any] | None) -> str | int:
    if not isinstance(payload, dict):
        return "server-error"
    candidate = payload.get("id", "server-error")
    if isinstance(candidate, (str, int)):
        return candidate
    return "server-error"


def _validate_portal_grant(headers: dict[str, str]) -> None:
    if not MCP_PORTAL_GRANT_TOKEN:
        raise ValueError("Server is missing required portal grant configuration.")
    supplied = headers.get(MCP_PORTAL_GRANT_HEADER, "")
    if not supplied:
        raise ValueError(f"Missing required header: {MCP_PORTAL_GRANT_HEADER}.")
    if not hmac.compare_digest(supplied, MCP_PORTAL_GRANT_TOKEN):
        raise ValueError("Invalid portal grant token.")


def _camel_to_snake(value: str) -> str:
    first_pass = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()


def _normalize_tool_arguments(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("method") != "tools/call":
        return payload
    params = payload.get("params")
    if not isinstance(params, dict):
        return payload
    tool_name = params.get("name")
    if not isinstance(tool_name, str):
        return payload
    if "." in tool_name:
        tool_name = tool_name.rsplit(".", 1)[-1]
    tool = _tool_registry().get(tool_name)
    if tool is None:
        return payload
    arguments = params.get("arguments")
    if not isinstance(arguments, dict):
        return payload
    allowed = set(tool.parameters.get("properties", {}).keys())
    normalized: dict[str, Any] = {}
    changed = False
    for key, value in arguments.items():
        target_key = key
        if key not in allowed:
            snake_key = _camel_to_snake(key)
            if snake_key in allowed and snake_key not in arguments:
                target_key = snake_key
                changed = True
        normalized[target_key] = value
    if changed:
        params["arguments"] = normalized
    return payload


async def _read_request_body(receive) -> bytes:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message.get("type") != "http.request":
            continue
        body = message.get("body") or b""
        if body:
            chunks.append(body)
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


def _replay_receive(body: bytes):
    emitted = False

    async def _inner():
        nonlocal emitted
        if not emitted:
            emitted = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    return _inner


async def _send_json(
    send,
    status: int,
    payload: dict[str, Any],
    *,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _send_jsonrpc_error(
    send,
    *,
    status: int,
    code: int,
    message: str,
    request_id: str | int,
    data: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
    if data is not None:
        payload["error"]["data"] = data
    await _send_json(send, status, payload)


def _accepts_media_type(accept_header: str, media_type: str) -> bool:
    if not accept_header:
        return False
    target = media_type.lower()
    for item in accept_header.lower().split(","):
        value = item.strip().split(";", 1)[0].strip()
        if not value:
            continue
        if value == "*/*" or value == target:
            return True
        if value.endswith("/*"):
            prefix = value[:-1]
            if target.startswith(prefix):
                return True
    return False


def build_hosted_mcp_http_wrapper(app):
    async def _wrapped(scope, receive, send):
        if scope.get("type") != "http":
            await app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = str(scope.get("method", "")).upper()

        if path == "/mcp/":
            await _send_json(
                send,
                410,
                {
                    "error": "deprecated_endpoint",
                    "message": "Deprecated MCP URL. Use /mcp (remove trailing slash).",
                },
                extra_headers=[(b"cache-control", b"no-store")],
            )
            return

        if path == "/health":
            tool_count = registered_tool_count()
            await _send_json(
                send,
                200,
                {
                    "ok": True,
                    "service": "google-mcp",
                    "version": os.getenv("MCP_SERVER_VERSION", "dev"),
                    "tool_count": tool_count,
                    "tools": {"total": tool_count},
                    "configuration": {
                        "portal_grant_configured": bool(MCP_PORTAL_GRANT_TOKEN),
                        "byok_headers_required": {
                            MCP_GOOGLE_CLIENT_ID_HEADER: MCP_REQUIRE_REQUEST_GOOGLE_CLIENT_ID,
                            MCP_GOOGLE_CLIENT_SECRET_HEADER: MCP_REQUIRE_REQUEST_GOOGLE_CLIENT_SECRET,
                            MCP_GOOGLE_REFRESH_TOKEN_HEADER: MCP_REQUIRE_REQUEST_GOOGLE_REFRESH_TOKEN,
                        },
                        "optional_headers": {
                            MCP_GOOGLE_MAPS_API_KEY_HEADER: bool(GOOGLE_MAPS_API_KEY),
                        },
                    },
                },
                extra_headers=[(b"cache-control", b"no-store")],
            )
            return

        if path != "/mcp":
            await app(scope, receive, send)
            return

        jsonrpc_id: str | int = "server-error"
        body = b""
        consumed_body = False
        if method in {"POST", "PUT", "PATCH"}:
            body = await _read_request_body(receive)
            consumed_body = True
            payload = _safe_parse_json(body)
            if payload is None:
                await _send_jsonrpc_error(
                    send,
                    status=400,
                    code=-32600,
                    message="Bad Request: body must be valid JSON-RPC object",
                    request_id=jsonrpc_id,
                )
                return
            jsonrpc_id = _extract_jsonrpc_id(payload)
            normalized_payload = _normalize_tool_arguments(payload)
            if normalized_payload is not payload or normalized_payload != payload:
                payload = normalized_payload
            body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode(
                "utf-8"
            )

        if method == "POST":
            content_type = _extract_header_from_scope(scope, "content-type").lower()
            if "application/json" not in content_type:
                await _send_jsonrpc_error(
                    send,
                    status=400,
                    code=-32600,
                    message="Bad Request: Content-Type must include application/json",
                    request_id=jsonrpc_id,
                )
                return
            accept = _extract_header_from_scope(scope, "accept")
            if not _accepts_media_type(accept, "application/json"):
                await _send_jsonrpc_error(
                    send,
                    status=406,
                    code=-32600,
                    message="Not Acceptable: Client must accept application/json",
                    request_id=jsonrpc_id,
                )
                return

        if method == "GET":
            accept = _extract_header_from_scope(scope, "accept")
            if not _accepts_media_type(accept, "text/event-stream"):
                await _send_jsonrpc_error(
                    send,
                    status=406,
                    code=-32600,
                    message="Not Acceptable: Client must accept text/event-stream",
                    request_id=jsonrpc_id,
                )
                return

        normalized_headers = _normalize_header_map(scope.get("headers", []) or [])
        try:
            _validate_portal_grant(normalized_headers)
        except ValueError as exc:
            await _send_jsonrpc_error(
                send,
                status=401,
                code=-32001,
                message=str(exc),
                request_id=jsonrpc_id,
                data={"required_headers": [MCP_PORTAL_GRANT_HEADER]},
            )
            return

        try:
            request_client, _ = _resolve_request_client(scope.get("headers", []) or [])
        except ValueError as exc:
            await _send_jsonrpc_error(
                send,
                status=401,
                code=-32001,
                message=str(exc),
                request_id=jsonrpc_id,
                data={
                    "required_headers": [
                        MCP_PORTAL_GRANT_HEADER,
                        MCP_GOOGLE_CLIENT_ID_HEADER,
                        MCP_GOOGLE_CLIENT_SECRET_HEADER,
                        MCP_GOOGLE_REFRESH_TOKEN_HEADER,
                    ],
                },
            )
            return

        token = ACTIVE_GOOGLE_CLIENT.set(request_client)
        headers_token = ACTIVE_REQUEST_HEADERS.set(normalized_headers)
        try:
            next_receive = _replay_receive(body) if consumed_body else receive
            await app(scope, next_receive, send)
        finally:
            ACTIVE_REQUEST_HEADERS.reset(headers_token)
            ACTIVE_GOOGLE_CLIENT.reset(token)

    return _wrapped


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
        return build_hosted_mcp_http_wrapper(app)

    uvicorn.run(
        build_app,
        host=MCP_BIND_ADDRESS,
        port=MCP_HTTP_PORT,
        factory=True,
        workers=MCP_WORKERS,
    )
