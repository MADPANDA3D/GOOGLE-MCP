import asyncio
import base64
import io
import json
import os
import threading
from typing import Any

from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload, MediaIoBaseDownload
from mcp.server.fastmcp import FastMCP
import uvicorn
from starlette.middleware.trustedhost import TrustedHostMiddleware


DEFAULT_SCOPES = (
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/presentations",
)

MCP_HTTP_PORT = int(os.getenv("MCP_HTTP_PORT", "8086"))
MCP_BIND_ADDRESS = os.getenv("MCP_BIND_ADDRESS", "0.0.0.0")
GOOGLE_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_CREDENTIALS_PATH", "fastmcp/.google/credentials.json"
)
GOOGLE_TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", "fastmcp/.google/token.json")
GOOGLE_SCOPES_RAW = os.getenv("GOOGLE_SCOPES", " ".join(DEFAULT_SCOPES))


mcp = FastMCP(
    name="google-mcp",
    stateless_http=True,
    json_response=True,
    host="0.0.0.0",
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

    def _save_token(self, creds: Credentials) -> None:
        token_dir = os.path.dirname(self.token_path)
        if token_dir:
            os.makedirs(token_dir, exist_ok=True)
        with open(self.token_path, "w", encoding="utf-8") as handle:
            handle.write(creds.to_json())

    def _load_credentials(self) -> Credentials:
        if not os.path.exists(self.token_path):
            raise FileNotFoundError(
                "Missing token.json. Run fastmcp/google_auth_local.py locally and copy it to the server."
            )
        with self._lock:
            creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)
            if not creds.valid:
                if creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                    self._save_token(creds)
                else:
                    raise RuntimeError(
                        "Token is invalid or expired without refresh token. Re-run the local auth flow."
                    )
            return creds

    def build_service(self, api_name: str, api_version: str):
        creds = self._load_credentials()
        return build(api_name, api_version, credentials=creds, cache_discovery=False)

    def authed_session(self) -> AuthorizedSession:
        creds = self._load_credentials()
        return AuthorizedSession(creds)


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
    return json.dumps(data, indent=2, sort_keys=True)


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

    def _request():
        session = client.authed_session()
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
            return payload
        try:
            text = response.text
            payload["text"] = text
        except UnicodeDecodeError:
            payload["content_base64"] = base64.b64encode(response.content).decode("ascii")
            payload["content_type"] = content_type
        return payload

    result = await run_blocking(_request)
    return json_dumps(result)


@mcp.tool()
async def drive_list_files(
    query: str = "",
    page_size: int = 100,
    fields: str = "files(id,name,mimeType,modifiedTime,size,parents),nextPageToken",
    order_by: str = "",
) -> str:
    """List files in Google Drive using the v3 API."""

    def _list_files():
        service = client.build_service("drive", "v3")
        request = service.files().list(
            q=query or None,
            pageSize=page_size,
            fields=fields,
            orderBy=order_by or None,
        )
        return request.execute()

    result = await run_blocking(_list_files)
    return json_dumps(result)


@mcp.tool()
async def drive_get_file(
    file_id: str,
    fields: str = "id,name,mimeType,modifiedTime,size,parents",
) -> str:
    """Get Drive file metadata."""

    if not file_id:
        raise ValueError("file_id cannot be empty")

    def _get_file():
        service = client.build_service("drive", "v3")
        request = service.files().get(fileId=file_id, fields=fields)
        return request.execute()

    result = await run_blocking(_get_file)
    return json_dumps(result)


@mcp.tool()
async def drive_create_folder(name: str, parent_id: str = "") -> str:
    """Create a Drive folder."""

    if not name:
        raise ValueError("name cannot be empty")

    def _create_folder():
        service = client.build_service("drive", "v3")
        body: dict[str, Any] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            body["parents"] = [parent_id]
        request = service.files().create(body=body, fields="id,name")
        return request.execute()

    result = await run_blocking(_create_folder)
    return json_dumps(result)


@mcp.tool()
async def drive_upload_file(
    name: str,
    content: str,
    mime_type: str = "text/plain",
    parent_id: str = "",
    is_base64: bool = False,
) -> str:
    """Upload a file to Drive from text or base64 content."""

    if not name:
        raise ValueError("name cannot be empty")
    if content is None:
        raise ValueError("content cannot be empty")

    def _upload():
        service = client.build_service("drive", "v3")
        data = (
            base64.b64decode(content.encode("ascii")) if is_base64 else content.encode("utf-8")
        )
        media = MediaInMemoryUpload(data, mimetype=mime_type, resumable=False)
        body: dict[str, Any] = {"name": name}
        if parent_id:
            body["parents"] = [parent_id]
        request = service.files().create(
            body=body,
            media_body=media,
            fields="id,name,mimeType,parents",
        )
        return request.execute()

    result = await run_blocking(_upload)
    return json_dumps(result)


@mcp.tool()
async def drive_download_file(
    file_id: str,
    export_mime_type: str = "",
) -> str:
    """Download a file from Drive. For Google Docs/Sheets/Slides, use export."""

    if not file_id:
        raise ValueError("file_id cannot be empty")

    def _download():
        service = client.build_service("drive", "v3")
        metadata = service.files().get(
            fileId=file_id, fields="id,name,mimeType,size"
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
            request = service.files().export(fileId=file_id, mimeType=download_mime)
        else:
            request = service.files().get_media(fileId=file_id)
            download_mime = mime_type

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        content_bytes = buffer.getvalue()

        return {
            "file": metadata,
            "download_mime_type": download_mime,
            "content_base64": base64.b64encode(content_bytes).decode("ascii"),
        }

    result = await run_blocking(_download)
    return json_dumps(result)


@mcp.tool()
async def docs_create_document(title: str) -> str:
    """Create a Google Doc."""

    if not title:
        raise ValueError("title cannot be empty")

    def _create_doc():
        service = client.build_service("docs", "v1")
        request = service.documents().create(body={"title": title})
        return request.execute()

    result = await run_blocking(_create_doc)
    return json_dumps(result)


@mcp.tool()
async def docs_get_document(document_id: str) -> str:
    """Fetch a Google Doc document."""

    if not document_id:
        raise ValueError("document_id cannot be empty")

    def _get_doc():
        service = client.build_service("docs", "v1")
        request = service.documents().get(documentId=document_id)
        return request.execute()

    result = await run_blocking(_get_doc)
    return json_dumps(result)


@mcp.tool()
async def docs_insert_text(document_id: str, text: str, index: int = 1) -> str:
    """Insert text into a Google Doc at the given index."""

    if not document_id:
        raise ValueError("document_id cannot be empty")
    if text is None:
        raise ValueError("text cannot be empty")

    def _insert():
        service = client.build_service("docs", "v1")
        body = {
            "requests": [
                {"insertText": {"location": {"index": index}, "text": text}}
            ]
        }
        request = service.documents().batchUpdate(documentId=document_id, body=body)
        return request.execute()

    result = await run_blocking(_insert)
    return json_dumps(result)


@mcp.tool()
async def docs_replace_text(
    document_id: str,
    contains_text: str,
    replace_text: str,
    match_case: bool = False,
) -> str:
    """Replace text in a Google Doc."""

    if not document_id:
        raise ValueError("document_id cannot be empty")
    if not contains_text:
        raise ValueError("contains_text cannot be empty")

    def _replace():
        service = client.build_service("docs", "v1")
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
        return request.execute()

    result = await run_blocking(_replace)
    return json_dumps(result)


@mcp.tool()
async def sheets_create_spreadsheet(title: str) -> str:
    """Create a Google Sheet."""

    if not title:
        raise ValueError("title cannot be empty")

    def _create_sheet():
        service = client.build_service("sheets", "v4")
        request = service.spreadsheets().create(body={"properties": {"title": title}})
        return request.execute()

    result = await run_blocking(_create_sheet)
    return json_dumps(result)


@mcp.tool()
async def sheets_get_spreadsheet(spreadsheet_id: str) -> str:
    """Fetch a Google Sheet spreadsheet."""

    if not spreadsheet_id:
        raise ValueError("spreadsheet_id cannot be empty")

    def _get_sheet():
        service = client.build_service("sheets", "v4")
        request = service.spreadsheets().get(spreadsheetId=spreadsheet_id)
        return request.execute()

    result = await run_blocking(_get_sheet)
    return json_dumps(result)


@mcp.tool()
async def sheets_get_values(spreadsheet_id: str, range_a1: str) -> str:
    """Read values from a Google Sheet range."""

    if not spreadsheet_id:
        raise ValueError("spreadsheet_id cannot be empty")
    if not range_a1:
        raise ValueError("range_a1 cannot be empty")

    def _get_values():
        service = client.build_service("sheets", "v4")
        request = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_a1,
        )
        return request.execute()

    result = await run_blocking(_get_values)
    return json_dumps(result)


@mcp.tool()
async def sheets_update_values(
    spreadsheet_id: str,
    range_a1: str,
    values: list[list[Any]],
    value_input_option: str = "RAW",
) -> str:
    """Write values to a Google Sheet range."""

    if not spreadsheet_id:
        raise ValueError("spreadsheet_id cannot be empty")
    if not range_a1:
        raise ValueError("range_a1 cannot be empty")
    if values is None:
        raise ValueError("values cannot be empty")

    def _update_values():
        service = client.build_service("sheets", "v4")
        body = {"values": values}
        request = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_a1,
            valueInputOption=value_input_option,
            body=body,
        )
        return request.execute()

    result = await run_blocking(_update_values)
    return json_dumps(result)


@mcp.tool()
async def slides_create_presentation(title: str) -> str:
    """Create a Google Slides presentation."""

    if not title:
        raise ValueError("title cannot be empty")

    def _create_presentation():
        service = client.build_service("slides", "v1")
        request = service.presentations().create(body={"title": title})
        return request.execute()

    result = await run_blocking(_create_presentation)
    return json_dumps(result)


@mcp.tool()
async def slides_get_presentation(presentation_id: str) -> str:
    """Fetch a Google Slides presentation."""

    if not presentation_id:
        raise ValueError("presentation_id cannot be empty")

    def _get_presentation():
        service = client.build_service("slides", "v1")
        request = service.presentations().get(presentationId=presentation_id)
        return request.execute()

    result = await run_blocking(_get_presentation)
    return json_dumps(result)


@mcp.tool()
async def slides_replace_text(
    presentation_id: str,
    contains_text: str,
    replace_text: str,
    match_case: bool = False,
) -> str:
    """Replace text across a Slides presentation."""

    if not presentation_id:
        raise ValueError("presentation_id cannot be empty")
    if not contains_text:
        raise ValueError("contains_text cannot be empty")

    def _replace():
        service = client.build_service("slides", "v1")
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
        return request.execute()

    result = await run_blocking(_replace)
    return json_dumps(result)


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
