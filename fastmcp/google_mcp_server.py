import asyncio
import base64
import io
import json
import os
import threading
from email.message import EmailMessage
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
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
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
    values: Values,
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


@mcp.tool()
async def gmail_list_labels() -> str:
    """List Gmail labels for the authenticated user."""

    def _list_labels():
        service = client.build_service("gmail", "v1")
        request = service.users().labels().list(userId="me")
        return request.execute()

    result = await run_blocking(_list_labels)
    return json_dumps(result)


@mcp.tool()
async def gmail_create_label(
    name: str,
    label_list_visibility: str = "labelShow",
    message_list_visibility: str = "show",
) -> str:
    """Create a Gmail label."""

    if not name:
        raise ValueError("name cannot be empty")

    def _create_label():
        service = client.build_service("gmail", "v1")
        body = {
            "name": name,
            "labelListVisibility": label_list_visibility,
            "messageListVisibility": message_list_visibility,
        }
        request = service.users().labels().create(userId="me", body=body)
        return request.execute()

    result = await run_blocking(_create_label)
    return json_dumps(result)


@mcp.tool()
async def gmail_delete_label(label_id: str) -> str:
    """Delete a Gmail label."""

    if not label_id:
        raise ValueError("label_id cannot be empty")

    def _delete_label():
        service = client.build_service("gmail", "v1")
        request = service.users().labels().delete(userId="me", id=label_id)
        return request.execute()

    result = await run_blocking(_delete_label)
    return json_dumps(result)


@mcp.tool()
async def gmail_list_messages(
    query: str = "",
    label_ids: list[str] | None = None,
    max_results: int = 100,
    include_spam_trash: bool = False,
) -> str:
    """List Gmail messages matching a query or labels."""

    def _list_messages():
        service = client.build_service("gmail", "v1")
        request = service.users().messages().list(
            userId="me",
            q=query or None,
            labelIds=label_ids or None,
            maxResults=max_results,
            includeSpamTrash=include_spam_trash,
        )
        return request.execute()

    result = await run_blocking(_list_messages)
    return json_dumps(result)


@mcp.tool()
async def gmail_get_message(message_id: str, format: str = "full") -> str:
    """Get a Gmail message by ID."""

    if not message_id:
        raise ValueError("message_id cannot be empty")

    def _get_message():
        service = client.build_service("gmail", "v1")
        request = service.users().messages().get(
            userId="me",
            id=message_id,
            format=format,
        )
        return request.execute()

    result = await run_blocking(_get_message)
    return json_dumps(result)


@mcp.tool()
async def gmail_list_threads(
    query: str = "",
    label_ids: list[str] | None = None,
    max_results: int = 50,
) -> str:
    """List Gmail threads."""

    def _list_threads():
        service = client.build_service("gmail", "v1")
        request = service.users().threads().list(
            userId="me",
            q=query or None,
            labelIds=label_ids or None,
            maxResults=max_results,
        )
        return request.execute()

    result = await run_blocking(_list_threads)
    return json_dumps(result)


@mcp.tool()
async def gmail_get_thread(thread_id: str, format: str = "full") -> str:
    """Get a Gmail thread by ID."""

    if not thread_id:
        raise ValueError("thread_id cannot be empty")

    def _get_thread():
        service = client.build_service("gmail", "v1")
        request = service.users().threads().get(
            userId="me",
            id=thread_id,
            format=format,
        )
        return request.execute()

    result = await run_blocking(_get_thread)
    return json_dumps(result)


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

    if not to:
        raise ValueError("to cannot be empty")
    if not subject:
        raise ValueError("subject cannot be empty")
    if body is None:
        raise ValueError("body cannot be empty")

    def _send():
        service = client.build_service("gmail", "v1")
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
        return request.execute()

    result = await run_blocking(_send)
    return json_dumps(result)


@mcp.tool()
async def gmail_send_raw_message(raw_base64: str, thread_id: str = "") -> str:
    """Send a Gmail message using a base64url-encoded raw MIME message."""

    if not raw_base64:
        raise ValueError("raw_base64 cannot be empty")

    def _send_raw():
        service = client.build_service("gmail", "v1")
        payload: dict[str, Any] = {"raw": raw_base64}
        if thread_id:
            payload["threadId"] = thread_id
        request = service.users().messages().send(userId="me", body=payload)
        return request.execute()

    result = await run_blocking(_send_raw)
    return json_dumps(result)


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

    if not to:
        raise ValueError("to cannot be empty")
    if not subject:
        raise ValueError("subject cannot be empty")
    if body is None:
        raise ValueError("body cannot be empty")

    def _create_draft():
        service = client.build_service("gmail", "v1")
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
        return request.execute()

    result = await run_blocking(_create_draft)
    return json_dumps(result)


@mcp.tool()
async def gmail_send_draft(draft_id: str) -> str:
    """Send an existing Gmail draft."""

    if not draft_id:
        raise ValueError("draft_id cannot be empty")

    def _send_draft():
        service = client.build_service("gmail", "v1")
        request = service.users().drafts().send(userId="me", body={"id": draft_id})
        return request.execute()

    result = await run_blocking(_send_draft)
    return json_dumps(result)


@mcp.tool()
async def gmail_modify_message_labels(
    message_id: str,
    add_label_ids: list[str] | None = None,
    remove_label_ids: list[str] | None = None,
) -> str:
    """Add or remove labels on a Gmail message."""

    if not message_id:
        raise ValueError("message_id cannot be empty")

    def _modify():
        service = client.build_service("gmail", "v1")
        body = {
            "addLabelIds": add_label_ids or [],
            "removeLabelIds": remove_label_ids or [],
        }
        request = service.users().messages().modify(
            userId="me",
            id=message_id,
            body=body,
        )
        return request.execute()

    result = await run_blocking(_modify)
    return json_dumps(result)


@mcp.tool()
async def gmail_trash_message(message_id: str) -> str:
    """Move a Gmail message to trash."""

    if not message_id:
        raise ValueError("message_id cannot be empty")

    def _trash():
        service = client.build_service("gmail", "v1")
        request = service.users().messages().trash(userId="me", id=message_id)
        return request.execute()

    result = await run_blocking(_trash)
    return json_dumps(result)


@mcp.tool()
async def gmail_untrash_message(message_id: str) -> str:
    """Restore a Gmail message from trash."""

    if not message_id:
        raise ValueError("message_id cannot be empty")

    def _untrash():
        service = client.build_service("gmail", "v1")
        request = service.users().messages().untrash(userId="me", id=message_id)
        return request.execute()

    result = await run_blocking(_untrash)
    return json_dumps(result)


@mcp.tool()
async def gmail_delete_message(message_id: str) -> str:
    """Permanently delete a Gmail message."""

    if not message_id:
        raise ValueError("message_id cannot be empty")

    def _delete():
        service = client.build_service("gmail", "v1")
        request = service.users().messages().delete(userId="me", id=message_id)
        return request.execute()

    result = await run_blocking(_delete)
    return json_dumps(result)


@mcp.tool()
async def calendar_list_calendars() -> str:
    """List calendars visible to the authenticated user."""

    def _list_calendars():
        service = client.build_service("calendar", "v3")
        request = service.calendarList().list()
        return request.execute()

    result = await run_blocking(_list_calendars)
    return json_dumps(result)


@mcp.tool()
async def calendar_get_calendar(calendar_id: str) -> str:
    """Get calendar metadata by ID."""

    if not calendar_id:
        raise ValueError("calendar_id cannot be empty")

    def _get_calendar():
        service = client.build_service("calendar", "v3")
        request = service.calendars().get(calendarId=calendar_id)
        return request.execute()

    result = await run_blocking(_get_calendar)
    return json_dumps(result)


@mcp.tool()
async def calendar_create_calendar(summary: str, description: str = "", time_zone: str = "") -> str:
    """Create a new calendar."""

    if not summary:
        raise ValueError("summary cannot be empty")

    def _create_calendar():
        service = client.build_service("calendar", "v3")
        body: dict[str, Any] = {"summary": summary}
        if description:
            body["description"] = description
        if time_zone:
            body["timeZone"] = time_zone
        request = service.calendars().insert(body=body)
        return request.execute()

    result = await run_blocking(_create_calendar)
    return json_dumps(result)


@mcp.tool()
async def calendar_delete_calendar(calendar_id: str) -> str:
    """Delete a calendar."""

    if not calendar_id:
        raise ValueError("calendar_id cannot be empty")

    def _delete_calendar():
        service = client.build_service("calendar", "v3")
        request = service.calendars().delete(calendarId=calendar_id)
        return request.execute()

    result = await run_blocking(_delete_calendar)
    return json_dumps(result)


@mcp.tool()
async def calendar_list_events(
    calendar_id: str = "primary",
    time_min: str = "",
    time_max: str = "",
    query: str = "",
    max_results: int = 100,
    single_events: bool = True,
    order_by: str = "startTime",
) -> str:
    """List events in a calendar."""

    def _list_events():
        service = client.build_service("calendar", "v3")
        request = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min or None,
            timeMax=time_max or None,
            q=query or None,
            maxResults=max_results,
            singleEvents=single_events,
            orderBy=order_by or None,
        )
        return request.execute()

    result = await run_blocking(_list_events)
    return json_dumps(result)


@mcp.tool()
async def calendar_get_event(calendar_id: str, event_id: str) -> str:
    """Get a calendar event by ID."""

    if not calendar_id:
        raise ValueError("calendar_id cannot be empty")
    if not event_id:
        raise ValueError("event_id cannot be empty")

    def _get_event():
        service = client.build_service("calendar", "v3")
        request = service.events().get(calendarId=calendar_id, eventId=event_id)
        return request.execute()

    result = await run_blocking(_get_event)
    return json_dumps(result)


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

    if not calendar_id:
        raise ValueError("calendar_id cannot be empty")
    if not summary:
        raise ValueError("summary cannot be empty")
    if not start_iso or not end_iso:
        raise ValueError("start_iso and end_iso are required")

    def _create_event():
        service = client.build_service("calendar", "v3")
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
        return request.execute()

    result = await run_blocking(_create_event)
    return json_dumps(result)


@mcp.tool()
async def calendar_update_event(
    calendar_id: str,
    event_id: str,
    event_patch: dict[str, Any],
    send_updates: str = "all",
) -> str:
    """Patch a calendar event with a partial update."""

    if not calendar_id:
        raise ValueError("calendar_id cannot be empty")
    if not event_id:
        raise ValueError("event_id cannot be empty")
    if event_patch is None:
        raise ValueError("event_patch cannot be empty")

    def _update_event():
        service = client.build_service("calendar", "v3")
        request = service.events().patch(
            calendarId=calendar_id,
            eventId=event_id,
            body=event_patch,
            sendUpdates=send_updates or None,
        )
        return request.execute()

    result = await run_blocking(_update_event)
    return json_dumps(result)


@mcp.tool()
async def calendar_delete_event(calendar_id: str, event_id: str, send_updates: str = "all") -> str:
    """Delete a calendar event."""

    if not calendar_id:
        raise ValueError("calendar_id cannot be empty")
    if not event_id:
        raise ValueError("event_id cannot be empty")

    def _delete_event():
        service = client.build_service("calendar", "v3")
        request = service.events().delete(
            calendarId=calendar_id,
            eventId=event_id,
            sendUpdates=send_updates or None,
        )
        return request.execute()

    result = await run_blocking(_delete_event)
    return json_dumps(result)


@mcp.tool()
async def calendar_quick_add(calendar_id: str, text: str) -> str:
    """Create an event from a natural language text string."""

    if not calendar_id:
        raise ValueError("calendar_id cannot be empty")
    if not text:
        raise ValueError("text cannot be empty")

    def _quick_add():
        service = client.build_service("calendar", "v3")
        request = service.events().quickAdd(calendarId=calendar_id, text=text)
        return request.execute()

    result = await run_blocking(_quick_add)
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
