FROM python:3.12-slim

ARG MCP_BUILD_SHA=unknown
ARG MCP_IMAGE_VERSION=google-2026.07.12.1

LABEL org.opencontainers.image.title="MADPANDA Google MCP" \
      org.opencontainers.image.description="Agent-ready Google Workspace MCP provider" \
      org.opencontainers.image.source="https://github.com/MADPANDA3D/GOOGLE-MCP" \
      org.opencontainers.image.revision="$MCP_BUILD_SHA" \
      org.opencontainers.image.version="$MCP_IMAGE_VERSION"

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip==26.1.2 \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=10001:10001 google_mcp_server.py ./
COPY --chown=10001:10001 google_auth_local.py ./
COPY --chown=10001:10001 tool_manifest.py ./

ENV MCP_HTTP_PORT="8086"
ENV MCP_BIND_ADDRESS="0.0.0.0"
ENV MCP_BUILD_SHA="$MCP_BUILD_SHA"
ENV MCP_SERVER_VERSION="$MCP_BUILD_SHA"
ENV GOOGLE_CREDENTIALS_PATH="/app/.google/credentials.json"
ENV GOOGLE_TOKEN_PATH="/app/.google/token.json"
ENV GOOGLE_SCOPES="https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/documents https://www.googleapis.com/auth/spreadsheets https://www.googleapis.com/auth/presentations https://www.googleapis.com/auth/gmail.modify https://www.googleapis.com/auth/gmail.send https://www.googleapis.com/auth/calendar"
ENV PYTHONDONTWRITEBYTECODE="1"
ENV PYTHONUNBUFFERED="1"

EXPOSE 8086

USER 10001:10001

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8086/health', timeout=3).read()"]

CMD ["python", "google_mcp_server.py"]
