# linux/amd64 Python 3.12.13 slim-bookworm pin. Update the digest deliberately.
FROM python:3.14.6-slim-bookworm@sha256:86f975aca15cf04a40b399eebede9aea7c82eae084d1f1a0a6ef6bcaae871a30

ARG BUILD_SHA=development
ARG SOURCE_FINGERPRINT=development

LABEL org.opencontainers.image.title="MADPANDA Google MCP" \
      org.opencontainers.image.description="Dual-mode, request-scoped BYOK Google MCP provider" \
      org.opencontainers.image.revision="${BUILD_SHA}" \
      org.opencontainers.image.source="https://github.com/MADPANDA3D/GOOGLE-MCP" \
      org.opencontainers.image.licenses="MIT" \
      com.madpanda.source-fingerprint="${SOURCE_FINGERPRINT}"

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && test "$(id -u app)" = "10001" \
    && test "$(id -g app)" = "10001"

WORKDIR /app

COPY requirements.lock ./requirements.lock
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      --require-hashes -r requirements.lock \
    && rm -rf /root/.cache

COPY --chown=10001:10001 fastmcp/google_mcp_server.py fastmcp/tool_manifest.py ./fastmcp/
COPY --chown=10001:10001 scripts/runtime_smoke.py ./scripts/runtime_smoke.py
COPY --chown=10001:10001 README.md CHANGELOG.md LICENSE NOTICE ./

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/fastmcp \
    HOME=/tmp \
    MCP_HTTP_PORT=8086 \
    MCP_BIND_ADDRESS=0.0.0.0 \
    MCP_MODE=standalone \
    MCP_BUILD_SHA="${BUILD_SHA}" \
    MCP_SERVER_VERSION="${BUILD_SHA}" \
    MCP_SOURCE_FINGERPRINT="${SOURCE_FINGERPRINT}" \
    MCP_EXPECTED_TOOL_COUNT=151 \
    MCP_PROVIDER_RESPONSE_MAX_BYTES=4194304 \
    MCP_MAX_BATCH_ITEMS=1000 \
    MCP_MAX_PROVIDER_CALLS_PER_TOOL=128 \
    MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK=true \
    MCP_BYOK_CLIENT_CACHE_SIZE=0 \
    MCP_BYOK_CLIENT_CACHE_TTL_SECONDS=0

EXPOSE 8086
USER 10001:10001

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import json,os,urllib.request; p=json.load(urllib.request.urlopen('http://127.0.0.1:8086/health',timeout=3)); raise SystemExit(0 if p.get('status')=='healthy' and p.get('tool_count')==int(os.environ['MCP_EXPECTED_TOOL_COUNT']) else 1)"]

CMD ["python", "-m", "google_mcp_server"]
