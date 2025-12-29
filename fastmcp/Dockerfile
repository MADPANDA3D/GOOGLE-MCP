FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY google_mcp_server.py ./
COPY google_auth_local.py ./

ENV MCP_HTTP_PORT="8086"
ENV MCP_BIND_ADDRESS="0.0.0.0"
ENV GOOGLE_CREDENTIALS_PATH="/app/.google/credentials.json"
ENV GOOGLE_TOKEN_PATH="/app/.google/token.json"
ENV GOOGLE_SCOPES="https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/documents https://www.googleapis.com/auth/spreadsheets https://www.googleapis.com/auth/presentations"

EXPOSE 8086

CMD ["python", "google_mcp_server.py"]
