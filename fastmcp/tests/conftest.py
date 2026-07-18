"""Deterministic, provider-free environment for the offline test suite."""

import os

TEST_PORTAL_GRANT_TOKEN = "test-portal-grant-0000000000000000"

os.environ.update(
    {
        "MCP_MODE": "portal",
        "MCP_PORTAL_GRANT_TOKEN": TEST_PORTAL_GRANT_TOKEN,
        "MCP_ACCESS_TOKEN": "",
        "MCP_ALLOWED_HOSTS": "localhost,127.0.0.1,[::1],testserver,google-mcp",
        "MCP_ALLOWED_ORIGINS": "",
        "MCP_ALLOW_REQUEST_OVERRIDES": "true",
        "MCP_REQUIRE_REQUEST_GOOGLE_CLIENT_ID": "true",
        "MCP_REQUIRE_REQUEST_GOOGLE_CLIENT_SECRET": "true",
        "MCP_REQUIRE_REQUEST_GOOGLE_REFRESH_TOKEN": "true",
        "MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK": "true",
        "MCP_BYOK_CLIENT_CACHE_SIZE": "0",
        "MCP_BYOK_CLIENT_CACHE_TTL_SECONDS": "0",
        "MCP_BUILD_SHA": "development",
        "MCP_SERVER_VERSION": "development",
        "MCP_SOURCE_FINGERPRINT": "development",
        "MCP_IMAGE_REFERENCE": "development",
        "GOOGLE_MAPS_API_KEY": "",
    }
)
