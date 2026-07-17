import asyncio
import copy
import hashlib
import json

import google_mcp_server as gm
from tool_manifest import CATALOG_VERSION, STANDARD_NAVIGATION_TOOLS


def _canonical_hash(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_provider_manifest_is_complete_deterministic_and_lossless():
    gm._current_tool_manifest.cache_clear()
    manifest = gm._current_tool_manifest()

    assert manifest["schemaVersion"] == "1.0.0"
    assert manifest["serviceId"] == "google"
    assert manifest["catalogVersion"] == CATALOG_VERSION
    assert manifest["counts"] == {
        "raw": 151,
        "agentReady": 144,
        "legacy": 5,
        "hidden": 2,
    }
    assert len(manifest["tools"]) == 151
    assert manifest["descriptorHash"] == _canonical_hash(manifest["tools"])
    assert STANDARD_NAVIGATION_TOOLS <= {
        item["nativeToolName"] for item in manifest["tools"]
    }

    identities = set()
    for descriptor in manifest["tools"]:
        assert descriptor["serviceId"] == "google"
        assert descriptor["canonicalName"] == (
            f"google.{descriptor['nativeToolName']}"
        )
        assert descriptor["title"]
        assert descriptor["description"]
        assert descriptor["category"]
        assert descriptor["documentationUrl"].startswith("https://")
        assert descriptor["tier"] in {"agent_ready", "legacy", "hidden"}
        assert descriptor["inputSchema"]["type"] == "object"
        assert descriptor["outputSchema"]["type"] == "object"
        assert set(descriptor["annotations"]) == {
            "readOnlyHint",
            "destructiveHint",
            "openWorldHint",
            "idempotentHint",
        }
        if descriptor["annotations"]["destructiveHint"]:
            assert descriptor["confirmation"]["required"] is True
            assert descriptor["confirmation"]["exactPhrase"].startswith(
                "CONFIRM GOOGLE "
            )
        else:
            assert descriptor["confirmation"]["required"] is False

        unhashed = copy.deepcopy(descriptor)
        supplied_hash = unhashed.pop("descriptorHash")
        assert supplied_hash == _canonical_hash(unhashed)

        for identity in (
            descriptor["nativeToolName"],
            descriptor["canonicalName"],
            *descriptor["aliases"],
        ):
            normalized = identity.lower()
            assert normalized not in identities
            identities.add(normalized)


def test_manifest_preserves_legacy_and_advanced_contract_tiers():
    manifest = gm._current_tool_manifest()
    by_name = {item["nativeToolName"]: item for item in manifest["tools"]}

    assert by_name["google_raw_request"]["tier"] == "hidden"
    assert by_name["gmail_signature_preflight"]["tier"] == "hidden"
    assert by_name["gmail_signature_preflight"]["annotations"]["readOnlyHint"] is True
    assert by_name["drive_purge_trash"]["tier"] == "legacy"
    assert by_name["drive_purge_trash"]["deprecation"]["replacement"] == (
        "drive_empty_trash"
    )
    assert by_name["google_mcp_list_capabilities"]["tier"] == "legacy"
    assert by_name["list_capabilities"]["tier"] == "agent_ready"


def test_list_capabilities_returns_counts_or_complete_descriptors():
    counts_only = json.loads(asyncio.run(gm.list_capabilities(False)))
    assert counts_only["counts"]["raw"] == 151
    assert counts_only["tools"] == []

    complete = json.loads(asyncio.run(gm.list_capabilities(True)))
    assert len(complete["tools"]) == 151
    assert complete["descriptorHash"] == _canonical_hash(complete["tools"])


def test_search_is_punctuation_normalized_multi_token_and_ranked():
    result = json.loads(
        asyncio.run(
            gm.find_tools(
                query="Drive: search files!",
                category="drive",
                risk="read",
            )
        )
    )
    assert result["ok"] is True
    names = [item["toolName"] for item in result["data"]["matches"]]
    assert names[0] == "drive_search_files"
    assert "google_raw_request" not in names


def test_tool_reference_resolves_canonical_name_and_is_lossless():
    result = json.loads(
        asyncio.run(gm.get_tool_usage("google.drive_upload_file"))
    )
    assert result["ok"] is True
    descriptor = result["data"]["descriptor"]
    expected = next(
        item
        for item in gm._current_tool_manifest()["tools"]
        if item["nativeToolName"] == "drive_upload_file"
    )
    assert descriptor == expected
    assert descriptor["inputSchema"]["properties"]["upload_mode"]["description"]


def test_health_reports_catalog_build_and_contract_counts():
    from starlette.testclient import TestClient

    async def _unused_app(scope, receive, send):
        raise AssertionError("health must not reach the MCP transport")

    client = TestClient(gm.build_hosted_mcp_http_wrapper(_unused_app))
    try:
        response = client.get("/health")
    finally:
        client.close()
    payload = response.json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["status"] == "healthy"
    assert payload["tool_count"] == 151
    assert payload["raw_tool_count"] == 151
    assert payload["exposed_tool_count"] == 151
    assert payload["agent_ready_tool_count"] == 144
    assert payload["documented_tool_count"] == 151
    assert payload["catalog_version"] == CATALOG_VERSION
    assert len(payload["descriptor_hash"]) == 64
    assert payload["configuration"]["ready"] is True
