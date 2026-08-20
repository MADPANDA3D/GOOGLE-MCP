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
    assert CATALOG_VERSION == "google-2026.07.18.2"
    assert manifest["counts"] == {
        "raw": 151,
        "agentReady": 144,
        "legacy": 5,
        "hidden": 2,
    }
    assert len(manifest["tools"]) == 151
    assert manifest["descriptorHash"] == _canonical_hash(manifest["tools"])
    assert manifest["descriptorHash"] == (
        "2c777ccf9f5528e8a3fcaea8de69535ca8a8aae8f85fa622fa55e7d76ffc76d0"
    )
    assert STANDARD_NAVIGATION_TOOLS <= {item["nativeToolName"] for item in manifest["tools"]}

    identity_projection = [
        {key: descriptor[key] for key in ("nativeToolName", "canonicalName", "aliases")}
        for descriptor in manifest["tools"]
    ]
    compatibility_projection = [
        {
            key: descriptor[key]
            for key in (
                "nativeToolName",
                "aliases",
                "inputSchema",
                "outputSchema",
                "annotations",
                "confirmation",
            )
        }
        for descriptor in manifest["tools"]
    ]
    assert _canonical_hash(identity_projection) == (
        "156235e3f91fa345ae4e11308e20bddcd209822cc2cc1740e120dd6788cf52b6"
    )
    assert _canonical_hash(compatibility_projection) == (
        "9f12a0b7bdc2df0b01ee1ecf6f8b3ff178b6b6bf56ad5ddab7f90be821b5b505"
    )

    identities = set()
    risks = {"read": 0, "write": 0, "destructive": 0}
    confirmed_tools = set()
    for descriptor in manifest["tools"]:
        assert descriptor["serviceId"] == "google"
        assert descriptor["canonicalName"] == (f"google.{descriptor['nativeToolName']}")
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
            risks["destructive"] += 1
        elif descriptor["annotations"]["readOnlyHint"]:
            risks["read"] += 1
        else:
            risks["write"] += 1

        confirmation_required = "confirm" in descriptor["inputSchema"].get("properties", {})
        if confirmation_required:
            confirmed_tools.add(descriptor["nativeToolName"])
            assert descriptor["confirmation"]["required"] is True
            assert descriptor["confirmation"]["parameter"] == "confirm"
            assert descriptor["confirmation"]["exactPhrase"].startswith("CONFIRM GOOGLE ")
            assert "confirm=true" in descriptor["confirmation"]["when"]
            assert "portal.call_destructive_tool" in descriptor["confirmation"]["when"]
        else:
            assert descriptor["confirmation"]["required"] is False
            assert descriptor["confirmation"]["parameter"] is None

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

    assert risks == {"read": 84, "write": 18, "destructive": 49}
    assert len(confirmed_tools) == 55
    assert {
        descriptor["nativeToolName"]
        for descriptor in manifest["tools"]
        if descriptor["confirmation"]["required"]
        and not descriptor["annotations"]["destructiveHint"]
    } == {
        "calendar_update_calendar",
        "calendar_update_calendar_list_entry",
        "drive_copy_file",
        "drive_create_comment",
        "gmail_modify_thread_labels",
        "gmail_untrash_message",
    }


def test_gmail_update_draft_is_destructive_and_fails_closed_without_confirmation():
    descriptor = next(
        item
        for item in gm._current_tool_manifest()["tools"]
        if item["nativeToolName"] == "gmail_update_draft"
    )
    assert descriptor["annotations"]["destructiveHint"] is True
    assert descriptor["confirmation"]["required"] is True

    result = json.loads(
        asyncio.run(
            gm.gmail_update_draft(
                draft_id="synthetic-draft",
                to="nobody@example.invalid",
                subject="Synthetic subject",
                body="Synthetic body",
            )
        )
    )
    assert result["ok"] is False
    assert "confirm=true" in result["error"]["message"]


def test_manifest_preserves_legacy_and_advanced_contract_tiers():
    manifest = gm._current_tool_manifest()
    by_name = {item["nativeToolName"]: item for item in manifest["tools"]}

    assert by_name["google_raw_request"]["tier"] == "hidden"
    assert by_name["gmail_signature_preflight"]["tier"] == "hidden"
    assert by_name["gmail_signature_preflight"]["annotations"] == {
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
        "idempotentHint": True,
    }
    assert "configured runtime mode" in by_name["find_tools"]["description"]
    assert "Portal grant" not in by_name["find_tools"]["description"]
    for tool_name in gm.BILLABLE_MAPS_TOOLS:
        descriptor = by_name[tool_name]
        assert "billable, confirmed Google Maps read" in descriptor["description"]
        assert "does not mutate or delete Google data" in descriptor["description"]
        assert "overwrite, send, move, remove" not in descriptor["description"]
    assert by_name["drive_purge_trash"]["tier"] == "legacy"
    assert by_name["drive_purge_trash"]["deprecation"]["replacement"] == ("drive_empty_trash")
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
    result = json.loads(asyncio.run(gm.get_tool_usage("google.drive_upload_file")))
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
