from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from typing import Any, Mapping


SCHEMA_VERSION = "1.0.0"
SERVICE_ID = "google"
CATALOG_VERSION = "google-2026.07.12.2"
DOCUMENTATION_URL = "https://github.com/MADPANDA3D/GOOGLE-MCP#readme"

STANDARD_NAVIGATION_TOOLS = frozenset(
    {
        "check_configuration",
        "list_capabilities",
        "get_endpoint_coverage",
        "get_tool_usage",
        "find_tools",
    }
)

NAVIGATION_ROLES = {
    "check_configuration": "configuration",
    "list_capabilities": "catalog",
    "get_endpoint_coverage": "coverage",
    "get_tool_usage": "reference",
    "find_tools": "discovery",
}

LEGACY_TOOLS = frozenset(
    {
        "drive_purge_trash",
        "google_mcp_welcome",
        "google_mcp_list_capabilities",
        "google_mcp_get_endpoint_coverage",
        "google_mcp_get_tool_usage",
    }
)
HIDDEN_TOOLS = frozenset({"google_raw_request"})

ALIASES: dict[str, tuple[str, ...]] = {
    "check_configuration": ("google_configuration_status",),
    "list_capabilities": ("google_capability_manifest",),
    "get_endpoint_coverage": ("google_endpoint_inventory",),
    "get_tool_usage": ("google_tool_reference",),
    "find_tools": ("search_google_tools",),
}

DEPRECATIONS: dict[str, dict[str, Any]] = {
    "drive_purge_trash": {
        "deprecated": True,
        "since": CATALOG_VERSION,
        "replacement": "drive_empty_trash",
        "sunsetAt": None,
        "message": "Compatibility alias retained for existing callers; use drive_empty_trash for new workflows.",
    },
    "google_mcp_welcome": {
        "deprecated": True,
        "since": CATALOG_VERSION,
        "replacement": "list_capabilities",
        "sunsetAt": None,
        "message": "Legacy navigation entry point retained for existing callers; use the standard navigation tools for new agents.",
    },
    "google_mcp_list_capabilities": {
        "deprecated": True,
        "since": CATALOG_VERSION,
        "replacement": "list_capabilities",
        "sunsetAt": None,
        "message": "Legacy prefixed navigation tool retained for compatibility.",
    },
    "google_mcp_get_endpoint_coverage": {
        "deprecated": True,
        "since": CATALOG_VERSION,
        "replacement": "get_endpoint_coverage",
        "sunsetAt": None,
        "message": "Legacy prefixed navigation tool retained for compatibility.",
    },
    "google_mcp_get_tool_usage": {
        "deprecated": True,
        "since": CATALOG_VERSION,
        "replacement": "get_tool_usage",
        "sunsetAt": None,
        "message": "Legacy prefixed navigation tool retained for compatibility.",
    },
}

DEFAULT_DEPRECATION = {
    "deprecated": False,
    "since": None,
    "replacement": None,
    "sunsetAt": None,
    "message": None,
}

GENERIC_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Google MCP normalized operation envelope.",
    "required": ["ok", "data", "error", "meta"],
    "properties": {
        "ok": {
            "type": "boolean",
            "description": "Whether the local or Google provider operation completed successfully.",
        },
        "data": {
            "description": "Bounded local navigation or Google provider result when ok is true.",
            "anyOf": [
                {"type": "object", "additionalProperties": True},
                {"type": "array", "items": {}},
                {"type": "string"},
                {"type": "number"},
                {"type": "boolean"},
                {"type": "null"},
            ],
        },
        "error": {
            "description": "Safe structured error when ok is false.",
            "anyOf": [
                {"type": "object", "additionalProperties": True},
                {"type": "null"},
            ],
        },
        "meta": {
            "type": "object",
            "description": "Safe timing, pagination, cache, request, and truncation metadata.",
            "additionalProperties": True,
        },
    },
    "additionalProperties": False,
}

TOOL_MANIFEST_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Complete provider-owned Google ToolManifest catalog.",
    "required": [
        "schemaVersion",
        "serviceId",
        "catalogVersion",
        "buildSha",
        "descriptorHash",
        "counts",
        "tools",
    ],
    "properties": {
        "schemaVersion": {
            "type": "string",
            "description": "Version of the shared ToolManifest wire contract.",
        },
        "serviceId": {
            "type": "string",
            "description": "Canonical Portal registry service identifier.",
        },
        "catalogVersion": {
            "type": "string",
            "description": "Immutable Google provider catalog version identifier.",
        },
        "buildSha": {
            "type": "string",
            "description": "Source revision used to build the running provider image.",
        },
        "descriptorHash": {
            "type": "string",
            "description": "SHA-256 digest of the canonical ordered descriptors.",
        },
        "counts": {
            "type": "object",
            "description": "Raw, agent-ready, legacy, and hidden descriptor counts.",
            "additionalProperties": {"type": "integer"},
        },
        "tools": {
            "type": "array",
            "description": "Complete descriptors when requested, or an empty array for counts-only reads.",
            "items": {"type": "object", "additionalProperties": True},
        },
    },
    "additionalProperties": False,
}


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_sha() -> str:
    for name in ("MCP_BUILD_SHA", "BUILD_SHA", "GIT_COMMIT_SHA", "SOURCE_VERSION"):
        value = os.getenv(name, "").strip()
        if value and re.fullmatch(r"[A-Za-z0-9._-]{7,128}", value):
            return value
    return "unknown"


def _title(name: str) -> str:
    return " ".join(
        part.capitalize()
        for part in name.replace("-", "_").split("_")
        if part
    )


def category_for(name: str) -> str:
    if name in STANDARD_NAVIGATION_TOOLS or name.startswith("google_mcp_"):
        return "navigation"
    if name == "mcp_health_check":
        return "configuration"
    if name == "google_raw_request":
        return "advanced-broker"
    prefixes = (
        ("business_profile_", "business-profile"),
        ("searchconsole_", "search-console"),
        ("drive_", "drive"),
        ("docs_", "docs"),
        ("sheets_", "sheets"),
        ("slides_", "slides"),
        ("gmail_", "gmail"),
        ("calendar_", "calendar"),
        ("youtube_", "youtube"),
        ("analytics_", "analytics"),
        ("maps_", "maps"),
        ("merchant_", "merchant"),
        ("adsense_", "adsense"),
    )
    for prefix, category in prefixes:
        if name.startswith(prefix):
            return category
    return "google-workspace"


def _annotations(tool: Any) -> dict[str, bool]:
    source = getattr(tool, "annotations", None)
    values = (
        source.model_dump(by_alias=True, exclude_none=True)
        if source is not None
        else {}
    )
    return {
        "readOnlyHint": values.get("readOnlyHint") is True,
        "destructiveHint": values.get("destructiveHint") is True,
        "openWorldHint": values.get("openWorldHint") is True,
        "idempotentHint": values.get("idempotentHint") is True,
    }


def _describe_nested_properties(schema: Any, path: str = "value") -> None:
    if not isinstance(schema, dict):
        return
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child in properties.items():
            if isinstance(child, dict):
                child.setdefault(
                    "description",
                    f"Documented {str(name).replace('_', ' ')} value within {path}.",
                )
                _describe_nested_properties(child, f"{path}.{name}")
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            for index, branch in enumerate(branches):
                _describe_nested_properties(branch, f"{path}.{keyword}[{index}]")
    items = schema.get("items")
    if isinstance(items, dict):
        _describe_nested_properties(items, f"{path}.items")
    for definitions_key in ("$defs", "definitions"):
        definitions = schema.get(definitions_key)
        if isinstance(definitions, dict):
            for name, definition in definitions.items():
                _describe_nested_properties(definition, f"{path}.{definitions_key}.{name}")


def _input_schema(tool: Any) -> dict[str, Any]:
    source = getattr(tool, "parameters", None)
    if not isinstance(source, Mapping):
        raise RuntimeError(f"Tool {getattr(tool, 'name', '')} has no input schema.")
    schema = copy.deepcopy(dict(source))
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("additionalProperties", False)
    _describe_nested_properties(schema, str(getattr(tool, "name", "tool")))
    return schema


def _risk(annotations: dict[str, bool]) -> str:
    if annotations["destructiveHint"]:
        return "destructive"
    if annotations["readOnlyHint"]:
        return "read"
    return "write"


def _complete_description(name: str, tool: Any, annotations: dict[str, bool]) -> str:
    source = " ".join(str(getattr(tool, "description", "") or "").split())
    if not source:
        source = f"Use this Google {category_for(name)} tool for {_title(name).lower()}."
    risk = _risk(annotations)
    if name in STANDARD_NAVIGATION_TOOLS:
        suffix = (
            " This is a local, read-only navigation operation. It requires a valid "
            "MAD MCP Portal grant but does not contact Google or require Google OAuth credentials."
        )
    elif risk == "read":
        suffix = (
            " It reads the configured user's Google data through per-request OAuth, "
            "does not mutate provider state, and returns a normalized bounded envelope."
        )
    elif risk == "destructive":
        suffix = (
            " It can overwrite, send, move, remove, or permanently delete Google data. "
            "Use it only after preview and the manifest-defined confirmation; it requires per-request Google OAuth."
        )
    else:
        suffix = (
            " It creates or updates Google provider state and requires per-request Google OAuth. "
            "The result is returned in a normalized bounded envelope."
        )
    return source.rstrip(".") + "." + suffix


def _confirmation(name: str, annotations: dict[str, bool]) -> dict[str, Any]:
    if not annotations["destructiveHint"]:
        return {
            "required": False,
            "parameter": None,
            "exactPhrase": None,
            "when": None,
        }
    phrase = "CONFIRM GOOGLE " + name.replace("_", " ").upper()
    return {
        "required": True,
        "parameter": None,
        "exactPhrase": phrase,
        "when": (
            "Supply this exact phrase out-of-band to portal.call_destructive_tool "
            "after preview; it is never a provider credential or native argument."
        ),
    }


def _tier(name: str) -> str:
    if name in HIDDEN_TOOLS:
        return "hidden"
    if name in LEGACY_TOOLS:
        return "legacy"
    return "agent_ready"


def build_tool_manifest(registered_tools: Mapping[str, Any]) -> dict[str, Any]:
    names = set(registered_tools)
    missing_navigation = STANDARD_NAVIGATION_TOOLS - names
    if missing_navigation:
        raise RuntimeError(
            "Google ToolManifest drift; missing navigation tools="
            f"{sorted(missing_navigation)}."
        )
    descriptors: list[dict[str, Any]] = []
    for name in sorted(names):
        tool = registered_tools[name]
        annotations = _annotations(tool)
        descriptor: dict[str, Any] = {
            "serviceId": SERVICE_ID,
            "nativeToolName": name,
            "canonicalName": f"{SERVICE_ID}.{name}",
            "aliases": list(ALIASES.get(name, ())),
            "title": _title(name),
            "description": _complete_description(name, tool, annotations),
            "category": category_for(name),
            "deprecation": copy.deepcopy(
                DEPRECATIONS.get(name, DEFAULT_DEPRECATION)
            ),
            "inputSchema": _input_schema(tool),
            "outputSchema": copy.deepcopy(
                TOOL_MANIFEST_OUTPUT_SCHEMA
                if name == "list_capabilities"
                else GENERIC_OUTPUT_SCHEMA
            ),
            "annotations": annotations,
            "confirmation": _confirmation(name, annotations),
            "documentationUrl": DOCUMENTATION_URL,
            "navigationRole": NAVIGATION_ROLES.get(name),
            "catalogVersion": CATALOG_VERSION,
            "tier": _tier(name),
        }
        descriptor["descriptorHash"] = canonical_hash(descriptor)
        descriptors.append(descriptor)
    counts = {
        "raw": len(descriptors),
        "agentReady": sum(item["tier"] == "agent_ready" for item in descriptors),
        "legacy": sum(item["tier"] == "legacy" for item in descriptors),
        "hidden": sum(item["tier"] == "hidden" for item in descriptors),
    }
    return {
        "schemaVersion": SCHEMA_VERSION,
        "serviceId": SERVICE_ID,
        "catalogVersion": CATALOG_VERSION,
        "buildSha": build_sha(),
        "descriptorHash": canonical_hash(descriptors),
        "counts": counts,
        "tools": descriptors,
    }


def resolve_descriptor(manifest: dict[str, Any], identity: str) -> dict[str, Any] | None:
    normalized = str(identity or "").strip().lower()
    for descriptor in manifest.get("tools", []):
        identities = {
            str(descriptor.get("nativeToolName", "")).lower(),
            str(descriptor.get("canonicalName", "")).lower(),
            *(str(alias).lower() for alias in descriptor.get("aliases", [])),
        }
        if normalized in identities:
            return descriptor
    return None


def normalize_search_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def search_manifest(
    manifest: dict[str, Any],
    *,
    query: str,
    category: str = "",
    risk: str = "",
    include_legacy: bool = False,
    limit: int = 8,
) -> list[dict[str, Any]]:
    tokens = normalize_search_tokens(query)
    if not tokens:
        return []
    category_filter = str(category or "").strip().lower()
    risk_filter = str(risk or "").strip().lower()
    matches: list[tuple[int, str, dict[str, Any]]] = []
    for descriptor in manifest.get("tools", []):
        tier = descriptor.get("tier")
        if tier == "hidden" or (tier == "legacy" and not include_legacy):
            continue
        annotations = descriptor.get("annotations", {})
        descriptor_risk = _risk(annotations)
        if category_filter and descriptor.get("category") != category_filter:
            continue
        if risk_filter and descriptor_risk != risk_filter:
            continue
        native_name = str(descriptor.get("nativeToolName", ""))
        normalized_name = " ".join(normalize_search_tokens(native_name))
        alias_text = " ".join(
            " ".join(normalize_search_tokens(alias))
            for alias in descriptor.get("aliases", [])
        )
        title = str(descriptor.get("title", "")).lower()
        category_text = str(descriptor.get("category", "")).lower()
        description = str(descriptor.get("description", "")).lower()
        haystack = " ".join(
            (normalized_name, alias_text, title, category_text, description)
        )
        if not all(token in haystack for token in tokens):
            continue
        score = 100
        normalized_query = " ".join(tokens)
        if normalized_query == normalized_name:
            score += 1_000
        if normalized_query in normalized_name:
            score += 500
        if normalized_query in alias_text:
            score += 450
        score += sum(80 for token in tokens if token in normalized_name)
        score += sum(40 for token in tokens if token in title)
        score += sum(20 for token in tokens if token in category_text)
        result = {
            "toolName": native_name,
            "canonicalName": descriptor.get("canonicalName"),
            "title": descriptor.get("title"),
            "category": descriptor.get("category"),
            "risk": descriptor_risk,
            "tier": tier,
            "summary": descriptor.get("description"),
            "descriptorHash": descriptor.get("descriptorHash"),
        }
        matches.append((score, native_name, result))
    matches.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in matches[: max(1, min(int(limit), 25))]]
