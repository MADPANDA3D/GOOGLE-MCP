#!/usr/bin/env python3
"""Fail CI when public source contains known private-boundary artifacts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATH_PARTS = {
    ".whoami",
    "AGENTS.md",
    "HANDOVER.md",
    "private-archives",
    "internal-audits",
    "tickets",
}
FORBIDDEN_PATH_FRAGMENTS = (
    "live-test",
    "live_workspace_report",
    "credential-export",
    "runtime-snapshot",
)
TEXT_RULES = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "Google OAuth access token": re.compile(r"\bya29\.[0-9A-Za-z._-]{20,}\b"),
    "Google OAuth client secret": re.compile(r"\bGOCSPX-[0-9A-Za-z_-]{20,}\b"),
    "private service path": re.compile(r"/(?:home/services|root/google-mcp)(?:/|\b)"),
}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".lock"}


def public_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode == 0 and result.stdout:
        return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and ".venv" not in path.parts
        and "__pycache__" not in path.parts
    ]


def main() -> None:
    violations: list[str] = []
    for path in public_files():
        relative = path.relative_to(ROOT)
        parts = set(relative.parts)
        rendered = relative.as_posix()
        if parts & FORBIDDEN_PATH_PARTS or any(
            fragment in rendered.lower() for fragment in FORBIDDEN_PATH_FRAGMENTS
        ):
            violations.append(f"forbidden public path: {rendered}")
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in TEXT_RULES.items():
            if pattern.search(text):
                violations.append(f"{label} in {rendered}")
    if violations:
        raise SystemExit(
            "Public-source safety gate failed:\n- " + "\n- ".join(sorted(set(violations)))
        )
    print(f"public-source safety gate passed ({len(public_files())} files)")


if __name__ == "__main__":
    main()
