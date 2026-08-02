"""PA-F1 grep gate — CI test that BLOCKS the build if point_adjust or pa_f1
appears anywhere under the project source or tests.

Per the implementation plan: "PA-F1 banned in code. A pytest-time grep gate
fails the build if point_adjust or pa_f1 appears anywhere under src/ or tests/."
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories to scan (all Python source files)
SCAN_DIRS = [
    PROJECT_ROOT / "domain",
    PROJECT_ROOT / "explore",
    PROJECT_ROOT / "evaluate",
    PROJECT_ROOT / "hypothesize",
    PROJECT_ROOT / "ingest",
    PROJECT_ROOT / "plan",
    PROJECT_ROOT / "twin",
    PROJECT_ROOT / "runstore",
    PROJECT_ROOT / "cli",
    PROJECT_ROOT / "dashboard",
    PROJECT_ROOT / "api",
    PROJECT_ROOT / "knowledge",
    PROJECT_ROOT / "tests",
]

# Patterns that are BANNED
BANNED_PATTERNS = [
    re.compile(r"\bpoint_adjust\b", re.IGNORECASE),
    re.compile(r"\bpa_f1\b", re.IGNORECASE),
    re.compile(r"\bpoint\.adjust\b", re.IGNORECASE),
]

# Files that are exempt (this test file itself)
EXEMPT_FILES = {
    Path(__file__).resolve(),
}


def test_no_pa_f1_anywhere():
    """Grep gate: ensure PA-F1 / point_adjust is banned from the codebase."""
    violations = []

    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for py_file in scan_dir.rglob("*.py"):
            if py_file.resolve() in EXEMPT_FILES:
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            for pattern in BANNED_PATTERNS:
                matches = pattern.findall(content)
                if matches:
                    for match in matches:
                        # Find line number
                        for i, line in enumerate(content.splitlines(), 1):
                            if pattern.search(line):
                                violations.append(
                                    f"  {py_file.relative_to(PROJECT_ROOT)}:{i} — '{match}'"
                                )

    assert not violations, (
        "PA-F1 / point_adjust is BANNED in this codebase.\n"
        "The following files contain banned terms:\n"
        + "\n".join(violations)
    )
