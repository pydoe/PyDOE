#!/usr/bin/env python3
"""Sync the Python versions declared in pyproject.toml with the versions
upstream still supports.

Reads the active (non-EOL) versions from the ACTIVE_VERSIONS environment
variable -- a JSON array of "3.x" labels as produced by endoflife.date --
and rewrites, in place:

  * project.classifiers, so the "Programming Language :: Python :: X.Y"
    entries drop versions that have gone EOL and pick up newly released
    ones;
  * project.requires-python, raised to the oldest still-supported version
    when the current floor has gone EOL.

The floor is only ever raised, never lowered. A project that deliberately
requires a newer Python than the oldest supported one -- say ">=3.12" while
3.10 is still alive -- keeps its own floor, and only the versions at or
above it are declared.

Nothing is written when the declaration already matches, so the calling
workflow opens a pull request only on a real change.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import NoReturn

import tomlkit


PYPROJECT = Path("pyproject.toml")
GENERIC_CLASSIFIER = "Programming Language :: Python :: 3"
VERSION_CLASSIFIER = re.compile(
    r"^Programming Language :: Python :: (\d+\.\d+)$"
)
FLOOR = re.compile(r">=\s*(\d+\.\d+)")


def key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def emit(name: str, value: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def fail(message: str) -> NoReturn:
    print(f"::error::{message}", file=sys.stderr)
    raise SystemExit(1)


def active_versions() -> list[str]:
    """Return the sorted active 3.x labels from ACTIVE_VERSIONS."""
    raw = os.environ.get("ACTIVE_VERSIONS", "").strip()
    if not raw:
        fail("ACTIVE_VERSIONS is unset")

    try:
        labels = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"ACTIVE_VERSIONS is not valid JSON: {exc}")

    # endoflife.date also carries the 2.x tail on some products; only the
    # 3.x line is meaningful for a project that declares "3 :: Only".
    active = sorted(
        {v for v in labels if re.fullmatch(r"3\.\d+", str(v))}, key=key
    )
    if not active:
        fail("no active Python 3.x versions in ACTIVE_VERSIONS")
    return active


def rebuild_classifiers(
    existing: list[str], positions: list[int], supported: list[str]
) -> list[str]:
    """Return `existing` with the per-version classifiers replaced."""
    wanted = [f"Programming Language :: Python :: {v}" for v in supported]
    if positions:
        # Every version classifier sits at or after the first one, so the
        # slice before it survives removal unshifted.
        anchor = positions[0]
        kept = [c for i, c in enumerate(existing) if i not in set(positions)]
        return kept[:anchor] + wanted + kept[anchor:]

    # No per-version entries yet: seed them after the generic
    # "Programming Language :: Python :: 3" marker when there is one.
    try:
        anchor = existing.index(GENERIC_CLASSIFIER) + 1
    except ValueError:
        anchor = len(existing)
    return existing[:anchor] + wanted + existing[anchor:]


def summarize(
    supported: list[str],
    declared: list[str],
    current_floor: str | None,
    floor: str,
) -> str:
    """Render the pull request body describing the rewrite."""
    added = [v for v in supported if v not in declared]
    dropped = [v for v in declared if v not in supported]

    lines = [
        "The Python versions declared in `pyproject.toml` no longer match the",
        "versions upstream supports, per [endoflife.date](https://endoflife.date/python).",
        "",
    ]
    if added:
        lines.append(f"- **Added:** {', '.join(added)}")
    if dropped:
        lines.append(f"- **Dropped (end-of-life):** {', '.join(dropped)}")
    if current_floor != floor:
        lines.append(
            f"- **`requires-python`:** `>={current_floor}` -> `>={floor}`"
        )
    lines += [
        "",
        f"Declared set is now: {', '.join(supported)}.",
        "",
        "Opened automatically by the `Python Versions Sync` workflow.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    active = active_versions()

    if not PYPROJECT.is_file():
        fail("pyproject.toml not found")

    document = tomlkit.parse(PYPROJECT.read_text(encoding="utf-8"))
    project = document.get("project")
    if project is None:
        fail("pyproject.toml has no [project] table")

    # --- requires-python -------------------------------------------------
    requires = str(project.get("requires-python", "")).strip()
    match = FLOOR.search(requires)
    current_floor = match.group(1) if match else None
    floor = max([f for f in (current_floor, active[0]) if f], key=key)
    supported = [v for v in active if key(v) >= key(floor)]
    if not supported:
        fail(f"requires-python floor {current_floor} excludes every version")

    # --- classifiers -----------------------------------------------------
    classifiers = project.get("classifiers")
    if classifiers is None:
        fail("[project] has no classifiers list")

    existing = [str(item) for item in classifiers]
    positions = [
        i for i, item in enumerate(existing) if VERSION_CLASSIFIER.match(item)
    ]
    declared = [
        VERSION_CLASSIFIER.match(existing[i]).group(1) for i in positions
    ]

    if declared == supported and current_floor == floor:
        joined = ", ".join(supported)
        print(f"Already in sync: {joined} (requires-python >={floor})")
        emit("changed", "false")
        return

    array = tomlkit.array()
    for item in rebuild_classifiers(existing, positions, supported):
        array.append(item)
    array.multiline(multiline=True)
    project["classifiers"] = array

    if current_floor != floor:
        project["requires-python"] = f">={floor}"

    PYPROJECT.write_text(tomlkit.dumps(document), encoding="utf-8")

    body = summarize(supported, declared, current_floor, floor)
    body_file = os.environ.get("PR_BODY_FILE")
    if body_file:
        Path(body_file).write_text(body, encoding="utf-8")

    emit("changed", "true")
    sys.stdout.write(body)


if __name__ == "__main__":
    main()
