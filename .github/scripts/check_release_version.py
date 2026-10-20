"""Check that release version declarations and an optional Git tag agree."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def match_version(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"could not find a version declaration in {path}")
    return match.group(1)


with (ROOT / "pyproject.toml").open("rb") as stream:
    package_version = str(tomllib.load(stream)["project"]["version"])

declared_versions = {
    "pyproject.toml": package_version,
    "src/mantishrimp/__init__.py": match_version(
        ROOT / "src/mantishrimp/__init__.py",
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
    ),
}

if len(set(declared_versions.values())) != 1:
    raise SystemExit(f"version declarations do not agree: {declared_versions}")

release_tag = os.environ.get("RELEASE_TAG")
if release_tag is not None and release_tag != f"v{package_version}":
    raise SystemExit(
        f"release tag {release_tag!r} does not match package version "
        f"v{package_version}"
    )

print(f"release version is consistent: {package_version}")
