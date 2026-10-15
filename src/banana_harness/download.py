"""Explicit provisioning operations. Never called by imports, doctor, or local task loading."""

from __future__ import annotations

from pathlib import Path

from banana_harness.types import HarnessError
from banana_harness.util import checked, write_json

SOURCES = {
    "deepswe": ("https://github.com/datacurve-ai/deep-swe.git", "tasks", "1.1"),
    "terminal-bench-2.1": ("https://github.com/harbor-framework/terminal-bench-2-1.git", "", "2.1"),
    "swebench-pro": ("https://github.com/scaleapi/SWE-bench_Pro-os.git", "v2/tasks", "2"),
}


def fetch(
    benchmark: str, destination: Path, *, allow_downloads: bool = False, revision: str | None = None
) -> Path:
    if not allow_downloads:
        raise HarnessError("Fetching task definitions requires explicit --allow-downloads.")
    if benchmark not in SOURCES:
        raise HarnessError("This suite loads from Hugging Face; use run --allow-downloads.")
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise HarnessError(f"Fetch destination already exists: {destination}")
    url, relative, version = SOURCES[benchmark]
    checked(["git", "clone", "--depth=1", "--", url, str(destination)], timeout=1800)
    if revision:
        checked(["git", "fetch", "--depth=1", "origin", revision], cwd=destination, timeout=600)
        checked(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=destination)
    commit = checked(["git", "rev-parse", "HEAD"], cwd=destination)
    path = destination / relative
    if not path.is_dir():
        raise HarnessError(
            f"Selected revision does not have the expected task directory: {relative}"
        )
    write_json(
        destination / "banana-source.json",
        {
            "benchmark": benchmark,
            "version": version,
            "url": url,
            "commit": commit,
            "requested_revision": revision,
            "source": str(path),
        },
    )
    return path
