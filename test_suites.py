"""
test_suites.py -- pytest entry point for the check suites in this repository.

WHY THIS FILE EXISTS

The suites in test_*.py are script-style. They run their assertions at module
level through a check(label, got, want, tol) helper, collect results into PASS
and FAIL lists, print a per-suite tally, and raise SystemExit(1) if anything
failed. That format predates pytest here and is kept on purpose: the assertions
are the valuable part of this repository, and rewriting thousands of lines of
them into pytest functions would risk silently changing what they verify.

This module lets a bare `pytest` run every one of them without altering a single
assertion. Each suite runs in its own subprocess and its exit code decides the
result. A failing suite prints its captured output so the specific failing check
is visible in the pytest report rather than hidden behind an exit code.

Each suite keeps its own databases in a temp directory, so nothing here touches
data/registry.db or data/market.db.

    pytest                          run every suite
    pytest -k registry              run one suite
    python engine/test_registry.py  run a suite directly, with its full output
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
_SELF = Path(__file__).resolve()
_SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", ".pytest_cache", "node_modules"}

# Each suite is a separate process, and some do real numerical work, so allow
# generous headroom rather than turning a slow machine into a false failure.
_TIMEOUT_SECONDS = 900


def _discover() -> list[str]:
    """Every check suite in the tree, found by name so the list cannot drift."""
    found: list[str] = []
    for path in ROOT.rglob("test_*.py"):
        if path.resolve() == _SELF:
            continue
        if _SKIP_DIRS & set(path.parts):
            continue
        found.append(path.relative_to(ROOT).as_posix())
    return sorted(found)


SUITES = _discover()


def test_suites_were_discovered() -> None:
    """Guard against a silent pass.

    If discovery ever returns nothing, the parametrised test below would
    generate zero cases and the run would report success having verified
    nothing at all. This makes that state a failure instead.
    """
    assert SUITES, f"no test_*.py check suites were found under {ROOT}"


@pytest.mark.parametrize("suite", SUITES, ids=SUITES)
def test_check_suite(suite: str) -> None:
    """Run one check suite and require a clean exit."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / suite)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"{suite} exited with code {proc.returncode}\n\n"
            f"----- stdout -----\n{proc.stdout}\n"
            f"----- stderr -----\n{proc.stderr}",
            pytrace=False,
        )
