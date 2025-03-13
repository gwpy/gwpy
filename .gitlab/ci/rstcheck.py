"""Run rstcheck and report output for codeclimate."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import typing
from pathlib import Path

from rstcheck_core.config import RstcheckConfig
from rstcheck_core.runner import RstcheckMainRunner

if typing.TYPE_CHECKING:
    from typing import Literal

    from rstcheck_core.types import LintError

MESSAGE_RE = re.compile(
    r"\((?P<severity>[A-Z]+)/[0-9]+\)(\s+)(?P<message>.*)",
)
SEVERITY: dict[str, str] = {
    "ERROR": "major",
    "SEVERE": "major",
    "INFO": "info",
    "WARNING": "minor",
}


class CodeQualityLocation(typing.TypedDict):
    """Location parameter in a `CodeQualityItem`."""

    path: str
    lines: dict[str, int]


class CodeQualityItem(typing.TypedDict):
    """Gitlab Code Quality item.

    See <https://docs.gitlab.com/ci/testing/code_quality/#code-quality-report-format>.
    """

    description: str
    check_name: str
    fingerprint: str
    severity: Literal["info", "minor", "major", "critical", "blocker"]
    location: CodeQualityLocation


def rstcheck(
    config_path: Path,
    check_paths: list[str | Path],
    *,
    recursive: bool = True,
) -> RstcheckMainRunner:
    """Create, execute, and return an `rstcheck_core.runner.RstcheckMainRunner`."""
    config = RstcheckConfig(
        config_path=config_path,
        recursive=recursive,
    )
    runner = RstcheckMainRunner(
        check_paths=list(map(Path, check_paths)),
        rstcheck_config=config,
    )
    runner.check()
    return runner


def codeclimate_item(item: LintError) -> CodeQualityItem:
    """Construct a codeclimate item for this rstcheck lint error."""
    fingerprint = hashlib.sha1(
        "".join(map(str, item.values())).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    match = MESSAGE_RE.match(item["message"]).groupdict()
    return {
        "check_name": match["message"],
        "description": match["message"],
        "fingerprint": fingerprint,
        "location": {
            "path": str(item["source_origin"]),
            "lines": {
                "begin": (lineno := int(item["line_number"])),
                "end": lineno,
            },
        },
        "severity": SEVERITY.get(match["severity"], "info"),
    }


def codeclimate(items: list[LintError]) -> list[CodeQualityItem]:
    """Parse a list of errors for codeclimate."""
    return [codeclimate_item(item) for item in items]


def main() -> None:
    """Run `rstcheck` and print a codeclimate JSON report."""
    results = rstcheck(
        Path("pyproject.toml"),
        ["docs"],
    )
    issues = codeclimate(results.errors)
    json.dump(issues, sys.stdout)


if __name__ == "__main__":
    main()
