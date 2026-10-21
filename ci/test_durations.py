#!/usr/bin/env python3
"""Rewrite ci/test_durations.json, the per-file test times the CI shards balance on.

    python3 ci/test_durations.py RUN.xml [RUN.xml ...]

Each argument is a pytest ``--junitxml`` report. A file's duration is the sum of
its tests' times; given several reports (the shards of one leg), the per-file
sums are added, since every file ran in exactly one of them. python-tests.yml
uploads one report per shard in its log artifacts, which is the measurement to
refresh from: it is the hardware the split is for.

The table only steers balance. conftest's shard split deals a file it does not
list by a hash of its path, so a stale or missing table leaves the shards
uneven and never leaves a test out. Stdlib only, like the rest of ci/.
"""

from __future__ import annotations

import collections
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parent / "test_durations.json"


def file_of(classname: str) -> str | None:
    """``python.tests.test_x.TestY`` (or ``test_x``) -> ``python/tests/test_x.py``."""
    module = next((part for part in classname.split(".") if part.startswith("test_")), None)
    return None if module is None else f"python/tests/{module}.py"


def durations(reports: list[Path]) -> dict[str, float]:
    per_file: collections.Counter[str] = collections.Counter()
    for report in reports:
        for case in ET.parse(report).getroot().iter("testcase"):
            path = file_of(case.get("classname", ""))
            if path is not None:
                per_file[path] += float(case.get("time", 0) or 0)
    return {path: round(seconds, 1) for path, seconds in sorted(per_file.items())}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    table = durations([Path(a) for a in argv])
    if not table:
        print("no test cases with a python/tests file in those reports", file=sys.stderr)
        return 1
    OUT.write_text(json.dumps(table, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{OUT}: {len(table)} files, {sum(table.values()):.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
