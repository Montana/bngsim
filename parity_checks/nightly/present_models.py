#!/usr/bin/env python3
"""Print the comma-separated model_ids of a BioModels job manifest whose SBML is present.

The nightly workflow can fetch only part of the BioModels corpus (EBI's REST API
refuses GitHub-hosted runners; see nightly-parity.yml), so rr-ode and amici-sens
pass this list to their runner's ``--models`` filter instead of failing on the
first missing file. Job ``model`` paths are relative to ``parity_checks/rr_parity``
for both manifests.

``--min-bytes`` / ``--max-bytes`` split the list by SBML file size, so the few
giant models can run in a pass of their own (four of them at once, each compiling
~20 MB of generated C, exhausted a 16 GB runner). ``--skip`` drops named models.

The list can be empty (an empty line); the count goes to stderr.

    python3 present_models.py parity_checks/rr_parity/ode_jobs.json --max-bytes 1000000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RR_PARITY = Path(__file__).resolve().parent.parent / "rr_parity"


def present(
    manifest: Path,
    root: Path = RR_PARITY,
    *,
    min_bytes: int = 0,
    max_bytes: int = 0,
    skip: frozenset[str] = frozenset(),
) -> tuple[list[str], int]:
    """(sorted present model_ids passing the filters, number of jobs in the manifest)."""
    jobs = json.loads(Path(manifest).read_text())["jobs"]
    ids = []
    for j in jobs:
        path = root / j["model"]
        if j["model_id"] in skip or not path.is_file():
            continue
        size = path.stat().st_size
        if size < min_bytes or (max_bytes and size > max_bytes):
            continue
        ids.append(j["model_id"])
    return sorted(dict.fromkeys(ids)), len(jobs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("manifest")
    ap.add_argument("--min-bytes", type=int, default=0, help="only SBML files at least this large")
    ap.add_argument("--max-bytes", type=int, default=0, help="only SBML files at most this large")
    ap.add_argument("--skip", default="", help="comma-separated model_ids to leave out")
    args = ap.parse_args(argv)
    skip = frozenset(s.strip() for s in args.skip.split(",") if s.strip())
    ids, n = present(
        Path(args.manifest), min_bytes=args.min_bytes, max_bytes=args.max_bytes, skip=skip
    )
    print(
        f"{len(ids)} of {n} jobs selected (SBML present, size and skip filters applied)",
        file=sys.stderr,
    )
    print(",".join(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
