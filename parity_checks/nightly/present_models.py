#!/usr/bin/env python3
"""Print the comma-separated model_ids of a BioModels job manifest whose SBML is present.

The nightly workflow can fetch only part of the BioModels corpus (EBI's REST API
refuses GitHub-hosted runners; see nightly-parity.yml), so rr-ode and amici-sens
pass this list to their runner's ``--models`` filter instead of failing on the
first missing file. Job ``model`` paths are relative to ``parity_checks/rr_parity``
for both manifests. Exits 1 if nothing is present.

    python3 present_models.py parity_checks/rr_parity/ode_jobs.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RR_PARITY = Path(__file__).resolve().parent.parent / "rr_parity"


def present(manifest: Path, root: Path = RR_PARITY) -> tuple[list[str], int]:
    jobs = json.loads(Path(manifest).read_text())["jobs"]
    ids = [j["model_id"] for j in jobs if (root / j["model"]).is_file()]
    return sorted(dict.fromkeys(ids)), len(jobs)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    ids, n = present(Path(argv[0]))
    print(f"{len(ids)} of {n} jobs have their SBML present", file=sys.stderr)
    if not ids:
        return 1
    print(",".join(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
