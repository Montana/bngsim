#!/usr/bin/env python3
"""Fetch the rr_parity corpus's BioModels-only SBML from bngsim's own mirror.

The 331 ``biomodels_dir`` jobs in ``ode_jobs.json`` read SBML that only BioModels
itself serves (the other 992 come from sys-bio/temp-biomodels; ``fetch_sedml.py``).
BioModels is behind Amazon CloudFront, which refuses every request from a
GitHub-hosted runner with its own "Request blocked" page (issue #807; measured by
``.github/workflows/biomodels-probe.yml``), so the nightly cannot download them
there. BioModels releases its whole dataset under CC0
(https://www.biomodels.org/termsofuse), so bngsim hosts a copy: one archive,
attached to the ``MIRROR_TAG`` release of this repository. It is not a bngsim
release, and its tag cannot trigger one (release.yml publishes ``v*`` tags only).

The archive holds the bytes ``benchmarks/suites/biomodels/fetch.py`` downloads
at their pinned versions, and nothing here trusts it: the archive's sha256 is
pinned below, and every file is checked against its record in that suite's
``biomodels_provenance.json``.

Modes:
    (default)      download the archive, check its sha256, write ``<id>.xml``
                   for every ``biomodels_dir`` job into --dest (gitignored
                   ``data/biomodels_sbml/`` by default), and check each one
                   against the provenance manifest.
    --build OUT    pack the archive from a local corpus (``--sbml-dir``, by
                   default where fetch.py writes it). Deterministic: the same
                   files give the same bytes, so a rebuild can be compared with
                   MIRROR_SHA256.

Usage:
    python fetch_biomodels_mirror.py
    python materialize.py --sbml-dir data/biomodels_sbml
    python fetch_biomodels_mirror.py --build biomodels-sbml-2026-07-03.tar.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

MIRROR_TAG = "data-biomodels-sbml-2026-07-03"
MIRROR_NAME = "biomodels-sbml-2026-07-03"  # the archive's file stem and its top directory
MIRROR_URL = f"https://github.com/lanl/bngsim/releases/download/{MIRROR_TAG}/{MIRROR_NAME}.tar.gz"
MIRROR_SHA256 = "09e4337984bb03737f6120380d8b4cb40b7d23f3206591bbab2bfcd641a1c407"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BIOMODELS_SUITE = REPO / "benchmarks" / "suites" / "biomodels"
PROVENANCE = BIOMODELS_SUITE / "biomodels_provenance.json"
DEFAULT_JOBS = HERE / "ode_jobs.json"
DEFAULT_SBML_DIR = BIOMODELS_SUITE / "data" / "sbml_downloads"
DEFAULT_DEST = HERE / "data" / "biomodels_sbml"

README = """\
BioModels SBML for bngsim's rr_parity corpus (the "biomodels_dir" jobs of
parity_checks/rr_parity/ode_jobs.json), at the versions pinned by
benchmarks/suites/biomodels/biomodels_provenance.json (snapshot 2026-07-03).

These are BioModels' files, unmodified. BioModels dedicates its whole dataset to
the public domain under CC0 1.0: https://www.biomodels.org/termsofuse

Hosted by bngsim because BioModels refuses GitHub-hosted runners (bngsim issue
#807). Built and checked by parity_checks/rr_parity/fetch_biomodels_mirror.py.
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mirrored_ids(jobs_path: Path) -> list[str]:
    """The model ids whose SBML only BioModels serves: the ``biomodels_dir`` jobs."""
    jobs = json.loads(jobs_path.read_text())["jobs"]
    return sorted(
        {j["model_id"] for j in jobs if j.get("params", {}).get("sbml_origin") == "biomodels_dir"}
    )


def pinned_sha256() -> dict[str, str]:
    """model id -> the sha256 the provenance manifest pins for its SBML."""
    return {r["id"]: r["sha256"] for r in json.loads(PROVENANCE.read_text())["records"]}


def build(out: Path, sbml_dir: Path, ids: list[str]) -> int:
    """Pack ``<id>.xml`` for every id into ``out``, refusing any file whose bytes are
    not the pinned ones. Fixed order, times, owners and modes, so it is reproducible."""
    pins = pinned_sha256()
    bad = []
    for mid in ids:
        f = sbml_dir / f"{mid}.xml"
        if not f.is_file() or _sha256(f.read_bytes()) != pins.get(mid):
            bad.append(mid)
    if bad:
        print(
            f"{len(bad)} of {len(ids)} files are missing from {sbml_dir} or not the pinned "
            f"bytes (run benchmarks/suites/biomodels/fetch.py first): {bad[:10]}",
            file=sys.stderr,
        )
        return 1

    def member(name: str, data: bytes) -> tuple[tarfile.TarInfo, io.BytesIO]:
        info = tarfile.TarInfo(f"{MIRROR_NAME}/{name}")
        info.size, info.mtime, info.mode = len(data), 0, 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info, io.BytesIO(data)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        tar.addfile(*member("README.txt", README.encode()))
        for mid in ids:
            tar.addfile(*member(f"{mid}.xml", (sbml_dir / f"{mid}.xml").read_bytes()))
    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(out, "wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz,
    ):
        gz.write(buf.getvalue())
    data = out.read_bytes()
    print(f"{out}: {len(ids)} models, {len(data) / 1e6:.1f} MB, sha256 {_sha256(data)}")
    return 0


def fetch(dest: Path, ids: list[str], url: str, want: str) -> int:
    """Download the archive, check it, and write each model's SBML into ``dest``.
    Returns the number of models missing or not the pinned bytes."""
    with tempfile.TemporaryFile() as tmp:
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as r:
            while chunk := r.read(1 << 20):
                digest.update(chunk)
                tmp.write(chunk)
        if digest.hexdigest() != want:
            print(f"archive sha256 {digest.hexdigest()}, pinned {want}: {url}", file=sys.stderr)
            return len(ids)
        tmp.seek(0)
        pins = pinned_sha256()
        dest.mkdir(parents=True, exist_ok=True)
        wanted = {f"{MIRROR_NAME}/{mid}.xml": mid for mid in ids}
        written = set()
        # Only the expected names are read, and each is written to a path built here,
        # so no member name from the archive becomes a path.
        with tarfile.open(fileobj=tmp, mode="r:gz") as tar:
            for info in tar:
                mid = wanted.get(info.name)
                if mid is None or not info.isfile():
                    continue
                data = tar.extractfile(info).read()
                if _sha256(data) != pins.get(mid):
                    print(f"  DRIFT: {mid} is not the pinned bytes", file=sys.stderr)
                    continue
                (dest / f"{mid}.xml").write_bytes(data)
                written.add(mid)
    missing = sorted(set(ids) - written)
    if missing:
        print(f"  {len(missing)} not in the archive or drifted: {missing[:10]}", file=sys.stderr)
    print(f"mirror: {len(written)}/{len(ids)} models written to {dest}, each the pinned bytes")
    return len(missing)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="write <id>.xml for each mirrored model here",
    )
    ap.add_argument("--build", type=Path, metavar="OUT", help="pack the archive instead")
    ap.add_argument("--sbml-dir", type=Path, default=DEFAULT_SBML_DIR, help="--build's source")
    ap.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    ap.add_argument("--url", default=MIRROR_URL, help="archive to fetch (default: the pin)")
    ap.add_argument("--sha256", default=MIRROR_SHA256, help="its expected sha256")
    args = ap.parse_args()

    ids = mirrored_ids(args.jobs)
    if args.build is not None:
        return build(args.build, args.sbml_dir, ids)
    return 1 if fetch(args.dest, ids, args.url, args.sha256) else 0


if __name__ == "__main__":
    sys.exit(main())
