"""Hermetic tests for fetch_biomodels_mirror.py (issue #807).

The nightly's corpus job trusts this script to put exactly the pinned BioModels
bytes on disk, from an archive bngsim hosts as a release asset. So it must refuse
an archive whose sha256 is not the pinned one and a member whose bytes are not the
provenance manifest's, and its --build must refuse a corpus that is not the pinned
bytes. Everything here is a synthetic three-model corpus, fetched by file:// URL:
no network, no real corpus.
"""

from __future__ import annotations

import hashlib
import json

import fetch_biomodels_mirror as M
import pytest


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Two biomodels_dir jobs (B1, B2) and one temp-biomodels job (T1), with B1 and
    B2 pinned by a provenance manifest; returns (sbml_dir, jobs, ids)."""
    sbml = tmp_path / "sbml_downloads"
    sbml.mkdir()
    records = []
    for mid in ("B1", "B2", "T1"):
        data = f'<?xml version="1.0"?><sbml><model id="{mid}"/></sbml>'.encode()
        (sbml / f"{mid}.xml").write_bytes(data)
        records.append({"id": mid, "sha256": hashlib.sha256(data).hexdigest()})
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"records": records}))
    monkeypatch.setattr(M, "PROVENANCE", prov)
    jobs = tmp_path / "ode_jobs.json"
    origins = {"B1": "biomodels_dir", "B2": "biomodels_dir", "T1": "temp-biomodels"}
    jobs.write_text(
        json.dumps(
            {"jobs": [{"model_id": m, "params": {"sbml_origin": o}} for m, o in origins.items()]}
        )
    )
    return sbml, jobs, M.mirrored_ids(jobs)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_only_biomodels_dir_jobs_are_mirrored(corpus):
    assert corpus[2] == ["B1", "B2"]


def test_build_is_reproducible_and_fetch_writes_the_pinned_bytes(corpus, tmp_path):
    sbml, _, ids = corpus
    a, b = tmp_path / "a" / "m.tar.gz", tmp_path / "b" / "m.tar.gz"
    assert M.build(a, sbml, ids) == 0
    assert M.build(b, sbml, ids) == 0
    assert a.read_bytes() == b.read_bytes()

    dest = tmp_path / "dest"
    assert M.fetch(dest, ids, a.as_uri(), _sha(a)) == 0
    assert sorted(p.name for p in dest.iterdir()) == ["B1.xml", "B2.xml"]
    for mid in ids:
        assert (dest / f"{mid}.xml").read_bytes() == (sbml / f"{mid}.xml").read_bytes()


def test_fetch_refuses_an_archive_that_is_not_the_pinned_one(corpus, tmp_path):
    sbml, _, ids = corpus
    archive = tmp_path / "m.tar.gz"
    M.build(archive, sbml, ids)
    dest = tmp_path / "dest"
    assert M.fetch(dest, ids, archive.as_uri(), "0" * 64) == len(ids)
    assert not dest.exists() or not any(dest.iterdir())


def test_fetch_refuses_a_member_that_is_not_the_pinned_bytes(corpus, tmp_path, monkeypatch):
    sbml, _, ids = corpus
    archive = tmp_path / "m.tar.gz"
    M.build(archive, sbml, ids)
    # The archive is intact, but the manifest now pins other bytes for B1: B1 must
    # be refused, and B2 still written.
    prov = json.loads(M.PROVENANCE.read_text())
    prov["records"][0]["sha256"] = "0" * 64
    M.PROVENANCE.write_text(json.dumps(prov))
    dest = tmp_path / "dest"
    assert M.fetch(dest, ids, archive.as_uri(), _sha(archive)) == 1
    assert sorted(p.name for p in dest.iterdir()) == ["B2.xml"]


def test_build_refuses_a_corpus_that_is_not_the_pinned_bytes(corpus, tmp_path):
    sbml, _, ids = corpus
    (sbml / "B2.xml").write_bytes(b"<sbml>drifted</sbml>")
    out = tmp_path / "m.tar.gz"
    assert M.build(out, sbml, ids) == 1
    assert not out.exists()


def test_a_download_that_never_succeeds_is_refused_not_raised(corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "RETRY_DELAY", 0.0)
    _, _, ids = corpus
    missing = (tmp_path / "no-such-archive.tar.gz").as_uri()
    assert M.fetch(tmp_path / "dest", ids, missing, "0" * 64) == len(ids)
