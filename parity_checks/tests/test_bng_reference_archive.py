"""The BNG ODE reference archive (issue #805).

``bng_ode_run.py --write-reference-archive`` records, per job, the network
BNG2.pl generated and the reference's side (run_network, or the native BNG2.pl
protocol for a carry-over model); ``--reference-archive`` replays from that
record with neither tool. An entry that no longer matches its job or its derived
spec is stale and never compared.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_BNG = Path(__file__).resolve().parent.parent / "bng_parity"
for _p in (_BNG.parent, _BNG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load():
    spec = importlib.util.spec_from_file_location("bng_ode_run_t", _BNG / "bng_ode_run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bng_ode_run_t"] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load()

SPEC = {
    "key": "fast/x/m.bngl:ode",
    "model_id": "fast/x/m.bngl",
    "method": "ode",
    "rtol": 1e-8,
    "atol": 1e-8,
    "inject": None,
}
ODE = {"t_start": 0.0, "t_end": 10.0, "n_steps": 10, "rtol": 1e-8, "atol": 1e-8}


def _rn():
    t = np.linspace(0, 10, 11)
    return t, np.column_stack([t, t**2]), ["A()", "B()"]


def test_a_recorded_job_comes_back_as_it_was(tmp_path):
    record = {
        **B.archive_input_identity(SPEC, "begin model\nend model\n"),
        "netgen_sec": 0.1,
        "net_text": "begin parameters\nend parameters\n",
        "spec": B.archive_spec_identity(ODE, False),
        "exception": "",
        "timing": {"x": 1},
    }
    B.archive_save(tmp_path, SPEC, record, _rn())
    got, rn = B.archive_load(tmp_path, SPEC)
    assert got == record
    for a, b in zip(rn[:2], _rn()[:2], strict=False):
        np.testing.assert_array_equal(a, b)
    assert rn[2] == ["A()", "B()"]


def test_a_network_failure_is_recorded_without_a_reference(tmp_path):
    record = {**B.archive_input_identity(SPEC, "x"), "netgen_error": "BNG2.pl: ABORT"}
    B.archive_save(tmp_path, SPEC, record)
    got, rn = B.archive_load(tmp_path, SPEC)
    assert rn is None and got["netgen_error"] == "BNG2.pl: ABORT"


def test_no_entry(tmp_path):
    assert B.archive_load(tmp_path, SPEC) == (None, None)


def test_the_inputs_and_the_spec_are_both_checked():
    recorded = B.archive_input_identity(SPEC, "model text")
    assert B._stale_fields(recorded, B.archive_input_identity(SPEC, "model text")) == []
    assert B._stale_fields(recorded, B.archive_input_identity(SPEC, "edited")) == ["bngl_sha256"]
    assert B._stale_fields(
        recorded, B.archive_input_identity({**SPEC, "rtol": 1e-12}, "model text")
    ) == ["rtol"]
    assert B._stale_fields(
        recorded, B.archive_input_identity({**SPEC, "inject": "simulate(...)"}, "model text")
    ) == ["inject_sha256"]
    spec_id = B.archive_spec_identity(ODE, False)
    assert B._stale_fields(spec_id, B.archive_spec_identity({**ODE, "n_steps": 20}, False)) == [
        "n_steps"
    ]
    assert B._stale_fields(spec_id, B.archive_spec_identity(ODE, True)) == ["mode"]


def _bng_available() -> bool:
    try:
        sys.path.insert(0, str(_BNG.parent))
        from _core import resolve_bng

        return resolve_bng().ok
    except Exception:
        return False


def _ids(names):
    raw = json.loads((_BNG / "jobs.json").read_text())
    jobs = raw if isinstance(raw, list) else raw["jobs"]
    return [
        j["model_id"]
        for j in jobs
        if j["method"] == "ode" and any(j["model_id"].endswith(f"/{n}.bngl") for n in names)
    ]


def _run(tmp_path, models, *args, env=None):
    out = tmp_path / f"r{len(list(tmp_path.glob('r*.json')))}.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_BNG / "bng_ode_run.py"),
            "--models",
            ",".join(models),
            "--workers",
            "2",
            "--out",
            str(out),
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=_BNG,
        timeout=900,
        env=env,
    )
    assert out.exists(), proc.stdout[-2000:] + proc.stderr[-2000:]
    return {r["model_id"]: r for r in json.loads(out.read_text())["results"]}


@pytest.mark.skipif(not _bng_available(), reason="recording needs BNG2.pl and run_network")
def test_a_replay_without_bng_reproduces_the_live_verdicts(tmp_path):
    """akt-signaling runs one segment; sonic-hedgehog-gradient carries state
    across its protocol's segments, so its reference is BNG2.pl's own protocol."""
    models = _ids(["akt-signaling", "sonic-hedgehog-gradient"])
    assert len(models) == 2
    arch = tmp_path / "arch"
    live = _run(tmp_path, models, "--write-reference-archive", str(arch))
    blocked = tmp_path / "no_bng" / "bionetgen"
    blocked.mkdir(parents=True)
    (blocked / "__init__.py").write_text("raise ImportError('blocked for this test')\n")
    env = {k: v for k, v in os.environ.items() if k not in ("BNG2_PL",)}
    env.update(BNGPATH=str(tmp_path / "nowhere"), PYTHONPATH=str(blocked.parent))
    replay = _run(tmp_path, models, "--reference-archive", str(arch), env=env)
    modes = set()
    for m, row in live.items():
        got = replay[m]
        assert (got["outcome"], got["value"], got["comment"]) == (
            row["outcome"],
            row["value"],
            row["comment"],
        ), m
        assert got["timing"]["run_network"]["replayed_from_archive"] is True
        modes.add(got["timing"]["spec"]["mode"])
    assert modes == {"single_segment", "multi_segment"}
    stale = _run(tmp_path, models, "--reference-archive", str(arch), "--rtol", "1e-7", env=env)
    assert {r["outcome"] for r in stale.values()} == {"REFERENCE_FAILED"}
    assert all("archive_stale: the job's rtol changed" in r["exception"] for r in stale.values())


# ─── The worker's replay, with the tools faked (runs where BNG2.pl is absent) ──
#
# A one-species network, A -> 0 at k1 = 1 from 10, so the truth is 10·e^(−t).
# The recording run has `generate_network` faked to write that ``.net`` and
# `run_network_ode` faked to bngsim's own run of it; the replay has both faked
# to raise, so a replay that reaches either fails loudly.

_NET = (
    "# Created by BioNetGen 2.9.3\nbegin parameters\n    1 k1    1  # Constant\n"
    "end parameters\nbegin species\n    1 A() 10\nend species\n"
    "begin reactions\n    1 1 0 k1 #R_decay\nend reactions\n"
    "begin groups\n    1 A_tot 1\nend groups\n"
)
_BNGL = 'simulate({method=>"ode",t_end=>10,n_steps=>50,rtol=>1e-8,atol=>1e-8})\n'


class _Queue:
    def __init__(self):
        self.rows = []

    def put(self, row):
        self.rows.append(row)


def _spec(tmp_path, **changes):
    bngl = tmp_path / "m.bngl"
    if not bngl.exists():
        bngl.write_text(_BNGL)
    return {
        "key": "fast/x/m.bngl:ode",
        "model_id": "fast/x/m.bngl",
        "method": "ode",
        "metric": "max_rel_err",
        "tol": 1e-4,
        "bngl": str(bngl),
        "rtol": 1e-8,
        "atol": 1e-8,
        "cap": 120.0,
        "sub_timeout": 120.0,
        "settings": None,
        "bng2_pl": None,
        "run_network_bin": None,
        "inject": None,
        "codegen": False,
        **changes,
    }


def _worker_row(spec):
    q = _Queue()
    B._worker(spec, q)
    return q.rows[0]


def _record(tmp_path, monkeypatch, scale=1.0, failure=None):
    """A live run with the tools faked; returns the archive directory."""
    arch = tmp_path / "arch"

    def generate_network(
        bngl_text, bng2_pl, workdir, timeout=None, gen_network=None, state_prefix=""
    ):
        out = Path(workdir) / "model.net"
        out.write_text(_NET)
        return out, 0.01, ""

    def run_network_ode(net_path, run_network_bin, t_start, t_end, n_steps, rtol, atol, **kw):
        if failure is not None:
            raise RuntimeError(failure)
        t, v, n, timing = B.bc.bn_ode_net(net_path, t_start, t_end, int(n_steps) + 1, rtol, atol)
        return t, v * scale, n, {**timing, "engine": "faked run_network"}

    monkeypatch.setattr(B.bc, "generate_network", generate_network)
    monkeypatch.setattr(B.bc, "run_network_ode", run_network_ode)
    monkeypatch.setattr(B.versions, "bng_version", lambda bng2_pl=None: "2.9.3-test")
    live = _worker_row(_spec(tmp_path, write_reference_archive=str(arch)))
    return arch, live


def _block_tools(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a BNG tool was called during a replay")

    monkeypatch.setattr(B.bc, "generate_network", boom)
    monkeypatch.setattr(B.bc, "run_network_ode", boom)
    monkeypatch.setattr(B.bc, "native_protocol_oracle", boom)


def test_a_replay_reproduces_the_live_verdict_without_the_tools(tmp_path, monkeypatch):
    arch, live = _record(tmp_path, monkeypatch)
    assert live["status"] == "pass"
    _block_tools(monkeypatch)
    row = _worker_row(_spec(tmp_path, reference_archive=str(arch)))
    assert (row["status"], row["value"], row["comment"]) == (
        live["status"],
        live["value"],
        live["comment"],
    )
    rn = row["timing"]["run_network"]
    assert rn["replayed_from_archive"] is True
    assert rn["reference_version"] == "2.9.3-test"
    assert row["reference_version"] == "2.9.3-test"
    assert row["timing"]["spec"]["mode"] == "single_segment"


def test_a_replay_can_still_fail(tmp_path, monkeypatch):
    """A reference 10 % high is a diff on replay as it would be live."""
    arch, live = _record(tmp_path, monkeypatch, scale=1.1)
    assert live["status"] == "diff"
    _block_tools(monkeypatch)
    row = _worker_row(_spec(tmp_path, reference_archive=str(arch)))
    assert row["status"] == "diff" and row["value"] == pytest.approx(live["value"])


def test_a_recorded_reference_failure_replays_as_one(tmp_path, monkeypatch):
    arch, live = _record(tmp_path, monkeypatch, failure="run_network: ABORT")
    assert live["status"] == "reference_failed"
    _block_tools(monkeypatch)
    row = _worker_row(_spec(tmp_path, reference_archive=str(arch)))
    assert row["status"] == "reference_failed"
    assert "run_network: ABORT" in row["exception"]


def test_a_stale_or_missing_entry_is_never_compared(tmp_path, monkeypatch):
    arch, _live = _record(tmp_path, monkeypatch)
    _block_tools(monkeypatch)
    row = _worker_row(_spec(tmp_path, reference_archive=str(arch), rtol=1e-7))
    assert row["status"] == "reference_failed"
    assert "archive_stale: the job's rtol changed" in row["exception"]
    (tmp_path / "m.bngl").write_text(_BNGL + "# edited\n")
    row = _worker_row(_spec(tmp_path, reference_archive=str(arch)))
    assert "archive_stale: the job's bngl_sha256 changed" in row["exception"]
    row = _worker_row(
        _spec(tmp_path, reference_archive=str(arch), key="other:ode", model_id="other")
    )
    assert "archive_stale: no entry for this job" in row["exception"]


def test_an_unreadable_entry_is_the_archive_s_fault(tmp_path, monkeypatch):
    """A corrupt file used to raise out of the worker, which died, and the row
    was EXCEPTION, blamed on bngsim. It is REFERENCE_FAILED, archive_unreadable."""
    arch, _live = _record(tmp_path, monkeypatch)
    _block_tools(monkeypatch)
    B._archive_file(arch, "fast/x/m.bngl:ode").write_bytes(b"not an npz")
    row = _worker_row(_spec(tmp_path, reference_archive=str(arch)))
    assert row["status"] == "reference_failed"
    assert "archive_unreadable" in row["exception"]
    assert "value" in row and row["value"] is None


def test_two_keys_that_sanitize_alike_get_their_own_files(tmp_path):
    """The file name was the key with every other character made ``_``, so
    ``a/b.bngl:ode`` and ``a_b.bngl:ode`` shared one file and the second
    recording overwrote the first."""
    one, two = B._archive_file(tmp_path, "a/b.bngl:ode"), B._archive_file(tmp_path, "a_b.bngl:ode")
    assert one != two
    assert one.name.startswith("a_b.bngl_ode-") and one.suffix == ".npz"
