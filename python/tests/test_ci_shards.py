"""python-tests.yml splits each leg's suite across jobs (conftest's shard split).

What has to hold: every test pytest collects runs in exactly one shard, so a
green set of shard jobs says what one unsharded job said. The split is by file,
balanced on ci/test_durations.json, and a file's shard depends on its path and
that table alone; another shard's files are not collected at all.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402

REPO_ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "python-tests.yml"
SCRIPT = REPO_ROOT / "ci" / "test_durations.py"


@pytest.mark.parametrize("value", ["1/1", "1/2", "2/2", "3/4", " 2/3"])
def test_a_shard_is_k_of_n(shards, value):
    k, n = shards.parse(value.strip())
    assert 1 <= k <= n


@pytest.mark.parametrize("value", ["0/2", "3/2", "2", "a/b", "1/0", "1/2/3", ""])
def test_anything_else_is_a_usage_error_naming_the_variable(shards, value):
    with pytest.raises(pytest.UsageError, match=shards.env):
        shards.parse(value)


FILES = [f"python/tests/test_{name}.py" for name in "abcdefghij"]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_every_file_is_in_exactly_one_shard(shards, n):
    durations = {f: float(i + 1) for i, f in enumerate(FILES[:6])}  # four unknown files
    assigned = shards.assign(FILES, durations, n)
    assert set(assigned) == set(FILES)
    assert all(1 <= s <= n for s in assigned.values())


def test_known_files_go_heaviest_first_onto_the_lighter_shard(shards):
    durations = dict(zip(FILES[:6], [10.0, 9.0, 8.0, 3.0, 1.0, 1.0], strict=True))
    assigned = shards.assign(FILES[:6], durations, 2)
    load = {1: 0.0, 2: 0.0}
    for f, s in assigned.items():
        load[s] += durations[f]
    # 10 to shard 1, 9 to shard 2; 8 joins the lighter 9 (17); 3, 1, 1 join 10 (15).
    assert load == {1: 15.0, 2: 17.0}
    assert assigned[FILES[0]] == 1 and assigned[FILES[1]] == 2


def test_an_unknown_file_goes_by_a_hash_of_its_path(shards):
    assigned = shards.assign(FILES, {}, 3)
    assert assigned == {f: zlib.crc32(f.encode()) % 3 + 1 for f in FILES}


def test_a_files_shard_does_not_depend_on_which_others_were_collected(shards):
    """Two jobs of one leg could collect different files (a module skipped in one
    of them). If the dealing ran over what was collected, one difference would
    reshuffle every file after it, and tests would land in both shards or none."""
    durations = dict(zip(FILES[:6], [10.0, 9.0, 8.0, 3.0, 1.0, 1.0], strict=True))
    whole = shards.assign(FILES, durations, 2)
    for subset in (FILES[:3], FILES[2:5], FILES[::2], FILES[6:]):
        assert shards.assign(subset, durations, 2) == {f: whole[f] for f in subset}


def test_the_split_does_not_depend_on_the_order_files_arrive_in(shards):
    durations = {f: float(len(f) % 4) for f in FILES[:7]}
    assert shards.assign(FILES, durations, 2) == shards.assign(FILES[::-1], durations, 2)


def test_the_committed_table_is_per_file_seconds(shards):
    table = json.loads(shards.durations_path.read_text(encoding="utf-8"))
    assert table, "an empty table balances nothing"
    for path, seconds in table.items():
        assert re.fullmatch(r"python/tests/test_\w+\.py", path), path
        assert isinstance(seconds, (int, float)) and seconds >= 0, (path, seconds)
    assert shards.read_durations() == {k: float(v) for k, v in table.items()}


def _collected(shard: str | None) -> list[str]:
    """Node ids pytest collects from python/tests, unsharded or as one shard."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("BNGSIM_TEST_SHARD", "PYTEST_ADDOPTS") and not k.startswith("PYTEST_XDIST")
    }
    if shard is not None:
        env["BNGSIM_TEST_SHARD"] = shard
    proc = subprocess.run(
        # -qq: addopts carries -v, which a single -q only cancels to the tree view.
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-qq",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            "python/tests",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return [line for line in proc.stdout.splitlines() if "::" in line]


def test_the_shards_of_the_real_suite_are_disjoint_and_cover_it():
    """The guarantee itself, on pytest's own collection of this checkout."""
    if not (REPO_ROOT / "python" / "tests").is_dir():
        pytest.skip("python/tests is not in this checkout")
    whole = _collected(None)
    one, two = _collected("1/2"), _collected("2/2")
    assert len(whole) > 1000, "the collection went blind"
    assert set(one).isdisjoint(two)
    assert sorted(one + two) == sorted(whole)
    # Both shards carry real weight, not one file each.
    assert min(len(one), len(two)) > len(whole) // 4


def _config(shards, rootpath: Path, shard, python_files=("test_*.py", "*_test.py")):
    """Just enough of a pytest.Config for the collection filter."""
    stash = pytest.Stash()
    stash[shards.key] = shard
    return SimpleNamespace(stash=stash, rootpath=rootpath, getini=lambda name: list(python_files))


def test_another_shards_test_modules_are_not_collected_and_nothing_else_is_touched(
    shards, tmp_path
):
    dealt = {"python/tests/test_a.py": 1, "python/tests/b_test.py": 2}
    one = _config(shards, tmp_path, (1, 2, dealt))
    two = _config(shards, tmp_path, (2, 2, dealt))
    for name, mine in (("test_a.py", one), ("b_test.py", two)):
        path = tmp_path / "python" / "tests" / name
        assert not shards.in_another_shard(path, mine)
        assert shards.in_another_shard(path, two if mine is one else one)
    for name in ("conftest.py", "__init__.py", "_helper.py", "data.json"):
        path = tmp_path / "python" / "tests" / name
        assert not shards.in_another_shard(path, one) and not shards.in_another_shard(path, two)
    # No shard set: nothing is filtered.
    assert not shards.in_another_shard(
        tmp_path / "python/tests/test_a.py", _config(shards, tmp_path, None)
    )


def test_a_module_outside_the_rootdir_is_in_exactly_one_shard(shards, tmp_path):
    outside = tmp_path / "elsewhere" / "test_far.py"
    root = tmp_path / "root"
    verdicts = [
        shards.in_another_shard(outside, _config(shards, root, (k, 3, {}))) for k in (1, 2, 3)
    ]
    assert verdicts.count(False) == 1, verdicts


def test_no_test_sees_the_shard_variable(shards):
    """conftest takes it out of the environment once collection is done, so a
    test that starts a pytest of its own sees the whole suite. In a CI shard job
    the variable is set for the job, and this is the test that it was taken out."""
    assert shards.env not in os.environ


def test_a_bad_shard_is_a_usage_error_before_any_worker_starts(tmp_path):
    """Read at configure time: under xdist, read later, it surfaced as the
    workers' INTERNALERROR (exit 3) and the message was lost."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_XDIST")}
    env["BNGSIM_TEST_SHARD"] = "3/2"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "python/tests/test_ci_shards.py",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 4, proc.stdout[-1500:] + proc.stderr[-1500:]
    assert "BNGSIM_TEST_SHARD='3/2'" in proc.stderr + proc.stdout


def test_the_workflow_runs_every_shard_the_variable_names():
    """The matrix's shards and the k/n it hands conftest must agree, and the
    job that sums the counts must wait for all of them."""
    if not WORKFLOW.is_file():
        pytest.skip(".github/ is not in this checkout")
    text = WORKFLOW.read_text(encoding="utf-8")
    shards = re.search(r"^\s+shard: \[([0-9, ]+)\]$", text, re.M)
    assert shards, "python-tests.yml has no shard axis"
    values = [int(v) for v in shards.group(1).split(",")]
    n = len(values)
    assert values == list(range(1, n + 1))
    assert f"BNGSIM_TEST_SHARD: ${{{{ matrix.shard }}}}/{n}" in text
    assert re.search(r"needs: \[?python-tests\]?", text), "nothing waits for the shards"
    # The counts job sums one leg's shards by label: a leg it does not list has
    # no guard at all, and a label no leg has fails on a missing log.
    legs = re.findall(r"^\s+label: ([\w.-]+)\s*(?:#.*)?$", text, re.M)
    counted = re.search(r"^\s+label: \[([^\]]+)\]$", text, re.M)
    assert counted, "the counts job lists no legs"
    assert sorted(legs) == sorted(s.strip() for s in counted.group(1).split(","))
    assert f"for shard in {' '.join(map(str, values))}; do" in text


def _durations_script():
    spec = importlib.util.spec_from_file_location("bngsim_ci_test_durations", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_table_is_rebuilt_from_junit_reports(tmp_path, monkeypatch):
    if not SCRIPT.is_file():
        pytest.skip("ci/test_durations.py lives in the source root")
    script = _durations_script()
    for i, cases in enumerate(
        [
            [("python.tests.test_a", "t1", "1.25"), ("python.tests.test_a.TestX", "t2", "0.5")],
            [("test_b", "t3", "2"), ("conftest", "setup", "9")],
        ]
    ):
        body = "".join(f'<testcase classname="{c}" name="{n}" time="{t}"/>' for c, n, t in cases)
        (tmp_path / f"r{i}.xml").write_text(
            f"<testsuites><testsuite>{body}</testsuite></testsuites>"
        )
    out = tmp_path / "table.json"
    monkeypatch.setattr(script, "OUT", out)
    assert script.main([str(tmp_path / "r0.xml"), str(tmp_path / "r1.xml")]) == 0
    assert json.loads(out.read_text()) == {
        "python/tests/test_a.py": 1.8,
        "python/tests/test_b.py": 2.0,
    }
    assert script.main([]) == 2
