"""The pre-push hook runs the whole suite on every core (ci/pre_push_pytest.py).

Serially the hook took 44m29s; under pytest-xdist, one file per worker, 4m42s
with the same result. These pin the two halves of that: the hook still runs all
of ``python/tests`` (nothing was dropped to get the speed), and a venv without
xdist still gets a run rather than pytest's bare "unrecognized arguments: -n".
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402

REPO_ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "ci" / "pre_push_pytest.py"

if not SCRIPT.is_file():  # an installed wheel carries no ci/ directory
    pytest.skip("ci/pre_push_pytest.py lives in the source root", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("bngsim_ci_pre_push_pytest", SCRIPT)
assert _spec and _spec.loader
pre_push = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pre_push)


def _run_main(monkeypatch: pytest.MonkeyPatch, *, have_xdist: bool) -> list[str]:
    """Call ``main()`` with xdist present or absent; return the command it ran."""
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "xdist":
            return object() if have_xdist else None
        return real_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    ran: list[list[str]] = []
    monkeypatch.setattr(pre_push.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(pre_push.subprocess, "call", lambda cmd: ran.append(cmd) or 7)
    assert pre_push.main() == 7, "main() must return the suite's own exit code"
    assert len(ran) == 1
    return ran[0]


def test_with_xdist_the_whole_suite_runs_one_file_per_worker(monkeypatch):
    monkeypatch.delenv(pre_push.WORKERS_ENV, raising=False)
    monkeypatch.setattr(pre_push, "default_workers", lambda: "12")
    cmd = _run_main(monkeypatch, have_xdist=True)
    assert cmd[:5] == [sys.executable, "-m", "pytest", "-q", "python/tests"]
    assert cmd[5:] == ["-n", "12", "--dist", "loadfile"]


def test_the_worker_count_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(pre_push.WORKERS_ENV, " 0 ")
    assert _run_main(monkeypatch, have_xdist=True)[5:] == ["-n", "0", "--dist", "loadfile"]
    monkeypatch.setattr(pre_push, "default_workers", lambda: "12")
    monkeypatch.setenv(pre_push.WORKERS_ENV, "")
    assert _run_main(monkeypatch, have_xdist=True)[5:7] == ["-n", "12"]


class _Sysctl:
    """A stand-in for ``subprocess.run`` that answers the perflevel0 query."""

    def __init__(self, stdout: str | None) -> None:
        self.stdout = stdout
        self.asked: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.asked.append(cmd)
        if self.stdout is None:
            raise FileNotFoundError(cmd[0])
        return pre_push.subprocess.CompletedProcess(cmd, 0, stdout=self.stdout, stderr="")


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [("12\n", "12"), ("", "auto"), ("0\n", "auto"), (None, "auto")],
    ids=["apple-silicon", "intel-mac", "zero", "no-sysctl"],
)
def test_on_a_mac_the_default_is_one_worker_per_performance_core(monkeypatch, stdout, expected):
    """16 workers on a 12+4-core Mac took 5m27s against 12 workers' 4m42s."""
    sysctl = _Sysctl(stdout)
    monkeypatch.setattr(pre_push.sys, "platform", "darwin")
    monkeypatch.setattr(pre_push.subprocess, "run", sysctl)
    assert pre_push.default_workers() == expected
    assert sysctl.asked == [["sysctl", "-n", "hw.perflevel0.physicalcpu"]]


def test_elsewhere_the_default_is_xdist_auto(monkeypatch):
    sysctl = _Sysctl("12\n")
    monkeypatch.setattr(pre_push.sys, "platform", "linux")
    monkeypatch.setattr(pre_push.subprocess, "run", sysctl)
    assert pre_push.default_workers() == "auto"
    assert sysctl.asked == []


def test_without_xdist_the_suite_still_runs_serially_and_says_how_to_fix_it(monkeypatch, capsys):
    monkeypatch.setenv(pre_push.WORKERS_ENV, "4")
    cmd = _run_main(monkeypatch, have_xdist=False)
    assert cmd == [sys.executable, "-m", "pytest", "-q", "python/tests"]
    err = capsys.readouterr().err
    assert "pytest-xdist is not installed" in err
    assert "uv sync --extra dev" in err


def test_the_push_hook_is_this_script():
    """The hook's entry is the script, and it stays a push-only hook."""
    config = REPO_ROOT / ".pre-commit-config.yaml"
    if not config.is_file():
        pytest.skip(".pre-commit-config.yaml is not in this checkout")
    hook = config.read_text(encoding="utf-8").split("- id: pytest\n")[1].split("- id: ")[0]
    assert "entry: uv run --no-sync python ci/pre_push_pytest.py" in hook
    assert "stages: [pre-push]" in hook


def test_the_test_extra_brings_xdist_so_a_synced_venv_runs_in_parallel():
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    test_extra = pyproject["project"]["optional-dependencies"]["test"]
    assert any(req.startswith("pytest-xdist") for req in test_extra)
