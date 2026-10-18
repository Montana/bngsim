"""The pre-push hook runs the whole suite in parallel (ci/pre_push_pytest.py).

Serially the hook took 44m29s; under pytest-xdist, one file per worker, 4m42s
with the same result. These pin what that rests on: the hook still runs all of
``python/tests`` (nothing was dropped to get the speed), a worktree reuses the
main checkout's test cache (a cold one took 19m15s, waiting on macOS to check
each new ``.dylib``), and a venv without xdist is told so at once instead of
running for 45 minutes behind a ``Passed``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
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

SERIAL = [sys.executable, "-m", "pytest", "-q", "python/tests"]


def _run_main(
    monkeypatch: pytest.MonkeyPatch, *, have_xdist: bool, shared: Path | None = None
) -> tuple[int, list[list[str]], str | None]:
    """Call ``main()``; return its exit code, what it ran, and the cache env it set."""
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "xdist":
            return object() if have_xdist else None
        return real_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    ran: list[list[str]] = []
    seen: list[str | None] = []

    def run(cmd: list[str]) -> int:
        ran.append(cmd)
        seen.append(os.environ.get(pre_push.CACHE_ENV))
        return 7

    monkeypatch.setattr(pre_push.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(pre_push, "run", run)
    monkeypatch.setattr(pre_push, "default_workers", lambda: "12")
    monkeypatch.setattr(pre_push, "shared_cache_dir", lambda: shared)
    # main() writes the cache variable into os.environ for its child; have
    # monkeypatch restore it rather than leak it into the rest of the session.
    monkeypatch.setenv(pre_push.CACHE_ENV, os.environ.get(pre_push.CACHE_ENV, ""))
    code = pre_push.main()
    return code, ran, seen[0] if seen else None


def test_with_xdist_the_whole_suite_runs_one_file_per_worker(monkeypatch):
    monkeypatch.delenv(pre_push.WORKERS_ENV, raising=False)
    code, ran, _ = _run_main(monkeypatch, have_xdist=True)
    assert code == 7, "main() must return the suite's own exit code"
    assert ran == [[*SERIAL, "-n", "12", "--dist", "loadfile"]]


def test_the_worker_count_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(pre_push.WORKERS_ENV, " 4 ")
    expected = [[*SERIAL, "-n", "4", "--dist", "loadfile"]]
    assert _run_main(monkeypatch, have_xdist=True)[1] == expected
    monkeypatch.setenv(pre_push.WORKERS_ENV, "")
    assert _run_main(monkeypatch, have_xdist=True)[1][0][5:7] == ["-n", "12"]


def test_zero_workers_is_the_plain_serial_run_and_needs_no_xdist(monkeypatch):
    monkeypatch.setenv(pre_push.WORKERS_ENV, "0")
    for have_xdist in (True, False):
        code, ran, _ = _run_main(monkeypatch, have_xdist=have_xdist)
        assert (code, ran) == (7, [SERIAL])


def test_without_xdist_the_push_stops_at_once_and_says_how_to_fix_it(monkeypatch, capsys):
    monkeypatch.delenv(pre_push.WORKERS_ENV, raising=False)
    code, ran, _ = _run_main(monkeypatch, have_xdist=False)
    assert code == 1
    assert ran == [], "it must not fall back to a 45-minute run nobody sees"
    err = capsys.readouterr().err
    assert "pytest-xdist is not installed" in err
    assert "uv sync --extra dev" in err
    assert f"{pre_push.WORKERS_ENV}=0" in err


def test_the_suite_runs_on_the_main_checkouts_test_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(pre_push.WORKERS_ENV, raising=False)
    monkeypatch.delenv(pre_push.CACHE_ENV, raising=False)
    shared = tmp_path / "main" / ".pytest_cache" / "d" / "bngsim"
    assert _run_main(monkeypatch, have_xdist=True, shared=shared)[2] == str(shared)


def test_an_explicit_test_cache_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.delenv(pre_push.WORKERS_ENV, raising=False)
    monkeypatch.setenv(pre_push.CACHE_ENV, str(tmp_path / "mine"))
    shared = tmp_path / "main" / ".pytest_cache" / "d" / "bngsim"
    assert _run_main(monkeypatch, have_xdist=True, shared=shared)[2] == str(tmp_path / "mine")


def test_with_no_main_checkout_found_the_cache_is_the_default(monkeypatch):
    monkeypatch.delenv(pre_push.WORKERS_ENV, raising=False)
    monkeypatch.delenv(pre_push.CACHE_ENV, raising=False)
    assert _run_main(monkeypatch, have_xdist=True, shared=None)[2] == ""


def _git(cwd: Path, *args: str) -> None:
    # GIT_DIR and friends would point these at the developer's own repository.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)


def test_a_worktree_finds_the_main_checkouts_cache(tmp_path, monkeypatch):
    main = tmp_path / "main"
    main.mkdir()
    (main / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    _git(main, "init", "-q")
    _git(
        main,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "x",
    )
    _git(main, "worktree", "add", "-q", "--detach", str(tmp_path / "wt" / "bngsim"))
    for name in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(name)
    expected = main.resolve() / ".pytest_cache" / "d" / "bngsim"
    for cwd in (main, tmp_path / "wt" / "bngsim"):
        monkeypatch.chdir(cwd)
        assert pre_push.shared_cache_dir() == expected
    # Not a checkout of anything with a pyproject: no guess.
    (main / "pyproject.toml").unlink()
    assert pre_push.shared_cache_dir() is None


class _Sysctl:
    """A stand-in for ``subprocess.run`` answering ``sysctl -n <name>``."""

    def __init__(self, values: dict[str, str] | None) -> None:
        self.values = values
        self.asked: list[str] = []

    def __call__(self, cmd, **kwargs):
        assert cmd[:2] == ["sysctl", "-n"]
        self.asked.append(cmd[2])
        if self.values is None:
            raise FileNotFoundError(cmd[0])
        out = self.values.get(cmd[2], "")
        return subprocess.CompletedProcess(cmd, 0 if out else 1, stdout=out + "\n", stderr="")


def _levels(*levels: tuple[str, int]) -> dict[str, str]:
    values = {"hw.nperflevels": str(len(levels))}
    for i, (name, count) in enumerate(levels):
        values[f"hw.perflevel{i}.name"] = name
        values[f"hw.perflevel{i}.physicalcpu"] = str(count)
    return values


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # This machine: 16 workers took 5m27s against 12 workers' 4m42s.
        (_levels(("Performance", 12), ("Efficiency", 4)), "12"),
        # A cluster ahead of the performance cores still counts; position is not the test.
        (_levels(("Super", 6), ("Performance", 12), ("Efficiency", 4)), "18"),
        # An Intel Mac reports one "Standard" level: nothing to leave out.
        (_levels(("Standard", 8)), "auto"),
        ({}, "auto"),
        (None, "auto"),
        (_levels(("Performance", 0), ("Efficiency", 4)), "auto"),
    ],
    ids=["p-and-e", "three-levels", "intel", "no-perflevels", "no-sysctl", "zero"],
)
def test_on_a_mac_the_default_leaves_out_the_efficiency_cores(monkeypatch, values, expected):
    sysctl = _Sysctl(values)
    monkeypatch.setattr(pre_push.sys, "platform", "darwin")
    monkeypatch.setattr(pre_push.subprocess, "run", sysctl)
    assert pre_push.default_workers() == expected
    assert sysctl.asked[0] == "hw.nperflevels"


def test_elsewhere_the_default_is_xdist_auto(monkeypatch):
    sysctl = _Sysctl(_levels(("Performance", 12), ("Efficiency", 4)))
    monkeypatch.setattr(pre_push.sys, "platform", "linux")
    monkeypatch.setattr(pre_push.subprocess, "run", sysctl)
    assert pre_push.default_workers() == "auto"
    assert sysctl.asked == []


def test_run_returns_the_childs_exit_code():
    assert pre_push.run([sys.executable, "-c", "raise SystemExit(5)"]) == 5


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
