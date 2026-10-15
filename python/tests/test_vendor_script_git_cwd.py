"""A vendor script's clean-destination guard must run git inside a work tree.

`scripts/vendor_*.py` once derived their paths from
``REPO_ROOT = Path(__file__).resolve().parents[2]`` — the directory that
*contains* the bngsim checkout — and took the checkout to be
``REPO_ROOT / "bngsim"``. That is the monorepo layout these scripts were written
for. In a standalone clone the parent is not a work tree at all.

`vendor_nfsim.py:ensure_clean_destination` ran ``git status`` with
``cwd=REPO_ROOT``, so in a standalone checkout the documented step-3 refresh died
with::

    Command '['git', 'status', '--porcelain', '--', 'bngsim/third_party/nfsim']'
    returned non-zero exit status 128

before writing anything. The guard never checked anything; it just aborted, and
the only way past it was ``--force``, which *skips* the check outright. So the
failure mode was not a broken guard but a disabled one — the refresh still ran,
with nothing standing between it and a dirty destination.

Running git from ``BNGSIM_ROOT`` fixed that, but ``BNGSIM_ROOT`` was still
``<parent>/bngsim``: a missing directory in a checkout named anything else (a
worktree such as ``bngsim-wt/<topic>``), and the *other* checkout in a clone
beside one (GH #864). Every script now takes the checkout from its own location,
``parents[1]``. These tests pin both invariants for the whole family.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
VENDOR_NFSIM = SCRIPTS_DIR / "vendor_nfsim.py"

pytestmark = pytest.mark.skipif(
    not VENDOR_NFSIM.exists(),
    reason="scripts/ is not in this checkout (installed package)",
)


def _load(path: Path, name: str) -> types.ModuleType:
    """Import a scripts/ module by path — scripts/ is not a package."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    return mod


def _vendor_nfsim() -> types.ModuleType:
    return _load(VENDOR_NFSIM, "_vendor_nfsim_under_test")


def _vendor_scripts_with_the_guard() -> list[Path]:
    return sorted(
        path
        for path in SCRIPTS_DIR.glob("vendor_*.py")
        if "def ensure_clean_destination" in path.read_text(encoding="utf-8")
    )


class TestGuardRunsGitInsideAWorkTree:
    def test_git_is_invoked_from_a_directory_git_recognizes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _vendor_nfsim()
        calls: list[tuple[list[str], object]] = []

        def fake_run(cmd: list[str], cwd: object = None, text: bool = True) -> object:
            calls.append((cmd, cwd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(mod, "run", fake_run)
        mod.ensure_clean_destination(force=False)

        assert calls, "the guard consulted git not at all"
        cmd, cwd = calls[0]
        assert cmd[:3] == ["git", "status", "--porcelain"], cmd
        assert cwd is not None, "the guard ran git with no explicit cwd"

        # The actual defect: this directory has to be one git will answer from.
        # `rev-parse` is the layout-independent form of that question -- it
        # succeeds for the repo root and for any subdirectory of the work tree,
        # so it holds in the monorepo layout too, and fails for REPO_ROOT in a
        # standalone checkout, which is exactly the bug.
        probe = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert probe.returncode == 0, (
            f"the guard runs git from {cwd}, which is not inside a Git work tree "
            f"({probe.stderr.strip()}). It will abort with exit 128 before "
            "checking anything, and --force is the only way past."
        )

    def test_pathspec_is_relative_to_the_cwd_it_uses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A cwd fix that left the pathspec relative to the *other* root would
        # silently scope the status query to a path that does not exist, and the
        # guard would report clean no matter how dirty the destination was.
        mod = _vendor_nfsim()
        calls: list[tuple[list[str], object]] = []

        def fake_run(cmd: list[str], cwd: object = None, text: bool = True) -> object:
            calls.append((cmd, cwd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(mod, "run", fake_run)
        mod.ensure_clean_destination(force=False)

        cmd, cwd = calls[0]
        pathspec = cmd[-1]
        assert (Path(str(cwd)) / pathspec).resolve() == mod.VENDOR_DIR.resolve(), (
            f"pathspec {pathspec!r} resolved against cwd {cwd} is not the vendor "
            f"directory {mod.VENDOR_DIR}"
        )

    def test_dirty_destination_still_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _vendor_nfsim()

        def fake_run(cmd: list[str], cwd: object = None, text: bool = True) -> object:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=" M third_party/nfsim/src/NFcore/reactionClass.cpp\n", stderr=""
            )

        monkeypatch.setattr(mod, "run", fake_run)
        with pytest.raises(RuntimeError, match="uncommitted changes"):
            mod.ensure_clean_destination(force=False)

    def test_force_short_circuits_without_consulting_git(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _vendor_nfsim()

        def explode(*args: object, **kwargs: object) -> object:
            raise AssertionError("--force must not shell out to git")

        monkeypatch.setattr(mod, "run", explode)
        mod.ensure_clean_destination(force=True)


class TestNoVendorScriptRunsGitOutsideTheCheckout:
    """The family-wide invariant, so this cannot drift back in one script."""

    def test_guard_scripts_do_not_use_repo_root_as_a_git_cwd(self) -> None:
        scripts = _vendor_scripts_with_the_guard()
        assert scripts, "no scripts/vendor_*.py defines ensure_clean_destination"

        offenders = [
            path.name for path in scripts if "cwd=REPO_ROOT" in path.read_text(encoding="utf-8")
        ]
        assert not offenders, (
            f"{', '.join(offenders)} runs a git subprocess with cwd=REPO_ROOT. "
            "REPO_ROOT is the parent of the bngsim checkout and is not a work "
            "tree in a standalone clone; use BNGSIM_ROOT."
        )

    def test_every_guard_script_defines_bngsim_root(self) -> None:
        for path in _vendor_scripts_with_the_guard():
            source = path.read_text(encoding="utf-8")
            assert "BNGSIM_ROOT" in source, (
                f"{path.name} has a clean-destination guard but no BNGSIM_ROOT to run git from"
            )


# ─── GH #864: the checkout is wherever scripts/ is, whatever it is called ────


def _vendor_scripts() -> list[Path]:
    # Evaluated at collection, so it must not raise: without scripts/ the module
    # skip above has to get the chance to apply.
    return sorted(SCRIPTS_DIR.glob("vendor_*.py"))


@pytest.mark.parametrize("script", _vendor_scripts(), ids=lambda p: p.stem)
def test_bngsim_root_is_this_checkout(script: Path) -> None:
    # The checkout as found by its pyproject.toml, not this file's walk-up:
    # run_tests.sh reaches scripts/ through a symlink, which the scripts resolve.
    checkout = bngsim_source_root()
    assert checkout is not None
    mod = _load(script, f"_{script.stem}_root_under_test")
    assert checkout == mod.BNGSIM_ROOT
    assert mod.VENDOR_DIR.parent == checkout / "third_party"
    assert mod.VENDOR_DIR.is_dir()


@pytest.mark.parametrize("script", _vendor_scripts(), ids=lambda p: p.stem)
def test_bngsim_root_does_not_depend_on_the_checkout_name(script: Path, tmp_path: Path) -> None:
    # A worktree named bngsim-wt/<topic>, with a clone called bngsim beside it:
    # <parent>/bngsim is the wrong checkout, and the path arithmetic this
    # replaces picked it. The suite's own checkout is usually named bngsim, so
    # only a copy run from elsewhere can show the difference.
    other = tmp_path / "bngsim"
    (other / "third_party").mkdir(parents=True)
    checkout = tmp_path / "bngsim-wt-864"
    (checkout / "scripts").mkdir(parents=True)
    copy = checkout / "scripts" / script.name
    copy.write_bytes(script.read_bytes())

    mod = _load(copy, f"_{script.stem}_renamed_under_test")
    assert checkout.resolve() == mod.BNGSIM_ROOT
    assert mod.VENDOR_DIR.is_relative_to(checkout.resolve())
    if hasattr(mod, "display_path"):
        # The recorded form must not pick up the checkout's name either; only a
        # checkout not named bngsim can tell.
        assert mod.display_path(mod.VENDOR_DIR) == f"bngsim/third_party/{mod.VENDOR_DIR.name}"


@pytest.mark.parametrize("script", _vendor_scripts(), ids=lambda p: p.stem)
def test_no_vendor_script_names_the_checkout(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    assert "parents[2]" not in source, f"{script.name} reaches above the checkout"
    assert 'ROOT / "bngsim"' not in source, f"{script.name} names the checkout as <parent>/bngsim"


@pytest.mark.parametrize("name", ["vendor_nfsim", "vendor_rulemonkey"])
def test_display_paths_keep_the_bngsim_prefix(name: str) -> None:
    # Messages and VENDOR.json record paths as bngsim/<path in the checkout>
    # (rulemonkey's exprtk_sync.verified_against_bngsim_paths among them), so the
    # metadata a refresh writes does not change with the checkout's name.
    mod = _load(SCRIPTS_DIR / f"{name}.py", f"_{name}_display_under_test")
    assert mod.display_path(mod.VENDOR_DIR) == f"bngsim/third_party/{mod.VENDOR_DIR.name}"


def test_vendor_dir_is_unchanged_by_the_bngsim_root_refactor() -> None:
    # The vendored paths land in this checkout, found by its pyproject.toml
    # rather than by the script's own arithmetic, exactly where they always did.
    checkout = bngsim_source_root()
    assert checkout is not None
    mod = _vendor_nfsim()
    assert checkout / "third_party" / "nfsim" == mod.VENDOR_DIR
    assert checkout / "scripts" / "nfsim_vendor_patches" == mod.PATCHES_DIR
    assert mod.VENDOR_DIR.is_dir()
    assert mod.PATCHES_DIR.is_dir()
