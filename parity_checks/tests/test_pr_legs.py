"""Issue #1387: a pull request runs only the nightly-parity legs its files feed.

Every PR that touched one of the workflow's PR paths ran all of its legs,
about 1.5 hours on ~10 runners. ``parity_checks/nightly/pr_legs.py`` maps the
PR's changed paths to the legs they feed, and the plan job runs those. These
pin the mapping on the issue's cases, measure each runner's imports again so
the table cannot drift from the code, and check that the workflow asks for
what the script answers.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "nightly-parity.yml"

_spec = importlib.util.spec_from_file_location(
    "pr_legs", REPO / "parity_checks/nightly/pr_legs.py"
)
pr_legs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr_legs)

ALL = set(pr_legs.LEGS)
PARITY = ALL - {"sbml"}

# The script each leg's job runs (the workflow's own command lines).
RUNNERS = {
    "sbml": ["benchmarks/suites/sbml_test_suite/run.py", "harness/run_dsmts.py"],
    "rr-ode": ["parity_checks/rr_parity/rr_run.py"],
    "bng-ode": ["parity_checks/bng_parity/bng_ode_run.py"],
    "rr-ssa": ["parity_checks/rr_parity/ssa_screen.py"],
    "fd-sens": ["parity_checks/fd_sens/fd_sens_run.py"],
    "differential": ["parity_checks/differential/differential_run.py"],
    "bng-stoch": ["parity_checks/bng_parity/bng_stoch_run.py"],
    "amici-sens": ["parity_checks/amici_parity/amici_sens_run.py"],
}


def _run(paths: list[str], *args: str) -> dict[str, bool]:
    done = subprocess.run(
        [sys.executable, str(REPO / "parity_checks/nightly/pr_legs.py"), *args],
        input="\n".join(paths),
        capture_output=True,
        text=True,
        check=True,
    )
    return {k: v == "true" for k, v in (line.split("=") for line in done.stdout.split())}


@pytest.mark.parametrize(
    ("paths", "legs", "corpus"),
    [
        (["parity_checks/differential/_differential.py"], {"differential"}, True),
        (["parity_checks/fd_sens/fd_sens_run.py"], {"fd-sens", "differential"}, True),
        (
            ["parity_checks/amici_parity/amici_sens_run.py"],
            {"amici-sens", "fd-sens", "differential"},
            True,
        ),
        (["parity_checks/bng_parity/bng_stoch_run.py"], {"bng-ode", "bng-stoch"}, False),
        (["parity_checks/_core/work.py"], PARITY, True),
        (["parity_checks/rr_parity/_rr_common.py"], PARITY, True),
        ([".github/workflows/nightly-parity.yml"], ALL, True),
        (["parity_checks/nightly/verdicts.py"], ALL, True),
        (["harness/run_dsmts.py"], {"sbml"}, False),
        (["parity_checks/nightly/baselines/differential.json"], set(), False),
        # Beside a baseline, the code change decides.
        (
            ["parity_checks/nightly/baselines/fd_sens.json", "parity_checks/fd_sens/x.py"],
            {"fd-sens", "differential"},
            True,
        ),
        # A PR's other files run no leg here, as a PR of only those runs none.
        (["changelog.d/1387.changed.md", "python/bngsim/_simulator.py"], set(), False),
        (
            ["parity_checks/differential/x.py", "changelog.d/1.fixed.md", "python/tests/t.py"],
            {"differential"},
            True,
        ),
    ],
    ids=[
        "differential only",
        "fd_sens only",
        "amici_parity only",
        "bng_parity only",
        "_core",
        "rr_parity",
        "the workflow",
        "a nightly script",
        "harness",
        "a baseline only",
        "a baseline beside code",
        "only files outside the nightly",
        "differential beside a changelog and a test",
    ],
)
def test_a_pull_requests_paths_run_the_legs_they_feed(paths, legs, corpus):
    out = _run(paths)
    assert {leg for leg in pr_legs.LEGS if out[leg]} == legs
    assert out["corpus"] is corpus


def test_a_scheduled_or_dispatched_run_runs_every_leg():
    out = _run([], "--all")
    assert all(out[leg] for leg in pr_legs.LEGS) and out["corpus"]


def _imported_dirs(script: str) -> set[str]:
    """The repo directories ``script`` loads modules from, loaded as its job
    runs it (its own directory first on sys.path), without running main()."""
    probe = textwrap.dedent(f"""
        import importlib.util, json, pathlib, sys
        repo = pathlib.Path({str(REPO)!r})
        script = repo / {script!r}
        sys.path.insert(0, str(script.parent))
        spec = importlib.util.spec_from_file_location("runner_under_test", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        files = [script] + [getattr(m, "__file__", None) for m in list(sys.modules.values())]
        dirs = set()
        for f in files:
            if not f:
                continue
            try:
                rel = pathlib.Path(f).resolve().relative_to(repo)
            except ValueError:
                continue
            if rel.parts[0] != "python":
                dirs.add(str(rel.parent))
        print(json.dumps(sorted(dirs)))
    """)
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO, timeout=300
    )
    if done.returncode != 0:
        pytest.skip(f"{script} does not import here: {done.stderr.strip().splitlines()[-1:]}")
    return set(json.loads(done.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize(
    ("leg", "script"), [(leg, s) for leg, scripts in RUNNERS.items() for s in scripts]
)
def test_every_directory_a_runner_imports_feeds_its_leg(leg, script):
    """The table is measured, not guessed: a runner that comes to import
    another directory fails here until the table says that directory feeds
    its leg."""
    for d in _imported_dirs(script):
        if d == "benchmarks":  # one module: the rule names the file
            continue
        assert leg in pr_legs.legs_for([f"{d}/any.py"]), (script, d)


def _workflow_triggers() -> list[str]:
    text = WORKFLOW.read_text()
    block = text.split("  pull_request:\n    paths:\n", 1)[1].split("\n\n", 1)[0]
    return re.findall(r'^\s+- "([^!"][^"]*)"', block, flags=re.M)


def test_the_script_reads_the_paths_the_workflow_is_triggered_on():
    """TRIGGERS is the workflow's pull_request.paths: a trigger the script did
    not read would run its PR with no leg."""
    as_prefixes = [p.removesuffix("**") for p in _workflow_triggers()]
    assert as_prefixes and sorted(as_prefixes) == sorted(pr_legs.TRIGGERS)


def test_an_unruled_trigger_runs_every_leg(monkeypatch):
    """When unsure, run the leg."""
    monkeypatch.setattr(pr_legs, "TRIGGERS", (*pr_legs.TRIGGERS, "parity_checks/copasi_parity/"))
    assert pr_legs.legs_for(["parity_checks/copasi_parity/copasi_run.py"]) == ALL


def test_every_path_the_workflow_triggers_on_has_a_rule():
    """A trigger the table does not name would run every leg, unmeasured."""
    triggers = _workflow_triggers()
    assert triggers
    for pattern in triggers:
        path = pattern.removesuffix("/**") + ("/any.py" if pattern.endswith("/**") else "")
        named = [prefix for prefix, _ in pr_legs.RULES if path.startswith(prefix)]
        assert named, pattern


def test_the_workflow_runs_each_leg_on_its_flag():
    """Each leg's job is gated on the plan job's output of the same name, and
    the plan job outputs every one the script prints."""
    text = WORKFLOW.read_text()
    for name in [*pr_legs.LEGS, "corpus"]:
        assert f"      {name}: ${{{{ steps.legs.outputs.{name} }}}}" in text, name
        assert f"    if: needs.plan.outputs.{name} == 'true'" in text, name
    assert "python3 parity_checks/nightly/pr_legs.py --all" in text
