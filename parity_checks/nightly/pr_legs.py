#!/usr/bin/env python3
"""Which nightly-parity legs a pull request's changed files feed (issue #1387).

A pull request that touched any of the workflow's PR paths ran every leg, about
1.5 hours on ~10 runners, whatever it changed. The plan job now asks this for
the PR's changed paths and runs a leg only where one of them feeds it. A
scheduled or dispatched run is not asked, and runs every leg.

``RULES`` is taken from what each leg's runner imports, measured by loading it
as its job runs it (``test_pr_legs.py`` measures it again, so the table cannot
drift from the code), plus the data a runner reads from another leg's
directory: ``differential`` runs fd_sens's models.

Only the paths the workflow is triggered on (``TRIGGERS``, the workflow's
``pull_request.paths``) are read. A PR's other files (its changelog fragment,
``python/``, its tests) run no leg here, as a PR with only those runs none.
A triggering path no rule names runs every leg: when unsure, run the leg.

Usage, in the plan job:  ``gh api .../pulls/N/files --jq '.[].filename' |
python3 parity_checks/nightly/pr_legs.py >> "$GITHUB_OUTPUT"``; with ``--all``
(no stdin), every leg.
"""

from __future__ import annotations

import sys

LEGS = (
    "sbml",
    "rr-ode",
    "bng-ode",
    "rr-ssa",
    "fd-sens",
    "differential",
    "bng-stoch",
    "amici-sens",
)
# The legs that read the BioModels corpus, which the corpus job fetches.
CORPUS_LEGS = frozenset({"rr-ode", "rr-ssa", "amici-sens", "differential"})
_PARITY = frozenset(LEGS) - {"sbml"}

# The workflow's pull_request.paths, as path prefixes ("dir/**" -> "dir/");
# test_pr_legs.py keeps the two the same.
TRIGGERS = (
    ".github/workflows/nightly-parity.yml",
    "parity_checks/nightly/",
    "parity_checks/_core/",
    "parity_checks/rr_parity/",
    "parity_checks/bng_parity/",
    "parity_checks/amici_parity/",
    "parity_checks/fd_sens/",
    "parity_checks/differential/",
    "benchmarks/suites/sbml_test_suite/",
    "benchmarks/_effort.py",
    "harness/",
)

# (path prefix, the legs it feeds), most specific first: a path takes the
# first rule whose prefix it starts with.
RULES: tuple[tuple[str, frozenset[str]], ...] = (
    # A verdict file a nightly already produced, copied in to re-baseline. The
    # workflow is not triggered on these alone (pull_request.paths excludes
    # them); beside other changes, they ask for no leg of their own.
    ("parity_checks/nightly/baselines/", frozenset()),
    ("parity_checks/nightly/flaky.json", frozenset()),
    # Shared by every leg: the workflow, compare.sh, verdicts.py,
    # reference_archive.py, present_models.py, and this file.
    (".github/workflows/nightly-parity.yml", frozenset(LEGS)),
    ("parity_checks/nightly/", frozenset(LEGS)),
    # Every parity runner imports _core; every one but sbml imports rr_parity's
    # _rr_common (and reference_archive.py does too).
    ("parity_checks/_core/", _PARITY),
    ("parity_checks/rr_parity/", _PARITY),
    ("parity_checks/bng_parity/", frozenset({"bng-ode", "bng-stoch"})),
    # fd_sens_run and differential_run import _amici_sens's scheduler.
    ("parity_checks/amici_parity/", frozenset({"amici-sens", "fd-sens", "differential"})),
    # differential runs fd_sens_jobs.json's models as well as its own checks.
    ("parity_checks/fd_sens/", frozenset({"fd-sens", "differential"})),
    ("parity_checks/differential/", frozenset({"differential"})),
    ("benchmarks/suites/sbml_test_suite/", frozenset({"sbml"})),
    ("benchmarks/_effort.py", frozenset({"sbml"})),
    ("harness/", frozenset({"sbml"})),
)


def legs_for(paths) -> set[str]:
    """The legs the changed ``paths`` feed. A path outside ``TRIGGERS`` feeds
    none; a triggering path no rule names feeds every leg."""
    out: set[str] = set()
    for path in paths:
        path = path.strip()
        if not path or not path.startswith(TRIGGERS):
            continue
        for prefix, legs in RULES:
            if path == prefix or path.startswith(prefix):
                out |= legs
                break
        else:
            out |= set(LEGS)
    return out


def outputs(legs: set[str]) -> list[str]:
    """``name=true|false`` for every leg, and for the corpus job, as
    ``$GITHUB_OUTPUT`` lines."""
    lines = [f"{leg}={'true' if leg in legs else 'false'}" for leg in LEGS]
    lines.append(f"corpus={'true' if legs & CORPUS_LEGS else 'false'}")
    return lines


def main(argv: list[str]) -> int:
    legs = set(LEGS) if "--all" in argv else legs_for(sys.stdin)
    print("\n".join(outputs(legs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
