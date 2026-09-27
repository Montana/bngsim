"""Every pybind11 binding appears in the committed type stub.

``python/bngsim/_bngsim_core.pyi`` is machine-written by ``pybind11_stubgen``
and committed, because it is the *only* description of the compiled extension
that a type checker or an editor can read — neither can see into a ``.so``. It
is regenerated as a side effect of ``scripts/rebuild_editable.py``, and until
this test nothing checked that it matched the bindings.

It drifted the first time someone forgot: PR #306 added
``SolverOptions.set_crossing_stops`` without regenerating, so main
described an API missing a method it had. Two costs, neither loud:

  * every rebuild dirties the working tree with the regenerated hunk, which
    reads as noise from the build script rather than as a real omission;
  * mypy believes the stub. It reports ``"SolverOptions" has no attribute
    "set_crossing_stops"`` — *correct code, flagged* — for any caller that
    reaches the method through a typed reference.

mypy did not catch that drift itself only because the one caller passes ``opts``
as an **untyped** parameter, so inside the helper it is ``Any`` and no attribute
is checked. That is a thin thread to hang the invariant on: it holds only while
every call site stays untyped, which is the opposite of what anyone wants.

Names, not signatures, on purpose. Regenerating the stub in CI and diffing would
also catch a changed *signature*, but it needs a built extension on the leg and
``pybind11_stubgen`` output moves with the tool version and the platform — the
strictness would be paid for in flakes. The realistic failure is somebody adding
a binding and not regenerating, and a name-level check catches that with a
regex, in milliseconds, on every leg, with no build.

The second failure is the same one in disguise: a binding added to the stub by
hand instead of by regenerating. It carries the right name, so the name check
passes, but it lands wherever it was pasted. #844 put
``net_refuse_parameters_that_read_state`` above ``net_file_structure``, and
every rebuild after that moved it back, a four-line diff with no content. The
generator's order is a rule rather than an accident: classes, then functions,
each by name; inside a class, attributes, then methods, then properties, each
by name. Checking the rule catches a hand placement just as cheaply, and it does
not move with the tool version the way signatures do.
"""

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402

# The source tree is READ here, never imported: the stub is text and the bindings
# are C++. run_tests.sh keeps python/bngsim/ off sys.path, which is what stops it
# shadowing the installed package — it does not, and need not, put the files out
# of reach. Resolving through the rig's env vars rather than a __file__ walk-up
# is what tells the two apart; the walk-up lands in the stand-in, where
# python/ holds tests/ and nothing else, and skipped these tests (issue #594).
_REPO = bngsim_source_root() or Path(__file__).resolve().parents[2]
_BINDINGS = _REPO / "src" / "_bngsim_core.cpp"
_STUB = _REPO / "python" / "bngsim" / "_bngsim_core.pyi"

# Each pybind11 spelling that publishes a *named* Python attribute. `.def(
# py::init<...>())` and the operator overloads carry no name literal and so
# match nothing, which is what we want.
_BINDING_PATTERNS = (
    r'\.def\(\s*"([A-Za-z_]\w*)"',
    r'\.def_static\(\s*"([A-Za-z_]\w*)"',
    r'\.def_readwrite\(\s*"([A-Za-z_]\w*)"',
    r'\.def_readonly\(\s*"([A-Za-z_]\w*)"',
    r'\.def_property\(\s*"([A-Za-z_]\w*)"',
    r'\.def_property_readonly\(\s*"([A-Za-z_]\w*)"',
)

_REBUILD = "uv run --no-sync python scripts/rebuild_editable.py"

# Dunders pybind11 synthesizes or that stubgen renders structurally rather than
# as a plain `def` line.
_EXEMPT = frozenset({"__init__", "__enter__", "__exit__", "__repr__", "__str__"})


def _bound_names() -> set[str]:
    source = _BINDINGS.read_text()
    names: set[str] = set()
    for pattern in _BINDING_PATTERNS:
        names |= set(re.findall(pattern, source))
    return names - _EXEMPT


def _declared_in_stub(name: str, stub: str) -> bool:
    # A method (`def name(`) or an attribute (`name: type`), at any indent.
    return (
        re.search(rf"(?m)^\s*(?:def\s+{re.escape(name)}\b|{re.escape(name)}\s*:)", stub)
        is not None
    )


# Guarded on BOTH files, because this one reads both (issue #590). The guard
# named only the bindings, so a tree carrying src/ but not the stub passed it and
# then died on a FileNotFoundError. That is a real wheel/subtree checkout, which
# is the case these reasons describe; it is no longer what run_tests.sh produces.
@pytest.mark.skipif(
    not _BINDINGS.is_file(),
    reason="src/_bngsim_core.cpp is not in this checkout (installed package)",
)
@pytest.mark.skipif(
    not _STUB.is_file(),
    reason=f"no committed stub at {_STUB}",
)
def test_every_binding_is_declared_in_the_committed_stub():
    """A binding the stub does not declare is a mypy error waiting for a caller.

    Fix by rebuilding: ``uv run --no-sync python scripts/rebuild_editable.py``
    regenerates the
    stub from the freshly built module. Commit the regenerated file with the
    binding that made it necessary.
    """
    stub = _STUB.read_text()
    missing = sorted(n for n in _bound_names() if not _declared_in_stub(n, stub))
    assert not missing, (
        f"{len(missing)} pybind11 binding(s) are missing from {_STUB.name}: "
        f"{missing}. Run `{_REBUILD}` and commit the regenerated stub "
        "alongside the binding."
    )


@pytest.mark.skipif(
    not _BINDINGS.is_file(),
    reason="src/_bngsim_core.cpp is not in this checkout (installed package)",
)
def test_the_scan_actually_finds_the_bindings():
    """Guard the guard: a regex that matched nothing would pass vacuously.

    The count is asserted as a floor rather than a number so adding bindings
    does not fail this, while a refactor that moves the binding block out of
    this file — or a pybind11 spelling change that stops matching — does.
    """
    names = _bound_names()
    assert len(names) > 150, f"only {len(names)} bindings found; the scan has gone blind"
    # Spot-check one binding of each kind that must be found by name.
    for expected in ("run", "set_crossing_stops", "rtol", "n_discontinuity_triggers"):
        assert expected in names, f"{expected!r} not seen by the binding scan"


def _stub_order_problems(stub: str) -> list[str]:
    """Every place ``stub`` departs from the order pybind11-stubgen writes.

    Module level: every class before every function, each group by name. Inside a
    class: a run of annotated attributes, then methods, then properties, each run
    by name and none split. A ``@x.setter`` belongs to its property, and an
    overloaded name repeats, so consecutive repeats count once.
    """
    problems = []
    top = re.findall(r"^(class|def) ([A-Za-z_]\w*)", stub, re.M)
    if [k for k, _ in top] != sorted(k for k, _ in top):
        problems.append("a module-level function comes before a class")
    for kind in ("class", "def"):
        names = [n for k, n in top if k == kind]
        if names != sorted(names):
            problems.append(f"module-level `{kind}` names out of order: {names}")
    for cls, body in re.findall(r"^class (\w+).*?:\n((?:(?:    .*)?\n)*)", stub, re.M):
        runs: list[str] = []
        names: dict[str, list[str]] = {}
        decorator = None
        for line in body.splitlines():
            if line.startswith("    @"):
                decorator = line.strip()
                continue
            if m := re.match(r"    def (\w+)", line):
                name, dec, decorator = m.group(1), decorator, None
                if dec and dec.endswith(".setter"):
                    continue
                group = {"@property": "property", "@staticmethod": "static"}.get(dec, "method")
            elif m := re.match(r"    (\w+): ", line):
                name, group = m.group(1), "attribute"
            else:
                continue
            seen = names.setdefault(group, [])
            if not seen or seen[-1] != name:
                seen.append(name)
            if not runs or runs[-1] != group:
                runs.append(group)
        if len(runs) != len(set(runs)):
            problems.append(f"{cls}: member groups split {runs}")
        problems += [f"{cls}: {g} names out of order" for g, n in names.items() if n != sorted(n)]
    return problems


@pytest.mark.skipif(
    not _STUB.is_file(),
    reason=f"no committed stub at {_STUB}",
)
def test_the_committed_stub_is_in_stubgen_order():
    """A hand-placed entry is a diff every rebuild will reproduce.

    Fix by regenerating rather than by moving lines: the rebuild writes the
    order this checks, and a stub that came from the generator passes it.
    """
    problems = _stub_order_problems(_STUB.read_text())
    assert not problems, (
        f"{_STUB.name} is not in pybind11-stubgen's order, so it was edited by hand "
        f"and every rebuild will rewrite it: {problems}. Run `{_REBUILD}` and "
        "commit the regenerated stub."
    )


@pytest.mark.skipif(
    not _STUB.is_file(),
    reason=f"no committed stub at {_STUB}",
)
def test_the_order_check_sees_the_stub_and_catches_a_swap():
    """Guard the guard: the class-body pattern must read real classes, and a
    misplaced line must fail it, or the order test passes vacuously."""
    stub = _STUB.read_text()
    classes = re.findall(r"^class (\w+)", stub, re.M)
    functions = re.findall(r"^def (\w+)", stub, re.M)
    assert len(classes) >= 10 and len(functions) >= 5, (classes, functions)
    assert "SolverOptions" in classes and "net_file_structure" in functions

    swapped = stub.replace("def net_file_structure(", "def zz_net_file_structure(", 1)
    assert any("module-level `def`" in p for p in _stub_order_problems(swapped))
    moved = re.sub(
        r"(?m)^(class SolverStats\b.*:\n)", r"\1    def zz_first(self) -> None: ...\n", stub
    )
    assert any(p.startswith("SolverStats:") for p in _stub_order_problems(moved))
