"""A leg that fires on one half of a mirrored invariant must fire on the other.

Issue #619. ``python/bngsim/_codegen.py`` emits the C ``func[]`` block in
dependency order and says so of itself:

    This mirrors ModelBuilder's Kahn sort (src/model_builder.cpp) so the
    emitted ``func[]`` block is ordered the same way.

Two files, one invariant — and the emitted order and the engine's order agree
only while the two sorts agree. ``_codegen.py`` was a registered path on
``mir.yml``, ``windows-klu.yml`` and ``windows-tail.yml``;
``src/model_builder.cpp`` was registered on none of them. So editing the
mirroring half ran the codegen legs on MIR and Windows, and editing the half
being mirrored did not.

Nothing else covers for it, though not for the reason it first appears. The
codegen cache key *does* hash ``codegen_data()`` (``_codegen.py``'s key docstring
says it "closes the hole" the source digest leaves). But the sort order is not in
``codegen_data()`` — that reports functions in *declaration* order, and
``var_param_bindings`` is not exported at all — so a change to the sort is
invisible to the key, and has to be caught by *running* something. Exporting the
order would close it at the root and make this trigger unnecessary.
#568 changed derived-parameter evaluation order in ``model_builder.cpp`` and did
run all thirteen checks, but only because it also touched ``cvode_simulator.cpp``
and ``steady_state.cpp``, which were registered; its follow-up #617, confined to
``model_builder.cpp``, ran six. The coverage came from which *other* files a PR
happened to touch.

**Why this one names its pairs where test_ci_run_list_coverage.py (issue #295)
derives its rule.** #295's rule — a leg that fires on a test file either runs it
or declares why not — is readable off any workflow's own shape, so a fourth leg
inherits it with no edit there. "These two files implement the same invariant"
is a fact about what the code *means*, and there is no shape to read it off:
``_codegen.py`` mentions several ``src/*.cpp`` files, in contexts ranging from a
passing cross-reference to a genuine claim of mirroring, and a regex over those
mentions cannot tell them apart. So the pairs are named below and justified one
at a time — while leg **discovery** stays structural exactly as #295 does it, so
a new leg registering either half is checked with no edit here.

The table is deliberately incomplete. ``_codegen.py`` also says it mirrors
``src/model.cpp`` (``compute_rxn_rate``, ``compute_derivs``), and adding that row
would fail today because ``mir.yml`` registers ``_codegen.py`` but not
``model.cpp``. That is a real question — is the RHS emitter's mirror of
``compute_derivs`` owed the same trigger? — and it is left open rather than
answered by a row nobody has justified. Do not read the single row as a survey.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ci_workflow import (  # noqa: E402
    REPO_ROOT,
    WORKFLOWS,
    paths_filter_entries,
    strip_comments,
)

# (one half, the other half, why they are one invariant). A leg registering
# either path must register both. Add a row only for a coupling that is real in
# the code — the premise test below refuses a row whose reason has evaporated.
MIRRORED_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "python/bngsim/_codegen.py",
        "src/model_builder.cpp",
        "the emitted func[] order mirrors ModelBuilder's Kahn sort, so the "
        "compiled RHS and evaluate_functions() agree only while the two sorts do "
        "(GH #76; #568 added derived_param_order beside it)",
    ),
)


def _legs() -> list[tuple[str, list[str]]]:
    """Every workflow with a ``paths:`` filter, and what it fires on.

    Structural, as issue #295 discovers its own: a leg added later is covered
    without an edit to this file.
    """
    out = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        entries = paths_filter_entries(path.read_text(encoding="utf-8"))
        if entries:
            out.append((path.name, entries))
    assert out, "no workflow has a paths: filter — the parser has gone blind"
    return out


_PAIR_IDS = [f"{left.split('/')[-1]}+{right.split('/')[-1]}" for left, right, _ in MIRRORED_PAIRS]


@pytest.mark.parametrize("left, right, why", MIRRORED_PAIRS, ids=_PAIR_IDS)
def test_both_halves_of_a_mirrored_invariant_fire_together(left: str, right: str, why: str):
    """Registering one half and not the other is the issue #619 gap."""
    for name, entries in _legs():
        has_left, has_right = left in entries, right in entries
        if has_left == has_right:
            continue
        present, missing = (left, right) if has_left else (right, left)
        pytest.fail(
            f"{name} fires on {present} but not on {missing}. They are one "
            f"invariant in two files: {why}. A change to {missing} would skip "
            f"this leg entirely, so the leg's verdict would be about the other "
            f"half only — add {missing} to this workflow's paths filter, or drop "
            f"the pair from MIRRORED_PAIRS if the coupling is gone."
        )


@pytest.mark.parametrize("left, right, why", MIRRORED_PAIRS, ids=_PAIR_IDS)
def test_the_pair_still_has_its_reason(left: str, right: str, why: str):
    """The table is only honest while the coupling it asserts is real.

    Both files must exist, and the Python half must still name the C++ half —
    which is how the coupling is declared in the first place. If the mirroring is
    ever deleted, this fails and the row gets revisited, instead of quietly
    outliving its reason and firing two expensive legs forever.
    """
    left_path, right_path = REPO_ROOT / left, REPO_ROOT / right
    assert left_path.is_file(), f"{left} no longer exists; revisit this pair"
    assert right_path.is_file(), f"{right} no longer exists; revisit this pair"
    # Not a skip: MIRRORED_PAIRS rows put the mirroring (Python) half first, and
    # this check reads that half for the declaration. A row shaped otherwise
    # needs this test rethought — reporting it as skipped would hide exactly the
    # "row outlives its reason" case the test exists to catch.
    assert left.endswith(".py"), (
        f"MIRRORED_PAIRS puts the mirroring (Python) half first; {left} is not one"
    )
    body = left_path.read_text(encoding="utf-8")
    assert re.search(rf"\b{re.escape(right)}\b", body), (
        f"{left} no longer names {right}, which is what declared them one "
        f"invariant ({why}). Either the mirroring moved — re-point this row at "
        f"where it lives now — or it is gone, and the row should go with it."
    )


def test_the_workflow_parser_sees_a_real_filter():
    """Guard the guard: a parser that silently matched nothing would make both
    tests above vacuously green, which is the one way a coverage check fails
    without saying so."""
    legs = dict(_legs())
    assert "mir.yml" in legs, "mir.yml has a paths filter; the parser missed it"
    # Comments are not entries — the reason each path is registered is prose.
    for name, entries in legs.items():
        assert all(not e.startswith("#") for e in entries), name
        assert entries == paths_filter_entries(
            strip_comments((WORKFLOWS / name).read_text(encoding="utf-8"))
        ), f"{name}: comment stripping changed the entry list"


def test_an_inline_comment_does_not_truncate_a_filter():
    """The parser used to ``break`` on any line its item regex missed, so one
    trailing ``# comment`` on a path entry silently dropped every entry below it
    — and every consumer reads a short list as "not registered", so the coverage
    assertions above turned GREEN. Comparing against ``strip_comments`` cannot
    catch it: that removes only whole-line comments, so both sides truncate
    identically."""
    body = (
        "on:\n  push:\n    paths: &p\n"
        '      - "src/a.cpp"  # why a is here\n'
        '      - "src/b.cpp"\n'
        '      - "python/c.py"\n'
    )
    assert paths_filter_entries(body) == ["src/a.cpp", "src/b.cpp", "python/c.py"]
