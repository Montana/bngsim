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

Nothing else covers for it. The codegen cache key hashes the Python emitters
only and says so in as many words — *"a C++ change that alters codegen_data() is
not caught here"* — so such a change has to be caught by *running* something.
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
``_codegen.py`` names four different ``src/*.cpp`` files and only one of them is
a mirrored counterpart. A regex over those mentions would demand registering the
other three, which is how a trigger filter stops meaning anything. So the pairs
are named below and justified one at a time — while leg **discovery** stays
structural exactly as #295 does it, so a new leg registering either half is
checked with no edit here.
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


@pytest.mark.parametrize("left, right, why", MIRRORED_PAIRS, ids=lambda v: v.split("/")[-1])
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


@pytest.mark.parametrize("left, right, why", MIRRORED_PAIRS, ids=lambda v: v.split("/")[-1])
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
    if not left.endswith(".py"):
        return
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
