"""A species an assignment rule sets, written to .net (issue #1274).

The writer emits it ``$``-fixed, beside the observable (a linear rule) or the
function (any other) that holds the rule under the species' own name. bngsim's
reader takes the pair as the rule (``_ar_report_map_from_net``, #515), so the
file is exact for bngsim. BioNetGen's ``run_network`` holds the species at its
initial value for the whole run instead, and so runs every rate law that reads
it wrong. The writer now says so, and still writes the file.
"""

from __future__ import annotations

import subprocess
import warnings

import bngsim
import numpy as np
import pytest
from bngsim._bngpath import resolve_bng
from bngsim.convert import capability_report, write_net

_BNG = resolve_bng()

RULE = "species A = 10; species B; k = 0.5; J: A -> ; k*A; B := 2*A"


def _write(antimony: str, path, **kw):
    model = bngsim.Model.from_antimony_string(antimony)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_net(model, str(path), **kw)
    return [str(w.message) for w in caught if issubclass(w.category, bngsim.ConversionWarning)]


@pytest.mark.parametrize(
    ("antimony", "kind"),
    [
        (RULE, "a linear rule, held by an observable"),
        (
            "species A = 10; species B; k = 0.5; J: A -> ; k*A; B := A*A",
            "any other, by a function",
        ),
    ],
)
def test_a_rule_species_is_warned_and_still_written(tmp_path, antimony, kind):
    path = tmp_path / "rule.net"
    said = _write(antimony, path)
    assert path.is_file()
    rule = [m for m in said if "only bngsim reads as written" in m]
    assert len(rule) == 1, said
    assert "'B'" in rule[0] and "run_network holds each at its initial value" in rule[0]


def test_strict_does_not_refuse_it(tmp_path):
    """The file is exact for bngsim's own reader: warned, never refused, so the
    default strict=True still writes it."""
    path = tmp_path / "rule.net"
    _write(RULE, path, strict=True)
    assert path.is_file()


@pytest.mark.parametrize(
    "antimony",
    [
        "species A = 10; species B = 0; k = 0.5; J: A -> B; k*A",
        # a boundary species a reaction makes is written $-fixed too, and is set
        # by no rule: held, as written, by bngsim and run_network alike
        "species A = 10; $B = 3; k = 0.5; J: A -> B; k*A",
    ],
    ids=["no-rule", "boundary-species"],
)
def test_a_model_with_no_rule_species_is_not_warned(tmp_path, antimony):
    path = tmp_path / "plain.net"
    said = _write(antimony, path)
    if "$B" in antimony:
        assert "$B 3.0" in path.read_text()  # the control is a $-fixed species
    assert not [m for m in said if "only bngsim reads as written" in m], said


def test_the_report_names_it_apart_from_lossy():
    report = capability_report(bngsim.Model.from_antimony_string(RULE))
    assert len(report["bngsim_only"]) == 1
    assert not any("assignment rule" in note for note in report["lossy"])


def test_bngsim_reads_the_rule_back(tmp_path):
    path = tmp_path / "rule.net"
    _write(RULE, path)
    r = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").run(
        t_span=(0, 5), n_points=2
    )
    names = list(r.species_names)
    a, b = np.asarray(r.species)[-1, [names.index("A"), names.index("B")]]
    assert b == pytest.approx(2 * a, rel=1e-6)


@pytest.mark.skipif(not _BNG.ok, reason=f"BNG2.pl unavailable: {_BNG.why_not()}")
def test_run_network_holds_it_at_its_initial_value(tmp_path):
    """What the warning says, as run_network does it: B = 20 at t = 5, where the
    rule gives 2·A = 1.64."""
    path = tmp_path / "rule.net"
    _write(RULE, path)
    run_network = _BNG.root / "bin" / "run_network"
    if not run_network.is_file():
        pytest.skip(f"BNG2.pl unavailable: no run_network at {run_network}")
    out = tmp_path / "rn"
    subprocess.run(
        [
            str(run_network),
            "-o",
            str(out),
            "-p",
            "cvode",
            "-a",
            "1e-12",
            "-r",
            "1e-10",
            str(path),
            "5",
            "1",
        ],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    rows = np.loadtxt(f"{out}.cdat")
    assert rows[-1, 2] == pytest.approx(20.0)  # B, held
