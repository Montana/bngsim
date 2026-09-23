"""GH #575 — a uniformRange's numberOfPoints counts intervals, not points.

SED-ML's ``numberOfPoints`` counts the steps between endpoints: a range expands
to ``numberOfPoints + 1`` values. ``write_sedml_protocol`` applies that to the
``uniformTimeCourse`` — ``n_points = 11`` is written as ``numberOfPoints="10"``
— and wrote the scan's ``scan_points`` verbatim.

bngsim's ``scan_points`` is the inclusive point count: ``_resolve_scan_values``
expands it with ``np.linspace``, and BNG2.pl computes ``delta = (max - min) /
(n_scan_pts - 1)`` over the same inclusive count. So a 5-point scan over [1, 2]
was written as ``numberOfPoints="5"``, which every standards-compliant consumer
expands to six values — 1, 1.2, 1.4, 1.6, 1.8, 2 — where bngsim ran five:
1, 1.25, 1.5, 1.75, 2. Not one extra sample at the end: a different parameter
value at every step but the first. The document is well formed and the sweep
looks reasonable, so nothing downstream had cause to complain.

bngsim's own round trip goes through the ``<annotation>`` that carries the
verbatim ProtocolSpec, which is why this never showed up in-house — the
standards elements are what other tools read.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from bngsim.convert import read_sedml_protocol, write_sedml_protocol
from bngsim.convert._protocol import Experiment, ProtocolSpec

START, END = 1.0, 2.0


def _scan_spec(scan_points: int, *, log: bool = False, n_points: int = 11) -> ProtocolSpec:
    return ProtocolSpec(
        steps=[
            Experiment(
                kind="scan",
                method="ode",
                t_span=(0.0, 10.0),
                n_points=n_points,
                scan_parameter="k",
                scan_min=START,
                scan_max=END,
                scan_points=scan_points,
                scan_log=log,
                reset_between=True,
            )
        ]
    )


def _attr(xml: str, tag: str, name: str) -> str | None:
    for el in ET.fromstring(xml).iter():
        if el.tag.rsplit("}", 1)[-1] == tag:
            return el.get(name)
    raise AssertionError(f"no <{tag}> in the document")


def _without_annotation(xml: str) -> str:
    """The same document with bngsim's verbatim-spec annotation removed, which
    is what another tool effectively reads: the standard elements alone."""
    root = ET.fromstring(xml)
    for parent in root.iter():
        for child in list(parent):
            if child.tag.rsplit("}", 1)[-1] == "annotation":
                parent.remove(child)
    return ET.tostring(root, encoding="unicode")


# ── Writing ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("scan_points, want", [(5, "4"), (2, "1"), (1, "0"), (21, "20")])
def test_the_range_is_written_as_intervals(scan_points, want):
    assert (
        _attr(write_sedml_protocol(_scan_spec(scan_points)), "uniformRange", "numberOfPoints")
        == want
    )


def test_the_time_course_conversion_is_unchanged():
    """It was already right; the scan branch is what differed from it."""
    xml = write_sedml_protocol(_scan_spec(5, n_points=11))
    assert _attr(xml, "uniformTimeCourse", "numberOfPoints") == "10"


def test_the_other_range_attributes_are_untouched():
    xml = write_sedml_protocol(_scan_spec(5, log=True))
    assert _attr(xml, "uniformRange", "start") == "1"
    assert _attr(xml, "uniformRange", "end") == "2"
    assert _attr(xml, "uniformRange", "type") == "log"


# ── What a standards-compliant consumer expands it to ────────────────────────


@pytest.mark.parametrize("scan_points", [2, 5, 21])
def test_a_spec_consumer_expands_it_to_the_values_bngsim_runs(scan_points):
    """The point of the whole issue: `numberOfPoints + 1` values, spaced by
    `(end - start) / numberOfPoints`, must be the linspace bngsim sweeps."""
    intervals = int(
        _attr(write_sedml_protocol(_scan_spec(scan_points)), "uniformRange", "numberOfPoints")
    )
    consumer = [START + i * (END - START) / intervals for i in range(intervals + 1)]
    ours = list(np.linspace(START, END, scan_points))
    assert consumer == pytest.approx(ours, rel=0, abs=1e-12)


# ── Reading ──────────────────────────────────────────────────────────────────


def test_the_standards_path_recovers_the_point_count():
    """With the annotation gone — what another tool's document looks like —
    the reconstruction must land back on the inclusive count."""
    xml = _without_annotation(write_sedml_protocol(_scan_spec(5)))
    step = read_sedml_protocol(xml).steps[0]
    assert step.scan_points == 5
    assert step.n_points == 11  # the time course, for contrast


def test_the_annotation_round_trip_is_still_exact():
    spec = _scan_spec(5)
    back = read_sedml_protocol(write_sedml_protocol(spec))
    assert back.steps[0].scan_points == 5
    assert back.steps[0].scan_min == START
    assert back.steps[0].scan_max == END


@pytest.mark.parametrize("scan_points", [1, 2, 5, 21])
def test_every_count_survives_the_standards_path(scan_points):
    xml = _without_annotation(write_sedml_protocol(_scan_spec(scan_points)))
    assert read_sedml_protocol(xml).steps[0].scan_points == scan_points


def test_a_foreign_documents_range_is_read_as_intervals():
    """A range another tool wrote: four intervals are five values."""
    xml = _without_annotation(write_sedml_protocol(_scan_spec(5)))
    xml = xml.replace('numberOfPoints="4"', 'numberOfPoints="9"')
    assert read_sedml_protocol(xml).steps[0].scan_points == 10


def test_an_unstated_count_stays_unknown():
    """Absent is not zero: a range with no numberOfPoints says nothing about
    the count, and must not come back as a 1-point scan."""
    xml = _without_annotation(write_sedml_protocol(_scan_spec(5)))
    xml = xml.replace(' numberOfPoints="4"', "")
    assert read_sedml_protocol(xml).steps[0].scan_points is None
