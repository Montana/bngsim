"""Issue #1364: parse_bngl_protocol reads simulate/parameter_scan as BNG2.pl does.

``BNGAction.pm`` (BNG2 2.9.3) defines the arguments: ``continue=>1`` starts at
the model time, where an earlier run ended; ``n_output_steps`` is ``n_steps``,
both is an error, and neither is one step; ``sample_times`` sets the grid where
neither is given, ending at its last time; a scan reads ``par_scan_vals`` where
``par_min``/``par_max`` are not given. The parser had none of this: (0, 20) for
a continued (10, 20), 101 points for 11 and for 2, a horizon of 0, and a scan of
one run, and ``strict=True`` raised nothing.

What the IR's uniform ``(t_span, n_points)`` and ``(min, max, points, log)``
hold, it now holds exactly; what they cannot (an uneven grid, uneven scan
values) is lossy, and ``strict=True`` refuses it.
"""

from __future__ import annotations

import re

import pytest
from bngsim import ConversionError, ConversionWarning
from bngsim.convert import parse_bngl_protocol, write_bngl_protocol, write_sedml_protocol


def _parse(*actions: str, strict: bool = True):
    return parse_bngl_protocol(
        "begin actions\n" + "\n".join(actions) + "\nend actions\n", strict=strict
    )


def _only(*actions: str):
    spec = _parse(*actions)
    assert spec.lossy == ()
    return spec.steps


def test_a_continued_run_starts_where_the_last_one_ended():
    first, second = _only(
        'simulate({method=>"ode",t_end=>10,n_steps=>10})',
        'simulate({method=>"ode",continue=>1,t_end=>20,n_steps=>10})',
    )
    assert first.t_span == (0.0, 10.0)
    assert second.t_span == (10.0, 20.0)


def test_a_continued_run_after_a_scan_starts_at_the_scans_end():
    """Each point of a scan is a simulate, which leaves the model time at its
    t_end."""
    _, cont = _only(
        'parameter_scan({parameter=>"k",par_min=>1,par_max=>2,n_scan_pts=>2,t_end=>5})',
        'simulate({method=>"ode",continue=>1,t_end=>8,n_steps=>3})',
    )
    assert cont.t_span == (5.0, 8.0)


def test_a_continued_run_with_its_t_start_keeps_it():
    _, cont = _only(
        'simulate({method=>"ode",t_end=>10,n_steps=>10})',
        'simulate({method=>"ode",continue=>1,t_start=>10,t_end=>30,n_steps=>4})',
    )
    assert cont.t_span == (10.0, 30.0)


def test_a_continuation_from_an_unknown_time_is_lossy():
    with pytest.raises(ConversionError, match=r"continue=>1 starts at the model time"):
        _parse(
            'simulate({method=>"ode",t_end=>tf,n_steps=>10})',
            'simulate({method=>"ode",continue=>1,t_end=>20,n_steps=>10})',
        )


def test_n_output_steps_is_n_steps():
    (exp,) = _only('simulate({method=>"ode",t_end=>10,n_output_steps=>10})')
    assert exp.n_points == 11
    assert "n_output_steps" not in exp.extra


def test_both_n_steps_and_n_output_steps_is_refused():
    with pytest.raises(ConversionError, match=r"'n_steps' and 'n_output_steps'"):
        _parse('simulate({method=>"ode",t_end=>10,n_steps=>5,n_output_steps=>5})')


def test_no_n_steps_is_one_step():
    (exp,) = _only('simulate({method=>"ode",t_end=>10})')
    assert exp.n_points == 2


def test_an_n_steps_the_parse_cannot_read_keeps_its_fallback():
    """Control: a parameter reference is not BNG2's default; the 100 the parse
    has always taken for it is kept."""
    (exp,) = _only('simulate({method=>"ode",t_end=>10,n_steps=>nsteps})')
    assert exp.n_points == 101


def test_an_even_sample_grid_from_t_start_is_the_time_course():
    (exp,) = _only('simulate({method=>"ode",sample_times=>[0,2,4,6,8,10]})')
    assert exp.t_span == (0.0, 10.0)
    assert exp.n_points == 6
    assert "sample_times" not in exp.extra


def test_n_steps_takes_precedence_over_sample_times():
    (exp,) = _only('simulate({method=>"ode",t_end=>10,n_steps=>5,sample_times=>[0,1,5,10]})')
    assert (exp.t_span, exp.n_points) == ((0.0, 10.0), 6)


def test_an_uneven_sample_grid_is_lossy():
    with pytest.raises(
        ConversionError, match=r"sample_times \[0.0, 1.0, 5.0, 10.0\] is not an even"
    ):
        _parse('simulate({method=>"ode",sample_times=>[0,1,5,10]})')
    with pytest.warns(ConversionWarning):
        spec = _parse('simulate({method=>"ode",sample_times=>[0,1,5,10]})', strict=False)
    assert spec.lossy == ("simulate",)
    assert spec.steps[0].t_span == (0.0, 10.0)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ("[1,2,3]", (1.0, 3.0, 3, False)),
        ("[0.1,1,10]", (0.1, 10.0, 3, True)),
        ("[3,1,2]", (1.0, 3.0, 3, False)),
    ],
    ids=["linear", "log", "unsorted"],
)
def test_par_scan_vals_are_the_scan(values, expected):
    (exp,) = _only(
        f'parameter_scan({{parameter=>"k",par_scan_vals=>{values},method=>"ode",t_end=>10,n_steps=>5}})'
    )
    assert (exp.scan_min, exp.scan_max, exp.scan_points, exp.scan_log) == pytest.approx(expected)
    assert "par_scan_vals" not in exp.extra


def test_par_min_and_par_max_take_precedence_over_par_scan_vals():
    (exp,) = _only(
        'parameter_scan({parameter=>"k",par_min=>1,par_max=>9,n_scan_pts=>5,'
        "par_scan_vals=>[1,2],t_end=>10})"
    )
    assert (exp.scan_min, exp.scan_max, exp.scan_points) == (1.0, 9.0, 5)


def test_uneven_par_scan_vals_are_lossy():
    with pytest.raises(
        ConversionError, match=r"par_scan_vals \[1.0, 2.0, 7.0\] is not evenly spaced"
    ):
        _parse('parameter_scan({parameter=>"k",par_scan_vals=>[1,2,7],t_end=>1})')


def test_the_sedml_export_carries_the_grid_and_the_scan():
    spec = _parse(
        'simulate({method=>"ode",t_end=>10,n_steps=>10})',
        'simulate({method=>"ode",continue=>1,t_end=>20,n_steps=>10})',
        'simulate({method=>"ode",sample_times=>[0,2,4,6,8,10]})',
        'parameter_scan({parameter=>"k",par_scan_vals=>[1,2,3],method=>"ode",t_end=>10,n_steps=>5})',
    )
    x = write_sedml_protocol(spec)
    grids = re.findall(
        r'outputStartTime="([^"]*)" outputEndTime="([^"]*)" numberOfPoints="([^"]*)"', x
    )
    assert ("10", "20", "10") in grids and ("0", "10", "5") in grids
    assert "repeatedTask" in x


def test_the_bngl_writer_round_trips_what_bng2_accepts():
    """Re-emitted, a continuation names the t_start it implies beside
    continue=>1 (BNG2.pl checks they agree), n_output_steps is one n_steps, and
    a scan is par_min/par_max/n_scan_pts; parsed again, the spec is the same.
    That BNG2.pl writes identical .gdat and .scan files from both is in the PR."""
    spec = _parse(
        'simulate({method=>"ode",t_end=>10,n_steps=>10})',
        'simulate({method=>"ode",continue=>1,t_end=>20,n_steps=>10})',
        'simulate({method=>"ode",t_end=>10,n_output_steps=>10})',
        'parameter_scan({parameter=>"k",par_scan_vals=>[0.1,1,10],method=>"ode",t_end=>10,n_steps=>5})',
    )
    text = write_bngl_protocol(spec)
    assert "t_start=>10,t_end=>20,n_steps=>10,continue=>1" in text
    assert "n_output_steps" not in text and "par_scan_vals" not in text
    assert parse_bngl_protocol(text, strict=True).steps == spec.steps
