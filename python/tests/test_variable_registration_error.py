"""What a name the evaluator refuses is told about it.

ExprTk's ``add_variable`` returns the same ``false`` for a name that is no
symbol as for one that is already taken, and ``define_variable`` read every
refusal as a duplicate: a ``.net`` parameter named ``k-1`` or ``2k`` failed to
load with "is already registered. Check the .net file for duplicate ... names",
sending the user to look for a duplicate that was not there. Found in the review
of #919, which made the same refusal a return value for the callers that skip
such a name (a ``.net`` species ``A()``).
"""

from __future__ import annotations

import bngsim
import pytest

NET = """\
begin parameters
    1 {param}  1.0
end parameters
begin species
    1 A() 1
end species
begin reactions
    1 1 0 {param}
end reactions
begin groups
    1 {group} 1
end groups
"""


def _load(tmp_path, param, group="Aobs"):
    p = tmp_path / "m.net"
    p.write_text(NET.format(param=param, group=group))
    return bngsim.Model.from_net(p)


@pytest.mark.parametrize("name", ["k-1", "2k"])
def test_a_name_that_is_no_identifier_is_told_so(tmp_path, name):
    with pytest.raises(bngsim.ModelError) as err:
        _load(tmp_path, name)
    msg = str(err.value)
    assert f"'{name}'" in msg
    assert "is not a valid identifier" in msg
    assert "already registered" not in msg


def test_a_name_that_is_taken_is_still_a_duplicate(tmp_path):
    # A parameter and an observable under one name.
    with pytest.raises(bngsim.ModelError) as err:
        _load(tmp_path, "k", group="k")
    msg = str(err.value)
    assert "'k'" in msg
    assert "is already registered" in msg
    assert "not a valid identifier" not in msg
