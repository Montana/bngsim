"""Issue #1022: method='rm' gives an Arrhenius (eBNGL) rule BNG2's rate.

BNG2 fires an energy rule at exp(-(Ea0 + phi*dG)), dG the free-energy change of
the reaction it writes and 1-phi for the reverse half of a reversible rule, with
energies in units of RT: there is no RT in the formula, and a model parameter
named RT is an ordinary parameter. RuleMonkey 3.10.1 divided every energy by RT,
the model's or 2.478 by default (richardposner/RuleMonkey#107), so every energy
model without RT = 1 ran at the wrong rates with no warning. On
arrhenius_coop.xml (A(b)+B(a) <-> AB and B(c)+C(b) <-> BC with a cooperativity
energy, 100 of each) rm held AB at 87 against network SSA's 74.7.

Oracles: network SSA on the BNG-generated .net for the transient, the exact
stationary distribution for the equilibrium (detailed balance gives it a product
form), and the closed form of a one-way unbinding rule, which is first order.

RuleMonkey 3.11.0 expanded some energy rules differently from BNG2: it dropped
or misapplied an energy pattern's context, and an Arrhenius rule's own reactant
context, and rm refused those by name. RuleMonkey 3.12.0 runs each at BNG2's
rate, expanded at load where that is exact and counted at each firing elsewhere
(richardposner/RuleMonkey#109), and issue #1058 lifted the refusal. The models
are in arrhenius/misread/, each one-way unbinding with a closed form, and
state_pattern, a reversible rule with a state in its energy pattern, which
settles on its exact stationary distribution.
"""

from __future__ import annotations

import functools
import math
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest

ARR = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim" / "arrhenius"

T = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 2.0, 5.0]
N_RM, N_SSA = 40, 200


def _has_rulemonkey() -> bool:
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_rulemonkey(), reason="RuleMonkey not compiled in")


@functools.cache
def _runs(method: str, name: str, n: int, observables: tuple[str, ...]) -> np.ndarray:
    """replicates x observables x times, a fresh Model or session per replicate.
    *method* is a Simulator method or ``"session"``, a RuleMonkeySession."""
    out = []
    for seed in range(1, n + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("error", bngsim.RuleMonkeyWarning)
            if method == "session":
                with bngsim.RuleMonkeySession(str(ARR / f"{name}.xml")) as s:
                    s.initialize(seed=seed)
                    r = s.simulate(sample_times=T)
            else:
                m = bngsim.Model.from_net(str(ARR / f"{name}.net"))
                kw = {"xml_path": str(ARR / f"{name}.xml")} if method == "rm" else {}
                r = bngsim.Simulator(m, method=method, **kw).run(sample_times=T, seed=seed)
        out.append([np.asarray(r.observables[o]) for o in observables])
    return np.asarray(out, dtype=float)


def _assert_same_mean(a: np.ndarray, b: np.ndarray) -> None:
    """Two-sample z-test per row, both columns of a row at the start equal."""
    se = np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b))
    diff = a.mean(axis=0) - b.mean(axis=0)
    fixed = se == 0
    assert np.all(diff[fixed] == 0), diff
    assert np.all(np.abs(diff[~fixed] / se[~fixed]) < 4.5), diff / np.where(fixed, 1, se)


@functools.cache
def _coop_equilibrium() -> dict[str, float]:
    """Exact stationary means of AB, BC and ABC for arrhenius_coop.

    Species A, B, C, AB, BC, ABC (AB = A.B, BC = B.C, ABC = A.B.C), 100 of each
    molecule. Detailed balance makes the stationary distribution
    prod_s w_s^n_s / n_s! with w = 1 for a monomer and exp(-G) for a complex,
    G its formation energy: 3.5, 3.5 and 3.5 + 3.5 - 2.
    """
    n = 100
    x = np.arange(n + 1)[:, None, None]  # AB
    y = np.arange(n + 1)[None, :, None]  # BC
    z = np.arange(n + 1)[None, None, :]  # ABC
    a, b, c = n - x - z, n - x - y - z, n - y - z
    ok = (a >= 0) & (b >= 0) & (c >= 0)
    log_fact = np.array([math.lgamma(k + 1.0) for k in range(n + 1)])

    def lgam(k):
        return log_fact[np.where(ok, k, 0)]

    logw = -3.5 * x - 3.5 * y - 5.0 * z - lgam(a) - lgam(b) - lgam(c) - lgam(x) - lgam(y) - lgam(z)
    logw = np.where(ok, logw, -np.inf)
    p = np.exp(logw - logw.max())
    p /= p.sum()
    return {
        "AB": float((p * (x + z)).sum()),
        "BC": float((p * (y + z)).sum()),
        "ABC": float((p * z).sum()),
    }


@pytest.mark.parametrize(
    "name",
    [
        "arrhenius_coop",
        # The same energies written in kJ/mol over a model parameter RT = 2.577:
        # RT is ordinary, so the model is arrhenius_coop.
        "arrhenius_coop_rt",
    ],
)
def test_reversible_energy_rules_match_network_ssa(name):
    obs = ("AB", "BC", "ABC")
    rm = _runs("rm", name, N_RM, obs)
    ssa = _runs("ssa", "arrhenius_coop", N_SSA, obs)
    for j in range(len(obs)):
        _assert_same_mean(rm[:, j, :], ssa[:, j, :])


@pytest.mark.parametrize("name", ["arrhenius_coop", "arrhenius_coop_rt"])
def test_reversible_energy_rules_settle_on_the_exact_equilibrium(name):
    obs = ("AB", "BC", "ABC")
    rm = _runs("rm", name, N_RM, obs)[:, :, -1]  # t = 5, long past relaxation
    exact = _coop_equilibrium()
    for j, o in enumerate(obs):
        se = rm[:, j].std(ddof=1) / math.sqrt(len(rm))
        assert abs(rm[:, j].mean() - exact[o]) < 4.5 * se, (o, rm[:, j].mean(), exact[o])


def test_the_exact_equilibrium_is_network_ssas():
    # The oracle above, checked against the generated network.
    ssa = _runs("ssa", "arrhenius_coop", N_SSA, ("AB", "BC", "ABC"))[:, :, -1]
    exact = _coop_equilibrium()
    for j, o in enumerate(("AB", "BC", "ABC")):
        se = ssa[:, j].std(ddof=1) / math.sqrt(len(ssa))
        assert abs(ssa[:, j].mean() - exact[o]) < 4.5 * se


def test_one_way_energy_rules_fire_one_way_at_bng2s_rate():
    """A one-way binding rule has no reverse, and a one-way unbinding rule fires
    at exp(-(Ea0 + phi*dG)) with dG = -Gf_CD = 2 the change it writes."""
    rm = _runs("rm", "arrhenius_one_way", N_RM, ("AB", "CD"))
    ssa = _runs("ssa", "arrhenius_one_way", N_SSA, ("AB", "CD"))
    _assert_same_mean(rm[:, 0, :], ssa[:, 0, :])
    # Binding never comes undone: AB only grows along every run.
    assert np.all(np.diff(rm[:, 0, :], axis=1) >= 0)
    # CD decays at k = exp(-(0.5 + 0.3*2)), each pair on its own.
    q = np.exp(-math.exp(-1.1) * np.asarray(T))
    cd = rm[:, 1, :]
    assert np.all(np.diff(cd, axis=1) <= 0)
    se = np.sqrt(100 * q * (1 - q) / N_RM)
    z = (cd.mean(axis=0)[1:] - 100 * q[1:]) / se[1:]
    assert np.all(np.abs(z) < 4.5), z


# Each .bngl in misread/ unbinds one-way and its header gives BNG2's rate for each
# kind of complex it seeds, 1000 of each, so an observable that counts one kind
# is binomial(1000, exp(-k t)) at t, and k = 0 holds it at 1000. RuleMonkey 3.11.0
# read all but plain_two_a and samename_explicit wrong (|z| from 24 to 142 against
# these, 40 seeds), and rm refused all fifteen until issue #1058.
E09, E125 = math.exp(-0.9), math.exp(-1.25)
MISREADS = {
    # A state in the energy pattern: A(b!1).B(a!1,s~1).
    "state_ctx": {"AB1": E125, "AB0": E09},
    # A site the pattern needs free: A(b!1,c).B(a!1).
    "freectx": {"AB1": E125, "AB0": E09},
    # Wildcard bonds only, L(r!+,r!+): both bonds of L.R.R break at e^-1.25.
    "wild": {"AB1": 2 * E125},
    # L(r!1,r!+).R(l!1) needs the other r bound, which L(r!1,r).R(l!1) lacks.
    "samename": {"AB1": E09},
    # L(r!1,r!2).R(l!1).S(l!2) applies only where the other r holds an S.
    "samename_partner": {"LRS": E125, "LRR": 2 * E09},
    # A's c binds C or D, and A(b!1,c!2).B(a!1).C(a!2) only C.
    "partner": {"ABC": E125, "ABD": E09},
    # Context two bonds out: A(b!1).B(a!1,c!2).C(b!2,d!3).D(c!3).
    "chain2ctx": {"ABCD": E125, "ABC": E09},
    # Two context bonds, both needed: A(b!1,c!2,d!3).B(a!1).C(a!2).D(a!3).
    "two_leaves": {"ABCD": E125, "ABC": E09},
    # A(x,c,c): one embedding of A(x!1,c!2).B(y!1).C(a!2) per bound c.
    "samectx": {"AB1": E125, "AB2": math.exp(-1.6)},
    # Z, not C, binds A.c, through a rule that synthesizes Z bound.
    "synth": {"AB": E09},
    # A state in the rule's own reactant pattern: only A(s~1).B unbinds.
    "rule_context": {"AB1": E09, "AB0": 0.0},
    # Unbinding written with '.': the ring opens, the chain never parts.
    "ringopen": {"Ring": E09, "Open": 0.0},
    # A unidirectional rule named like a reverse half, which BNG2 gives phi.
    "revname": {"AB": E09},
    # Two A on one B, A(b!1).B(a1!1,a2!2).A(b!2); only the a1 bond breaks.
    "plain_two_a": {"Full": E125},
    # A site named twice, with explicit bonds: L(r!1,r!2).R(l!1).R(l!2).
    "samename_explicit": {"Full": 2 * E125},
}


def _assert_decays(counts: np.ndarray, k: float) -> None:
    """*counts* (replicates x times) of 1000 complexes that each leave at rate *k*."""
    if k == 0:
        assert np.all(counts == 1000), counts.min()
        return
    q = np.exp(-k * np.asarray(T))
    assert np.all(counts[:, 0] == 1000)
    assert np.all(np.diff(counts, axis=1) <= 0)  # one-way: nothing comes back
    se = np.sqrt(1000 * q[1:] * (1 - q[1:]) / len(counts))
    z = (counts.mean(axis=0)[1:] - 1000 * q[1:]) / se
    assert np.all(np.abs(z) < 4.5), z


@pytest.mark.parametrize("method", ["rm", "session", "ssa"])
@pytest.mark.parametrize("name", sorted(MISREADS))
def test_energy_context_fires_at_bng2s_closed_form(name, method):
    # ssa, on the generated network, checks the closed form itself.
    obs = tuple(MISREADS[name])
    runs = _runs(method, f"misread/{name}", N_SSA if method == "ssa" else N_RM, obs)
    for j, o in enumerate(obs):
        _assert_decays(runs[:, j, :], MISREADS[name][o])


@pytest.mark.parametrize("name", sorted(MISREADS))
def test_energy_context_matches_network_ssa(name):
    obs = tuple(MISREADS[name])
    rm = _runs("rm", f"misread/{name}", N_RM, obs)
    ssa = _runs("ssa", f"misread/{name}", N_SSA, obs)
    for j in range(len(obs)):
        _assert_same_mean(rm[:, j, :], ssa[:, j, :])


@functools.cache
def _state_pattern_equilibrium() -> dict[str, float]:
    """Exact stationary means of AB0 and AB1 for state_pattern.

    100 A, 100 B(s~0) and 100 B(s~1), and nothing changes s. Detailed balance
    makes the stationary distribution prod_s w_s^n_s / n_s! with w = 1 for a
    monomer, exp(-3.5) for A.B(s~0) and exp(-(3.5 - 2)) for A.B(s~1).
    """
    n = 100
    x = np.arange(n + 1)[:, None]  # AB0
    y = np.arange(n + 1)[None, :]  # AB1
    a = n - x - y
    ok = a >= 0
    log_fact = np.array([math.lgamma(k + 1.0) for k in range(n + 1)])

    def lgam(k):
        return log_fact[np.where(ok, k, 0)]

    logw = -3.5 * x - 1.5 * y - lgam(a) - lgam(n - x) - lgam(n - y) - lgam(x) - lgam(y)
    logw = np.where(ok, logw, -np.inf)
    p = np.exp(logw - logw.max())
    p /= p.sum()
    return {"AB0": float((p * x).sum()), "AB1": float((p * y).sum())}


def test_a_reversible_rule_with_a_state_energy_pattern_matches_network_ssa():
    obs = ("AB0", "AB1")
    rm = _runs("rm", "state_pattern", N_RM, obs)
    ssa = _runs("ssa", "state_pattern", N_SSA, obs)
    for j in range(len(obs)):
        _assert_same_mean(rm[:, j, :], ssa[:, j, :])


@pytest.mark.parametrize("method", ["rm", "ssa"])
def test_a_reversible_rule_with_a_state_energy_pattern_settles_on_the_exact_equilibrium(
    method,
):
    # ssa, on the generated network, checks the oracle itself.
    obs = ("AB0", "AB1")
    runs = _runs(method, "state_pattern", N_SSA if method == "ssa" else N_RM, obs)[:, :, -1]
    exact = _state_pattern_equilibrium()
    for j, o in enumerate(obs):
        se = runs[:, j].std(ddof=1) / math.sqrt(len(runs))
        assert abs(runs[:, j].mean() - exact[o]) < 4.5 * se, (o, runs[:, j].mean(), exact[o])


def test_rulemonkey_names_an_energy_rule_shape_it_refuses_itself():
    """A state-change Arrhenius rule is refused by RuleMonkey's own load check,
    whose message names the rule and says what it is."""
    xml = str(ARR / "state_change.xml")
    m = bngsim.Model.from_net(str(ARR / "state_change.net"))
    for build in (
        lambda: bngsim.Simulator(m, method="rm", xml_path=xml),
        lambda: bngsim.RuleMonkeySession(xml),
    ):
        with pytest.raises(
            RuntimeError, match="rule '_R1' is outside the shapes RM runs: it changes a state"
        ):
            build()


def test_cooperativity_on_both_reaction_centers_matches_network_ssa():
    """Two reversible energy rules whose patterns each hold one leaf on a center,
    patterns both rules read, and a mass-action rule binding a leaf site."""
    obs = ("AB", "AC", "ABC", "ABD")
    rm = _runs("rm", "coop2", N_RM, obs)
    ssa = _runs("ssa", "coop2", N_SSA, obs)
    for j in range(len(obs)):
        _assert_same_mean(rm[:, j, :], ssa[:, j, :])


def test_a_pair_pattern_on_a_bivalent_ligand_matches_its_closed_form():
    # L(r,r) with only L(r!1).R(l!1): each bond breaks at exp(-0.9).
    full = _runs("rm", "bivalent_pair", N_RM, ("Full",))[:, 0, :]
    q = np.exp(-2 * math.exp(-0.9) * np.asarray(T))
    z = (full.mean(axis=0)[1:] - 1000 * q[1:]) / np.sqrt(1000 * q[1:] * (1 - q[1:]) / N_RM)
    assert np.all(np.abs(z) < 4.5), z
