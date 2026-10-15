"""Checks a BNG XML must pass before a network-free backend is handed it.

Issue #862: NFsim cannot prepare a function that reads ``time()`` and fails at
setup with ``Error preparing function f in class GlobalFunction``, which names
neither the cause nor an alternative. It is refused here, before the backend is
built, naming the functions. The clock is read under either spelling,
``time()`` or a bare ``time``, and both are refused.

RuleMonkey runs such a function since 3.11.0, which integrates a rate that reads
the clock between events (richardposner/RuleMonkey#86), and it runs on absolute
model time (issue #1022). So does a table function indexed by time
(``type="TFUN"`` with a clock ``ctrName``), which keeps its clock in an
attribute while its ``<Expression>`` reads a placeholder. NFsim runs one, but
refreshes its value only when a reaction fires, so a rate that is 0 at the
start is not seen (issue #892), and it reads elapsed time, BNG2.pl's
simulate_nf semantics. That is left to the fix there, since refusing it would
take away a table function NFsim reads correctly whenever reactions fire often.

A RuleMonkey session uses :func:`clock_readers` to refuse a segment whose labels
are not the clock the model reads.

Issue #1058: RuleMonkey 3.11.0 gives an Arrhenius rule BNG2's rate,
exp(-(Ea0 + phi*dG)), where every energy pattern the rule's bond is part of is
that bonded pair of sites, alone or with one more bond to a third molecule
whose site binds nothing else, and the rule names only those two sites. It
expands each energy rule at load, and outside that it drops or misapplies the
context (richardposner/RuleMonkey#109), so the rule fires at the wrong rate
with no warning. :func:`refuse_rulemonkey_energy_misreads` refuses such a model
by name until the upstream fix is vendored.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from bngsim._exceptions import ModelError

# `time()` or a bare `time`: the engine binds the clock under both spellings.
_TIME_REF = re.compile(r"(?<![A-Za-z0-9_])time(?![A-Za-z0-9_])")

# The counter names RuleMonkey reads as the clock for a table function, checked
# before any parameter or observable of that name (resolve_tfun_counter in
# third_party/rulemonkey/cpp/rulemonkey/simulator.cpp).
_RULEMONKEY_CLOCK_COUNTERS = frozenset({"time", "t", "time()", "t()"})

_BACKENDS = {
    "nfsim": (
        "NFsim (method='nf' / 'nf_reject')",
        "NFsim cannot prepare a function that reads time() and fails at setup",
    ),
}


def _functions(xml_path: str | Path) -> list[ET.Element] | None:
    """Every ``<Function>`` element in *xml_path*, or ``None`` when the file
    cannot be parsed: the backend then reports its own parse error, which is
    more specific than anything these checks could say."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return None
    return [
        el
        for el in root.iter()
        if el.tag.endswith("Function") and not el.tag.endswith("ListOfFunctions")
    ]


def _reads_time(func: ET.Element) -> bool:
    return any(
        expr.tag.endswith("Expression") and _TIME_REF.search(expr.text or "") for expr in func
    )


def _indexed_by_time(func: ET.Element) -> bool:
    return (func.get("type") or "").lower() == "tfun" and (
        (func.get("ctrName") or "").strip() in _RULEMONKEY_CLOCK_COUNTERS
    )


def time_reading_functions(xml_path: str | Path) -> list[str]:
    """The ids of the functions in *xml_path* whose expression reads the clock,
    as ``time()`` or as a bare ``time``. Empty when the file cannot be parsed."""
    return [f.get("id", "?") for f in _functions(xml_path) or () if _reads_time(f)]


def time_indexed_table_functions(xml_path: str | Path) -> list[str]:
    """The ids of the table functions in *xml_path* whose counter RuleMonkey
    reads as the clock. Empty when the file cannot be parsed."""
    return [f.get("id", "?") for f in _functions(xml_path) or () if _indexed_by_time(f)]


def clock_readers(xml_path: str | Path) -> list[str]:
    """The ids of the functions in *xml_path* that read the clock, through
    ``time()``, a bare ``time``, or a table indexed by time. Empty when the file
    cannot be parsed."""
    return [
        f.get("id", "?")
        for f in _functions(xml_path) or ()
        if _reads_time(f) or _indexed_by_time(f)
    ]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(el: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in el if _local(c.tag) == name]


def _graph(el: ET.Element) -> tuple[list[ET.Element], dict[str, tuple[int, str, str]], list]:
    """The molecules of a pattern or species, its sites (component id ->
    molecule index, molecule type, component name) and its bonds (pairs of
    component ids)."""
    molecules = [m for lst in _children(el, "ListOfMolecules") for m in lst]
    sites = {}
    for i, mol in enumerate(molecules):
        for lst in _children(mol, "ListOfComponents"):
            for comp in lst:
                sites[comp.get("id") or ""] = (i, mol.get("name") or "", comp.get("name") or "")
    bonds = [
        (b.get("site1") or "", b.get("site2") or "")
        for lst in _children(el, "ListOfBonds")
        for b in lst
    ]
    return molecules, sites, bonds


def _components(mol: ET.Element) -> list[ET.Element]:
    return [c for lst in _children(mol, "ListOfComponents") for c in lst]


def _plain_bond_pattern(pattern: ET.Element) -> bool:
    """Whether a pattern names nothing but explicit bonds between its own
    molecules: every component bound once to a partner the pattern lists, with
    no state, no site named twice on one molecule, and no compartment."""
    molecules, sites, bonds = _graph(pattern)
    if not molecules or pattern.get("compartment"):
        return False
    ends = [site for bond in bonds for site in bond]
    for mol in molecules:
        comps = _components(mol)
        names = [c.get("name") for c in comps]
        if mol.get("compartment") or not comps or len(set(names)) != len(names):
            return False
        for comp in comps:
            if comp.get("state") is not None or comp.get("numberOfBonds") != "1":
                return False
            if ends.count(comp.get("id") or "") != 1:
                return False
    return all(a in sites and b in sites for a, b in bonds)


Site = tuple[str, str]  # (molecule type, component name)


def _bond_partners(root: ET.Element) -> dict[Site, set[Site]]:
    """For every site, the sites a bond in the model can join it to: the bonds
    of the seed species and every bond a rule adds."""
    partners: dict[Site, set[Site]] = {}

    def add(sites: dict, a: str, b: str) -> None:
        if a in sites and b in sites:
            sa, sb = sites[a][1:], sites[b][1:]
            partners.setdefault(sa, set()).add(sb)
            partners.setdefault(sb, set()).add(sa)

    for el in root.iter():
        if _local(el.tag) == "Species":
            _, sites, bonds = _graph(el)
            for a, b in bonds:
                add(sites, a, b)
        elif _local(el.tag) == "ReactionRule":
            sites = {}
            for lst in _children(el, "ListOfReactantPatterns"):
                for rp in lst:
                    sites.update(_graph(rp)[1])
            for lst in _children(el, "ListOfOperations"):
                for op in lst:
                    if _local(op.tag) == "AddBond":
                        add(sites, op.get("site1") or "", op.get("site2") or "")
    return partners


def _arrhenius_centers(root: ET.Element) -> tuple[list[tuple[Site, Site]], list[str]]:
    """The bond each Arrhenius rule makes or breaks, as a pair of sites, and
    the rules whose reactant patterns name anything besides those two sites.
    A rule that is not a single AddBond or DeleteBond between two molecule
    types is left to RuleMonkey, which refuses it itself."""
    centers: list[tuple[Site, Site]] = []
    with_context: list[str] = []
    for rule in root.iter():
        if _local(rule.tag) != "ReactionRule" or not any(
            (rl.get("type") or "").lower() == "arrhenius" for rl in _children(rule, "RateLaw")
        ):
            continue
        reactants = [rp for lst in _children(rule, "ListOfReactantPatterns") for rp in lst]
        molecules: list[ET.Element] = []
        sites: dict[str, tuple[int, str, str]] = {}
        for rp in reactants:
            mols, rp_sites, _ = _graph(rp)
            sites.update({k: (len(molecules) + v[0], v[1], v[2]) for k, v in rp_sites.items()})
            molecules += mols
        ops = [op for lst in _children(rule, "ListOfOperations") for op in lst]
        if len(ops) != 1 or _local(ops[0].tag) not in ("AddBond", "DeleteBond"):
            continue
        a, b = ops[0].get("site1") or "", ops[0].get("site2") or ""
        if a not in sites or b not in sites or sites[a][1] == sites[b][1]:
            continue
        centers.append((sites[a][1:], sites[b][1:]))
        bare = len(molecules) == 2 and not any(rp.get("compartment") for rp in reactants)
        for mol in molecules:
            comps = _components(mol)
            if (
                mol.get("compartment")
                or len(comps) != 1
                or comps[0].get("state") is not None
                or (comps[0].get("id") or "") not in (a, b)
            ):
                bare = False
        if not bare:
            with_context.append(rule.get("name") or rule.get("id") or "?")
    return centers, with_context


def _expanded_exactly(
    pattern: ET.Element, center: tuple[Site, Site], partners: dict[Site, set[Site]]
) -> bool:
    """Whether RuleMonkey's expansion of an energy rule with bond *center*
    reads the plain pattern *pattern* as BNG2 does: it is the center's two
    sites bonded, with at most one more bond, from one of them to a molecule of
    a third type that names only that site, which nothing else binds."""
    molecules, sites, bonds = _graph(pattern)
    kind = set(center)
    on_center = [bd for bd in bonds if {sites[bd[0]][1:], sites[bd[1]][1:]} == kind]
    if not on_center:
        return True  # it holds no bond the rule makes or breaks
    if len(on_center) > 1:
        return False
    core = {sites[on_center[0][0]][0], sites[on_center[0][1]][0]}
    others = [bd for bd in bonds if bd != on_center[0]]
    leaves = [i for i in range(len(molecules)) if i not in core]
    if not leaves:
        return not others
    if len(leaves) > 1 or len(others) != 1:
        return False
    a, b = others[0]
    leaf_end, core_end = (a, b) if sites[a][0] == leaves[0] else (b, a)
    if sites[leaf_end][0] != leaves[0] or sites[core_end][0] not in core:
        return False
    if sites[leaf_end][1] in {site[0] for site in center}:
        return False
    if len(_components(molecules[leaves[0]])) != 1:
        return False
    # RuleMonkey tests the core site as bound to anything (energy_expand.cpp).
    return partners.get(sites[core_end][1:], set()) <= {sites[leaf_end][1:]}


def rulemonkey_energy_misreads(xml_path: str | Path) -> tuple[list[str], list[str]]:
    """The energy patterns (as BNGL) and the Arrhenius rules (by name) of
    *xml_path* that RuleMonkey 3.11.0 expands differently from BNG2 (issue
    #1058). Both empty for a model with no Arrhenius rule, or when the
    file cannot be parsed."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return [], []
    centers, with_context = _arrhenius_centers(root)
    if not centers and not with_context:
        return [], []
    partners = _bond_partners(root)
    patterns = []
    for ep in root.iter():
        if _local(ep.tag) != "EnergyPattern":
            continue
        graphs = _children(ep, "Pattern") or [ep]
        if not all(
            _plain_bond_pattern(g) and all(_expanded_exactly(g, c, partners) for c in centers)
            for g in graphs
        ):
            patterns.append(ep.get("pattern") or ep.get("id") or "?")
    return patterns, with_context


def refuse_rulemonkey_energy_misreads(xml_path: str | Path) -> None:
    """Raise :class:`ModelError` if RuleMonkey would expand an Arrhenius rule of
    *xml_path* differently from BNG2 (issue #1058).

    RuleMonkey 3.11.0 drops or misapplies an energy pattern's context, and an
    Arrhenius rule's own reactant context, when it expands the rule at load
    (richardposner/RuleMonkey#109). Delete this check, its calls in
    ``Simulator`` and ``RuleMonkeySession``, and the helpers only it uses, once
    the upstream fix is vendored.
    """
    patterns, rules = rulemonkey_energy_misreads(xml_path)
    if not patterns and not rules:
        return
    parts = []
    if patterns:
        parts.append("energy patterns " + ", ".join(f"'{n}'" for n in patterns))
    if rules:
        parts.append("the reactant context of rules " + ", ".join(f"'{n}'" for n in rules))
    raise ModelError(
        "RuleMonkey (method='rm' / 'nf_exact') cannot run this Arrhenius model: it would "
        f"drop or misapply {' and '.join(parts)} when it expands the energy rules at load "
        "(richardposner/RuleMonkey#109), so they would fire at the wrong rate. It runs an "
        "energy pattern that is the bonded pair of sites a rule joins or parts, alone or "
        "with one more bond to a molecule of a third type whose site binds nothing else, "
        "on rules that name only those two sites. Simulate it with method='ode' or "
        "method='ssa' on the generated network (issue #1058)."
    )


def refuse_time_dependent_rate_laws(xml_path: str | Path, backend: str) -> None:
    """Raise :class:`ModelError` if *xml_path* has a function reading the clock.

    *backend* is ``"nfsim"``, the one network-free backend that cannot run one.
    """
    reads = time_reading_functions(xml_path)
    if not reads:
        return
    label, why = _BACKENDS[backend]
    raise ModelError(
        f"{label} cannot run a model whose functions read time(): "
        + ", ".join(f"'{n}'" for n in reads)
        + f". {why}. Simulate it with method='rm', or with method='ode' or "
        "method='ssa' on the generated network, which read time() as the model "
        "time (issue #862)."
    )
