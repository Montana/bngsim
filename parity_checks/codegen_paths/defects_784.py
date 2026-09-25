"""Does the model codegen path retire each #784 defect (and #608)?

Each reproduction from the issues runs through the interpreter, the ``.net``
codegen path and the model codegen path -- the same public ``Simulator``, with
``model._net_path`` cleared for the model path, the one switch ``Simulator``
reads -- and is checked against an oracle that shares no code with bngsim: a
closed form, or scipy's ``solve_ivp``. The ExprTk-spelling cases (#734) use the
interpreter, whose reading BNG2.pl's own parenthesisation confirms.

Since #803 step 3 that switch selects nothing -- every model compiles from the
built model -- so on a current bngsim the ``.net`` column repeats the model one.
Run it against a bngsim from before that step to see the two paths apart.

    python defects_784.py            # markdown table; exit 0 whatever it finds

Needs a C compiler (codegen) and scipy. No BNG2.pl.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="defects784-"))
os.environ["BNGSIM_CODEGEN_CACHE_DIR"] = str(WORK / "cg")  # cold; read at import
warnings.simplefilter("ignore")

import bngsim  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

ROWS: list[dict] = []
PATHS = ("interp", "net", "model")


def w(name: str, text: str) -> str:
    p = WORK / name
    p.write_text(text)
    return str(p)


def sim_run(net, path, *, sens=None, t_end=1.0, n=3, override=None, env=None, model=None):
    """One run on ``path`` ('interp' | 'net' | 'model'): (species, sensitivities)."""
    m = model if model is not None else bngsim.Model.from_net(net)
    for k, v in (override or {}).items():
        m.set_param(k, v)
    if path == "model":
        m._net_path = ""
    kw = {"sensitivity_params": sens} if sens else {"codegen": path != "interp"}
    old = {k: os.environ.get(k) for k in env or {}}
    os.environ.update(env or {})
    try:
        r = bngsim.Simulator(m, method="ode", **kw).run(
            t_span=(0.0, t_end), n_points=n, rtol=1e-10, atol=1e-12
        )
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return np.asarray(r.species), (np.asarray(r.sensitivities) if sens else None)


def record(case, issue, path, got, want, tol=1e-6):
    wv = np.atleast_1d(np.asarray(want, float))
    if isinstance(got, Exception):
        ok, shown = False, f"{type(got).__name__}: {str(got).splitlines()[0][:70]}"
    else:
        g = np.atleast_1d(np.asarray(got, float))
        ok = bool(np.allclose(g, wv, rtol=tol, atol=tol * max(1.0, float(np.max(np.abs(wv))))))
        shown = np.array2string(g, precision=6)
    ROWS.append(
        {
            "issue": issue,
            "case": case,
            "path": path,
            "ok": ok,
            "got": shown,
            "want": np.array2string(wv, precision=6),
        }
    )


def each_path(case, issue, fn, want, tol=1e-6, paths=PATHS):
    for path in paths:
        try:
            got = fn(path)
        except Exception as e:  # noqa: BLE001 - a refusal is an outcome
            got = e
        record(case, issue, path, got, want, tol)


def sens_paths(case, issue, fn, want, tol=1e-6):
    each_path(case, issue, fn, want, tol, paths=("net", "model"))


# ── #689: scientific-notation and leading-dot stat factors, Sat, Hill ─────────
LR = w(
    "lr_comp.net",
    """begin parameters
    1 NA     6.022e23  # Constant
    2 V      1e-15  # Constant
    3 kon    1e6  # Constant
    4 koff   0.01  # Constant
end parameters
begin species
    1 @C::L(r) 1000
    2 @C::R(l) 500
    3 @C::L(r!1).R(l!1) 0
end species
begin reactions
    1 1,2 3 1.6605779e-09*kon #_R1 unit_conversion=1/(V*NA)
    2 3 1,2 koff #_reverse__R1
end reactions
begin groups
    1 Bound                3
end groups
""",
)


def lr_scipy(kon=1e6, koff=0.01):
    def f(t, y):
        v = 1.6605779e-09 * kon * y[0] * y[1] - koff * y[2]
        return [-v, -v, v]

    return solve_ivp(f, (0, 1), [1000, 500, 0], method="LSODA", rtol=1e-12, atol=1e-12).y[2, -1]


b1 = lr_scipy()
dkon = (lr_scipy(kon=1e6 * (1 + 1e-5)) - lr_scipy(kon=1e6 * (1 - 1e-5))) / (2e6 * 1e-5)
dkoff = (lr_scipy(koff=0.01 * (1 + 1e-5)) - lr_scipy(koff=0.01 * (1 - 1e-5))) / (2 * 0.01 * 1e-5)
each_path("stat factor 1.6605779e-09*kon: Bound(1)", 689, lambda p: sim_run(LR, p)[0][-1, 2], b1)
sens_paths(
    "stat factor 1.6605779e-09*kon, sensitivity run: Bound(1), dBound/d(kon, koff)",
    689,
    lambda p: (lambda r: np.r_[r[0][-1, 2], r[1][-1, 2]])(sim_run(LR, p, sens=["kon", "koff"])),
    [b1, dkon, dkoff],
    tol=1e-5,
)
DECAY = (
    "begin parameters\n    1 k 1\nend parameters\nbegin species\n    1 A() 10\nend species\n"
    "begin reactions\n    1 1 0 RATE\nend reactions\nbegin groups\n    1 Atot 1\nend groups\n"
)
for tok, sf in ((".5*k", 0.5), ("5e-1*k", 0.5), ("5E+0*k", 5.0)):
    f = w(f"decay_{tok.replace('*', '_')}.net", DECAY.replace("RATE", tok))
    each_path(
        f"stat factor {tok}: A(1)", 689, lambda p, f=f: sim_run(f, p)[0][-1, 0], 10 * math.exp(-sf)
    )

SAT = w(
    "sat.net",
    "begin parameters\n    1 k3  1.52\n    2 K4  114.418\nend parameters\n"
    "begin species\n    1 S() 100\n    2 P() 0\nend species\nbegin reactions\n    1 1 2 Sat k3 K4\n"
    "end reactions\nbegin groups\n    1 Stot 1\nend groups\n",
)
HILL = w(
    "hill.net",
    "begin parameters\n    1 V   1.0\n    2 K   50.0\n    3 n   2.0\nend parameters\n"
    "begin species\n    1 S() 100\n    2 P() 0\nend species\nbegin reactions\n    1 1 2 Hill V K n\n"
    "end reactions\nbegin groups\n    1 Stot 1\nend groups\n",
)


def s10(dsdt, q):
    return solve_ivp(lambda t, y: [dsdt(y[0], q)], (0, 10), [100.0], rtol=1e-12, atol=1e-12).y[
        0, -1
    ]


for net, sp, p0, dsdt in (
    (SAT, "k3", [1.52, 114.418], lambda S, p: -p[0] * S / (p[1] + S)),
    (HILL, "V", [1.0, 50.0, 2.0], lambda S, p: -p[0] * S ** p[2] / (p[1] ** p[2] + S ** p[2])),
):
    ref = s10(dsdt, p0)
    dref = (s10(dsdt, [p0[0] + 1e-6, *p0[1:]]) - s10(dsdt, [p0[0] - 1e-6, *p0[1:]])) / 2e-6
    nm = Path(net).stem.capitalize()
    each_path(
        f"legacy {nm}: S(10)", 689, lambda p, net=net: sim_run(net, p, t_end=10)[0][-1, 0], ref
    )
    sens_paths(
        f"legacy {nm}, sensitivity run: S(10), dS/d{sp}",
        689,
        lambda p, net=net, sp=sp: (lambda r: np.r_[r[0][-1, 0], r[1][-1, 0, 0]])(
            sim_run(net, p, sens=[sp], t_end=10)
        ),
        [ref, dref],
        tol=1e-5,
    )

# ── #694: set_param override of a derived parameter ───────────────────────────
ELEM = w(
    "elem.net",
    "begin parameters\n    1 k1   1     # Constant\n    2 k2   2*k1  # ConstantExpression\n"
    "end parameters\nbegin species\n    1 A() 10\n    2 B() 0\nend species\n"
    "begin reactions\n    1 1 2 k2\nend reactions\n",
)
FUNC = w(
    "func.net",
    "begin parameters\n    1 k1   1     # Constant\n    2 k2   2*k1  # ConstantExpression\n"
    "end parameters\nbegin species\n    1 A() 10\n    2 B() 0\nend species\nbegin groups\n"
    "    1 Atot 1\nend groups\nbegin functions\n    1 frate() k2*Atot\nend functions\n"
    "begin reactions\n    1 1 2 frate\nend reactions\n",
)
sens_paths(
    "Elementary k2 = 2*k1 pinned to 5: A(1), dA/dk1",
    694,
    lambda p: (lambda r: np.r_[r[0][-1, 0], r[1][-1, 0, 0]])(
        sim_run(ELEM, p, sens=["k1"], override={"k2": 5.0})
    ),
    [10 * math.exp(-5), 0.0],
)
CROSS = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
import bngsim
net, mode, path = sys.argv[1:4]
m = bngsim.Model.from_net(net)
if mode == "override":
    m.set_param("k2", 5.0)
if path == "model":
    m._net_path = ""
r = bngsim.Simulator(m, method="ode", sensitivity_params=["k1"]).run(
    t_span=(0.0, 1.0), n_points=3, rtol=1e-10, atol=1e-12)
print("ROW" + json.dumps([float(r.species[-1, 0]), float(r.sensitivities[-1, 0, 0])]))
"""
# dA/dt = -k2 A^2: A = 10/(1 + 10 k2 t); pristine dA/dk1 = 2 dA/dk2; pinned k2: dA/dk1 = 0
want = {"pristine": [10 / 21, -200 / 21**2], "override": [10 / 51, 0.0]}
for path in ("net", "model"):
    for order in (("override", "pristine"), ("pristine", "override")):
        env = dict(os.environ, BNGSIM_CODEGEN_CACHE_DIR=str(WORK / f"cg694_{path}_{order[0]}"))
        for i, mode in enumerate(order):
            out = subprocess.run(
                [sys.executable, "-c", CROSS, FUNC, mode, path],
                capture_output=True,
                text=True,
                env=env,
            )
            rows = [line[3:] for line in out.stdout.splitlines() if line.startswith("ROW")]
            got = np.array(json.loads(rows[-1])) if rows else RuntimeError(out.stderr[-200:])
            record(
                f"Functional k2*Atot, one cache, {order[0]} first: run {i + 1} ({mode}) A(1), dA/dk1",
                694,
                path,
                got,
                want[mode],
            )

# ── #699: a function that reads one declared after it ─────────────────────────
FWD = w(
    "fwd.net",
    "begin parameters\n    1 k 2\nend parameters\nbegin species\n    1 A() 10\nend species\n"
    "begin reactions\n    1 1 0 y\nend reactions\nbegin groups\n    1 Atot 1\nend groups\n"
    "begin functions\n    1 y() x*3\n    2 x() k*0.5\nend functions\n",
)
tt = np.array([0.0, 0.5, 1.0])
for chunk in ("off", "on"):
    env = {"BNGSIM_CODEGEN_CHUNK": chunk}
    each_path(
        f"forward function reference, chunk={chunk}: A(0, .5, 1)",
        699,
        lambda p, env=env: sim_run(FWD, p, env=env)[0][:, 0],
        10 * np.exp(-3 * tt),
    )
    sens_paths(
        f"forward function reference, chunk={chunk}, sensitivity run: A, dA/dk",
        699,
        lambda p, env=env: (lambda r: np.r_[r[0][:, 0], r[1][:, 0, 0]])(
            sim_run(FWD, p, sens=["k"], env=env)
        ),
        np.r_[10 * np.exp(-3 * tt), -15 * tt * np.exp(-3 * tt)],
    )

# ── #721: tfun time index spelled T ───────────────────────────────────────────
TF = WORK / "tfT"
TF.mkdir()
(TF / "drive.tfun").write_text("# T  drv\n0       1.0\n1       2.0\n2       4.0\n5       10.0\n")
M721 = (
    "begin parameters\n    1 T   0.5\n    2 k   0.1\nend parameters\nbegin functions\n"
    "    1 drv()  tfun('drive.tfun', T)\nend functions\nbegin species\n    1 A() 1\n    2 B() 0\n"
    "end species\nbegin reactions\n    1 1 1,2 drv #R1\nend reactions\nbegin groups\n"
    "    1 A_tot  1\n    2 B_tot  2\nend groups\n"
)
(TF / "m.net").write_text(M721)
(TF / "m_noT.net").write_text(
    M721.replace("    1 T   0.5\n", "").replace("    2 k   0.1", "    1 k   0.1")
)
# B' = drv(t) * A with A = 1, so B(3) = the integral of the table over [0, 3] = 1.5 + 3 + 5
for fn in ("m.net", "m_noT.net"):
    net = str(TF / fn)
    label = "with a parameter named T" if fn == "m.net" else "no symbol named T"
    each_path(
        f"tfun index T, {label}: B(3)",
        721,
        lambda p, net=net: sim_run(net, p, t_end=3)[0][-1, 1],
        9.5,
    )
    sens_paths(
        f"tfun index T, {label}, sensitivity run: B(3)",
        721,
        lambda p, net=net: sim_run(net, p, t_end=3, sens=["k"])[0][-1, 1],
        9.5,
    )

# ── #730: the .net is rewritten between load and Simulator ────────────────────
V1 = (
    "begin parameters\n    1 k1 1\n    2 k2 5\nend parameters\nbegin species\n    1 X() 10\n    2 Y() 0\n"
    "end species\nbegin reactions\n    1 1 2 RATE #_R1\nend reactions\nbegin groups\n    1 Xtot 1\nend groups\n"
)
for path in ("net", "model"):
    for sens in (None, ["k1"]):
        p730 = w(f"stale_{path}_{bool(sens)}.net", V1.replace("RATE", "k1"))
        m = bngsim.Model.from_net(p730)
        Path(p730).write_text(V1.replace("RATE", "k2"))
        try:
            sp, se = sim_run(p730, path, sens=sens, model=m)
            got = np.r_[sp[-1, 0], se[-1, 0, 0]] if sens else sp[-1, 0]
        except Exception as e:  # noqa: BLE001
            got = e
        want730 = [10 * math.exp(-1), -10 * math.exp(-1)] if sens else 10 * math.exp(-1)
        record(
            f"file rewritten after load{', sensitivity run' if sens else ''}: X(1)"
            f"{', dX/dk1' if sens else ''}",
            730,
            path,
            got,
            want730,
        )

# ── #731: same-mtime rewrite within one process ───────────────────────────────
for path in ("net", "model"):
    v1 = w(
        f"v1_{path}.net",
        "begin parameters\n    1 k 1\nend parameters\nbegin species\n    1 A() 10\n"
        "    2 B() 0\n    3 C() 0\nend species\nbegin reactions\n    1 1 2 k\nend reactions\n",
    )
    v2 = w(f"v2_{path}.net", Path(v1).read_text().replace("1 1 2 k", "1 1 3 k"))
    st = os.stat(v1)
    os.utime(v2, ns=(st.st_atime_ns, st.st_mtime_ns))
    tgt = str(WORK / f"model731_{path}.net")
    shutil.copy2(v1, tgt)
    sim_run(tgt, path)
    shutil.copy2(v2, tgt)
    try:
        got = sim_run(tgt, path)[0][-1]
    except Exception as e:  # noqa: BLE001
        got = e
    record(
        "same-mtime rewrite A->B then A->C: (A, B, C)(1)",
        731,
        path,
        got,
        [10 * math.exp(-1), 0, 10 * (1 - math.exp(-1))],
    )

# ── #734: ExprTk-only spellings in function text ──────────────────────────────
OPS = """begin parameters
    1 k 3
    2 a 0.5
    3 b 2
end parameters
begin species
    1 $Src() 1
    2 S0() 0
    3 S1() 0
    4 S2() 0
    5 D() 0
end species
begin functions
    1 f0() F0
    2 f1() F1
    3 f2() F2
end functions
begin reactions
    1 1 1,2 f0
    2 1 1,3 f1
    3 1 1,4 f2
    4 1 1,5 k
end reactions
"""
# ExprTk reads a relational chain left to right at one level (BNG2.pl writes
# if(((a==b)<1),...) for the same text), '=' as equality, '--k' as -(-k).
for i, (label, f0, f1, f2, want734) in enumerate(
    (
        (
            "relational chain a==b<1, k!=b>1",
            "if(a==b<1,1,2)",
            "if(k!=b>1,1,2)",
            "b*k",
            [1, 2, 6, 3],
        ),
        ("single '=' as equality", "if(k=2,1,2)", "1", "b*k", [2, 1, 6, 3]),
        ("'--k' as double negation", "1", "1", "b*--k", [1, 1, 6, 3]),
    )
):
    f = w(f"ops_{i}.net", OPS.replace("F0", f0).replace("F1", f1).replace("F2", f2))
    each_path(
        f"{label}: (S0, S1, S2, D)(1)", 734, lambda p, f=f: sim_run(f, p)[0][-1, 1:], want734
    )
    sens_paths(
        f"{label}, sensitivity run: (S0, S1, S2, D)(1)",
        734,
        lambda p, f=f: sim_run(f, p, sens=["b"])[0][-1, 1:],
        want734,
    )

# ── #608: unindexed blocks (writeFile({format=>"net"})'s shape) ───────────────
UNIDX = w(
    "unindexed.net",
    "begin parameters\n    kcat 0.3\nend parameters\nbegin species\n    A() 100\n"
    "    B() 0\nend species\nbegin groups\n    Atot 1\nend groups\nbegin functions\n"
    "    1 fA() kcat*Atot\nend functions\nbegin reactions\n    1 1 2 fA\nend reactions\n",
)
# dA/dt = -kcat A^2: A = 100/(1 + 100 kcat t), dA/dkcat = -1e4 t/(1 + 100 kcat t)^2
each_path("unindexed blocks: A(1)", 608, lambda p: sim_run(UNIDX, p)[0][-1, 0], 100 / 31)
sens_paths(
    "unindexed blocks, sensitivity run: A(1), dA/dkcat",
    608,
    lambda p: (lambda r: np.r_[r[0][-1, 0], r[1][-1, 0, 0]])(sim_run(UNIDX, p, sens=["kcat"])),
    [100 / 31, -1e4 / 31**2],
)

# ── report ───────────────────────────────────────────────────────────────────
print("| issue | model path | .net path | interpreter |\n|---|---|---|---|")
for issue in sorted({r["issue"] for r in ROWS}):
    cells = []
    for path in ("model", "net", "interp"):
        rs = [r for r in ROWS if r["issue"] == issue and r["path"] == path]
        cells.append(f"{sum(r['ok'] for r in rs)}/{len(rs)}" if rs else "-")
    print(f"| #{issue} | " + " | ".join(cells) + " |")
print("\n| issue | path | | case | got | want |\n|---|---|---|---|---|---|")
for r in ROWS:
    print(
        f"| #{r['issue']} | {r['path']} | {'ok' if r['ok'] else '**FAIL**'} | {r['case']} | "
        f"`{r['got']}` | `{r['want']}` |"
    )
shutil.rmtree(WORK, ignore_errors=True)
