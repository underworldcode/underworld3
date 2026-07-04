"""Warm-start puzzle: on the adapted mesh, default GAMG Stokes
takes 2 SNES iters whether started cold (x=0) or warm (remapped
V,P from snapshot). Even with snes_atol = rtol·‖F0‖ both paths
remain at 2 iters. Why?

Controlled probe: feed the *exact converged* steady-state V,P
back into the solver as the warm guess. If SNES still does 2
iters, the problem isn't the guess quality — something deeper
about how SNES is consuming the convergence criteria.

Sequence:
  1. Cold solve from x=0 → V_steady, P_steady (the truth for
     this exact operator/RHS).
  2. Warm solve from V_steady → should converge in 0 SNES iters
     via snes_atol path. If it doesn't, we've isolated the bug.
  3. Verbose KSP+SNES monitor on a 4th solve to see the
     residual-drop trace.
"""
from __future__ import annotations
import os
import time
import numpy as np
import sympy

import underworld3 as uw


SRC = os.path.expanduser(
    '~/+Simulations/StagnantLid/adapted_R15_Ra1e7_dEta1e4')
STEM = "adapted"
Ra = 1.0e7
theta_FK = float(np.log(1.0e4))

mesh = uw.discretisation.Mesh(
    os.path.join(SRC, f"{STEM}.mesh.00000.h5"))
X = mesh.CoordinateSystem.X
r_sym = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
unit_r = mesh.CoordinateSystem.unit_e_0

T = uw.discretisation.MeshVariable(
    "T_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=3,
    continuous=True)
V = uw.discretisation.MeshVariable(
    "V_v2p1", mesh, vtype=uw.VarType.VECTOR, degree=2,
    continuous=True)
P = uw.discretisation.MeshVariable(
    "P_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=1,
    continuous=True)
T.read_timestep(STEM, "T_v2p1", 0, outputPath=SRC)
V.read_timestep(STEM, "V_v2p1", 0, outputPath=SRC)
P.read_timestep(STEM, "P_v2p1", 0, outputPath=SRC)
V_remap = np.asarray(V.data).copy()
P_remap = np.asarray(P.data).copy()


def make_stokes(verbose=False):
    s = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = (
        sympy.exp(theta_FK * (1 - T.sym[0])))
    s.tolerance = 1.0e-5
    s.penalty = 0.0
    s.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    KFS = 1.0e6
    fs_term = (KFS * V.sym.dot(unit_r) * unit_r)
    s.add_natural_bc(fs_term, mesh.boundaries.Upper.name)
    T_cond = sympy.log(r_sym / 1.0) / sympy.log(0.5 / 1.0)
    s.bodyforce = Ra * (T.sym[0] - T_cond) * unit_r
    if verbose:
        s.petsc_options["snes_monitor"] = None
        s.petsc_options["ksp_monitor"] = None
        s.petsc_options["snes_converged_reason"] = None
    return s


def fnorm_at(stokes):
    """Read ‖F‖ history after a solve."""
    try:
        rh, _ = stokes.snes.getConvergenceHistory()
        return list(rh) if rh is not None else []
    except Exception as e:
        return [f"err: {e}"]


def run(label, x_init_zero, V_init=None, P_init=None,
        snes_atol=None, verbose=False):
    """Standalone solve from given IC; report reason, iters,
    wall, and full ‖F‖ history."""
    if x_init_zero:
        V.data[...] = 0.0
        P.data[...] = 0.0
    else:
        V.data[...] = V_init
        P.data[...] = P_init
    s = make_stokes(verbose=verbose)
    if snes_atol is not None:
        s.petsc_options["snes_atol"] = float(snes_atol)
    t0 = time.time()
    try:
        s.solve(zero_init_guess=x_init_zero)
        s.snes.setConvergenceHistory(reset=True)
        # Re-solve to capture history (matches catalogue pattern)
        if x_init_zero:
            V.data[...] = 0.0
            P.data[...] = 0.0
        else:
            V.data[...] = V_init
            P.data[...] = P_init
        s.solve(zero_init_guess=x_init_zero)
    except Exception as e:
        print(f"  [{label}] EXC {e!r}")
        return None
    wall = time.time() - t0
    reason = int(s.snes.getConvergedReason())
    its = int(s.snes.getIterationNumber())
    rh = fnorm_at(s)
    vmax = float(np.sqrt(V.data[:, 0] ** 2
                         + V.data[:, 1] ** 2).max())
    print(f"  [{label}]  reason={reason}  its={its}  "
          f"wall={wall:.2f}s  |v|max={vmax:.3e}")
    print(f"    ‖F‖ history: " + ", ".join(
        f"{x:.3e}" if isinstance(x, float) else str(x)
        for x in rh))
    return s, vmax


print("=" * 72)
print("WARM-PUZZLE PROBE on adapted mesh, Ra=1e7, Δη=1e4")
print("=" * 72)

# Step 1: cold solve to get the TRUTH (V_steady, P_steady)
print("\n[1] Cold solve (x=0) → capture V_steady, P_steady")
s1, vmax1 = run("cold-from-zero", x_init_zero=True)
V_steady = np.asarray(V.data).copy()
P_steady = np.asarray(P.data).copy()

# Step 2: warm from the EXACT solution we just found
print("\n[2] Warm from V_steady (exact converged guess)")
run("warm-from-exact-steady",
    x_init_zero=False, V_init=V_steady, P_init=P_steady)

# Step 3: warm from snapshot remap (what the sweep did)
print("\n[3] Warm from snapshot remap (V_remap, P_remap)")
run("warm-from-remap", x_init_zero=False,
    V_init=V_remap, P_init=P_remap)

# Step 4: warm-from-steady WITH snes_atol set
print("\n[4] Same as [2] but with snes_atol = rtol·‖F0‖ "
      "(catalogue gated fix)")
# F0 from step 1's history
try:
    rh1, _ = s1.snes.getConvergenceHistory()
    F0 = float(rh1[0])
    atol = 1.0e-5 * F0
    print(f"    F0 from cold = {F0:.4e}  ⇒  atol = {atol:.4e}")
except Exception:
    F0 = None
    atol = None
if atol is not None:
    run("warm-steady+atol", x_init_zero=False,
        V_init=V_steady, P_init=P_steady, snes_atol=atol)

# Step 5: verbose monitor on warm-from-remap
print("\n[5] Verbose KSP/SNES monitor on warm-from-remap "
      "(snes_atol set)")
run("verbose-warm-remap", x_init_zero=False,
    V_init=V_remap, P_init=P_remap,
    snes_atol=atol, verbose=True)
