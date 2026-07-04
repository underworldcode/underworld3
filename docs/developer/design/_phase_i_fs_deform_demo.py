"""Minimal deform()-native free-surface convection — validates the foolproof
mesh-coordinate-mutation path (Mesh.deform + the SemiLagrangian DDt's on_remesh
ALE) end to end, with NO hand-rolled v_mesh.

Each step:
  1. stokes.solve()                         — V on the current mesh
  2. surface radial increment u_n·dt, harmonic-extended by the diffuser → X_new
     (computed as an array; the diffuser solve does NOT move the mesh)
  3. mesh.deform(X_new, dt=dt)              — the ONE sanctioned move: REMAP user
     fields + on_remesh ALE for the SL history stack (carry psi_star + v_mesh pulse)
  4. stokes.solve()                         — re-solve V on the new mesh
  5. adv_diff.solve(dt)                     — DDt applies the v_mesh pulse

Compare Nu(t)/T-bounds to the free-slip ground truth in
~/+Simulations/fs_convection_goal4/freeslip_worklog.csv (Nu ignites to ~49).

    python _phase_i_fs_deform_demo.py [--t-carry] [--n-steps N] [--res R]
"""
import os
import sys
import importlib.util
import numpy as np
import sympy
import underworld3 as uw

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "fs_zoo", os.path.join(HERE, "_phase_i_fs_convection_zoo.py"))
zoo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zoo)

t_carry = "--t-carry" in sys.argv
def _arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
n_steps = _arg("--n-steps", 30)
res = _arg("--res", 20)
Ra = _arg("--Ra", 1.0e5)

state = zoo._build(res=res, Ra=Ra, rho_g=Ra, t_degree=3, stokes_tol=1.0e-5,
                   nitsche_penalty=0.0)
mesh = state['mesh']; v = state['v']; t_soln = state['t_soln']
Vr = state['Vr']; diffuser = state['diffuser']; stokes = state['stokes']
adv_diff = state['adv_diff']; r_o = state['r_o']
internal_idx = state['internal_idx']; internal_th = state['internal_th']
th = state['th_sym']
unit_r = mesh.CoordinateSystem.unit_e_0

adv_diff.DuDt.theta = 0.5; adv_diff.DFDt.theta = 0.5
adv_diff.DuDt.monotone_mode = "clamp"; adv_diff.DFDt.monotone_mode = "clamp"

from underworld3.discretisation.remesh import RemeshPolicy
if t_carry:
    # Hypothesis: for smooth ALE the advected field should ride the mesh
    # (CARRY) so the on_remesh v_mesh pulse compensates it consistently with
    # its psi_star history — instead of being REMAP'd (double-handled).
    t_soln.remesh_policy = RemeshPolicy.CARRY
print(f"=== deform-demo: res={res} Ra={Ra:g} n_steps={n_steps} "
      f"t_soln.remesh_policy={getattr(t_soln,'remesh_policy', 'REMAP(default)')} ===",
      flush=True)


def trap_w(thw):
    n = len(thw); d = np.empty(n)
    d[1:-1] = 0.5 * (thw[2:] - thw[:-2])
    d[0] = 0.5 * (thw[1] - (thw[-1] - 2 * np.pi))
    d[-1] = 0.5 * ((thw[0] + 2 * np.pi) - thw[-2])
    return d


def fourier_decomp(values, n_modes):
    dthw = trap_w(internal_th)
    a = np.zeros(n_modes + 1); b = np.zeros(n_modes + 1)
    a[0] = float(np.sum(values * dthw) / (2 * np.pi))
    for m in range(1, n_modes + 1):
        a[m] = float(np.sum(values * np.cos(m * internal_th) * dthw) / np.pi)
        b[m] = float(np.sum(values * np.sin(m * internal_th) * dthw) / np.pi)
    return a, b


def fourier_to_sympy(a, b, theta_sym):
    expr = sympy.Float(a[0])
    for m in range(1, len(a)):
        if abs(a[m]) > 1e-12:
            expr = expr + a[m] * sympy.cos(m * theta_sym)
        if abs(b[m]) > 1e-12:
            expr = expr + b[m] * sympy.sin(m * theta_sym)
    return expr


def _nu():
    gT_n = (t_soln.sym[0].diff(mesh.X[0]) * unit_r[0]
            + t_soln.sym[0].diff(mesh.X[1]) * unit_r[1])
    return -float(uw.maths.BdIntegral(
        mesh, gT_n, mesh.boundaries.Upper.name).evaluate())


t_sim = 0.0
for s in range(n_steps):
    stokes.solve(zero_init_guess=False)
    dt = 0.5 * float(stokes.estimate_dt())
    dt = min(dt, float(adv_diff.estimate_dt()))

    coords = np.asarray(mesh.X.coords)
    u_n = np.asarray(uw.function.evaluate(
        v.sym.dot(unit_r), coords[internal_idx])).flatten()
    inc = u_n * dt
    n_modes = max(2, len(internal_th) // 3)
    a, b = fourier_decomp(inc, n_modes)
    inc_fn = fourier_to_sympy(a, b, th)
    diffuser._reset()
    diffuser.add_essential_bc(sympy.Matrix([inc_fn]), mesh.boundaries.Upper.name)
    diffuser.add_essential_bc(sympy.Matrix([0.0]), mesh.boundaries.Lower.name)
    diffuser.solve(zero_init_guess=False)
    f = np.asarray(uw.function.evaluate(
        Vr.sym * unit_r, coords)).reshape(-1, 2)
    X_new = coords + f

    # THE one sanctioned move: transfers T + SL history (ALE) coherently.
    mesh.deform(X_new, dt=dt)

    stokes.solve(zero_init_guess=False)
    adv_diff.solve(timestep=dt, zero_init_guess=False)
    t_sim += dt

    Tmin = float(t_soln.data[:, 0].min()); Tmax = float(t_soln.data[:, 0].max())
    hmax = float(np.abs(np.sqrt((mesh.X.coords[internal_idx] ** 2).sum(1)) - r_o).max())
    print(f"s{s+1:2d} t={t_sim:.4f} Nu={_nu():+.2f} T=[{Tmin:+.3f},{Tmax:+.3f}] "
          f"h_max={hmax:.3e}", flush=True)
