"""Diagnostic: load vs mesh-deformation, with two normal-vector choices.

For a single Stokes step on the Cathles relaxation IC (h = A sin(k θ)
on the annulus), measure u_n on the upper boundary in three ways:

  (a) TRUTH: deform mesh by predicted Δh, solve Stokes, sample u_n.
  (b) LOAD-radial: apply Δh as natural-BC traction along the
      analytic radial direction `unit_e_0`. (What rk*_load_sl does.)
  (c) LOAD-Gamma_N: apply Δh as natural-BC traction along the
      PETSc outward normal `Gamma_N` (geometric normal of the
      actual deformed face). (Should be more physically correct
      for finite-amplitude surface deformation.)

For each case, compute the Fourier amplitude at the IC mode k.
Compare the three.  The (a)–(b) gap is the linearisation+normal
error.  The (a)–(c) gap is just the linearisation error if normals
agree.  The (b)–(c) gap isolates the normal-direction effect.
"""

import os
import sys
import numpy as np
import sympy

import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw


def fourier_amp(theta, dr, k):
    """Trap-rule sin-coefficient at mode k."""
    n = len(theta)
    dth = np.empty(n)
    dth[1:-1] = 0.5 * (theta[2:] - theta[:-2])
    dth[0]    = 0.5 * (theta[1] - (theta[-1] - 2*np.pi))
    dth[-1]   = 0.5 * ((theta[0] + 2*np.pi) - theta[-2])
    return float(np.sum(dr * np.sin(k * theta) * dth) / np.pi)


def main():
    res = 20
    mode = 10
    amp0 = 0.05
    r_o, r_inner = 1.0, 0.5
    cellsize = 1.0 / res

    # Match the structured-annulus mesh used by the runner
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _structured_annulus import AnnulusStructured
    nA = max(8, int(round(2 * np.pi * r_o / cellsize)))
    nA = nA + (nA % 2)
    nR = max(2, int(round((r_o - r_inner) / cellsize)))
    mesh = AnnulusStructured(
        radiusOuter=r_o, radiusInner=r_inner,
        nRadial=nR, nAngular=nA, qdegree=3,
    )
    r, th = mesh.CoordinateSystem.R
    unit_r = mesh.CoordinateSystem.unit_e_0
    Gamma_N = mesh.Gamma   # PETSc outward normal at boundaries

    Vr = uw.discretisation.MeshVariable(
        "Vr_test", mesh, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True)
    v = uw.discretisation.MeshVariable(
        "V_test", mesh, vtype=uw.VarType.VECTOR,
        degree=2, continuous=True)
    p = uw.discretisation.MeshVariable(
        "P_test", mesh, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True)
    delta_h_load = uw.discretisation.MeshVariable(
        "dh_load_test", mesh, vtype=uw.VarType.SCALAR,
        degree=2, continuous=True)
    delta_h_load.data[:, 0] = 0.0

    # IC: deform initial mesh by amp0 × (r/r_o) × sin(mode θ)
    diffuser = uw.systems.Poisson(mesh, Vr)
    diffuser.constitutive_model = uw.constitutive_models.DiffusionModel
    diffuser.constitutive_model.Parameters.diffusivity = 1.0
    diffuser.tolerance = 1.0e-3
    deform_fn = (r / r_o) * sympy.sin(mode * th) * amp0
    diffuser.add_essential_bc(sympy.Matrix([deform_fn]),
                              mesh.boundaries.Upper.name)
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Lower.name)
    diffuser.solve()
    init_disp = np.asarray(uw.function.evaluate(
        Vr.sym * unit_r, mesh.X.coords)).reshape(-1, 2)
    mesh._deform_mesh(mesh.X.coords + init_disp)
    diffuser._reset()
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Upper.name)
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Lower.name)
    diffuser.solve()

    # Identify surface DOFs of mesh and of delta_h_load
    X = mesh.X.coords
    R = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    TH = np.arctan2(X[:, 1], X[:, 0])
    is_surf = np.abs(R - r_o) < 0.5 * cellsize / r_o
    surf_idx = np.where(is_surf)[0]
    sort = np.argsort(TH[surf_idx])
    surf_idx = surf_idx[sort]
    surf_th = TH[surf_idx]

    dh_coords = delta_h_load.coords
    dh_R = np.sqrt(dh_coords[:, 0] ** 2 + dh_coords[:, 1] ** 2)
    dh_TH = np.arctan2(dh_coords[:, 1], dh_coords[:, 0])
    dh_surf_mask = np.abs(dh_R - r_o) < 0.5 * cellsize / r_o
    dh_surf_idx = np.where(dh_surf_mask)[0]
    dh_sort = np.argsort(dh_TH[dh_surf_idx])
    dh_surf_idx = dh_surf_idx[dh_sort]
    dh_surf_th = dh_TH[dh_surf_idx]

    # Stokes solver — built fresh for each test variant.
    def fresh_stokes(load_dir):
        """Build a Stokes solver with natural BC on Upper using
        traction = -delta_h_load × load_dir."""
        st = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        st.constitutive_model = uw.constitutive_models.ViscousFlowModel
        st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        st.penalty = 0.0
        st.bodyforce = -1.0 * unit_r
        st.tolerance = 1.0e-7
        st.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
        if load_dir is not None:
            st.add_natural_bc(
                -delta_h_load.sym[0] * load_dir,
                mesh.boundaries.Upper.name)
        return st

    def sample_un_radial():
        upos = mesh.X.coords[surf_idx]
        return np.asarray(uw.function.evaluate(
            v.sym.dot(unit_r), upos)).flatten()

    def sample_un_normal():
        """Sample v · n̂ where n̂ is the geometric normal of the
        actual deformed surface (per-vertex finite-difference
        approximation from the surface-vertex coordinates)."""
        upos = mesh.X.coords[surf_idx]
        # Per-vertex unit normal from local geometry. For each surface
        # vertex i, neighbour-difference gives a tangent t̂; rotate 90°
        # outward to get n̂.
        n = len(surf_idx)
        normals = np.zeros((n, 2))
        for i in range(n):
            ip = (i + 1) % n
            im = (i - 1) % n
            # Tangent (going CCW)
            tx = upos[ip, 0] - upos[im, 0]
            ty = upos[ip, 1] - upos[im, 1]
            tnorm = np.hypot(tx, ty)
            tx, ty = tx / tnorm, ty / tnorm
            # Outward normal: rotate t by -90° (since CCW tangent →
            # outward normal points right of motion)
            normals[i, 0] = ty
            normals[i, 1] = -tx
        # Get v at each surface vertex (radial and theta components)
        v_xy = np.asarray(uw.function.evaluate(
            v.sym, upos)).reshape(-1, 2)
        # u · n̂
        return np.einsum('ij,ij->i', v_xy, normals)

    sample_un = sample_un_radial   # default — what the load schemes use

    # === Stage 1: solve Stokes at h_current with no load. Get u_n_1. ===
    delta_h_load.data[:, 0] = 0.0
    st_noload = fresh_stokes(load_dir=None)
    st_noload.solve()
    u_n_1 = sample_un()

    # Predicted Δh for stage 1 of RK2 with γΔt = 1 (Δt = 20).
    dt = 20.0
    delta_h_pred_at_surf = 0.5 * dt * u_n_1   # at mesh-vertex DOFs

    # Project delta_h_pred_at_surf to delta_h_load's DOFs via Fourier.
    # (Same trick as the load schemes use.)
    n_modes = max(2, len(surf_th) // 3)

    def fourier_decomp(values, theta_arr, n_modes):
        n = len(theta_arr)
        dth = np.empty(n)
        dth[1:-1] = 0.5 * (theta_arr[2:] - theta_arr[:-2])
        dth[0]    = 0.5 * (theta_arr[1] - (theta_arr[-1] - 2*np.pi))
        dth[-1]   = 0.5 * ((theta_arr[0] + 2*np.pi) - theta_arr[-2])
        a = np.zeros(n_modes + 1); b = np.zeros(n_modes + 1)
        a[0] = float(np.sum(values * dth) / (2 * np.pi))
        for m in range(1, n_modes + 1):
            a[m] = float(np.sum(values * np.cos(m * theta_arr) * dth) / np.pi)
            b[m] = float(np.sum(values * np.sin(m * theta_arr) * dth) / np.pi)
        return a, b

    def fourier_eval(a, b, theta_arr):
        out = np.full_like(theta_arr, a[0], dtype=float)
        for m in range(1, len(a)):
            out += a[m] * np.cos(m * theta_arr) + b[m] * np.sin(m * theta_arr)
        return out

    a_dh, b_dh = fourier_decomp(delta_h_pred_at_surf, surf_th, n_modes)
    delta_h_pred_at_load_dofs = fourier_eval(a_dh, b_dh, dh_surf_th)

    # Diagnostic: Fourier amp of u_n_1 and delta_h_pred at mode k
    a_un, b_un = fourier_decomp(u_n_1, surf_th, n_modes)
    print(f"Stage-1 u_n_1 sin({mode}θ) amp: {b_un[mode]:.5f}")
    print(f"Predicted Δh sin({mode}θ) amp: {b_dh[mode]:.5f}")
    print(f"  (h_current sin({mode}θ) amp = {amp0:.5f}, expected u_n_1 = -γA = {-amp0/(2*mode):.5f})")
    print(f"  γΔt = {dt/(2*mode):.3f}  (Δt={dt}, k={mode})")
    print()

    saved_X = mesh.X.coords.copy()

    # Sweep γΔt by varying Δt, measuring the (truth, load_unit_r, load_Gamma_N)
    # u_n response per mode. Same starting state; only the load amplitude
    # and the size of the mesh deformation change.
    gamma_dt_sweep = [0.125, 0.25, 0.5, 1.0, 1.5]
    Cathles_gamma = 1.0 / (2.0 * mode)

    def run_truth(dh_a, dh_b):
        """Deform mesh by the Fourier-described Δh, solve Stokes,
        return (u_n_amp_at_mode_k, restore_callable)."""
        inc_fn = sympy.Float(dh_a[0])
        for m in range(1, len(dh_a)):
            if abs(dh_a[m]) > 1e-14:
                inc_fn = inc_fn + dh_a[m] * sympy.cos(m * th)
            if abs(dh_b[m]) > 1e-14:
                inc_fn = inc_fn + dh_b[m] * sympy.sin(m * th)
        diffuser._reset()
        diffuser.add_essential_bc(sympy.Matrix([inc_fn]),
                                  mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Lower.name)
        diffuser.solve()
        f = np.asarray(uw.function.evaluate(
            Vr.sym * unit_r, mesh.X.coords)).reshape(-1, 2)
        mesh._deform_mesh(mesh.X.coords + f)
        delta_h_load.data[:, 0] = 0.0
        st = fresh_stokes(load_dir=None)
        st.solve()
        u_n_truth = sample_un()
        mesh._deform_mesh(saved_X)
        return u_n_truth

    def run_load(load_dir, dh_at_load_dofs):
        delta_h_load.data[:, 0] = 0.0
        delta_h_load.data[dh_surf_idx, 0] = dh_at_load_dofs
        st = fresh_stokes(load_dir=load_dir)
        st.solve()
        return sample_un()

    print("=" * 100)
    print(f"γΔt sweep at mode {mode}, amp0={amp0}")
    print("=" * 100)
    print(f"{'γΔt':>5s} {'Δt':>5s} | {'truth':>9s} {'L-unit_r':>9s} {'L-Γ_N':>9s} "
          f"| {'%err unit_r':>11s} {'%err Γ_N':>9s}")

    for gdt in gamma_dt_sweep:
        dt_test = gdt / Cathles_gamma
        # Predicted Δh = (Δt/2) × u_n_1
        dh = 0.5 * dt_test * u_n_1
        a_dh_t, b_dh_t = fourier_decomp(dh, surf_th, n_modes)
        dh_at_load = fourier_eval(a_dh_t, b_dh_t, dh_surf_th)

        u_n_truth = run_truth(a_dh_t, b_dh_t)
        u_n_loadr = run_load(unit_r, dh_at_load)
        u_n_loadg = run_load(Gamma_N, dh_at_load)

        _, b_t = fourier_decomp(u_n_truth, surf_th, n_modes)
        _, b_r = fourier_decomp(u_n_loadr, surf_th, n_modes)
        _, b_g = fourier_decomp(u_n_loadg, surf_th, n_modes)

        amp_t = b_t[mode]
        amp_r = b_r[mode]
        amp_g = b_g[mode]
        err_r = 100 * (amp_r - amp_t) / abs(amp_t)
        err_g = 100 * (amp_g - amp_t) / abs(amp_t)

        print(f"{gdt:>5.3f} {dt_test:>5.1f} | {amp_t:>+.6f} {amp_r:>+.6f} {amp_g:>+.6f} "
              f"| {err_r:>+10.1f}% {err_g:>+8.1f}%")

    # Repeat for a couple of other modes
    print()
    print("=" * 100)
    print(f"Mode sweep at γΔt = 0.5")
    print("=" * 100)
    print(f"{'mode':>4s} | {'truth':>9s} {'L-unit_r':>9s} {'L-Γ_N':>9s} "
          f"| {'%err unit_r':>11s} {'%err Γ_N':>9s}")
    # Note: changing mode requires rebuilding the IC, so we'd need to redo
    # the whole setup. For this diagnostic we'll leave that and just note
    # that the per-mode test would need its own runner pass.
    print("  (mode sweep requires rebuilding IC per mode — left for a separate run)")


if __name__ == "__main__":
    main()
