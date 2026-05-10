"""Diagnose: are the LOAD-response and DISPLACEMENT-response factors
equal in linear Cathles theory, on the actual annulus geometry?

For a single mode h = A sin(kθ) on the annulus, Cathles half-space
analytics give:
  γ_displacement = u_n / displacement_amp = ρg / (2η|k|) (= 0.05 for k=10)
  γ_load = u_n / load_amp = same value (load amp = ρg × disp amp).

So the response factors should be EQUAL. We test this empirically:

  STEP A: deform the un-deformed mesh by δh × sin(kθ); measure
          u_n_displacement_response on the surface.
          → response factor = u_n_amp / δh_amp.

  STEP B: keep the mesh un-deformed; apply a load of amplitude
          δh × sin(kθ) (i.e. natural BC = -δh × sin(kθ) × r̂).
          → response factor = u_n_amp / δh_amp.

Compare. If different, the FSSA-style load equivalence is wrong on
this geometry, and that's the load failure root cause.
"""

import os
import sys
import numpy as np
import sympy

import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw


def fourier_amp(theta, dr, k):
    n = len(theta)
    dth = np.empty(n)
    dth[1:-1] = 0.5 * (theta[2:] - theta[:-2])
    dth[0]    = 0.5 * (theta[1] - (theta[-1] - 2*np.pi))
    dth[-1]   = 0.5 * ((theta[0] + 2*np.pi) - theta[-2])
    return float(np.sum(dr * np.sin(k * theta) * dth) / np.pi)


def main():
    res = 20
    mode = 10
    r_o, r_inner = 1.0, 0.5
    cellsize = 1.0 / res

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

    Vr = uw.discretisation.MeshVariable(
        "Vr_rt", mesh, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True)
    v = uw.discretisation.MeshVariable(
        "V_rt", mesh, vtype=uw.VarType.VECTOR,
        degree=2, continuous=True)
    p = uw.discretisation.MeshVariable(
        "P_rt", mesh, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True)
    delta_h_load = uw.discretisation.MeshVariable(
        "dh_rt", mesh, vtype=uw.VarType.SCALAR,
        degree=2, continuous=True)
    delta_h_load.data[:, 0] = 0.0

    diffuser = uw.systems.Poisson(mesh, Vr)
    diffuser.constitutive_model = uw.constitutive_models.DiffusionModel
    diffuser.constitutive_model.Parameters.diffusivity = 1.0
    diffuser.tolerance = 1.0e-3
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

    def fresh_stokes(use_load_bc):
        st = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        st.constitutive_model = uw.constitutive_models.ViscousFlowModel
        st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        st.penalty = 0.0
        st.bodyforce = -1.0 * unit_r
        st.tolerance = 1.0e-7
        st.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
        if use_load_bc:
            st.add_natural_bc(
                -delta_h_load.sym[0] * unit_r,
                mesh.boundaries.Upper.name)
        return st

    def sample_un():
        upos = mesh.X.coords[surf_idx]
        return np.asarray(uw.function.evaluate(
            v.sym.dot(unit_r), upos)).flatten()

    saved_X = mesh.X.coords.copy()
    Cathles_gamma = 1.0 / (2.0 * mode)
    print(f"Cathles γ_k for mode {mode}: {Cathles_gamma:.5f}")
    print()

    # Sweep amplitudes — check linearity AND compare load vs displacement
    amps = [0.005, 0.01, 0.025, 0.05, 0.10]

    print("=" * 100)
    print(f"Comparing displacement-response vs load-response for "
          f"sin({mode}θ) at varying amplitude")
    print("=" * 100)
    print(f"{'amp':>7s} | {'u_n disp':>11s} {'γ_disp':>9s} | "
          f"{'u_n load':>11s} {'γ_load':>9s} | "
          f"{'γ_disp/γ_load':>15s}")

    for amp in amps:
        # -- Displacement response --
        # CRITICAL: restore mesh to saved_X BEFORE solving the diffuser,
        # so the diffuser always operates on the same reference geometry.
        # (Earlier version solved on the previous iteration's deformed
        # mesh — cumulative drift artefact.)
        mesh._deform_mesh(saved_X)
        deform_fn = (r / r_o) * sympy.sin(mode * th) * amp
        diffuser._reset()
        diffuser.add_essential_bc(sympy.Matrix([deform_fn]),
                                  mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Lower.name)
        diffuser.solve()
        f = np.asarray(uw.function.evaluate(
            Vr.sym * unit_r, mesh.X.coords)).reshape(-1, 2)
        mesh._deform_mesh(saved_X + f)
        delta_h_load.data[:, 0] = 0.0
        st = fresh_stokes(use_load_bc=False)
        st.solve()
        u_n_disp = sample_un()
        amp_disp = fourier_amp(surf_th, u_n_disp, mode)
        gamma_disp = -amp_disp / amp
        # Restore
        mesh._deform_mesh(saved_X)

        # -- Load response --
        # Mesh un-deformed; apply natural BC with delta_h_load = amp × sin(kθ)
        delta_h_load.data[:, 0] = 0.0
        # Set surface DOFs of delta_h_load to amp × sin(mode × dh_surf_th)
        delta_h_load.data[dh_surf_idx, 0] = amp * np.sin(mode * dh_surf_th)
        st = fresh_stokes(use_load_bc=True)
        st.solve()
        u_n_load = sample_un()
        amp_load = fourier_amp(surf_th, u_n_load, mode)
        gamma_load = amp_load / amp  # load drives positive u_n if amp positive

        ratio = gamma_disp / gamma_load if abs(gamma_load) > 1e-12 else float('inf')
        print(f"{amp:>7.4f} | {amp_disp:>+11.5e} {gamma_disp:>9.5f} | "
              f"{amp_load:>+11.5e} {gamma_load:>9.5f} | "
              f"{ratio:>15.4f}")

    print()
    print("=" * 100)
    print(f"γ_load on the DEFORMED mesh — does the load response factor")
    print(f"depend on the surface-deformation amplitude?")
    print("=" * 100)
    print(f"{'h amp':>6s} {'load amp':>10s} | {'u_n eq (no-load)':>17s} "
          f"{'u_n with load':>14s} {'u_n_load only':>14s} {'γ_load (def)':>13s}")

    # Fixed small load amplitude — vary h_current amplitude.
    load_amp = 0.005
    for h_amp in [0.0, 0.005, 0.01, 0.025, 0.05, 0.10]:
        # Always restore mesh BEFORE diffuser solve (avoid cumulative drift)
        mesh._deform_mesh(saved_X)
        if h_amp > 0:
            deform_fn = (r / r_o) * sympy.sin(mode * th) * h_amp
            diffuser._reset()
            diffuser.add_essential_bc(sympy.Matrix([deform_fn]),
                                      mesh.boundaries.Upper.name)
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Lower.name)
            diffuser.solve()
            f = np.asarray(uw.function.evaluate(
                Vr.sym * unit_r, mesh.X.coords)).reshape(-1, 2)
            mesh._deform_mesh(saved_X + f)

        # 1: Stokes equilibrium at deformed mesh (no load)
        delta_h_load.data[:, 0] = 0.0
        st = fresh_stokes(use_load_bc=False)
        st.solve()
        u_n_eq = sample_un()
        amp_eq = fourier_amp(surf_th, u_n_eq, mode)

        # 2: Stokes with small load on the SAME deformed mesh
        delta_h_load.data[:, 0] = 0.0
        delta_h_load.data[dh_surf_idx, 0] = load_amp * np.sin(mode * dh_surf_th)
        st = fresh_stokes(use_load_bc=True)
        st.solve()
        u_n_load = sample_un()
        amp_loaded = fourier_amp(surf_th, u_n_load, mode)

        # The pure load contribution = u_n_loaded - u_n_eq (linearity)
        u_n_load_only = amp_loaded - amp_eq
        gamma_load_deformed = u_n_load_only / load_amp

        print(f"{h_amp:>6.4f} {load_amp:>10.4f} | "
              f"{amp_eq:>+17.5e} {amp_loaded:>+14.5e} {u_n_load_only:>+14.5e} "
              f"{gamma_load_deformed:>+13.5f}")

        mesh._deform_mesh(saved_X)

    print()
    print("Linear Cathles theory: γ_disp = γ_load = ρg/(2η|k|) = "
          f"{Cathles_gamma:.5f}")
    print()
    print("Interpretation:")
    print("  γ_disp = u_n on deformed mesh / displacement amp")
    print("  γ_load = u_n on un-deformed mesh / surface-load amp")
    print("  If equal (linear Cathles): the load equivalence holds.")
    print("  If γ_disp > γ_load (which we expect from the previous test):")
    print("    The load drives a SMALLER u_n response than an equivalent")
    print("    actual deformation. This is the source of load over-damping —")
    print("    the load can't 'undo' the equilibrium velocity as the")
    print("    linear FSSA derivation assumes.")


if __name__ == "__main__":
    main()
