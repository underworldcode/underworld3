"""Probe: does MMPDE interior cleaning stabilize the per-step held-lid
dynamic-topography measurement on the deforming surface?

Earlier the per-step held-lid h_eq flipped sign and grew (-0.13) over a ~1.5%
deflection, even with a fixed Eulerian-blob body force — pointing at
mesh distortion from the diffuser deformation, not physics. Test: relax the
surface RADIALLY (pole stays radial, so û=r̂ — isolates the distortion effect
from the deformed-normal/velocity question) and re-measure the held-lid h_eq
at the pole each step, WITH vs WITHOUT an MMPDE interior smooth
(smooth_mesh_interior, surface pinned). If the clean interior keeps h_eq
stable (≈0.0227 target, decaying toward 0 as h→equilibrium with the full
body force), the mesh mover is the fix.
"""

import numpy as np
import sympy
import underworld3 as uw

import nest_asyncio
nest_asyncio.apply()

res = 16
r_i, r_o = 0.5, 1.0
cellsize = 1.0 / res
x_b, y_b, sigma_b, blob_amp = 0.7, 0.0, 0.08, 0.6


def build():
    mesh = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i,
                              cellSize=cellsize, qdegree=3)
    unit_r = mesh.CoordinateSystem.unit_e_0
    v = uw.discretisation.MeshVariable("V", mesh, vtype=uw.VarType.VECTOR,
                                       degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("P", mesh, vtype=uw.VarType.SCALAR,
                                       degree=1, continuous=True)
    rho_var = uw.discretisation.MeshVariable("rho", mesh,
                                             vtype=uw.VarType.SCALAR,
                                             degree=0, continuous=False)
    topo = uw.discretisation.MeshVariable("topo", mesh,
                                          vtype=uw.VarType.SCALAR, degree=1,
                                          continuous=True)
    cen = mesh._centroids
    blob = blob_amp * np.exp(-((cen[:, 0] - x_b) ** 2 + (cen[:, 1] - y_b) ** 2)
                             / (2.0 * sigma_b ** 2))
    rho_var.data[:, 0] = 1.0 - blob
    r_node = sympy.sqrt(mesh.X[0] ** 2 + mesh.X[1] ** 2)
    rho_ref = sympy.Piecewise((1.0, r_node < r_o), (0.0, True))
    bf_full = -(rho_var.sym[0] - rho_ref) * unit_r
    bf_blob = blob_amp * sympy.exp(
        -((mesh.X[0] - x_b) ** 2 + (mesh.X[1] - y_b) ** 2)
        / (2.0 * sigma_b ** 2)) * unit_r
    return mesh, unit_r, v, p, rho_var, topo, bf_full, bf_blob


def upper_index(mesh):
    X0 = mesh.X.coords
    R0 = np.sqrt((X0 ** 2).sum(1))
    TH = np.arctan2(X0[:, 1], X0[:, 0])
    up = np.where(R0 > r_o - 0.5 * cellsize / r_o)[0]
    up = up[np.argsort(TH[up])]
    return up, TH[up], int(np.argmin(np.abs(TH[up])))


def held_heq_pole(mesh, unit_r, topo, bodyforce, up, ip, kid):
    """Fresh held-lid Nitsche solve; projected n·σ·n at pole → h_eq."""
    vh = uw.discretisation.MeshVariable(f"Vh{kid}", mesh,
                                        vtype=uw.VarType.VECTOR, degree=2)
    ph = uw.discretisation.MeshVariable(f"Ph{kid}", mesh,
                                        vtype=uw.VarType.SCALAR, degree=1)
    sh = uw.systems.Stokes(mesh, velocityField=vh, pressureField=ph)
    sh.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sh.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    sh.penalty = 1.0
    sh.bodyforce = bodyforce
    sh.tolerance = 1.0e-6
    sh.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    sh.add_nitsche_bc(mesh.boundaries.Upper.name, gamma=10.0)
    sh.solve()
    pr = uw.systems.Projection(mesh, topo)
    pr.uw_function = (unit_r * sh.stress * unit_r.T)[0, 0]
    pr.smoothing = 0.0
    pr.solve()
    sig = np.asarray(uw.function.evaluate(
        topo.sym[0], mesh.X.coords[up])).flatten()
    heq = -sig            # h_eq = -σ_rr/(Δρg)
    return heq[ip], heq[ip] - heq.mean()    # absolute, mean-relative


def run(with_smooth, label):
    mesh, unit_r, v, p, rho_var, topo, bf_full, bf_blob = build()
    up, uth, ip = upper_index(mesh)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.penalty = 1.0
    stokes.bodyforce = bf_full
    stokes.tolerance = 1.0e-6
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    stokes.solve()
    dt = 0.5 * stokes.estimate_dt()

    def hpole():
        pos = mesh.X.coords[up[ip]]
        return float(np.sqrt(pos[0] ** 2 + pos[1] ** 2) - r_o)

    print(f"\n=== {label} (interior smooth={with_smooth}) ===")
    print(f"  blob-only body force (no topographic-load double-count).")
    print(f"  {'step':>4} {'h_pole':>10} {'heq_pole(abs)':>14} "
          f"{'heq_pole-mean':>14}")
    kid = 0
    for step in range(11):
        absv, relv = held_heq_pole(mesh, unit_r, topo, bf_blob, up, ip, kid)
        kid += 1
        print(f"  {step:>4} {hpole():>+10.5f} {absv:>+14.5e} "
              f"{relv:>+14.5e}", flush=True)
        # RADIAL deflection by dt * u_n at each node (scaled by r/r_o inward)
        un = np.asarray(uw.function.evaluate(
            v.sym.dot(unit_r), mesh.X.coords)).flatten()
        rr = np.sqrt((mesh.X.coords ** 2).sum(1))
        # propagate surface radial velocity inward linearly (no diffuser):
        # scale node radial move by (r - r_i)/(r_o - r_i) so Lower stays fixed
        # and Upper moves by dt*u_n there. Use the surface u_n mapped by angle.
        THn = np.arctan2(mesh.X.coords[:, 1], mesh.X.coords[:, 0])
        un_surf = np.asarray(uw.function.evaluate(
            v.sym.dot(unit_r), mesh.X.coords[up])).flatten()
        un_at_node = np.interp(THn, uth, un_surf, period=2 * np.pi)
        scale = (rr - r_i) / (r_o - r_i)
        dr = dt * un_at_node * scale
        newc = mesh.X.coords + (dr[:, None]) * (mesh.X.coords / rr[:, None])
        mesh._deform_mesh(newc)
        if with_smooth:
            uw.meshing.smooth_mesh_interior(
                mesh, pinned_labels=[mesh.boundaries.Upper.name,
                                     mesh.boundaries.Lower.name],
                method=None, n_iters=5, alpha=0.5)
        stokes.solve(zero_init_guess=False)


run(False, "NO-SMOOTH")
print("\n  equilibrium ≈ 0.0227. heq_full: should DECAY to ~0 at equilibrium")
print("  (it's the residual out-of-balance). heq_blob: should stay ~0.0227")
print("  (the constant blob-driven target). Which is stable on the deformed mesh?")
