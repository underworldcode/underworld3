"""Compare current vs direction-aware (anisotropic-aware) CFL
estimate on the existing adapted meshes.

Current UW3:  dt_c = mesh._radii[c] / |v_c|
Proposed:     dt_c = (max(s_i) - min(s_i)) / |v_c|
              where s_i = (x_i - centroid) · v̂_c
              over the cell's vertices.

For isotropic cells the ratio ≈ 1 (slightly favours the proposed
formula since it uses the full v-direction diameter, not the
inradius). For anisotropic cells stretched along v, the proposed
formula can be substantially larger — exactly the savings we
want to capture on adapted meshes.
"""
import os
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import _tri_cells


def cell_extents_along_v(coords, tris, centroids, v_per_cell):
    """h_eff per cell = max(s_i) - min(s_i)
    where s_i = (x_i - centroid) · v̂."""
    # Per-cell velocity directions
    vmag = np.linalg.norm(v_per_cell, axis=1)
    vhat = np.where(vmag[:, None] > 0,
                    v_per_cell / np.maximum(vmag[:, None], 1e-30),
                    0.0)
    # Vertex offsets from centroid (shape: ncell, 3 verts, dim)
    V = coords[tris]                      # (ncell, 3, dim)
    D = V - centroids[:, None, :]         # (ncell, 3, dim)
    # Projection onto vhat per cell
    s = np.einsum('cvd,cd->cv', D, vhat)  # (ncell, 3)
    h_eff = s.max(axis=1) - s.min(axis=1)
    return h_eff, vmag


def isotropic_radii(coords, tris):
    """Cell inradius = 2A / (perimeter)."""
    a = coords[tris[:, 0]]
    b = coords[tris[:, 1]]
    c = coords[tris[:, 2]]
    A = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                     - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    perim = ab + bc + ca
    return 2.0 * A / perim, A


def cell_max_edge(coords, tris):
    """Longest edge — the "extent in best direction" upper bound."""
    a = coords[tris[:, 0]]
    b = coords[tris[:, 1]]
    c = coords[tris[:, 2]]
    return np.maximum(np.maximum(
        np.linalg.norm(b - a, axis=1),
        np.linalg.norm(c - b, axis=1)),
        np.linalg.norm(a - c, axis=1))


BASE = os.path.expanduser(
    '~/+Simulations/StagnantLid/R_compare')
R_LIST = [1.0, 1.5, 2.0, 3.0, 6.0, 10.0]

print(f"{'R':>5} {'h_eff/h_iso':>13} {'h_eff/h_iso':>13} "
      f"{'h_eff/h_iso':>13} {'dt_aniso':>10} {'dt_iso':>10} "
      f"{'GLOBAL':>8}")
print(f"{'':>5} {'(min)':>13} {'(median)':>13} "
      f"{'(max)':>13} {'(s)':>10} {'(s)':>10} {'gain':>8}")
print("-" * 90)

for R in R_LIST:
    src = os.path.join(BASE, f"R{R}")
    m = uw.discretisation.Mesh(
        os.path.join(src, "adapted.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep("adapted", "T_v2p1", 0, outputPath=src)
    # Estimate v on this mesh from a fresh Stokes solve
    V = uw.discretisation.MeshVariable(
        "V_v2p1", m, vtype=uw.VarType.VECTOR,
        degree=2, continuous=True)
    P = uw.discretisation.MeshVariable(
        "P_v2p1", m, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True)
    X = m.CoordinateSystem.X
    r_sym = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    unit_r = m.CoordinateSystem.unit_e_0
    s = uw.systems.Stokes(m, velocityField=V, pressureField=P)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    theta = float(np.log(1.0e4))
    s.constitutive_model.Parameters.shear_viscosity_0 = (
        sympy.exp(theta * (1 - T.sym[0])))
    s.tolerance = 1.0e-5
    s.penalty = 0.0
    s.add_essential_bc((0.0, 0.0), m.boundaries.Lower.name)
    KFS = 1.0e6
    fs = (KFS * V.sym.dot(unit_r) * unit_r)
    s.add_natural_bc(fs, m.boundaries.Upper.name)
    T_cond = sympy.log(r_sym) / sympy.log(0.5)
    s.bodyforce = 1.0e7 * (T.sym[0] - T_cond) * unit_r
    V.data[...] = 0.0
    P.data[...] = 0.0
    s.solve(zero_init_guess=True)

    coords = np.asarray(m.X.coords)
    tris = _tri_cells(m.dm)
    centroids = coords[tris].mean(axis=1)
    inrad, A_actual = isotropic_radii(coords, tris)
    h_long = cell_max_edge(coords, tris)
    v_per_cell = np.asarray(uw.function.evaluate(
        V.sym, centroids)).reshape(centroids.shape[0], 2)
    h_eff, vmag = cell_extents_along_v(
        coords, tris, centroids, v_per_cell)

    # Compare INRADIUS (what UW3 currently uses via _radii) to h_eff
    # (proposed) per cell. Mask out cells with negligible velocity.
    active = vmag > vmag.max() * 1e-3
    h_i = inrad[active]
    h_e = h_eff[active]
    h_l = h_long[active]

    ratio_eff_over_iso = h_e / h_i
    # Global dt = min over cells. CFL=0.5 factor for safety.
    CFL = 0.5
    vmag_active = vmag[active]
    dt_iso = CFL * float((h_i / vmag_active).min())
    dt_aniso = CFL * float((h_e / vmag_active).min())
    print(f"{R:>5.1f} "
          f"{ratio_eff_over_iso.min():>13.3f} "
          f"{np.median(ratio_eff_over_iso):>13.3f} "
          f"{ratio_eff_over_iso.max():>13.3f} "
          f"{dt_aniso:>10.3e} {dt_iso:>10.3e} "
          f"{dt_aniso / dt_iso:>8.3f}", flush=True)
