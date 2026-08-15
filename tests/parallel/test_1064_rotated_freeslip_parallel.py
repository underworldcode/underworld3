"""Parallel regression test for rotated strong free-slip (``add_rotated_freeslip_bc``).

The rotated free-slip solve (per-node DOF rotation → strong ``v_n=0`` on the rotated
normal rows → rotate back → gauge removal) was previously validated serially only
(``tests/test_1018_rotated_freeslip.py``). At np>1 it crashed: the RHS constrained-row
zeroing indexed the *local* slice with *global* row indices, which overflows on any rank
whose ownership does not start at 0 — an asymmetric crash that masqueraded as a hang.
With that fixed, the whole global system (and hence the velocity solve and the wall-normal
leakage) is partition-independent.

This test verifies that the parallel solve reproduces ITS OWN np=1 answer to a tight
tolerance for two geometries:

  * **box** — 4-wall rotated free-slip on axis-aligned walls (GAMG velocity block); the
    velocity L2 ``∫ v·v`` must match serial (bit-identical up to the parallel reduction
    order), and the analytic SolCx velocity error must stay small.
  * **annulus** — per-node *radial* free-slip on both arcs with the analytic normal (the
    rotation-nullspace + gauge-removal path); the velocity L2 and the radial leakage
    ``∫ (v·r̂)²`` on each arc must match serial.

All diagnostics use the parallel-safe ``uw.maths.Integral`` / ``BdIntegral`` reductions
(no rank-local ``v.data``, which is per-partition, and no ``uw.function.evaluate`` on
arbitrary points, which deadlocks np>1).

Every "must match serial" assertion below compares against a np=1 run of THIS FILE,
computed in THIS environment by ``serial_reference`` — not against a constant recorded
on a developer's machine. The distinction is not cosmetic. Five of these tests carried
hardcoded goldens and messages reading "differs serial vs np=N", and on CI they failed:
annulus velocity L2 recorded 1.897011154231e-02, measured 1.897329151624e-02. Running
the SAME diagnostic at BOTH rank counts on one CI host gives np=1 1.897329151623790e-02
and np=2 1.897329151623740e-02 — agreement to the 13th significant figure, with the
leakages identical to every digit — and both differ from the recorded golden by exactly
the same +1.676e-04. The rotated solve was never partition dependent. gmsh builds a
different triangulation on the Linux runner than on macOS/arm64, the goldens were
recorded on macOS, and because these tests are ``mpi(min_size=2)`` CI never ran np=1 to
notice. A day of investigation went into a defect that did not exist, and the
misleading part was the assertion message.

Where a genuine ABSOLUTE check is wanted — the analytic SolCx velocity error, the
sigma_nn accuracy against the exact solution, the iteration-count ceiling — it is kept
and labelled as an accuracy check, with a tolerance loose enough to survive a
cross-host mesh change. Those are NOT partition-independence checks and must not be
conflated with them again.

Run with:
    mpirun -n 2 python -m pytest --with-mpi \\
      tests/parallel/test_1064_rotated_freeslip_parallel.py
    mpirun -n 4 python -m pytest --with-mpi \\
      tests/parallel/test_1064_rotated_freeslip_parallel.py
"""

import numpy as np
import sympy
import pytest

import underworld3 as uw
from underworld3.function import analytic as A
from underworld3.utilities import custom_mg

from serial_reference import (
    accuracy_anchor, compare, emit, mesh_fingerprint, serial_reference)

# The timeout covers the np=1 child each partition test spawns as well as the
# parallel solve itself.
pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(900)]

# ABSOLUTE accuracy anchors, gated on the mesh fingerprint.
#
# `compare` proves the answer does not depend on the PARTITION; it says nothing about
# whether the answer is RIGHT. A rotated constraint that stopped constraining equally
# on every rank, an FMG hierarchy converging to the wrong place, a Zhong l=2 benchmark
# coefficient drifting — all of those pass a self-referential test. These are the
# pre-#568 goldens, kept for that second job, and gated so a host whose gmsh
# triangulates differently SKIPS the accuracy claim instead of failing it (which is
# what made them misleading in the first place — see the module docstring).
#
# rtol is 1e-2: not a reproducibility gate, just "is this still the same answer".
# Fingerprints are (owned cell count, integral 1 dV); recompute any line with
# `python <thisfile> <kind>`.
_MESH_BOX24 = [576, 1.0000000000000004]        # StructuredQuadBox(24, 24)
_MESH_BOX8 = [64, 0.9999999999999999]          # StructuredQuadBox(8, 8)
_MESH_ANNULUS = [600, 2.356187202481425]       # Annulus(0.5, 1.0, cellSize=0.1)
_MESH_ANNULUS_FMG = [2304, 2.356078287527854]  # cellSize=0.2, refined twice
_MESH_SHELL = [1417, 3.4521585981151097]       # SphericalShell(0.55, 1.0, cs=0.25)
_MESH_SHELL_INT = [2016, 3.4521585981151106]   # SphericalShellInternalBoundary

ANCHORS = {
    "box": {"fingerprint": _MESH_BOX24,
            "values": (1.275109036912e-03,)},
    "annulus": {"fingerprint": _MESH_ANNULUS,
                "values": (1.897011154231e-02, 4.563841e-05, 9.341699e-06)},
    "annulus_fmg": {"fingerprint": _MESH_ANNULUS_FMG,
                    "values": (1.906961759626e-02, 5.428193e-06, 1.177002e-06)},
    "spherical3d": {"fingerprint": _MESH_SHELL,
                    "values": (4.069689334228e-03,)},
    "spherical3d_topo": {"fingerprint": _MESH_SHELL_INT,
                         "values": (4.149689252074e-01, 3.952301937705e-01,
                                    4.215939953379e-01, 7.932177563075e-01,
                                    8.426041179682e-01, 7.762363224500e-01)},
    "nonlinear": {"fingerprint": _MESH_BOX8,
                  "values": (8.069396188270e-04, 8)},
    "sigma": {"fingerprint": _MESH_BOX24,
              "values": (5.554578e-02, 0.998466)},
    "topo": {"fingerprint": _MESH_BOX24,
             "values": (2.553916470e-01,)},
}


def _wrap(dm, m0):
    """Wrap a refined DMPlex as a UW3 mesh carrying the base mesh's boundaries and
    coordinate system (for building a nested annulus MG hierarchy)."""
    return uw.discretisation.Mesh(
        dm.clone(), simplex=True,
        coordinate_system_type=m0.CoordinateSystem.coordinate_type,
        qdegree=3, boundaries=m0.boundaries)


def _box_diagnostics():
    """Box SolCx with rotated free-slip on all four axis-aligned walls (GAMG velocity
    block). Returns (velocity L2, analytic velocity error)."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(24, 24), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)
    v = uw.discretisation.MeshVariable("vB", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pB", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    L2 = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
    verr = float(sol.velocity_error(v))
    return (L2, verr), mesh_fingerprint(mesh)


def _annulus_diagnostics():
    """Annulus with per-node radial free-slip on both arcs (analytic normal). Returns
    (velocity L2, radial-leakage L2 on Lower arc, radial-leakage L2 on Upper arc)."""
    RI, RO = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.1, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    th = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable("Va", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pa", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.bodyforce = sympy.Matrix([[x / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0,
                                 y / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0]])
    nhat = sympy.Matrix([[x / r, y / r]])
    s.add_rotated_freeslip_bc(0, "Lower", normal=nhat)
    s.add_rotated_freeslip_bc(0, "Upper", normal=nhat)
    s.tolerance = 1e-9
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    L2 = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
    vr = v.sym[0] * x / r + v.sym[1] * y / r      # radial velocity v·r̂
    leak_lo = float(np.sqrt(uw.maths.BdIntegral(
        mesh=mesh, fn=vr**2, boundary="Lower").evaluate()))
    leak_up = float(np.sqrt(uw.maths.BdIntegral(
        mesh=mesh, fn=vr**2, boundary="Upper").evaluate()))
    return (L2, leak_lo, leak_up), mesh_fingerprint(mesh)


def _spherical3d_diagnostics():
    """3D spherical shell with per-node radial free-slip on BOTH boundaries (the
    Zhong #248 configuration): all three rigid rotations are nullspace modes.
    Returns (velocity L2, outer KSP its, converged reason)."""
    RI, RO = 0.55, 1.0
    mesh = uw.meshing.SphericalShell(radiusInner=RI, radiusOuter=RO,
                                     cellSize=0.25, qdegree=2)
    x, y, z = mesh.X
    r = sympy.sqrt(x**2 + y**2 + z**2)
    v = uw.discretisation.MeshVariable("Vs", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Ps", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    ylm = (3 * (z / r) ** 2 - 1) / 2
    s.bodyforce = ylm * (r - RI) * (RO - r) * 20.0 / r * sympy.Matrix([[x, y, z]])
    nhat = sympy.Matrix([[x / r, y / r, z / r]])
    s.add_rotated_freeslip_bc(0, "Lower", normal=nhat)
    s.add_rotated_freeslip_bc(0, "Upper", normal=nhat)
    s.tolerance = 1e-7
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    L2 = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
    info = s._rotated_freeslip_info
    # the unified rotated loop reports one KSP count per Newton increment
    return ((L2,), mesh_fingerprint(mesh),
            max(info["ksp_its"]), int(info["ksp_reason"]))


def _spherical3d_topography_diagnostics(cell_size=0.25):
    """Zhong l=2 topography from the 3D rotated-constraint reaction."""
    RI, RO, RINT = 0.55, 1.0, 0.775
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusOuter=RO, radiusInternal=RINT, radiusInner=RI,
        cellSize=cell_size, qdegree=2, degree=1)
    v = uw.discretisation.MeshVariable(
        "Vst", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable(
        "Pst", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    theta = mesh.CoordinateSystem.xR[1]
    unit_r = mesh.CoordinateSystem.unit_e_0
    harmonic = sympy.assoc_legendre(2, 0, sympy.cos(theta))
    s.add_natural_bc(harmonic * unit_r, "Internal")
    s.add_rotated_freeslip_bc(0, "Upper", normal=unit_r)
    s.add_rotated_freeslip_bc(0, "Lower", normal=-unit_r)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.tolerance = 1.0e-5
    s.solve()

    dm = mesh.dm
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, mesh.dim)
    vertex_start, vertex_end = dm.getDepthStratum(0)
    local_vertex_keys = {
        tuple(np.round(cvec[csec.getOffset(point) // mesh.dim], 12))
        for point in range(vertex_start, vertex_end)
    }
    vertex_keys = set()
    for rank_keys in uw.mpi.comm.allgather(local_vertex_keys):
        vertex_keys.update(rank_keys)

    def harmonic_coefficients(boundary, response_sign):
        xs, sigma_nn = s.boundary_normal_traction(boundary)
        local = {
            tuple(np.round(x, 12)): -float(value)
            for x, value in zip(xs, sigma_nn)
        }
        samples = {}
        for rank_values in uw.mpi.comm.allgather(local):
            samples.update(rank_values)
        coords = np.asarray(list(samples))
        topography = np.asarray(list(samples.values()))
        radii = np.linalg.norm(coords, axis=1)
        harmonic_values = 0.5 * (3.0 * (coords[:, 2] / radii) ** 2 - 1.0)
        is_vertex = np.array([key in vertex_keys for key in samples], dtype=bool)

        def fit(mask):
            return float(
                response_sign
                * np.dot(topography[mask], harmonic_values[mask])
                / np.dot(harmonic_values[mask], harmonic_values[mask])
            )

        return fit(np.ones(len(coords), dtype=bool)), fit(is_vertex), fit(~is_vertex)

    return (
        *harmonic_coefficients("Upper", 1.0),
        *harmonic_coefficients("Lower", -1.0),
    ), mesh_fingerprint(mesh)


def _annulus_fmg_diagnostics(mesh_owned=False):
    """Annulus radial free-slip whose velocity block is CUSTOM GEOMETRIC FMG on a
    nested hierarchy (coarse annulus -> refine -> refine) — no GAMG, no direct
    solve. Returns (velocity L2, leakage L2 on Lower, on Upper).

    ``mesh_owned`` selects how the hierarchy reaches the solver: ``False`` is an
    explicit ``set_custom_fmg`` registration, ``True`` is the MESH-owned coarse
    tail an ``adapt()`` child carries. Both must produce the same solve (#467)."""
    RI, RO = 0.5, 1.0
    m0 = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.2, qdegree=3)
    dm1 = m0.dm.refine()
    dm2 = dm1.refine()
    coarse = [m0, _wrap(dm1, m0)]
    fine = _wrap(dm2, m0)
    if mesh_owned:
        fine._custom_mg_coarse_meshes = coarse
        fine._custom_mg_builder = "barycentric"

    x, y = fine.X
    r = sympy.sqrt(x**2 + y**2)
    th = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable("Vf", fine, fine.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pf", fine, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(fine, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.bodyforce = sympy.Matrix([[x / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0,
                                 y / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0]])
    nhat = sympy.Matrix([[x / r, y / r]])
    s.add_rotated_freeslip_bc(0, "Lower", normal=nhat)
    s.add_rotated_freeslip_bc(0, "Upper", normal=nhat)
    s.tolerance = 1e-9
    s.saddle_preconditioner = 1.0
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    if not mesh_owned:
        custom_mg.set_custom_fmg(s, coarse, builder="barycentric", field_id=0)
    s.solve()

    assert s._rotated_freeslip_info["ksp_reason"] > 0, "custom-FMG rotated solve did not converge"
    assert s._rotated_freeslip_info["velocity_pc"] == "custom-FMG", (
        "rotated velocity block fell back to "
        f"{s._rotated_freeslip_info['velocity_pc']} instead of geometric MG")
    L2 = float(np.sqrt(uw.maths.Integral(fine, v.sym.dot(v.sym)).evaluate()))
    vr = v.sym[0] * x / r + v.sym[1] * y / r
    leak_lo = float(np.sqrt(uw.maths.BdIntegral(
        mesh=fine, fn=vr**2, boundary="Lower").evaluate()))
    leak_up = float(np.sqrt(uw.maths.BdIntegral(
        mesh=fine, fn=vr**2, boundary="Upper").evaluate()))
    return (L2, leak_lo, leak_up), mesh_fingerprint(fine)


def _box_nonlinear_diagnostics():
    """NONLINEAR box: power-law viscosity eta = eps_II^(1/n-1) with rotated free-slip
    on all four walls, solved by the manual Newton loop (consistent tangent, GAMG
    velocity block). Returns (velocity L2, nonlinear iteration count) — both must be
    partition-independent (the loop's ptap / rotate / constrain / increment-solve are
    all collective and ownership-relative)."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("vNLp", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pNLp", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    g = sympy.Matrix([[v.sym[0].diff(x), v.sym[0].diff(y)],
                      [v.sym[1].diff(x), v.sym[1].diff(y)]])
    e = 0.5 * (g + g.T)
    eII = sympy.sqrt(0.5 * (e[0, 0] ** 2 + e[1, 1] ** 2) + e[0, 1] ** 2 + 1.0e-12)
    s.constitutive_model.Parameters.shear_viscosity_0 = eII ** (1.0 / 3.0 - 1.0)
    s.bodyforce = sympy.Matrix([[0.0, -2.0 * sympy.cos(sympy.pi * x)]])
    s.penalty = 0.0
    s.tolerance = 1e-7
    s.petsc_use_pressure_nullspace = True
    s.consistent_jacobian = True                 # Newton tangent (few iterations)
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.solve()

    L2 = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
    return ((L2, int(s._rotated_freeslip_info["nonlinear_iterations"])),
            mesh_fingerprint(mesh))


def _box_sigma_diagnostics():
    """Recover sigma_nn on Top via boundary_normal_traction and compare to the exact
    SolCx sigma_yy over the WHOLE boundary. The per-rank local (xs, sigma) are gathered
    + de-duplicated on rank 0, the metric is computed there and broadcast, so every rank
    returns the same (relL2, |corr|) — a direct partition-independence check."""
    res = 24
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)
    v = uw.discretisation.MeshVariable("vS", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pS", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    xs, sig = s.boundary_normal_traction("Top")
    comm = uw.mpi.comm
    gx = comm.gather(np.asarray(xs).reshape(-1, 2), root=0)
    gs = comm.gather(np.asarray(sig).reshape(-1), root=0)
    result = None
    if uw.mpi.rank == 0:
        allx = np.concatenate(gx) if len(gx) else np.zeros((0, 2))
        alls = np.concatenate(gs) if len(gs) else np.zeros(0)
        seen = {}
        for xc, sc in zip(allx, alls):
            seen[(round(float(xc[0]), 9), round(float(xc[1]), 9))] = (xc, sc)
        X = np.array([u[0] for u in seen.values()])
        S = np.array([u[1] for u in seen.values()])
        syy = np.asarray(sol.evaluate_stress(X))[:, 1]
        syy = syy - syy.mean()
        corr = float(np.dot(S, syy) / (np.linalg.norm(S) * np.linalg.norm(syy)))
        S = S if corr >= 0 else -S
        relL2 = float(np.linalg.norm(S - syy) / np.linalg.norm(syy))
        result = (relL2, abs(corr), len(X))
    return comm.bcast(result, root=0), mesh_fingerprint(mesh)


def _box_topography_bdl2():
    """dynamic_topography onto a P1 surface field; return the (collective) BdIntegral L2
    of h over Top — a partition-independent scalar functional of the topography field."""
    res = 24
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)
    v = uw.discretisation.MeshVariable("vTf", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pTf", mesh, 1, degree=1, continuous=False)
    hf = uw.discretisation.MeshVariable("hTf", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()
    s.dynamic_topography("Top", hf, buoyancy_scale=1.0)
    bdl2 = float(np.sqrt(uw.maths.BdIntegral(
        mesh=mesh, fn=hf.sym[0] ** 2, boundary="Top").evaluate()))
    return (bdl2,), mesh_fingerprint(mesh)


def test_rotated_freeslip_box_partition_independent():
    """Box: the parallel rotated free-slip solve reproduces its OWN np=1 velocity L2,
    and the analytic velocity error stays small."""
    values, fingerprint = _box_diagnostics()
    compare(values[:1], _reference("box", 1), rtols=(1e-8,), labels=("velocity L2",),
            fingerprint=fingerprint, what="box rotated free-slip")
    accuracy_anchor(values[:1], ANCHORS["box"], fingerprint, ("velocity L2",),
                    what="box rotated free-slip")
    # ACCURACY check, not a partition check: the SolCx error is a property of the
    # discretisation, and the gate is loose enough to survive a cross-host mesh.
    assert values[1] < 1e-3, (
        f"box velocity error {values[1]:.2e} too large at np={uw.mpi.size}")


def test_rotated_freeslip_annulus_partition_independent():
    """Annulus: the parallel radial free-slip solve reproduces its OWN np=1 velocity L2
    and radial leakage on both arcs.

    Measured on ONE CI host, this diagnostic: np=1 1.897329151623790e-02, np=2
    1.897329151623740e-02, leakages identical to every digit. The 1.7e-04 this test
    used to report was the distance to a golden recorded on a different host, not a
    partition effect — see the module docstring.
    """
    values, fingerprint = _annulus_diagnostics()
    # velocity L2 is iterative-solver-tolerance reproducible (~1e-8 rel), not the
    # box's 1e-10 — the annulus carries a rotation null space + gauge removal.
    labels = ("velocity L2", "Lower radial leakage", "Upper radial leakage")
    compare(values, _reference("annulus", 3), rtols=(1e-6, 1e-4, 1e-4),
            labels=labels, fingerprint=fingerprint, what="annulus rotated free-slip")
    # The LEAKAGE anchors are the point here: a rotated constraint that stopped
    # constraining would keep every partition agreeing with every other.
    accuracy_anchor(values, ANCHORS["annulus"], fingerprint, labels,
                    what="annulus rotated free-slip")


def test_rotated_freeslip_annulus_fmg_partition_independent():
    """The full stack: rotated radial free-slip on the annulus with the velocity block
    driven by CUSTOM GEOMETRIC FMG (set_custom_fmg) reproduces its own np=1 velocity L2
    and radial leakage in parallel — FMG x rotated x annulus x np>1."""
    values, fingerprint = _annulus_fmg_diagnostics()
    labels = ("velocity L2", "Lower radial leakage", "Upper radial leakage")
    compare(values, _reference("annulus_fmg", 3), rtols=(1e-6, 1e-4, 1e-4),
            labels=labels, fingerprint=fingerprint,
            what="custom-FMG annulus rotated free-slip")
    accuracy_anchor(values, ANCHORS["annulus_fmg"], fingerprint, labels,
                    what="custom-FMG annulus rotated free-slip")


def test_rotated_freeslip_mesh_owned_fmg_pickup():
    """#467, in parallel: a coarse tail owned by the MESH — what ``adapt()``
    leaves on a refinement child — must drive geometric MG under rotated
    free-slip, on every partition. The rotated path used to consult only an
    explicit ``set_custom_fmg`` registration and fell back to GAMG silently.

    The transfers are built cross-partition, so this is not implied by the serial
    pickup test (``tests/test_1021_mg_option_bundle.py``); the tail is attached by
    hand because what is under test is whether the rotated dispatch consults it,
    not how ``adapt()`` produces it.

    The np=1 reference is the EXPLICIT-registration one: #467 is precisely the claim
    that the two routes produce the same solve, so comparing the mesh-owned parallel
    run against the explicitly-registered serial run asserts both properties at once.
    """
    values, fingerprint = _annulus_fmg_diagnostics(mesh_owned=True)
    labels = ("velocity L2", "Lower radial leakage", "Upper radial leakage")
    compare(values, _reference("annulus_fmg", 3), rtols=(1e-6, 1e-4, 1e-4),
            labels=labels, fingerprint=fingerprint,
            what="mesh-owned FMG annulus rotated free-slip")
    accuracy_anchor(values, ANCHORS["annulus_fmg"], fingerprint, labels,
                    what="mesh-owned FMG annulus rotated free-slip")


def test_rotated_freeslip_spherical3d_partition_independent():
    """3D spherical shell (free-slip inner+outer, all three rotation nullspace
    modes): the parallel solve reproduces its own np=1 velocity L2, converges, and
    stays within the bounded outer iteration count (the 1/mu-mass Schur
    preconditioner — issue #248's rotated blow-out was ~44 its)."""
    values, fingerprint, its, reason = _spherical3d_diagnostics()
    # ABSOLUTE checks on the solve itself, not partition comparisons.
    #
    # INVARIANT: everything asserted before the `compare` below must be identical on
    # every rank, because `compare` calls the COLLECTIVE `serial_reference`. A gate
    # that fails on one rank only would take that rank out of the broadcast and hang
    # the others instead of failing the test. `reason` and `its` come from the
    # solver's own collective telemetry and `nnodes` (in the sigma test) is bcast, so
    # all of them satisfy it today — but nothing enforces it, so put new absolute
    # gates AFTER the compare unless you have checked.
    assert reason > 0, f"3D spherical rotated solve diverged: reason {reason}"
    assert its <= 25, f"3D spherical Schur iteration blow-out: {its} outer its"
    compare(values, _reference("spherical3d", 1), rtols=(1e-5,),
            labels=("velocity L2",), fingerprint=fingerprint,
            what="3D spherical rotated free-slip")
    accuracy_anchor(values, ANCHORS["spherical3d"], fingerprint, ("velocity L2",),
                    what="3D spherical rotated free-slip")


def test_rotated_freeslip_spherical3d_topography_partition_independent():
    """3D boundary-mass recovery gives partition-independent topography coefficients."""
    values, fingerprint = _spherical3d_topography_diagnostics()
    labels = ("surface all", "surface vertices", "surface midpoints",
              "CMB all", "CMB vertices", "CMB midpoints")
    compare(values, _reference("spherical3d_topo", 6), rtols=(1e-6,) * 6,
            labels=labels, fingerprint=fingerprint,
            what="3D spherical rotated topography")
    # Zhong l=2 benchmark coefficients — physics numbers, and the reason an absolute
    # anchor matters more here than anywhere else in this file.
    accuracy_anchor(values, ANCHORS["spherical3d_topo"], fingerprint, labels,
                    what="3D spherical rotated topography")


def test_rotated_freeslip_box_nonlinear_partition_independent():
    """NONLINEAR rotated free-slip is partition-independent: a power-law box solved by
    the manual Newton/Picard loop reproduces its own np=1 velocity L2 AND iteration
    count at np=2/4 — the rotated residual/Jacobian, the increment solve and the
    constraint zeroing are all parallel-safe (ownership-relative indexing, collective
    norms)."""
    values, fingerprint = _box_nonlinear_diagnostics()
    # rtol=0 on the iteration count: it is an integer and the claim is exact equality.
    labels = ("velocity L2", "nonlinear iteration count")
    compare(values, _reference("nonlinear", 2), rtols=(1e-6, 0.0), labels=labels,
            fingerprint=fingerprint, what="nonlinear box rotated free-slip")
    accuracy_anchor(values, ANCHORS["nonlinear"], fingerprint, labels,
                    what="nonlinear box rotated free-slip")


def test_rotated_freeslip_box_sigma_nn_partition_independent():
    """sigma_nn (boundary_normal_traction) recovery is partition-independent: the whole-
    boundary relL2 / |corr| vs analytic SolCx sigma_yy reproduce the np=1 run (and stay
    accurate) in parallel — the reaction read + consistent-mass de-smear are
    parallel-safe."""
    (relL2, corr, nnodes), fingerprint = _box_sigma_diagnostics()
    # ABSOLUTE checks: the gathered node set, and the accuracy against the analytic
    # solution. Neither is a partition comparison.
    assert nnodes == 49, f"expected 49 top nodes, gathered {nnodes} at np={uw.mpi.size}"
    assert relL2 < 0.10, f"sigma_nn relL2 vs analytic {relL2:.3f} too large"
    labels = ("sigma_nn relL2", "sigma_nn |corr|")
    compare((relL2, corr), _reference("sigma", 2), rtols=(1e-4, 1e-4), labels=labels,
            fingerprint=fingerprint, what="box sigma_nn recovery")
    accuracy_anchor((relL2, corr), ANCHORS["sigma"], fingerprint, labels,
                    what="box sigma_nn recovery")


def test_rotated_freeslip_dynamic_topography_partition_independent():
    """The dynamic_topography surface field is partition-independent: the collective
    BdIntegral L2 of h over Top reproduces the np=1 run at np=2/4. Guards the field
    write (a per-node write would deadlock when a rank owns none of the boundary) and
    the parallel reaction recovery underneath."""
    values, fingerprint = _box_topography_bdl2()
    compare(values, _reference("topo", 1), rtols=(1e-6,),
            labels=("topography BdIntegral L2",), fingerprint=fingerprint,
            what="box dynamic topography")
    accuracy_anchor(values, ANCHORS["topo"], fingerprint,
                    ("topography BdIntegral L2",), what="box dynamic topography")


_DIAGNOSTICS = {
    "box": _box_diagnostics,
    "annulus": _annulus_diagnostics,
    "annulus_fmg": _annulus_fmg_diagnostics,
    "spherical3d": _spherical3d_diagnostics,
    "spherical3d_topo": _spherical3d_topography_diagnostics,
    "nonlinear": _box_nonlinear_diagnostics,
    "sigma": _box_sigma_diagnostics,
    "topo": _box_topography_bdl2,
}


def _reference(kind, count):
    """The np=1 payload for ``kind``, with its length asserted. A reference of the
    wrong length would silently shorten the ``zip`` in ``compare`` and drop
    assertions."""
    reference = serial_reference(__file__, kind)
    assert len(reference["values"]) >= count, (
        f"serial reference for {kind!r} has {len(reference['values'])} values, "
        f"expected at least {count}")
    reference = dict(reference)
    reference["values"] = reference["values"][:count]
    return reference


if __name__ == "__main__":
    # Single-rank child of the parallel run (see serial_reference), and a
    # human-readable recompute:
    #   `python <thisfile> {box,annulus,annulus_fmg,spherical3d,spherical3d_topo,
    #                       nonlinear,sigma,topo}`.
    import sys
    _kind = sys.argv[1] if len(sys.argv) > 1 else "box"
    # `python <thisfile> spherical3d_topo 0.3` still works: the extra argument is a
    # debugging affordance (a coarser shell for a quick look) and is NOT reachable from
    # serial_reference, which always calls the default so the anchor stays comparable.
    _extra = [float(a) for a in sys.argv[2:]]
    _result = _DIAGNOSTICS[_kind](*_extra)
    # _spherical3d_diagnostics carries its solver telemetry after the fingerprint.
    _values, _fingerprint = _result[0], _result[1]
    emit(_values, _fingerprint)
    uw.mpi.pprint(f"DIAG_{_kind.upper()} "
                  + " ".join(f"{v:.12e}" for v in _values)
                  + f" [cells={_fingerprint[0]:.0f} vol={_fingerprint[1]:.12g}]")
