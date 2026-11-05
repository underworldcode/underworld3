"""Probe for #941: does a rotated wall's normal load at a partly pinned node depend on
the partition?

A box sheared so Top is tilted (y' = y (1 + a x)), with rotated free slip on Top and a
Dirichlet Left wall meeting it. Investigation only: it asserts nothing, it prints.

Arms:
  geometric-pin-vx    normal=None, Left pins v_x only: the case #941 predicts is affected
  geometric-pin-both  normal=None, Left pins both components: predicted unaffected
  analytic-pin-vx     the exact normal, Left pins v_x only: predicted unaffected

Printed per arm (rank 0):
  INT       the integral of sigma_nn * (1 + x^2) on Top, with and without mean removal
  CORNER    each rank's (load, owned, holds_a_Top_facet) at the Top/Left corner node
  SPREAD    the largest disagreement between ranks over any Top node, and at which point

Run: mpirun -n N python scripts/sessions/probe_941_partly_pinned_normal.py [cellSize [tolerance]]

Written on hyperion (2026-11-06) against development eddab707. Results at np 1, 2, 4 and 8,
cellSize 0.1 to 0.033: the #941 condition (the corner node on a rank holding no Top facet)
never occurred, so #941 is neither confirmed nor ruled out; a forced partition is needed
(-petscpartitioner_type simple gave the default's numbers). Separately, the mean-removed
integral varied with the partition (-0.1066155 to -0.1066242 at cellSize 0.1, np 1 to 8),
tracking the raw integral at a ratio of about -2.7e-3; not solver tolerance.
"""
import sys

import numpy as np
import sympy

import underworld3 as uw
import underworld3.utilities.rotated_bc as rbc

A = 0.3
CELL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0e-10
NTILT = np.array([-A, 1.0]) / np.hypot(A, 1.0)
ARMS = {
    "geometric-pin-vx": (None, (0.0, None)),
    "geometric-pin-both": (None, (0.0, 0.05)),
    "analytic-pin-vx": (tuple(NTILT), (0.0, None)),
}
CORNER = np.array([0.0, 1.0])


def tilted_box():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=CELL, qdegree=2)
    xy = np.array(mesh.X.coords, dtype=float)
    xy[:, 1] *= 1.0 + A * xy[:, 0]
    mesh._deform_mesh(xy)
    return mesh


def solve(name, normal, left):
    mesh = tilted_box()
    v = uw.discretisation.MeshVariable("V" + name, mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P" + name, mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, _y = mesh.X
    s.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * x)]])
    s.add_dirichlet_bc(left, "Left")
    s.add_rotated_freeslip_bc(0, "Top", normal=normal)
    s.add_rotated_freeslip_bc(0, "Bottom", normal=(0.0, -1.0))
    s.add_rotated_freeslip_bc(0, "Right", normal=(1.0, 0.0))
    s.tolerance = TOL
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()
    return mesh, s


def top_facets_here(mesh):
    """Whether this rank holds any facet of Top (the rank #941 says gets it right)."""
    label = mesh.dm.getLabel("Top")
    if label is None:
        return False
    fs, fe = mesh.dm.getHeightStratum(1)
    value = mesh.boundaries.Top.value
    # a rank with none of the wall gets a null IS, and getIndices on that segfaults
    if label.getStratumSize(value) == 0:
        return False
    return any(fs <= q < fe for q in label.getStratumIS(value).getIndices())


comm = uw.mpi.comm
for name, (normal, left) in ARMS.items():
    mesh, s = solve(name.replace("-", ""), normal, left)
    x, _y = mesh.X
    weight = 1 + x**2
    with_mean = s.boundary_normal_traction_integral("Top", weight, remove_mean=False)
    no_mean = s.boundary_normal_traction_integral("Top", weight, remove_mean=True)

    xs, loads, owned = rbc._wall_normal_loads(s, "Top", s._rotated_freeslip_info)
    holds = top_facets_here(mesh)
    rows = comm.gather(
        [(comm.rank, tuple(np.round(q, 10)), float(l), bool(o), holds)
         for q, l, o in zip(xs, loads, owned)], root=0)
    if comm.rank == 0:
        by_node = {}
        for rank_rows in rows:
            for r, key, load, own, hold in rank_rows:
                by_node.setdefault(key, []).append((r, load, own, hold))
        spreads = {k: max(e[1] for e in v) - min(e[1] for e in v) for k, v in by_node.items()}
        worst = max(spreads, key=spreads.get) if spreads else None
        corner = min(by_node, key=lambda k: np.linalg.norm(np.array(k) - CORNER))
        shared = sum(1 for v in by_node.values() if len(v) > 1)
        print(f"INT {name} np={comm.size} raw={with_mean:.12e} mean_removed={no_mean:.12e}")
        print(f"CORNER {name} np={comm.size} node={corner} "
              + " ".join(f"r{r}:load={l:.10e},owned={o},topfacets={h}"
                         for r, l, o, h in by_node[corner]))
        print(f"SPREAD {name} np={comm.size} top_nodes={len(by_node)} shared={shared} "
              f"max={spreads[worst]:.3e} at={worst}")
    comm.barrier()
