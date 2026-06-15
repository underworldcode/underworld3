"""Contract test: surface submesh extraction.

Mirrors the loud-failure stance of ``test_refined_pair_contract.py``.
Surface extraction requires:

  1. A boundary label that exists on the parent.
  2. That label's face stratum is non-empty.

Both contracts must raise rather than degrade — there is deliberately no
geometric / KDTree fallback that would silently produce a degenerate
submesh.

Also asserts that on a valid extraction, the submesh has the expected
geometry: ``dim = parent.dim - 1``, ``cdim = parent.cdim``, vertex
coordinates lie on the parent surface to machine precision.

Run:
    pixi run -e amr-dev python -u \\
        docs/examples/submesh_investigation/test_surface_submesh_contract.py
"""

import numpy as np

import underworld3 as uw

import surface_submesh_prototype as ssp


def banner(msg):
    uw.pprint(0, f"\n{'='*70}\n{msg}\n{'='*70}")


def main():
    banner("Parent mesh")
    shell = uw.meshing.SphericalShell(
        radiusOuter=1.0,
        radiusInner=0.5,
        cellSize=0.25,
    )
    uw.pprint(
        0,
        f"shell: dim={shell.dim}, cdim={shell.cdim}, "
        f"boundaries={[b.name for b in shell.boundaries]}",
    )

    banner("CONTRACT 1: unknown label -> ValueError, no degraded mesh")
    raised = False
    try:
        ssp.extract_surface(shell, "Bogus")
    except ValueError as e:
        raised = True
        uw.pprint(0, f"raised ValueError as required:\n  {e}")
    assert raised, "extract_surface must raise on an unknown label"

    banner("CONTRACT 2: empty face stratum -> ValueError")
    # Construct a parent boundary label name that exists but has an
    # empty face stratum. We test this by passing an out-of-range
    # label_value to bypass the name-lookup and hit the empty-stratum
    # check directly.
    raised = False
    try:
        ssp.extract_surface(shell, "Upper", label_value=99999)
    except ValueError as e:
        raised = True
        uw.pprint(0, f"raised ValueError as required:\n  {e}")
    assert raised, "extract_surface must raise on an empty face stratum"

    banner("CONTRACT 3: valid extraction -> dim=2, cdim=3, on the sphere")
    surface = ssp.extract_surface(shell, "Upper")
    assert surface.parent is shell, "surface must reference its parent"
    assert surface in shell._registered_submeshes, "surface must be registered"
    assert surface.dim == shell.dim - 1, (
        f"surface dim should be parent.dim - 1, got {surface.dim} "
        f"(parent {shell.dim})"
    )
    assert surface.cdim == shell.cdim, (
        f"surface cdim should equal parent.cdim, got {surface.cdim} "
        f"(parent {shell.cdim})"
    )

    radii = np.linalg.norm(np.asarray(surface.X.coords), axis=1)
    max_dev = np.abs(radii - 1.0).max()
    assert max_dev < 1.0e-10, (
        f"surface vertices not on parent sphere: max deviation {max_dev:.3e}"
    )
    uw.pprint(
        0,
        f"valid extraction: dim={surface.dim}, cdim={surface.cdim}, "
        f"max|r-1| = {max_dev:.3e}",
    )

    banner("CONTRACT 4: round-trip is bit-exact at surface DOFs")
    T_p = uw.discretisation.MeshVariable("T_p", shell, 1, degree=1)
    T_s = uw.discretisation.MeshVariable("T_s", surface, 1, degree=1)
    T_back = uw.discretisation.MeshVariable("T_back", shell, 1, degree=1)

    parent_coords = np.asarray(T_p.coords)
    new_T = np.zeros_like(T_p.data)
    # plant a recognisable 3-coord function
    new_T[:, 0] = parent_coords[:, 0] + 2 * parent_coords[:, 1] - parent_coords[:, 2]
    T_p.pack_raw_data_to_petsc(new_T, sync=True)

    surface.restrict(T_p, T_s)
    surface.prolongate(T_s, T_back)

    nonzero = np.where(np.abs(T_back.data[:, 0]) > 0)[0]
    diff = T_p.data[nonzero, 0] - T_back.data[nonzero, 0]
    max_err = np.abs(diff).max() if diff.size else 0.0
    n_surface_dofs = T_s.data.shape[0]
    uw.pprint(
        0,
        f"DOFs touched in round-trip: {nonzero.shape[0]} "
        f"(surface has {n_surface_dofs} DOFs)",
    )
    uw.pprint(0, f"max round-trip error: {max_err:.3e}")
    assert nonzero.shape[0] >= n_surface_dofs, (
        "round-trip should touch at least every surface DOF on the parent"
    )
    assert max_err < 1.0e-12, (
        f"round-trip not bit-exact: max error {max_err:.3e}"
    )

    banner("CONTRACT 5: cell navigation returns valid surface cells")
    # On a convex manifold (sphere) with on-surface query points,
    # centroid kd-tree ranks faces by chord distance which agrees with
    # the owning-face ranking to first order. With the phantom DAG
    # stripped (two-stage filter), the chart is clean and the standard
    # closest-cells path works.
    cS, cE = surface.dm.getHeightStratum(0)
    n_surface_cells = cE - cS
    # Use surface vertex coordinates as query points — they are
    # on-surface by construction.
    query = np.asarray(surface.X.coords)[:5]
    cells = surface.get_closest_cells(query)
    uw.pprint(0,
              f"get_closest_cells on 5 surface vertices -> {cells.tolist()}")
    assert cells.shape == (5,), (
        f"closest_cells should return a 1-D array, got shape {cells.shape}"
    )
    assert (cells >= 0).all() and (cells < n_surface_cells).all(), (
        f"returned cell ids {cells.tolist()} fall outside surface cell "
        f"range [0, {n_surface_cells})"
    )

    banner("CONTRACT 6: evaluate(expr, surface_coords) works")
    # Plant a 3-coord expression on a surface variable, then evaluate
    # the symbolic expression at the surface DOF coords. Result should
    # match the direct numpy eval at those coords.
    T_evaluate = uw.discretisation.MeshVariable(
        "T_evaluate", surface, 1, degree=1,
    )
    surface_coords = np.asarray(T_evaluate.coords)
    expected = (
        surface_coords[:, 0]
        + 2.0 * surface_coords[:, 1]
        - surface_coords[:, 2]
    )
    new_T = np.zeros_like(T_evaluate.data)
    new_T[:, 0] = expected
    T_evaluate.pack_raw_data_to_petsc(new_T, sync=True)

    # Evaluate at a subset of the DOF coordinates — query points lie
    # on the manifold by construction.
    query_pts = surface_coords[::13]  # sample
    vals = np.asarray(
        uw.function.evaluate(T_evaluate.sym[0], query_pts)
    ).reshape(-1)
    expected_at_query = (
        query_pts[:, 0] + 2.0 * query_pts[:, 1] - query_pts[:, 2]
    )
    max_err = np.abs(vals - expected_at_query).max()
    uw.pprint(0,
              f"evaluate at {query_pts.shape[0]} surface points: "
              f"max |val - expected| = {max_err:.3e}")
    assert max_err < 1.0e-10, (
        f"evaluate disagreement on the surface: max error {max_err:.3e}"
    )

    banner("CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
