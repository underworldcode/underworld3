"""P1 manual barycentric evaluator — band-aid for the cell-hinting fix.

Validates: if we evaluate T using the cell *we* picked (via centroid
kdtree) and manual barycentric, do the streak / antipode artefacts
disappear?

Test cases:
  (a) DOF 2107's trace-back endpoint, where DMInterpolation returned
      0.098 instead of 0 — should now return 0.
  (b) Sweep over all DOFs after 1 step and compare manual eval to
      DMInterpolation eval. Identify all DOFs where they disagree.
  (c) Compare both to analytic-rotated-Gaussian.
"""
import numpy as np
import underworld3 as uw


def manual_p1_evaluate(mesh, T, coords):
    """Evaluate a scalar P1 MeshVariable T at arbitrary coords on a
    manifold using cell hint from centroid kdtree + barycentric.

    Per-coord loop, O(n_query). Vectorisation possible but not done.

    The barycentric is computed via least-squares projection (Moore-
    Penrose pseudo-inverse of the 3×2 Jacobian) — matches what PETSc-FE
    does internally (DMPlexLocatePoint_Simplex_2D_Internal). For an
    in-plane point this gives exact barycentric; for an off-plane
    point it gives the perpendicular projection's barycentric.
    """
    mesh._build_kd_tree_index()
    nav_dm = mesh._nav_dm if mesh._nav_dm is not None else mesh.dm
    nav_coords = mesh._nav_coords
    cStart, _ = nav_dm.getHeightStratum(0)
    pStart, _ = nav_dm.getDepthStratum(0)
    cell_num_points = mesh.element.entities[mesh.dim]

    _, closest = mesh._centroid_index.query(coords, k=1, sqr_dists=False)
    closest = np.asarray(closest).reshape(-1).astype(int)

    # T.coords for DOF lookup. Use a kdtree of T DOF coords for quick
    # mapping vertex_coord → DOF index.
    T_coords = np.asarray(T.coords)
    T_dof_kdtree = uw.kdtree.KDTree(np.ascontiguousarray(T_coords))

    n = coords.shape[0]
    values = np.zeros(n)
    for i in range(n):
        cid = closest[i]
        cone_pts = nav_dm.getTransitiveClosure(cStart + cid)[0][-cell_num_points:]
        vtx = nav_coords[cone_pts - pStart]
        a, b, c = vtx[0], vtx[1], vtx[2]

        # Barycentric via 2×2 normal-equations (LS projection along cell normal).
        ab = b - a
        ac = c - a
        ap = coords[i] - a
        gram = np.array([[ab @ ab, ab @ ac], [ab @ ac, ac @ ac]])
        rhs = np.array([ab @ ap, ac @ ap])
        sol = np.linalg.solve(gram, rhs)
        beta_, gamma_ = sol
        alpha_ = 1.0 - beta_ - gamma_

        # Find DOF indices at the three vertices via the kdtree.
        _, dof_idxs = T_dof_kdtree.query(vtx, k=1, sqr_dists=False)
        dof_idxs = np.asarray(dof_idxs).reshape(-1).astype(int)

        T_a = float(T.data[dof_idxs[0], 0])
        T_b = float(T.data[dof_idxs[1], 0])
        T_c = float(T.data[dof_idxs[2], 0])

        values[i] = alpha_ * T_a + beta_ * T_b + gamma_ * T_c

    return values


def main():
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.075)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    coords = np.asarray(T.coords)

    sigma = 0.3
    T.data[:, 0] = np.exp(
        -((coords[:, 0] - 1.0)**2 + coords[:, 1]**2 + coords[:, 2]**2) / (2 * sigma**2)
    )

    print("=== (a) DOF 2107's bad trace-back endpoint ===")
    # End point from earlier diagnostic.
    end_pt = np.array([[-0.79784511, -0.59911955, 0.06707759]])
    print(f"   coord: {end_pt[0]}  (antipode region)")
    dmi_val = float(np.asarray(uw.function.evaluate(T.sym, end_pt)).reshape(-1)[0])
    manual_val = float(manual_p1_evaluate(mesh, T, end_pt)[0])
    analytic_val = float(np.exp(
        -((end_pt[0, 0] - 1.0)**2 + end_pt[0, 1]**2 + end_pt[0, 2]**2) / (2 * sigma**2)
    ))
    print(f"   DMInterpolation: {dmi_val:+.6f}   ← expected to be wrong")
    print(f"   Manual P1:       {manual_val:+.6f}   ← should be ~0")
    print(f"   Analytic:        {analytic_val:+.6f}")

    print("\n=== (b) Compare manual P1 vs DMInterpolation across all DOFs ===")
    # Trace-back from each DOF: end_pt = coord - V * dt where V = (-y, x, 0).
    dt = 2 * np.pi / 72  # one step = 5°
    end_pts = np.zeros_like(coords)
    end_pts[:, 0] = coords[:, 0] + coords[:, 1] * dt
    end_pts[:, 1] = coords[:, 1] - coords[:, 0] * dt
    end_pts[:, 2] = coords[:, 2]
    # Radial projection to r=1.
    r = np.linalg.norm(end_pts, axis=1)
    end_pts = end_pts / r.reshape(-1, 1)

    T_dmi = np.asarray(uw.function.evaluate(T.sym, end_pts)).reshape(-1)
    T_manual = manual_p1_evaluate(mesh, T, end_pts)

    # Analytic: T at the trace-back endpoint (one-step backward rotation).
    T_analytic = np.exp(
        -((end_pts[:, 0] - 1.0)**2 + end_pts[:, 1]**2 + end_pts[:, 2]**2) / (2 * sigma**2)
    )

    # Errors vs analytic.
    err_dmi = np.abs(T_dmi - T_analytic)
    err_manual = np.abs(T_manual - T_analytic)

    print(f"  DMInterpolation: max|err| = {err_dmi.max():.4e}, "
          f"mean|err| = {err_dmi.mean():.4e}")
    print(f"  Manual P1:       max|err| = {err_manual.max():.4e}, "
          f"mean|err| = {err_manual.mean():.4e}")

    # Identify DOFs where DMInterpolation is bad.
    wild_dmi = np.where(err_dmi > 0.05)[0]
    print(f"\n  DOFs where DMInterpolation has |err| > 0.05: {len(wild_dmi)}")
    print(f"  DOFs where Manual P1 has |err| > 0.05:        "
          f"{int((err_manual > 0.05).sum())}")

    # Spot-check: show a few wild DMI DOFs and what Manual gave there.
    if len(wild_dmi) > 0:
        worst = wild_dmi[np.argsort(err_dmi[wild_dmi])[::-1][:6]]
        print(f"\n  Top wild DMInterpolation DOFs and Manual values:")
        print(f"    {'idx':>5} {'analytic':>9} {'dmi':>9} {'manual':>9} "
              f"{'|dmi-an|':>9} {'|man-an|':>9}")
        for idx in worst:
            print(f"    {int(idx):>5} {T_analytic[idx]:>+9.4f} {T_dmi[idx]:>+9.4f} "
                  f"{T_manual[idx]:>+9.4f} {err_dmi[idx]:>9.4e} {err_manual[idx]:>9.4e}")


if __name__ == "__main__":
    main()
