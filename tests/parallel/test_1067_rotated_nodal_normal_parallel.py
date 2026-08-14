"""Parallel regression for the measure-weighted rotated nodal normal (issue #560).

The nodal normal is a sum over ALL facets meeting the node, so it only survives
partitioning if the sum is completed across ranks. Each boundary facet is labelled on
exactly one rank, so a node on a partition seam sees only a SUBSET of its facets
locally: accumulating rank-locally gives a seam node a different normal on each rank
and makes the answer rank-count dependent. Two further traps live here — a rank can OWN
a boundary node whose labelled facets are all on neighbours, and an outward-orientation
test taken against the mean of the rank's own coordinates can orient two facets of the
same node oppositely so that they CANCEL in the sum.

Measured on the skewed annulus (both arcs exactly circular, nodes moved to non-uniform
angles so only the facet LENGTHS change), geometric normal, ``|A z|/|A|_F``:

    np   before        after
     1   1.541e-07     6.624e-20
     2   1.611e-06     6.623e-20
     4   1.392e-05     6.690e-20

Run with::

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1067_rotated_nodal_normal_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1067_rotated_nodal_normal_parallel.py
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw
import underworld3.utilities.rotated_bc as rbc

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(300)]


def _skewed_annulus(tag, skew=0.04):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.55, cellSize=0.15,
                              qdegree=3)
    c = mesh.X.coords.copy()
    th = np.arctan2(c[:, 1], c[:, 0])
    r = np.hypot(c[:, 0], c[:, 1])
    th = th + skew * np.sin(3.0 * th)
    mesh.deform(np.column_stack([r * np.cos(th), r * np.sin(th)]))
    v = uw.discretisation.MeshVariable(f"vN{tag}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"pN{tag}", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    s.bodyforce = sympy.Matrix([[x, y]]) * sympy.cos(4 * sympy.atan2(y, x))
    s.penalty = 0.0
    s.tolerance = 1e-9
    s.add_rotated_freeslip_bc(0, "Upper")            # geometric normal
    s.add_rotated_freeslip_bc(0, "Lower")
    s.petsc_use_pressure_nullspace = True
    return s


class _ConstantPressureResidual:
    """``|A z|/|A|_F`` for the attached constant pressure, by PETSc matvec only.

    The Frobenius norm is not sigma_max, so this is NOT comparable with the serial
    dense-SVD number in ``tests/test_1018_rotated_nodal_normal.py`` — it is only
    comparable with itself across rank counts, which is exactly what is at issue."""

    def __enter__(self):
        self.value = None
        self._orig = rbc._solve_rotated_iterative

        def _capture(solver, Ahat, bhat, Q, Qt, normal_rows, **kw):
            if self.value is None and kw.get("nsp") is not None:
                z = kw["nsp"].getVecs()[0].copy()
                z.normalize()
                r = Ahat.createVecLeft()
                Ahat.mult(z, r)
                self.value = r.norm() / Ahat.norm(2)     # 2 == NORM_FROBENIUS
                z.destroy()
                r.destroy()
            return self._orig(solver, Ahat, bhat, Q, Qt, normal_rows, **kw)

        rbc._solve_rotated_iterative = _capture
        return self

    def __exit__(self, *exc):
        rbc._solve_rotated_iterative = self._orig
        return False


def test_nodal_normal_is_bit_identical_across_ranks():
    """Every rank holding a shared boundary node must compute the SAME normal, to the
    last bit — otherwise the constraint frame depends on the partition."""
    s = _skewed_annulus("B")
    s.solve()
    dm, dim = s.dm, s.mesh.dim
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)

    local = []
    for wall in ("Upper", "Lower"):
        for q, nrm in rbc._boundary_velocity_nodes(s, wall):
            c = rbc._point_coord(dm, dim, cvec, csec, v0, v1, q)
            local.append((wall, tuple(round(float(t), 9) for t in c),
                          tuple(float(t) for t in nrm)))

    gathered = uw.mpi.comm.gather(local, root=0)
    if uw.mpi.rank == 0:
        seen = {}
        shared = disagreeing = 0
        worst = 0.0
        for rows in gathered:
            for wall, coord, nrm in rows:
                key = (wall, coord)
                if key in seen:
                    shared += 1
                    d = max(abs(a - b) for a, b in zip(seen[key], nrm))
                    worst = max(worst, d)
                    disagreeing += d > 0.0
                else:
                    seen[key] = nrm
        assert shared > 0, "no boundary node is shared — the test proves nothing"
        assert disagreeing == 0, (
            f"{disagreeing} of {shared} shared boundary nodes disagree between "
            f"ranks, max |dn| = {worst:.3e}; the nodal normal is partition dependent")


def test_constant_pressure_is_a_null_vector_in_parallel():
    """The serial result must survive partitioning: rank-locally accumulated normals
    read 1.6e-06 at np=2 and 1.4e-05 at np=4 against 6.6e-20 here."""
    with _ConstantPressureResidual() as cap:
        _skewed_annulus("R").solve()
    assert cap.value is not None, "the operator capture never fired"
    assert cap.value < 1e-15, (
        f"|A z|/|A|_F = {cap.value:.3e} at np={uw.mpi.size}; the constant pressure is "
        "not a null vector of the constrained operator (#560)")
