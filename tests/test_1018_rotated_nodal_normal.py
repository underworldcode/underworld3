"""The rotated free-slip nodal normal must be consistent with the assembly (issue #560).

A boundary node on more than one facet gets ONE nodal normal while the assembler
integrates the boundary term facet by facet. If that normal is the plain bisector of
its facet normals, a kinked boundary with unequal facets leaves a residual in the
node's FREE tangential row, the exact constant-pressure vector stops being a null
vector of the constrained rotated operator, and the pressure gauge goes unpinned.

The measurable is ``|A z| / |A|`` with ``z`` the constant-pressure vector UW3 attaches
as the null space and ``A`` the constrained operator handed to the KSP. It is the right
metric because it scales smoothly with the deformation (amplitude cubed) — ``mean(p)``
and the run-to-run velocity move are the ROUND-OFF amplitude of an unpinned direction
and are chaotic, so they report the presence of the defect but not its size.

Each check carries its negative control: the same measurement with the pre-fix
bisector normal, which must FAIL the tolerance. Without that the tests would pass on
a metric that cannot see the defect.
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw
import underworld3.utilities.rotated_bc as rbc

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

_MACHINE = 1e-15         # |A z|/|A| at machine level; the straight box sits at 3e-18


def _bisector_nodes(solver, boundary, normal=None):
    """The pre-#560 accumulation, frozen here as the negative control.

    Identical to ``rbc._boundary_velocity_nodes`` except that each facet normal is
    normalised to unit length BEFORE accumulating, so a node's normal is the bisector
    of its facet normals rather than the measure-weighted average. The analytic-normal
    path is delegated to the real function — it is unchanged by the fix.
    """
    if normal is not None:
        return rbc._boundary_velocity_nodes(solver, boundary, normal=normal)
    dm = solver.dm
    dim = solver.mesh.dim
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    lsec = dm.getLocalSection()
    interior_ref = cvec.mean(axis=0)
    sis = rbc._boundary_stratum_is(dm, solver.mesh, boundary)
    if not (sis and sis.getSize() > 0):
        return []
    fS, fE = dm.getHeightStratum(1)
    nacc, pts = {}, set()
    for f in (int(z) for z in sis.getIndices()):
        if not (fS <= f < fE):
            continue
        _, cent, nrm = dm.computeCellGeometryFVM(f)
        ne = np.asarray(nrm, dtype=float)
        ne = ne / (np.linalg.norm(ne) + 1e-30)
        if np.dot(ne, np.asarray(cent) - interior_ref) < 0:
            ne = -ne
        for q in (int(c) for c in dm.getTransitiveClosure(f)[0]):
            if lsec.getFieldDof(q, rbc._VELOCITY_FIELD) <= 0:
                continue
            nacc[q] = nacc.get(q, np.zeros(dim)) + ne
            pts.add(q)
    return [(q, nacc[q] / (np.linalg.norm(nacc[q]) + 1e-30)) for q in pts]


class _CaptureOperator:
    """Capture the constrained rotated operator and its attached null space.

    The operator only exists inside the rotated solve, so the linear-solve entry point
    is wrapped for the duration of the ``with`` block. Nothing else can see it.
    """

    def __init__(self):
        self.A = None
        self.nsp = None

    def __enter__(self):
        self._orig = rbc._solve_rotated_iterative

        def _capture(solver, Ahat, bhat, Q, Qt, normal_rows, **kw):
            if self.A is None:
                n = Ahat.getSize()[0]
                ai, aj, av = Ahat.getValuesCSR()   # never MatConvert: it breaks the PC
                dense = np.zeros((n, n))
                for r in range(n):
                    dense[r, aj[ai[r]:ai[r + 1]]] = av[ai[r]:ai[r + 1]]
                self.A = dense
                self.nsp = [np.asarray(v.array_r).copy()
                            for v in kw["nsp"].getVecs()]
            return self._orig(solver, Ahat, bhat, Q, Qt, normal_rows, **kw)

        rbc._solve_rotated_iterative = _capture
        return self

    def __exit__(self, *exc):
        rbc._solve_rotated_iterative = self._orig
        return False

    @property
    def constant_pressure_residual(self):
        """``|A z| / |A|`` for the attached constant-pressure vector."""
        z = self.nsp[0] / np.linalg.norm(self.nsp[0])
        return np.linalg.norm(self.A @ z) / np.linalg.norm(self.A, 2)


class _BisectorNormals:
    """Run the enclosed solves with the pre-#560 bisector nodal normal."""

    def __enter__(self):
        self._orig = rbc._boundary_velocity_nodes
        rbc._boundary_velocity_nodes = _bisector_nodes
        return self

    def __exit__(self, *exc):
        rbc._boundary_velocity_nodes = self._orig
        return False


def _deformed_box(tag, amplitude, res=10):
    """Configuration B of issue #560: a box whose top is bowed by
    ``amplitude * y * sin(pi x)`` — a kinked rotated wall with unequal facets."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    if amplitude != 0.0:
        c = mesh.X.coords.copy()
        c[:, 1] += amplitude * c[:, 1] * np.sin(np.pi * c[:, 0])
        mesh.deform(c)
    return _rotated_box_solver(tag, mesh)


def _rotated_box_solver(tag, mesh):
    v = uw.discretisation.MeshVariable(f"vK{tag}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"pK{tag}", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    s.bodyforce = sympy.Matrix([[0.0, sympy.sin(sympy.pi * x) * sympy.cos(sympy.pi * y)]])
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    return s


def _skewed_annulus(tag, skew):
    """Both boundaries stay exactly circular; the nodes are moved to non-uniform
    angles, so only the facet LENGTHS change. Geometric (default) normals."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.55, cellSize=0.15,
                              qdegree=3)
    if skew != 0.0:
        c = mesh.X.coords.copy()
        th = np.arctan2(c[:, 1], c[:, 0])
        r = np.hypot(c[:, 0], c[:, 1])
        th = th + skew * np.sin(3.0 * th)
        mesh.deform(np.column_stack([r * np.cos(th), r * np.sin(th)]))
    v = uw.discretisation.MeshVariable(f"vA{tag}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"pA{tag}", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    s.bodyforce = sympy.Matrix([[x, y]]) * sympy.cos(4 * sympy.atan2(y, x))
    s.penalty = 0.0
    s.tolerance = 1e-9
    s.add_rotated_freeslip_bc(0, "Upper")
    s.add_rotated_freeslip_bc(0, "Lower")
    s.petsc_use_pressure_nullspace = True
    return s


def _residual(build, *args):
    with _CaptureOperator() as cap:
        build(*args).solve()
        return cap.constant_pressure_residual


def test_deformed_box_constant_pressure_stays_a_null_vector():
    """Configuration B: the attached constant pressure must annihilate the constrained
    operator on a kinked rotated wall, as it already does on a straight one."""
    straight = _residual(_deformed_box, "S", 0.0)
    deformed = _residual(_deformed_box, "D", 0.02)
    assert straight < _MACHINE, f"straight box control |Az|/|A| = {straight:.3e}"
    assert deformed < _MACHINE, (
        f"deformed rotated free-slip box leaves |Az|/|A| = {deformed:.3e}; the "
        "constant pressure is not a null vector and the gauge is unpinned (#560)")


def test_bisector_normal_fails_the_same_measurement():
    """Negative control: with the pre-fix bisector normal the metric must FIRE.

    Without this the test above could pass on a measurement that cannot see the
    defect at all."""
    with _BisectorNormals():
        deformed = _residual(_deformed_box, "N", 0.02)
        straight = _residual(_deformed_box, "M", 0.0)
    assert deformed > 1e-11, (
        f"the bisector normal gave |Az|/|A| = {deformed:.3e}; the metric is not "
        "detecting the defect it is supposed to detect")
    assert straight < _MACHINE, (
        "the straight box is clean under BOTH normals — the negative control must "
        "isolate the kink, not the whole rotated path")


def test_skewed_annulus_constant_pressure_stays_a_null_vector():
    """A curved rotated wall is clean while its facets are equal and acquires the
    defect as soon as they are not — the case that separates the mechanism (unequal
    kinked facets) from the symptom (a deformed box)."""
    skewed = _residual(_skewed_annulus, "K", 0.04)
    with _BisectorNormals():
        skewed_bisector = _residual(_skewed_annulus, "B", 0.04)
    assert skewed < _MACHINE, (
        f"skewed annulus leaves |Az|/|A| = {skewed:.3e} (#560)")
    assert skewed_bisector > 1e-11, (
        f"negative control did not fire on the skewed annulus "
        f"({skewed_bisector:.3e})")


def test_flat_walls_and_analytic_normals_are_bit_identical():
    """The weighting is a no-op wherever a node's facets share a normal, so the
    straight box and the analytic-normal override must be unchanged to the last bit."""
    straight = _deformed_box("I", 0.0)
    straight.solve()
    for wall in ("Top", "Bottom", "Left", "Right"):
        fixed = dict(rbc._boundary_velocity_nodes(straight, wall))
        old = dict(_bisector_nodes(straight, wall))
        assert fixed.keys() == old.keys()
        for q in fixed:
            assert np.array_equal(fixed[q], old[q]), (
                f"straight wall {wall!r} node {q} normal moved: "
                f"{fixed[q]} vs {old[q]}")

    annulus = _skewed_annulus("J", 0.04)
    annulus.solve()
    x, y = annulus.mesh.X
    radial = sympy.Matrix([[x, y]]) / sympy.sqrt(x**2 + y**2)
    for arc in ("Upper", "Lower"):
        fixed = dict(rbc._boundary_velocity_nodes(annulus, arc, normal=radial))
        old = dict(_bisector_nodes(annulus, arc, normal=radial))
        for q in fixed:
            assert np.array_equal(fixed[q], old[q]), (
                f"analytic normal on {arc!r} node {q} moved")

    # and the control: on the KINKED geometric path the two must differ, or the
    # comparison above is vacuous.
    kinked = _deformed_box("H", 0.02)
    kinked.solve()
    fixed = dict(rbc._boundary_velocity_nodes(kinked, "Top"))
    old = dict(_bisector_nodes(kinked, "Top"))
    assert any(not np.array_equal(fixed[q], old[q]) for q in fixed), (
        "the deformed top gave identical normals under both rules — the "
        "bit-identity checks above prove nothing")


@pytest.mark.level_2
def test_residual_does_not_grow_with_deformation_amplitude():
    """Before the fix ``|A z|`` grew as amplitude cubed. After it, the deformation
    must not drive it at all."""
    amplitudes = (0.0, 0.005, 0.02, 0.05)
    fixed = [_residual(_deformed_box, f"A{k}", a) for k, a in enumerate(amplitudes)]
    assert max(fixed) < _MACHINE, (
        "|Az|/|A| over the amplitude sweep " +
        ", ".join(f"{a:g}: {r:.3e}" for a, r in zip(amplitudes, fixed)))

    with _BisectorNormals():
        control = [_residual(_deformed_box, f"C{k}", a) for k, a in enumerate(amplitudes)]
    assert control[-1] > 100 * control[1], (
        "the negative control did not reproduce the growth with amplitude: " +
        ", ".join(f"{a:g}: {r:.3e}" for a, r in zip(amplitudes, control)))
