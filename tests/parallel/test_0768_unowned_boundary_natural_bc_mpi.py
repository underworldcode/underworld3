"""Natural BCs and boundary integrals on a boundary a rank owns NO part of.

This is the case the removed ``Null_Boundary`` workaround claimed to need. It
added a fake natural BC on a label marking every vertex, because "we need a
surface integral term somewhere on every process if we have a contribution from
anywhere". The workaround was removed after measuring that it changed no answer
at any rank count — value 666 marks no facet, so it integrated nothing, and
natural BCs are registered against ``UW_Boundaries`` unconditionally on every
rank anyway.

**What these tests are.** A tripwire on current behaviour, not a proof that the
PETSc problem the workaround was aimed at is solved. Removing a failed fix does
not establish that the thing it failed to fix is gone. If it resurfaces, these
tests are what will show it — and the answer then is a real fix, not a sentinel
label that integrates nothing.

**Why the suite needed them.** Before this file, the only parallel test creating
a natural BC was ``test_0770_submesh_extract_mpi``, whose annulus boundaries are
owned by every rank at np=2 and np=4 (``Internal`` [26,26] and [12,14,13,13]).
No test exercised a rank owning none, so nothing would catch a rank-local skip
being added to the DS registration — which the dead
``bc_is = bc_label.getStratumIS(value)`` in ``_setup_discretisation`` invites.

The geometry is a LONG FLAT box, so the partitioner cuts along x and the short
end faces land on one rank each. ``test_the_unowned_case_is_exercised`` asserts
that actually happened: without it a partitioner change could make every other
test here pass vacuously.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [
    pytest.mark.level_2,
    pytest.mark.tier_a,
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(300),
]

LX, LY = 4.0, 0.25
CELL = 0.05


def _mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(LX, LY), cellSize=CELL,
        regular=False, qdegree=3)


def _owned_facets(mesh, name):
    """Facets of ``name`` this rank OWNS, gathered to a per-rank list on rank 0.

    Owned, not merely present: a seam facet sits on every rank sharing it, and a
    local count reports it once per sharer. A point is a ghost exactly when it is
    a leaf of the point star-forest.
    """
    dm = mesh.dm
    ghosts = set()
    if uw.mpi.size > 1:
        _nroots, ilocal, iremote = dm.getPointSF().getGraph()
        ghosts = (set(range(0 if iremote is None else len(iremote)))
                  if ilocal is None else set(int(p) for p in ilocal))

    fS, fE = dm.getHeightStratum(1)
    n = 0
    label = dm.getLabel(name)
    if label is not None:
        ist = label.getStratumIS(int(mesh.boundaries[name].value))
        # A rank owning NO point with this value gets back a valid IS of SIZE
        # ZERO, and `.getIndices()` on it segfaults (issue #291). `is not None`
        # does not catch it, and the rank that trips it is precisely the one
        # this module is about.
        if ist and ist.getSize() > 0:
            pts = np.asarray(ist.getIndices(), dtype=np.int64)
            pts = pts[(pts >= fS) & (pts < fE)]
            n = sum(1 for p in pts if int(p) not in ghosts)
    return uw.mpi.comm.allgather(n)


def test_the_unowned_case_is_exercised():
    """Some rank must own no facet of `Right`, or the rest of this file is vacuous."""
    counts = _owned_facets(_mesh(), "Right")
    assert sum(counts) > 0, f"nobody owns the boundary at all: {counts}"
    assert min(counts) == 0, (
        f"every rank owns part of Right ({counts}); this file is meant to test "
        f"the case where at least one rank owns none, and no longer does")


def test_boundary_integral_over_an_unowned_boundary():
    """`BdIntegral` is exact on a boundary most ranks hold no part of."""
    mesh = _mesh()
    x, y = mesh.X
    _T = uw.discretisation.MeshVariable("Tbi", mesh, 1, degree=2)

    assert min(_owned_facets(mesh, "Left")) == 0

    assert np.isclose(uw.maths.BdIntegral(mesh, 1.0, "Left").evaluate(),
                      LY, rtol=1e-12)
    assert np.isclose(uw.maths.BdIntegral(mesh, y, "Left").evaluate(),
                      0.5 * LY ** 2, rtol=1e-12)
    # A boundary every rank DOES own, as the control on the same mesh.
    assert np.isclose(uw.maths.BdIntegral(mesh, 1.0, "Top").evaluate(),
                      LX, rtol=1e-12)


def test_natural_bc_on_an_unowned_boundary_is_assembled():
    """A Neumann term on a boundary most ranks own nothing of reaches the residual.

    Checked against the manufactured solution ``T = x^2``, which P2 reproduces
    EXACTLY, with the tolerance tightened well below the bundle default of
    ``snes_rtol`` 1e-4. The error then sits at the solver tolerance and a dropped
    surface term is orders of magnitude, not a wobble: zeroing the Right flux
    moves it from 3.6e-11 to 8.0e-03.

    Convergence alone would prove nothing — a silently dropped surface term
    still converges, just to the wrong field.
    """
    mesh = _mesh()
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("Tnbc", mesh, 1, degree=2)

    assert min(_owned_facets(mesh, "Right")) == 0

    poisson = uw.systems.Poisson(mesh=mesh, u_Field=T, degree=2)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = -2.0
    poisson.add_dirichlet_bc([x ** 2], "Bottom")
    poisson.add_dirichlet_bc([x ** 2], "Top")
    poisson.add_natural_bc([-2 * x], "Right")
    poisson.add_natural_bc(0.0, "Left")
    # `petsc_options["ksp_rtol"]` is silently overridden; use `.tolerance`.
    poisson.tolerance = 1.0e-10
    poisson.solve()

    assert poisson.snes.getConvergedReason() > 0

    exact = x ** 2
    err = uw.maths.Integral(mesh, (T.sym[0] - exact) ** 2).evaluate() ** 0.5
    norm = uw.maths.Integral(mesh, exact ** 2).evaluate() ** 0.5
    assert err / norm < 1.0e-8, f"relative L2 error {err / norm:.3e}"

    # No condition is manufactured behind the user's back.
    assert [bc.boundary for bc in poisson.natural_bcs] == ["Right", "Left"]


def test_vector_natural_bc_on_an_unowned_boundary_changes_the_answer():
    """The VECTOR path: a Stokes traction on an unowned boundary is live.

    The removed workaround built its fake condition as ``(0,)*u.shape[1]``, so
    the vector case is the one it was shaped for. Asserted by DIFFERENCE against
    the same problem with no condition on Right, rather than against a stored
    number: if the traction on the boundary most ranks own nothing of never
    reached the residual, the two solves would agree.

    Lid-driven, and the lid profile vanishes at both ends. A uniform body force
    in a closed incompressible box is hydrostatic — v is identically zero, and a
    probe whose answer is zero by construction cannot detect a dropped term
    however many ranks own nothing of the boundary.
    """
    def _vrms(with_traction):
        mesh = _mesh()
        x, y = mesh.X
        tag = "t" if with_traction else "n"
        v = uw.discretisation.MeshVariable(f"v{tag}", mesh, mesh.dim, degree=2)
        p = uw.discretisation.MeshVariable(f"p{tag}", mesh, 1, degree=1,
                                           continuous=True)
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        stokes.bodyforce = sympy.Matrix([0.0, 0.0])
        stokes.add_dirichlet_bc((sympy.sin(sympy.pi * x / LX), 0.0), "Top")
        stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((0.0, 0.0), "Left")
        if with_traction:
            stokes.add_natural_bc((0.0, -0.5), "Right")
        stokes.tolerance = 1.0e-10
        stokes.solve()
        assert stokes.snes.getConvergedReason() > 0
        return (uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()
                / (LX * LY)) ** 0.5

    free = _vrms(False)
    loaded = _vrms(True)
    assert abs(loaded - free) / free > 1.0e-5, (
        f"the traction on Right did not reach the residual: "
        f"{free:.12e} vs {loaded:.12e}")
