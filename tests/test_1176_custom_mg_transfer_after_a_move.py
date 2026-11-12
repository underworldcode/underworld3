"""A move that keeps the topology keeps the custom multigrid transfers.

``mesh.deform`` and the movers change where the nodes of the finest level sit
and nothing else: every cell keeps its vertices, and every node its parent cell
on the level below. The transfer between the two levels is a statement about
that relation, so it is the same before and after the move. Rebuilt from the
moved coordinates it is not: a node of the moved level is located among the
unmoved coarse nodes and the weights follow the displacement. On the
split-level fixture below that takes the solve from 5 velocity iterations to
13 at the first move, and on the faulted fixture of test_1152 it takes the
smallest diagonal entry of a level operator from 2.4 to 4.5e-4.

The hierarchy builds a moved level's transfers on the geometry the level was
made in, so the transfers after a move are those before it. That holds on each
route a level pair can take: the node-cloud builder of a placed level, the
exact embedding of a native refine pair, and the cell-located transfer between
split levels.

A move that changes the topology makes a new mesh, and a new mesh has its own
as-built geometry: nothing is kept across it, and a mesh made on a moved one is
read where it is. A relax of a mesh that stands where it was made is part of
making it, and coordinates that were written and put back are not a move.

The file is for one rank: the transfers are compared as sequential matrices.
"""
import numpy as np
import pytest
import scipy.sparse as sp
import sympy

import underworld3 as uw
from underworld3.utilities import custom_mg, fault_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

TRACE = np.array([[0.3, 0.45], [0.7, 0.55]])


def _placed_poisson(move_the_base=None):
    """A split fault placed on a refined box: the finest pair is the node-cloud
    builder's, the pair under it a native refine pair. ``move_the_base`` moves
    the box before the fault is placed on it."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=False,
        qdegree=2, refinement=1)
    levels = base._coarse_level_meshes()[:-1] + [base]
    if move_the_base is not None:
        move_the_base(base)
    mesh = base.add_fault(("Flt", TRACE))
    s = uw.systems.Poisson(mesh)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.f = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc(0, wall)
    s.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(s, levels)
    return s, mesh


def _native_stokes():
    """A twice-refined box on its own levels: every pair is a native refine
    pair, the finest included."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 4, regular=False,
        qdegree=2, refinement=2)
    x, y = mesh.X
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc((y - sympy.Rational(1, 2), 0), wall)
    s.bodyforce = sympy.Matrix([0, sympy.sin(sympy.pi * x)])
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(s, list(mesh.multigrid_levels[:-1]), field_id=0)
    return s, mesh


def _split_level_stokes(move_the_host=None):
    """A fault on an adapt child: the add-on levels are split, and the finest
    pair takes the cell-located transfer. ``move_the_host`` moves the child
    before the fault is cut into it."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, refinement=1,
        qdegree=2)
    pts = np.column_stack([np.linspace(*TRACE[:, 0], 9), np.linspace(*TRACE[:, 1], 9),
                           np.zeros(9)])
    guide = uw.meshing.Surface(f"guide_1176_{move_the_host is not None}", base, pts)
    guide.discretize()
    child = base.adapt(guide.refinement_metric_function(
        h_near=0.05 / 4, h_far=0.05, width=0.08, profile="linear"), max_levels=2)
    if move_the_host is not None:
        move_the_host(child)
    mesh = child.add_fault([("Flt", TRACE)])
    assert fault_mg.is_split_level(mesh._custom_mg_coarse_meshes[-1])
    return _fault_stokes(mesh), mesh


def _fault_stokes(mesh):
    x, y = mesh.X
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    s.tolerance = 1.0e-6
    s.petsc_use_pressure_nullspace = True
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_dirichlet_bc((y - 0.5, 0.0), wall)
    s.add_fault_bc(0, "Flt")
    return s


def _deform(mesh, axis=0, amplitude=0.02):
    c = np.asarray(mesh.X.coords).copy()
    c[:, axis] += amplitude * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])   # zero on the walls
    mesh.deform(c)


def _deform_across_the_fault(mesh):
    # A host that moved by 0.005 or more before the cut is refused: the split
    # level under the new mesh is cut where the host's coarse level stands, and
    # the dependent-column check (#1101) finds coarse unknowns with too few
    # fine images.
    _deform(mesh, axis=1, amplitude=0.002)


def _redistribute(mesh):
    x, y = mesh.X
    uw.meshing.node_redistribution(
        mesh, 1 + 6 * sympy.exp(-(((x - 0.45) ** 2 + (y - 0.5) ** 2) / 0.03)))


def _transfers(solver):
    """Copies of the hierarchy's transfers as built, coarsest pair first. On
    one rank only: the rows read are those the rank owns."""
    out = []
    for P in solver._custom_mg["hierarchy"].transfers:
        indptr, indices, values = P.getValuesCSR()
        out.append(sp.csr_matrix((values, indices, indptr), shape=P.getSize()))
    return out


@pytest.mark.parametrize("build,move", [
    (_placed_poisson, _deform),
    (_placed_poisson, _redistribute),
    (_native_stokes, _deform),
    (_split_level_stokes, _deform),
], ids=["placed-deform", "placed-mover", "native-deform", "split-level-deform"])
def test_the_transfers_after_a_move_are_those_before_it(build, move):
    s, mesh = build()
    s.solve()
    assert s.snes.getConvergedReason() > 0
    before = _transfers(s)
    at_rest = np.asarray(mesh.X.coords).copy()

    move(mesh)
    # the fixture is only a test if the move is far larger than the tolerance
    assert np.abs(np.asarray(mesh.X.coords) - at_rest).max() > 5.0e-3
    s.solve()
    assert s.snes.getConvergedReason() > 0
    after = _transfers(s)

    # bit for bit: the moved level is read on the vertices it was built with;
    # rebuilt from the moved coordinates the weights changed by 0.65 to 1.1
    assert len(after) == len(before)
    for level, (P0, P1) in enumerate(zip(before, after), start=1):
        assert P1.shape == P0.shape, level
        assert abs(P1 - P0).max() == 0, (level, abs(P1 - P0).max())


def test_a_split_level_solve_after_a_move_keeps_its_iterations():
    """5 velocity iterations at rest and 6 after each of three moves, which is
    what a solver built on the moved mesh takes; with the transfer rebuilt from
    the moved coordinates the first move takes 13. The fixture carries no FAC
    patch."""
    s, mesh = _split_level_stokes()
    s.solve()
    counts = [max(s._rotated_freeslip_info["vel_its_last"])]
    for _ in range(3):
        _deform(mesh)
        s.solve()
        assert s.snes.getConvergedReason() > 0
        counts.append(max(s._rotated_freeslip_info["vel_its_last"]))
    assert counts == [5, 6, 6, 6]

    built_after = _fault_stokes(mesh)
    built_after.solve()
    assert max(built_after._rotated_freeslip_info["vel_its_last"]) == 6


def test_a_fault_cut_into_a_moved_mesh_is_read_where_it_is():
    """The child moved before the fault was cut into it, so the fault lies on
    its trace in the moved geometry on the new mesh and on the split level
    under it. The new mesh has not moved since it was made and is read on its
    own coordinates: 6 velocity iterations (5 with the child at rest)."""
    s, mesh = _split_level_stokes(move_the_host=_deform_across_the_fault)
    s.solve()
    assert s.snes.getConvergedReason() > 0
    assert max(s._rotated_freeslip_info["vel_its_last"]) == 6
    assert not mesh._moved_since_built
    assert fault_mg.as_built_nodes(mesh, 1, True) is None


def test_a_moved_quadrilateral_level_is_read_where_it_is():
    """The kept transfer is for simplex levels. The transfers of a quadrilateral
    level are rebuilt from the moved coordinates: 2 iterations at rest and
    after each of two moves."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0, 0), maxCoords=(1, 1), qdegree=2,
        refinement=2)
    s = uw.systems.Poisson(mesh)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.f = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc(0, wall)
    s.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(s, list(mesh.multigrid_levels[:-1]))
    s.solve()
    assert s.snes.getKSP().getIterationNumber() == 2
    for _ in range(2):
        _deform(mesh)
        assert fault_mg.as_built_nodes(mesh, 1, True) is None
        s.solve()
        assert s.snes.getConvergedReason() > 0
        assert s.snes.getKSP().getIterationNumber() == 2


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("continuous", [True, False])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_a_moved_level_gives_its_nodes_where_it_was_built(degree, continuous, dim):
    """Bit for bit. The node-cloud builder is not continuous in the nodes it is
    given: one on a face of its triangulation is read in either neighbour, and
    nodes 2e-16 from the built ones changed a weight of a 3-D transfer by
    1.4e-4 on 2, 4 and 8 ranks."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0,) * dim, maxCoords=(1,) * dim, cellSize=1 / 6 if dim == 2 else 1 / 3,
        regular=False, qdegree=3, refinement=1)
    # at rest the current coordinates are the as-built ones: nothing is mapped
    assert fault_mg.as_built_nodes(mesh, degree, continuous) is None
    built = np.asarray(mesh._get_coords_for_basis(degree, continuous)).copy()

    _deform(mesh)
    moved = np.asarray(mesh._get_coords_for_basis(degree, continuous))
    assert np.abs(moved - built).max() > 5.0e-3
    as_built = fault_mg.as_built_nodes(mesh, degree, continuous)
    assert np.array_equal(as_built, built), np.abs(as_built - built).max()


def _adapt_child(relax):
    """An adapt child, relaxed at the end of the adapt or not at all. The relax
    moves every interior vertex of the child."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, refinement=1,
        qdegree=2)
    h_near, h_far, width = 0.012, 0.30, 0.10
    slope = np.sqrt(h_far**2 - h_near**2) / width

    def metric(centroids):
        d = np.abs(np.asarray(centroids)[:, 1] - 0.5)
        return 1.0 / np.minimum(np.sqrt(h_near**2 + (slope * d) ** 2), h_far) ** 2

    return base.adapt(metric, max_levels=3, relax=relax)


def _box_poisson(mesh):
    x, y = mesh.X
    s = uw.systems.Poisson(mesh, degree=2)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.constitutive_model.Parameters.diffusivity = 1
    s.f = sympy.sin(3 * x) * sympy.cos(2 * y)
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc(0, wall)
    s.tolerance = 1.0e-8
    return s


def test_a_relaxed_adapt_child_is_read_where_the_relax_left_it():
    """The relax that finishes an adapt is part of making the child, not a move
    of it: the child's transfers are built on the relaxed coordinates and the
    solve takes 2 iterations. Read on the coordinates from before the relax it
    takes 3. A move of the child after that keeps those transfers. The parent of
    a cell on the level below is still read where the child was refined."""
    child = _adapt_child(relax=True)
    assert fault_mg.as_built_nodes(child, 2, True) is None
    refined = fault_mg.reference_vertices(child, child._custom_mg_coarse_meshes[-1])
    assert np.abs(refined - child._transfer_vertices).max() > 1.0e-3
    s = _box_poisson(child)
    s.solve()
    assert s.snes.getLinearSolveIterations() == 2
    before = _transfers(s)

    _deform(child)
    s.solve()
    assert s.snes.getConvergedReason() > 0
    for level, (P0, P1) in enumerate(zip(before, _transfers(s)), start=1):
        assert abs(P1 - P0).max() == 0, (level, abs(P1 - P0).max())


def test_a_relax_in_a_call_of_its_own_is_part_of_making_the_mesh():
    """``child = mesh.adapt(...)`` then ``child.relax()`` is the relax of the
    adapt in two calls, and a second relax is more of the same: 2 iterations
    after each, as for the relax inside the adapt. A relax of a mesh that has
    moved is one more move, taken back to where the transfers were built."""
    child = _adapt_child(relax=False)
    s = _box_poisson(child)
    for _ in range(2):
        child.relax()
        assert fault_mg.as_built_nodes(child, 2, True) is None
        s.solve()
        assert s.snes.getLinearSolveIterations() == 2

    relaxed = np.asarray(child._get_coords_for_basis(2, True)).copy()
    _deform(child)
    child.relax()
    taken_back = fault_mg.as_built_nodes(child, 2, True)
    assert np.array_equal(taken_back, relaxed), np.abs(taken_back - relaxed).max()


def test_coordinates_put_back_are_not_a_move():
    """A level whose coordinates are the ones it was made with is read on them,
    whatever wrote them: a trial move that is undone (``ephemeral_coords``, the
    old-frame evaluation of a semi-Lagrangian step) or a deform back to rest."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 6, regular=False,
        qdegree=2, refinement=1)
    at_rest = np.asarray(mesh.X.coords).copy()
    with mesh.ephemeral_coords():
        pass
    assert fault_mg.as_built_nodes(mesh, 2, True) is None

    _deform(mesh)
    assert fault_mg.as_built_nodes(mesh, 2, True) is not None
    mesh.deform(at_rest)
    assert fault_mg.as_built_nodes(mesh, 2, True) is None

    # the comparison is exact: one vertex off by 1e-9 is a move
    nudged = at_rest.copy()
    nudged[np.argmin(np.abs(at_rest - 0.5).sum(axis=1)), 0] += 1.0e-9
    mesh.deform(nudged)
    assert fault_mg.as_built_nodes(mesh, 2, True) is not None


def test_a_moved_level_under_the_finest_keeps_its_transfer():
    """The box moved before the fault was placed on it. The placed mesh is a
    new mesh on the moved box, and the box is a moved level under the finest:
    the pair under the box is the one it has at rest."""
    s, _ = _placed_poisson()
    s.solve()
    at_rest = _transfers(s)[0]

    s, mesh = _placed_poisson(move_the_base=_deform)
    s.solve()
    assert s.snes.getConvergedReason() > 0
    assert not mesh._moved_since_built
    under_the_box = _transfers(s)[0]
    assert under_the_box.shape == at_rest.shape
    assert abs(under_the_box - at_rest).max() == 0, abs(under_the_box - at_rest).max()


def _raise_the_top_wall(mesh):
    c = np.asarray(mesh.X.coords).copy()
    c[:, 1] *= 1 + 0.05 * np.sin(np.pi * c[:, 0])
    mesh.deform(c)


@pytest.mark.parametrize("raised,iterations", [(False, 2), (True, 4)], ids=["at-rest", "wall-raised"])
def test_a_base_moved_before_an_adapt_goes_into_the_child_where_it_stands(raised, iterations):
    """An adapt is a change of topology: the child's hierarchy is made from the
    base as it stands, and the base's finest level goes into it as a new mesh
    on the moved coordinates. The pair under that level is built from them: 4
    iterations with the top wall raised three times before the adapt, 2 with
    the box at rest."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 4, regular=False,
        qdegree=2, refinement=2)
    if raised:
        for _ in range(3):
            _raise_the_top_wall(base)

    def metric(centroids):
        d = np.abs(np.asarray(centroids)[:, 1] - 0.5)
        return 1.0 / np.minimum(0.02 + 0.5 * d, 0.2) ** 2

    child = base.adapt(metric, max_levels=2)
    tail = list(child._custom_mg_coarse_meshes)
    assert not any(level._moved_since_built for level in tail)
    s = _box_poisson(child)
    custom_mg.set_custom_fmg(s, tail)
    s.solve()
    assert s.snes.getLinearSolveIterations() == iterations


def test_the_fmg_preconditioner_keeps_its_iterations_over_moves():
    """``preconditioner="fmg"`` builds the same hierarchy on the mesh's own
    levels: 2 iterations at rest and after each of three moves. With the finest
    pair rebuilt from the moved coordinates it takes 3 after each move."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 6, regular=False,
        qdegree=2, refinement=2)
    s = _box_poisson(mesh)
    s.preconditioner = "fmg"
    s.solve()
    assert s.snes.getKSP().getPC().getType() == "mg"
    counts = [s.snes.getLinearSolveIterations()]
    for _ in range(3):
        _deform(mesh)
        s.solve()
        assert s.snes.getConvergedReason() > 0
        counts.append(s.snes.getLinearSolveIterations())
    assert counts == [2, 2, 2, 2]


def _wall_stokes(mesh):
    x, y = mesh.X
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    for wall in ("Bottom", "Left", "Right"):
        s.add_dirichlet_bc((0, 0), wall)
    s.bodyforce = sympy.Matrix([0, sympy.sin(sympy.pi * x)])
    s.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(s, list(mesh.multigrid_levels[:-1]), field_id=0)
    return s


def _velocity_iterations(stokes):
    return stokes.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getIterationNumber()


def test_a_moving_wall_keeps_the_velocity_iterations():
    """The top wall rises by 5 % of the depth in each of three steps and the
    interior stretches with it, so the finest level leaves the box its coarse
    levels fill: 3 velocity iterations at rest and after each step, and 3 for a
    solver first built on the moved mesh. With the transfers rebuilt from the
    moved coordinates the steps take 5, 6 and 7, and the new solver 7."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 4, regular=False,
        qdegree=2, refinement=2)
    s = _wall_stokes(mesh)
    s.solve()
    counts = [_velocity_iterations(s)]
    for _ in range(3):
        _raise_the_top_wall(mesh)
        s.solve()
        assert s.snes.getConvergedReason() > 0
        counts.append(_velocity_iterations(s))
    assert counts == [3, 3, 3, 3]

    built_after = _wall_stokes(mesh)
    built_after.solve()
    assert _velocity_iterations(built_after) == 3


def test_a_moved_tetrahedral_level_keeps_its_transfers():
    """The transfers of a refined box of tetrahedra after two moves are those
    before them, and the solve stays at 3 iterations. The count is 3 with the
    pair rebuilt as well; here the transfers tell the two apart."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=1 / 3, regular=False,
        qdegree=2, refinement=1)
    x, y, z = mesh.X
    s = uw.systems.Poisson(mesh, degree=2)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.constitutive_model.Parameters.diffusivity = 1
    s.f = sympy.sin(3 * x) * sympy.cos(2 * y)
    for wall in ("Bottom", "Top", "Left", "Right", "Front", "Back"):
        s.add_dirichlet_bc(0, wall)
    s.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(s, list(mesh.multigrid_levels[:-1]))
    s.solve()
    before = _transfers(s)
    counts = [s.snes.getLinearSolveIterations()]
    for _ in range(2):
        c = np.asarray(mesh.X.coords).copy()
        c[:, 0] += 0.03 * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1]) * np.sin(np.pi * c[:, 2])
        mesh.deform(c)
        s.solve()
        assert s.snes.getConvergedReason() > 0
        counts.append(s.snes.getLinearSolveIterations())
    for level, (P0, P1) in enumerate(zip(before, _transfers(s)), start=1):
        assert abs(P1 - P0).max() == 0, (level, abs(P1 - P0).max())
    assert counts == [3, 3, 3]


def test_a_reloaded_mesh_is_read_on_its_rebuilt_refinement(tmp_path):
    """A reload rebuilds the levels by refinement and then gives the finest the
    saved coordinates. Where the two differ the reloaded mesh is a moved level,
    read on the refinement it was rebuilt by. A refined annulus differs where
    the refinement that made it put new boundary nodes on the circle: 2
    iterations, against 3 on the mesh that was saved and 3 with the finest pair
    rebuilt from the saved coordinates."""
    def poisson(mesh):
        x, y = mesh.X
        s = uw.systems.Poisson(mesh, degree=2)
        s.constitutive_model = uw.constitutive_models.DiffusionModel
        s.constitutive_model.Parameters.diffusivity = 1
        s.f = sympy.sin(3 * x) * sympy.cos(2 * y)
        s.add_dirichlet_bc(0, "Upper")
        s.add_dirichlet_bc(0, "Lower")
        s.tolerance = 1.0e-8
        custom_mg.set_custom_fmg(s, list(mesh.multigrid_levels[:-1]))
        return s

    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.3,
                              qdegree=2, refinement=2)
    assert fault_mg.as_built_nodes(mesh, 2, True) is None
    s = poisson(mesh)
    s.solve()
    assert s.snes.getLinearSolveIterations() == 3
    T = uw.discretisation.MeshVariable("T_1176", mesh, 1, degree=1)
    mesh.write_checkpoint("reload", outputPath=str(tmp_path), meshVars=[T])
    saved = sorted(tmp_path.glob("*mesh*.h5"))[0]

    again = uw.discretisation.Mesh(str(saved))
    now = np.asarray(again._get_coords_for_basis(2, True))
    rebuilt = fault_mg.as_built_nodes(again, 2, True)
    assert np.abs(rebuilt - now).max() > 1.0e-2
    s = poisson(again)
    s.solve()
    assert s.snes.getLinearSolveIterations() == 2
