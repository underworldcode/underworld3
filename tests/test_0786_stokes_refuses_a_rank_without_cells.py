"""The decision and the text of the Stokes refusal on a rank without cells (#962).

The refusal needs two ranks or more to be seen whole
(``tests/parallel/test_0786_stokes_refuses_a_rank_without_cells_parallel.py``),
and there a regression shows as a stopped or hung run. Here the same method runs
on one rank against a stand-in DM whose communicator reports several ranks, so a
wrong decision or a wrong text fails an assertion; and a real solver shows that a
full set-up calls it.
"""
import types

import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1]


class _Comm:
    def __init__(self, counts):
        self.counts, self.size, self.gathers = list(counts), len(counts), 0

    def allgather(self, value):
        self.gathers += 1
        return list(self.counts)


def _outcome(counts, rank):
    """What the refusal does on ``rank`` of a DM with ``counts`` cells per rank."""
    comm = _Comm(counts)
    dm = types.SimpleNamespace(comm=types.SimpleNamespace(tompi4py=lambda: comm),
                               getHeightStratum=lambda height: (0, counts[rank]))
    try:
        uw.systems.Stokes._refuse_a_rank_without_cells(types.SimpleNamespace(dm=dm))
        return None, comm.gathers
    except RuntimeError as error:
        return str(error), comm.gathers


@pytest.mark.tier_b
@pytest.mark.parametrize("counts, gathers", [
    ((5,), 0), ((0,), 0),                 # one rank: PETSc's ranks cannot disagree
    ((5, 1, 7), 1), ((3, 3), 1),          # every rank holds cells
    ((0, 0), 1),                          # no cells anywhere: every rank leaves alike
])
def test_no_refusal(counts, gathers):
    for rank in range(len(counts)):
        assert _outcome(counts, rank) == (None, gathers), (counts, rank)


@pytest.mark.tier_b
@pytest.mark.parametrize("counts, named", [
    ((5, 0, 7), "ranks without cells of this mesh: 1 of 3 (12 cells in all)"),
    ((0, 1), "ranks without cells of this mesh: 1 of 2 (1 cells in all)"),
    # the 14-cell box of #962 on 8 ranks
    ((0, 3, 2, 2, 0, 3, 2, 2), "ranks without cells of this mesh: 2 of 8 (14 cells in all)"),
])
def test_refused_with_the_same_text_on_every_rank(counts, named):
    outcomes = [_outcome(counts, rank) for rank in range(len(counts))]
    assert len(set(outcomes)) == 1, outcomes
    text, gathers = outcomes[0]
    assert gathers == 1
    assert text is not None, "not refused"
    assert named in text, text
    assert "DMPlexComputeJacobianByKey" in text and "#962" in text, text


@pytest.mark.tier_b
def test_a_full_set_up_reaches_the_refusal(monkeypatch):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.5)
    v = uw.discretisation.MeshVariable("V0786s", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P0786s", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([[0.0, 1.0]])
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_essential_bc((0.0, 0.0), wall)

    def refuse():
        raise RuntimeError("the refusal was reached")

    monkeypatch.setattr(stokes, "_refuse_a_rank_without_cells", refuse)
    with pytest.raises(RuntimeError, match="the refusal was reached"):
        stokes.solve()
