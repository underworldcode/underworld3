"""Submesh boundary names must survive on ranks owning no boundary points."""

from petsc4py import PETSc
import underworld3 as uw


def test_surviving_labels_use_global_presence_and_create_empty_local_labels():
    dm = PETSc.DMPlex().createBoxMesh([2, 2], simplex=False, comm=uw.mpi.comm)
    if uw.mpi.rank == 0:
        dm.createLabel('OnlyOnZero')
        dm.setLabelValue('OnlyOnZero', dm.getChart()[0], 42)
    surviving = uw.discretisation.Mesh._surviving_labels(
        None, dm, {'OnlyOnZero': 42, 'AbsentEverywhere': 99}
    )
    assert surviving == {'OnlyOnZero': 42}
    assert dm.getLabel('OnlyOnZero') is not None
    assert 'AbsentEverywhere' not in [dm.getLabelName(i) for i in range(dm.getNumLabels())]
    if uw.mpi.rank > 0:
        values = dm.getLabel('OnlyOnZero').getValueIS()
        assert values is None or len(values.getIndices()) == 0
    dm.destroy()


def test_extracted_region_boundary_enums_are_identical_on_every_rank():
    mesh = uw.meshing.BoxInternalBoundary(
        elementRes=(12,12), zelementRes=(10,2), zintCoord=.85,
        minCoords=(0.,0.), maxCoords=(1.,1.), simplex=True, qdegree=2)
    child = mesh.extract_region('Outer')
    names = [(b.name,b.value) for b in child.boundaries]
    assert all(other == names for other in uw.mpi.comm.allgather(names))
    assert 'Top' in dict(names)
