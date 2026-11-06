"""Shared observer ownership and exact cylindrical mixed-gauge oracles."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from petsc4py import PETSc
import underworld3 as uw
from underworld3.utilities._cylindrical_stokes_state import CylindricalSolveState

pytestmark = pytest.mark.level_1


@pytest.mark.tier_b
def test_shared_registration_excludes_geometry_alias_and_restores_on_error():
    stokes = SimpleNamespace()
    observer = uw.diagnostics.StokesObserver()
    with pytest.raises(ValueError, match='deliberate'):
        with uw.diagnostics.observe_stokes(stokes, observer):
            with pytest.raises(RuntimeError, match='already registered'):
                with uw.diagnostics.observe_spherical_stokes(stokes, observer):
                    pass
            raise ValueError('deliberate')
    assert not hasattr(stokes, '_stokes_observer')
    assert not hasattr(stokes, '_spherical_gap_observer')


@pytest.mark.tier_b
@pytest.mark.parametrize('components', [1, 2])
def test_cylindrical_original_equations_and_exact_pressure_rotation_gauges(components):
    comm = PETSc.COMM_WORLD
    pressure_mode = np.array([0.,0.,0.,0.,1.,1.,1.,1.])
    rotation_mode = np.array([1.,0.,0.,1.,0.,0.,0.,0.])
    exact = np.array([1.,2.,3.,-1.,-.1,-.2,7.,8.])
    modes = np.column_stack([pressure_mode,rotation_mode] if components == 1 else [pressure_mode])
    q,_ = np.linalg.qr(modes)
    projection = np.eye(8)-q@q.T
    dense = projection@np.diag(np.arange(1.,9.))@projection
    A = PETSc.Mat().createAIJ((8,8),nnz=8,comm=comm);A.setUp()
    lo,hi = A.getOwnershipRange()
    for row in range(lo,hi):A.setValues(row,np.arange(8,dtype=PETSc.IntType),dense[row])
    A.assemble()
    vectors=[]
    def vector(data):
        v=A.createVecRight();v.array[:]=data[lo:hi];vectors.append(v);return v
    rhs=vector(dense@exact);pm=vector(pressure_mode);rm=vector(rotation_mode)
    mean=vector(np.array([0.,0.,0.,0.,1.,1.,0.,0.]));rw=vector(rotation_mode)
    initial=exact+2.5*pressure_mode+(3*rotation_mode if components==1 else 0)
    x=vector(initial)
    state=CylindricalSolveState(dict(matrix=A,rhs=rhs,mpi=comm.tompi4py(),pressure_jump_integral=.3,
        lift_gap=0.,pmean=mean,pressure_mode=pm,rotation_weight=rw,rotation_weights=[rw],
        rotation_mode=rm,components=components,nu=4,npres=2))
    try:
        y=state.gauged_copy(x)
        try:
            np.testing.assert_allclose(y.array,exact[lo:hi],atol=3e-15,rtol=3e-15)
            np.testing.assert_array_equal(x.array,initial[lo:hi])
            assert state.residual(y)['original_relative_residual']<3e-15
            assert abs(state.residual(y)['physical_pressure_integral'])<3e-15
        finally:y.destroy()
    finally:
        for v in vectors:v.destroy()
        A.destroy()


@pytest.mark.tier_b
def test_solver_implementation_does_not_depend_on_diagnostics_namespace():
    import underworld3.utilities._cylindrical_gap as cyl
    import underworld3.utilities._spherical_stokes_gap as sph
    for module in (cyl,sph):
        tree=ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or '').startswith('underworld3.diagnostics')
            if isinstance(node,ast.Attribute):
                assert not (isinstance(node.value,ast.Name) and node.value.id=='uw' and node.attr=='diagnostics')
