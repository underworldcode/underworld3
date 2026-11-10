"""Hard-baseline tests of native observers, monitor ownership and runtime units."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from underworld3.diagnostics import (
    SphericalSolveState, SphericalStokesObserver, observe_spherical_stokes,
    solve_with_observer, collective_file_sha256,
)
from underworld3.diagnostics import runtime
from underworld3.utilities import _spherical_stokes_gap as native

pytestmark = pytest.mark.level_1


def algebra():
    comm = PETSc.COMM_WORLD
    g = np.array([0.,0.,0.,1.,1.,-1.,0.,0.])
    projection = np.eye(8)-np.outer(g,g)/(g@g)
    dense = projection@np.diag(np.arange(1.,9.))@projection
    exact = np.array([1.,-2.,.5,3.,-3.,4.,5.,6.])
    matrix = PETSc.Mat().createAIJ((8,8),nnz=8,comm=comm);matrix.setUp()
    lo, hi = matrix.getOwnershipRange()
    for row in range(lo,hi):matrix.setValues(row,np.arange(8,dtype=PETSc.IntType),dense[row])
    matrix.assemble()
    rhs=matrix.createVecRight();rhs.array[:]=(dense@exact)[lo:hi]
    mode=rhs.duplicate();mode.array[:]=g[lo:hi]
    mean=rhs.duplicate();mean.array[:]=np.array([0,0,0,1,1,0,0,0])[lo:hi]
    normalized=mode.copy();normalized.scale(1/normalized.norm())
    space=PETSc.NullSpace().create(vectors=[normalized],comm=comm);matrix.setNullSpace(space)
    ksp=PETSc.KSP().create(comm=comm);ksp.setOperators(matrix);ksp.setType('fgmres')
    ksp.getPC().setType('none');ksp.setPCSide(PETSc.PC.Side.RIGHT)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED);ksp.setTolerances(rtol=1e-12,atol=1e-50,max_it=20)
    state=SphericalSolveState(dict(matrix=matrix,rhs=rhs,mpi=comm.tompi4py(),pmean=mean,
        pressure_mode=mode,modes=[mode],rotation_weights=[],pressure_jump_integral=0.,lift_gap=0.,
        nu=3,npres=2,nboundary=1,components=3,umap=np.arange(3,dtype=PETSc.IntType),
        pmap=np.arange(2,dtype=PETSc.IntType),compressed=np.array([0],dtype=PETSc.IntType),coordinates=np.array([[1.,0.,0.]])))
    return state,ksp,(matrix,rhs,mode,mean,normalized,space),exact


@pytest.mark.tier_b
@pytest.mark.parametrize('fail',[False,True])
def test_original_operator_gauge_solution_and_monitor_ownership(fail):
    state,ksp,objects,exact=algebra();matrix,rhs,*_=objects
    solution=rhs.duplicate();rhs_before=rhs.copy();matrix_before=matrix.copy()
    external=[]
    def external_monitor(solver,iteration,norm):external.append(iteration)
    ksp.setMonitor(external_monitor);prior=list(ksp.getMonitor())
    class Observer(SphericalStokesObserver):
        def solve_started(self,ksp,state):self.iterations=[]
        def iteration(self,ksp,iteration,norm,state):
            self.iterations.append(iteration)
            vector=state.build_iterate(ksp)
            before=vector.copy();physical=state.gauged_copy(vector)
            try:
                before.axpy(-1,vector);assert before.norm()==0
                assert abs(state.residual(physical)['physical_pressure_integral'])<1e-12
            finally:
                before.destroy();physical.destroy();vector.destroy()
        def solve_finished(self,ksp,vector,state):
            physical=state.gauged_copy(vector)
            try:
                lo,hi=physical.getOwnershipRange()
                np.testing.assert_allclose(physical.array,exact[lo:hi],rtol=2e-12,atol=2e-12)
                info=state.residual(physical)
                assert info['original_relative_residual']<2e-12
                assert set(info['blocks'])=={'velocity','pressure','multiplier'}
                assert abs(info['physical_pressure_integral'])<1e-12
            finally:physical.destroy()
            if fail:raise RuntimeError('deliberate observer completion failure')
    observer=Observer()
    try:
        if fail:
            with pytest.raises(RuntimeError,match='deliberate observer'):
                solve_with_observer(ksp,rhs,solution,state,observer)
        else:solve_with_observer(ksp,rhs,solution,state,observer)
        assert external==observer.iterations
        assert ksp.getMonitor()==prior
        rhs_before.axpy(-1,rhs);assert rhs_before.norm()==0
        matrix_before.axpy(-1,matrix);assert matrix_before.norm()==0
    finally:
        ksp.destroy()
        for obj in (solution,rhs_before,matrix_before,*objects):obj.destroy()


@pytest.mark.tier_b
def test_observer_scope_restores_registration_on_exception():
    stokes=SimpleNamespace()
    observer=SphericalStokesObserver()
    with pytest.raises(ValueError,match='intentional'):
        with observe_spherical_stokes(stokes,observer):
            assert stokes._spherical_gap_observer is observer
            with pytest.raises(RuntimeError,match='already registered'):
                with observe_spherical_stokes(stokes,observer):pass
            raise ValueError('intentional')
    assert not hasattr(stokes,'_spherical_gap_observer')
    stokes._spherical_gap_observer=None
    with observe_spherical_stokes(stokes,observer):pass
    assert stokes._spherical_gap_observer is None


@pytest.mark.tier_b
def test_shared_hash_known_sha256_and_collective_missing_file():
    comm=MPI.COMM_WORLD
    folder=comm.bcast(tempfile.mkdtemp(prefix='uw3-hash-') if comm.rank==0 else None,root=0)
    path=Path(folder)/'abc'
    if comm.rank==0:path.write_bytes(b'abc')
    comm.Barrier()
    assert collective_file_sha256(path,comm)=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    with pytest.raises(OSError,match='Shared provenance hash failed'):
        collective_file_sha256(Path(folder)/'absent',comm)


@pytest.mark.tier_b
@pytest.mark.parametrize('system,factor',[('darwin',1),('linux',1024)])
def test_runtime_peak_memory_exact_units_and_rank_max(monkeypatch,system,factor):
    comm=MPI.COMM_WORLD
    monkeypatch.setattr(runtime,'sys',SimpleNamespace(platform=system,argv=['known value','--x'],version='Python test'))
    monkeypatch.setattr(runtime.resource,'getrusage',lambda who:SimpleNamespace(ru_maxrss=comm.rank+1))
    info=runtime.runtime_info(comm=comm)
    assert info['peak_resident_set_size_bytes_rank_max']==factor*comm.size
    assert info['command_line']=="'known value' --x"
    assert info['python_version']=='Python test'
    assert set(json.loads(info['package_versions_json']))=={'numpy','scipy','sympy','h5py','mpi4py','petsc4py','gmsh','underworld3'}


@pytest.mark.tier_b
def test_source_integrity_is_owned_by_native_observer(monkeypatch):
    observer=SphericalStokesObserver()
    original=native._solve_constraints
    metadata=observer.solver_source
    metadata['native_source_sha256']='tampered copy'
    stokes=SimpleNamespace()
    with monkeypatch.context() as patch:
        with pytest.raises(RuntimeError,match='changed during observation'):
            with observe_spherical_stokes(stokes,observer):
                patch.setattr(native,'_solve_constraints',object())
        assert not hasattr(stokes,'_spherical_gap_observer')
    assert native._solve_constraints is original
    assert observer.solver_source['native_source_sha256']!='tampered copy'
