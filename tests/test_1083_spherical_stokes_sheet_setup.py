"""Lazy SNES setup, independent radial-force moments and owned RHS lifetime."""
import os
from contextlib import ExitStack
import numpy as np
import sympy as sp
import pytest
from scipy.integrate import dblquad
from petsc4py import PETSc
import underworld3 as uw


pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.mark.parametrize('viscosity_case', ('uniform', 'smooth_spatial', 'layered_spatial'))
def test_sheet_adapter_sets_up_lazy_snes_and_releases_rhs(viscosity_case):
    root_pid=uw.mpi.comm.bcast(os.getpid() if uw.mpi.rank==0 else None,root=0)
    mesh=uw.meshing.UnstructuredSimplexBox(minCoords=(1.,1.,1.),maxCoords=(2.,2.,2.),
        cellSize=.5,regular=True,qdegree=4,filename=f'/tmp/uw-sheet-setup-{root_pid}.msh')
    u=uw.discretisation.MeshVariable('SheetSetupU',mesh,3,degree=2,vtype=uw.VarType.VECTOR)
    p=uw.discretisation.MeshVariable('SheetSetupP',mesh,1,degree=1)
    stokes=uw.systems.Stokes(mesh,velocityField=u,pressureField=p)
    stokes.constitutive_model=uw.constitutive_models.ViscousFlowModel
    x,y,z=mesh.X
    viscosity={'uniform':sp.Float(1.),'smooth_spatial':1+x*x,
               'layered_spatial':sp.Piecewise((sp.Float(1e4),x>1.5),(sp.Float(1.),True))}[viscosity_case]
    stokes.constitutive_model.Parameters.shear_viscosity_0=viscosity
    stokes.bodyforce=sp.zeros(1,3)
    stokes.petsc_use_nullspace=False
    stokes.petsc_use_pressure_nullspace=False
    if viscosity_case!='uniform':
        # Independently verify that the actual native residual is linear and
        # matches its assembled Jacobian before replacing only the sheet RHS.
        stokes._build(False, False, None)
        mesh.update_lvec();stokes.dm.setAuxiliaryVec(mesh.lvec,None)
        stokes._update_constants(record=True);stokes.snes.setUp()
        with ExitStack() as cleanup:
            zero=stokes.dm.createGlobalVec();cleanup.callback(zero.destroy);zero.set(0.)
            vector=zero.duplicate();cleanup.callback(vector.destroy)
            lo, hi = vector.getOwnershipRange();vector.array[:]=1e-4*np.sin(np.arange(lo,hi))
            native=zero.duplicate();cleanup.callback(native.destroy)
            linear=zero.duplicate();cleanup.callback(linear.destroy)
            J,Jp=stokes.snes.getJacobian()[:2]
            stokes.snes.computeJacobian(zero,J,Jp)
            stokes.snes.computeFunction(vector,native)
            J.mult(vector,linear);linear.axpy(-1.,native)
            assert linear.norm()<1e-12*max(native.norm(),1.)
    # Top is z=2 for this box. No solve/nullspace setup can hide lazy matrices.
    arguments=dict(interface='Top',harmonic_degree=1,density_coefficient=1.)
    rhs=stokes.install_linear_spherical_sheet_load(**arguments)
    assert bool(stokes.snes.getJacobian()[0]) and rhs.norm()>0.
    # The installed vector is a residual contribution, so F(0)=+b and
    # F(x)-Jx=b. In particular no sheet source enters incompressibility.
    with ExitStack() as cleanup:
        zero = rhs.duplicate()
        cleanup.callback(zero.destroy)
        zero.set(0.)
        residual = rhs.duplicate()
        cleanup.callback(residual.destroy)
        expected = rhs.duplicate()
        cleanup.callback(expected.destroy)
        vector = rhs.duplicate()
        cleanup.callback(vector.destroy)
        stokes.snes.computeFunction(zero, residual)
        residual.axpy(-1., rhs)
        assert residual.norm() < 1e-13 * rhs.norm()
        pressure_is, _ = stokes._subdict['pressure']
        pressure = rhs.getSubVector(pressure_is)
        try:
            assert pressure.norm() == 0.
        finally:
            rhs.restoreSubVector(pressure_is, pressure)
        lo, hi = vector.getOwnershipRange()
        vector.array[:] = 1e-4 * np.sin(np.arange(lo, hi))
        J = stokes.snes.getJacobian()[0]
        J.mult(vector, expected)
        expected.axpy(1., rhs)
        stokes.snes.computeFunction(vector, residual)
        residual.axpy(-1., expected)
        assert residual.norm() < 1e-13 * max(expected.norm(), rhs.norm())
        # Rejection before setup must leave the existing load and callback valid.
        for overrides, error, match in (
            ({'interface': 'UnknownSheetInterface'}, ValueError, 'Unknown interface'),
            ({'density_coefficient': np.nan}, ValueError, 'Invalid mode density'),
            ({'harmonic_degree': 0}, ValueError, 'degree'),
        ):
            with pytest.raises(error, match=match):
                stokes.install_linear_spherical_sheet_load(**(arguments | overrides))
            assert stokes._linear_spherical_sheet_rhs is rhs and bool(rhs)
            stokes.snes.computeFunction(zero, residual)
            residual.axpy(-1., rhs)
            assert residual.norm() < 1e-13 * rhs.norm()
    iset,subdm=stokes._subdict['velocity']
    sub=rhs.getSubVector(iset)
    try:
        with ExitStack() as cleanup:
            local=u.vec.duplicate();cleanup.callback(local.destroy)
            global_=subdm.createGlobalVec();cleanup.callback(global_.destroy)
            for axis in range(3):
                local.array[:]=0.;local.array.reshape(-1,3)[:,axis]=1.
                global_.set(0.)
                subdm.localToGlobal(local,global_,addv=PETSc.InsertMode.INSERT_VALUES)
                actual=float(sub.dot(global_))
                # Independent P_1(cos(theta))=z/r times the radial unit vector.
                expected=dblquad(lambda y,x: [x,y,2.][axis]*2./(x*x+y*y+4),
                    1.,2.,lambda x:1.,lambda x:2.,epsabs=1e-12,epsrel=1e-12)[0]
                np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-13)
            # A radial force has zero work against every rigid rotation,
            # independently of facet quadrature or the viscosity profile.
            cx,cy,cz=u.coords.T
            null=np.zeros_like(cx)
            for mode in (np.column_stack((null,-cz,cy)),
                         np.column_stack((cz,null,-cx)),
                         np.column_stack((-cy,cx,null))):
                local.array.reshape(-1,3)[:]=mode
                global_.set(0.)
                subdm.localToGlobal(local,global_,addv=PETSc.InsertMode.INSERT_VALUES)
                assert abs(float(sub.dot(global_)))<1e-12*max(rhs.norm(),1.)
    finally:rhs.restoreSubVector(iset,sub)
    with ExitStack() as cleanup:
        independent = rhs.copy()
        cleanup.callback(independent.destroy)
        saved_local = independent.array.copy()
        second = stokes.install_linear_spherical_sheet_load(
            **(arguments | {'density_coefficient': -.7}))
        assert not bool(rhs) and second.norm() > 0.
        difference = second.copy()
        cleanup.callback(difference.destroy)
        difference.axpy(.7, independent)
        assert difference.norm() < 1e-13 * independent.norm()
        zero = second.duplicate()
        cleanup.callback(zero.destroy)
        zero.set(0.)
        residual = second.duplicate()
        cleanup.callback(residual.destroy)
        stokes.snes.computeFunction(zero, residual)
        residual.axpy(-1., second)
        assert residual.norm() < 1e-13 * second.norm()
        # A full DM rebuild must release the borrowed load, while caller-owned
        # copies retain their own PETSc storage and values across all teardown.
        stokes.is_setup = False
        stokes._needs_dm_rebuild = True
        stokes._build(False, False, None)
        assert not bool(second) and stokes._linear_spherical_sheet_rhs is None
        assert bool(independent)
        np.testing.assert_array_equal(independent.array, saved_local)
        third = stokes.install_linear_spherical_sheet_load(**arguments)
        stokes._reset()
        assert not bool(third) and stokes.snes is None
        assert stokes._linear_spherical_sheet_rhs is None
        assert bool(independent)
        np.testing.assert_array_equal(independent.array, saved_local)


@pytest.mark.parametrize('dependence', ('velocity', 'pressure', 'velocity_gradient'))
def test_sheet_adapter_rejects_unknown_dependent_viscosity(dependence):
    root_pid=uw.mpi.comm.bcast(os.getpid() if uw.mpi.rank==0 else None,root=0)
    mesh=uw.meshing.UnstructuredSimplexBox(minCoords=(1.,1.,1.),maxCoords=(2.,2.,2.),
        cellSize=.5,regular=True,qdegree=4,filename=f'/tmp/uw-sheet-nonlinear-{root_pid}.msh')
    u=uw.discretisation.MeshVariable('SheetGuardU',mesh,3,degree=2,vtype=uw.VarType.VECTOR)
    p=uw.discretisation.MeshVariable('SheetGuardP',mesh,1,degree=1)
    stokes=uw.systems.Stokes(mesh,velocityField=u,pressureField=p)
    stokes.constitutive_model=uw.constitutive_models.ViscousFlowModel
    coefficient={'velocity':u.sym[0], 'pressure':p.sym[0],
                 'velocity_gradient':sp.diff(u.sym[0],mesh.X[0])}[dependence]
    stokes.constitutive_model.Parameters.shear_viscosity_0=1+coefficient**2
    stokes.bodyforce=sp.zeros(1,3)
    with pytest.raises(ValueError,match='solution-independent viscosity'):
        stokes.install_linear_spherical_sheet_load(interface='Top',
            harmonic_degree=1,density_coefficient=1.)
