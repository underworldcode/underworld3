"""Independent spherical oracle comparisons, bulk equations, walls and jumps."""
from types import SimpleNamespace
import numpy as np
import pytest
import sympy as sp
from underworld3.analytic.spherical.kramer import SphericalStokes
from underworld3.analytic.kramer import assess_available
from underworld3.analytic import _validation
from underworld3.maths.spherical_harmonics import real_harmonic_values, solid_harmonic_sympy

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

@pytest.fixture
def mesh():
    return SimpleNamespace(dim=3, X=sp.Matrix([sp.symbols('x y z',real=True)]))

@pytest.mark.skipif(not assess_available(), reason='optional assess oracle')
@pytest.mark.parametrize('density,boundary', [('delta','free'),('delta','zero'),('smooth','free'),('smooth','zero')])
@pytest.mark.parametrize('ell,m,component', [(2,0,'cos'),(4,1,'cos'),(4,2,'sin')])
def test_assess_fields(mesh,density,boundary,ell,m,component):
    import assess
    sol=SphericalStokes(mesh,ell=ell,m=m,component=component,normalization='real_orthonormal',
                        density=density,boundary=boundary,viscosity=2.3,forcing=.61)
    family=getattr(assess,f'SphericalStokesSolution{density.capitalize()}{"FreeSlip" if boundary=="free" else "ZeroSlip"}')
    common=dict(Rp=sol.r_outer,Rm=sol.r_inner,nu=sol.viscosity,g=sol.g)
    branches=[family(ell,m,sign,rp=sol.r_int,**common) for sign in (-1,1)] if density=='delta' else [family(ell,m,2,**common)]*2
    pts=sol.sample_points(19)
    # assess uses Re(Y_lm), so the orthonormal real basis gains sqrt(2) for m>0.
    # Rotate coordinates by pi/(2m) to independently compare the sine component.
    angle=-np.pi/(2*m) if component=='sin' else 0
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    rotated=pts@rotation.T
    scale=np.sqrt(2) if m else 1
    for field,method in [('velocity','velocity_cartesian'),('pressure','pressure_cartesian'),('radial_normal_stress','radial_stress_cartesian')]:
        expected=np.asarray([getattr(branches[int(np.linalg.norm(p)>=sol.r_int)],method)(p) for p in rotated])*scale
        if field=='velocity':expected=expected@rotation
        np.testing.assert_allclose(sol.evaluate(field,pts),expected,rtol=3e-11,atol=3e-12)

@pytest.mark.parametrize('density,boundary', [('delta','free'),('delta','zero'),('smooth','free'),('smooth','zero')])
def test_bulk_physics(mesh,density,boundary):
    sol=SphericalStokes(mesh,density=density,boundary=boundary)
    pts=sol.sample_points(8)
    assert _validation.incompressibility_residual(sol,pts)<1e-10
    assert _validation.momentum_residual(sol,pts)<1e-9
    assert _validation.strainrate_consistency(sol,pts)<1e-10

@pytest.mark.parametrize('density,boundary', [('delta','free'),('delta','zero'),('smooth','free'),('smooth','zero')])
@pytest.mark.parametrize('m', [0,1,2])
def test_walls_and_poles(mesh,density,boundary,m):
    sol=SphericalStokes(mesh,m=m,normalization='real_orthonormal',density=density,boundary=boundary)
    directions=np.vstack((np.eye(3),-np.eye(3), sol.sample_points(12)/np.linalg.norm(sol.sample_points(12),axis=1)[:,None]))
    for r in (sol.r_inner,sol.r_outer):
        pts=r*directions
        v=sol.evaluate('velocity',pts)
        stress=sol.evaluate('stress',pts)
        assert np.all(np.isfinite(stress))
        np.testing.assert_allclose(np.sum(v*directions,axis=1),0,atol=2e-12)
        if boundary=='zero':np.testing.assert_allclose(v,0,atol=2e-12)
        else:np.testing.assert_allclose(sol.evaluate('radial_shear_stress',pts),0,atol=2e-12)
        # Radial contraction of independently differentiated Cartesian stress.
        np.testing.assert_allclose(np.einsum('ni,nij,nj->n',directions,stress,directions),
                                   sol.evaluate('radial_normal_stress',pts),atol=2e-12)

@pytest.mark.parametrize('boundary', ['free','zero'])
def test_sheet_jumps(mesh,boundary):
    sol=SphericalStokes(mesh,m=1,normalization='real_orthonormal',boundary=boundary)
    pts=sol.r_int*sol.sample_points(17)/np.linalg.norm(sol.sample_points(17),axis=1)[:,None]
    for field in ('velocity','radial_shear_stress','radial_normal_stress'):
        expressions=[b[field] for b in sol._cartesian_fields]
        jump=expressions[1]-expressions[0]
        terms=list(jump) if isinstance(jump,sp.MatrixBase) else [jump]
        fn=sp.lambdify((*sol._xyz,sol._r),terms,'numpy',cse=True)
        values=np.array([np.broadcast_to(v,len(pts)) for v in fn(*pts.T,sol.r_int)])
        expected=real_harmonic_values(pts,degree=2,order=1,normalization='real_orthonormal') if field=='radial_normal_stress' else 0
        np.testing.assert_allclose(values, np.broadcast_to(expected,values.shape),atol=2e-12)

@pytest.mark.parametrize('ell,m,component', [(2,0,'cos'),(3,1,'sin'),(4,2,'cos'),(4,4,'sin')])
def test_cartesian_harmonics(mesh,ell,m,component):
    pts=np.array([[0,0,2],[0,0,-2],[.7,.2,1.1],[-.4,.8,-1.]])
    H=solid_harmonic_sympy(list(mesh.X),degree=ell,order=m,component=component,normalization='real_orthonormal')
    actual=sp.lambdify(list(mesh.X),H,'numpy')(*pts.T)/np.linalg.norm(pts,axis=1)**ell
    np.testing.assert_allclose(actual,real_harmonic_values(pts,degree=ell,order=m,component=component,normalization='real_orthonormal'),atol=2e-14)

@pytest.mark.parametrize('kwargs', [dict(ell=1),dict(m=3),dict(k=1,density='smooth'),dict(r_int=3),dict(viscosity=0),dict(m=0,component='sin')])
def test_rejects_invalid_and_resonant(mesh,kwargs):
    with pytest.raises((ValueError,NotImplementedError)):SphericalStokes(mesh,**kwargs)
