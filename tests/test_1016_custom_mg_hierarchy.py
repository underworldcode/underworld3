"""Layer-1 generalized FMG hierarchy: CustomMGHierarchy + set_custom_fmg with
BC-per-level reduction, over a uniform + SBR-targeted nested hierarchy."""
import numpy as np, sympy, pytest, underworld3 as uw
from petsc4py import PETSc
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

def _sbr(dm, band=0.2):
    # scoped SBR (restores dm_plex_transform_type so uniform refine() still works)
    return custom_mg.sbr_refine_where(dm, lambda cen: abs(cen[0]-0.5) < band)
def _wrap(dm,m0):
    return uw.discretisation.Mesh(dm.clone(), simplex=True,
        coordinate_system_type=m0.CoordinateSystem.coordinate_type, qdegree=3, boundaries=m0.boundaries)
def _poisson(mesh):
    p=uw.systems.Poisson(mesh); p.constitutive_model=uw.constitutive_models.DiffusionModel
    x,y=mesh.X
    p.constitutive_model.Parameters.diffusivity=sympy.Piecewise((1.0,x<0.5),(1000.0,True))
    p.f=8*sympy.pi**2*sympy.sin(2*sympy.pi*x)*sympy.sin(2*sympy.pi*y)
    for b in ("Bottom","Top","Left","Right"): p.add_dirichlet_bc(0.0,b)
    p.petsc_options["ksp_rtol"]=1e-8; p.petsc_options["ksp_type"]="cg"
    return p

def _hierarchy():
    m0=uw.meshing.UnstructuredSimplexBox(minCoords=(0,0),maxCoords=(1,1),cellSize=0.5,regular=True,qdegree=3)
    dm0=m0.dm; dm1=dm0.refine(); dm2=_sbr(dm1)          # 2 uniform-ish + 1 SBR
    return m0, [_wrap(dm0,m0), _wrap(dm1,m0)], _wrap(dm2,m0)

def test_custom_fmg_hierarchy_converges():
    m0, coarse, fine = _hierarchy()
    s=_poisson(fine)
    custom_mg.set_custom_fmg(s, coarse, builder="barycentric")
    s.solve()
    assert s.snes.getKSP().getPC().getType()=="mg"
    assert s.snes.getKSP().getPC().getMGLevels()==3
    assert s.snes.getConvergedReason()>0
    # matches a GAMG solve of the same fine problem
    g=_poisson(_wrap(fine.dm,m0)); g.preconditioner="gamg"; g.solve()
    rel=np.linalg.norm(s.Unknowns.u.data-g.Unknowns.u.data)/np.linalg.norm(g.Unknowns.u.data)
    assert rel < 1e-4

def test_bc_per_level_reduction_no_zero_columns():
    """build() must produce transfers with no zero columns (BCs at every level)."""
    m0, coarse, fine = _hierarchy()
    s=_poisson(fine); s._build(False,False,None); s.snes.setUp()
    h=custom_mg.CustomMGHierarchy(coarse+[fine], builder="barycentric")
    Ps=h.build(s)                          # raises if any transfer has zero columns
    assert len(Ps)==2
    for P in Ps:
        Pc=P.getValuesCSR(); import scipy.sparse as sp
        M=sp.csr_matrix(Pc[::-1], shape=P.getSize())
        assert int((np.asarray((M!=0).sum(axis=0)).ravel()==0).sum())==0

def test_rbf_builder_also_works():
    m0, coarse, fine = _hierarchy()
    s=_poisson(fine)
    custom_mg.set_custom_fmg(s, coarse, builder="rbf")
    s.solve()
    assert s.snes.getKSP().getPC().getType()=="mg"
    assert s.snes.getConvergedReason()>0
