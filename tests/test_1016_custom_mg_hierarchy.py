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


# --------------------------------------------------------------------------- #
#  Operator-faithful finest reduced map (adapt-child regression)
# --------------------------------------------------------------------------- #
def test_finest_map_operator_mismatch_raises():
    """The finest reduced map must span exactly the assembled operator's rows.
    A stale/oversized finest map (adapt-child section inconsistency) must fail
    with an actionable error, not a bare PETSc PtAP error 60."""
    m0, coarse, fine = _hierarchy()
    s=_poisson(fine); s._build(False,False,None); s.snes.setUp()
    op_n = int(s.snes.getJacobian()[0].getSize()[0])
    # A correct finest map (length == op_n) passes; a doctored oversized one raises.
    good = np.arange(op_n)
    custom_mg.CustomMGHierarchy._assert_finest_matches_operator(s, good, parallel=False)
    bad = np.arange(op_n + 17)
    with pytest.raises(RuntimeError, match="finest reduced-map size"):
        custom_mg.CustomMGHierarchy._assert_finest_matches_operator(s, bad, parallel=False)


def test_advdiff_on_nvb_adapt_child_gets_custom_mg():
    """TASK C: a semi-Lagrangian AdvDiffusion on an NVB adapt() child now installs
    custom-P geometric MG via the mesh-owned auto-pickup (previously skipped by a
    DuDt guard because the finest reduced map read from the DM section could
    disagree with the assembled operator). The finest map is now read after the
    SNES section is finalized and validated against the operator, so the transfer
    is faithful on adapt children -> the PC becomes 'mg', the solve converges, and
    the result matches a default-preconditioner solve."""
    pytest.importorskip(
        "underworld3.utilities._nvb_transform",
        reason="native uwnvb transform not built (needs the custom-PETSc/amr env)")
    if PETSc.COMM_WORLD.getSize() > 1:
        pytest.skip("serial regression; parallel adapt covered elsewhere")

    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0,0), maxCoords=(1,1), cellSize=0.15, regular=False,
        qdegree=3, refinement=1)
    x, y = base.CoordinateSystem.X

    def metric(coords):
        d = np.abs(coords[:,0]-0.5)
        h = np.where(d<0.1, 0.03, np.minimum(0.03+(0.12-0.03)*(d-0.1)/0.2, 0.12))
        return 1.0/h**2

    child = base.adapt(metric, max_levels=3, engine="nvb")
    assert child._custom_mg_coarse_meshes is not None   # adapt child carries a tail

    def make(name):
        T = uw.discretisation.MeshVariable(name, child, 1, degree=2)
        T.data[:,0] = np.asarray(uw.function.evaluate(
            sympy.exp(-(((x-0.3)**2+(y-0.7)**2)/(2*0.05**2))), T.coords)).reshape(-1)
        V = uw.discretisation.MeshVariable(name+"V", child, child.dim, degree=2)
        V.data[:,0] = 0.4
        adv = uw.systems.AdvDiffusionSLCN(child, u_Field=T, V_fn=V.sym, order=1,
                                          monotone_mode="clamp")
        adv.constitutive_model = uw.constitutive_models.DiffusionModel
        adv.constitutive_model.Parameters.diffusivity = 2.0e-4
        adv.f = 0.0
        adv.add_dirichlet_bc(0.0, "Top")
        return adv, T

    # reference: mesh-owned pickup disabled (clear the coarse tail)
    advR, TR = make("Tc_ref")
    saved = child._custom_mg_coarse_meshes
    child._custom_mg_coarse_meshes = None
    advR.solve(timestep=0.01)
    ref = np.asarray(TR.data[:,0]).copy()
    child._custom_mg_coarse_meshes = saved

    # custom-P via the real auto-pickup path (no monkeypatch)
    advC, TC = make("Tc_cmg")
    advC.solve(timestep=0.01)
    cust = np.asarray(TC.data[:,0])

    assert advC.snes.getKSP().getPC().getType() == "mg"    # custom-P installed
    assert advC._custom_mg is not None                     # registered, not skipped
    assert advC.snes.getConvergedReason() > 0
    # same solution up to iterative tolerance (different preconditioners)
    rel = np.linalg.norm(cust-ref)/max(np.linalg.norm(ref), 1e-30)
    assert rel < 1e-4
