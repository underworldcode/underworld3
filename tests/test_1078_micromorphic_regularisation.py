"""Micromorphic yield-law regularisation as a constitutive option.

``cm.Parameters.plastic_regularisation = Micromorphic(length, modulus)`` gives the
yield law a length scale through a field chi solved with the flow. The model asks for
the field; the solver registers it as a coupled field before set-up. These tests hold
the option to the hand-built formulation it replaces (``add_coupled_field`` plus the
shift written into the yield stress), to round-off.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.constitutive_models import Micromorphic

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

ELL = 0.1


def _probe():
    g = np.linspace(0.05, 0.95, 7)
    return np.array([[2 * a, b] for a in g for b in g])


def _shear_box():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0), cellSize=1.0 / 12, qdegree=3)
    stokes = uw.systems.Stokes(mesh)
    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    x, y = mesh.X
    # a weak spot at the base seeds a band
    tau0 = 0.6 + 0.4 * sympy.exp(-((x - 1) ** 2 + y ** 2) / 0.02)
    return mesh, stokes, tau0


def _solve(stokes):
    stokes.consistent_jacobian = True
    stokes.petsc_options["snes_rtol"] = 1.0e-10
    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0
    return np.asarray(uw.function.global_evaluate(stokes.Unknowns.u.sym, _probe()))


def test_option_reproduces_the_hand_built_formulation():
    # hand-built: the field, its equation and the shift written by the user
    mesh, stokes, tau0 = _shear_box()
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    chi = uw.discretisation.MeshVariable("chi_hand", mesh, 1, degree=1)
    e = cm._strainrate_inv_II
    H = 0.1 * cm.Parameters.shear_viscosity_0
    stokes.add_coupled_field(
        chi, F0=H * (chi.sym[0] - e),
        F1=H * ELL ** 2 * chi.sym.jacobian(mesh.CoordinateSystem.N))
    cm.Parameters.yield_stress = tau0 + H * (e - chi.sym[0])
    u_hand = _solve(stokes)
    its_hand = stokes.snes.getIterationNumber()
    chi_hand = np.asarray(uw.function.global_evaluate(chi.sym, _probe()))

    # the option: the default modulus is 0.1 x shear_viscosity_0
    mesh, stokes, tau0 = _shear_box()
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.yield_stress = tau0
    reg = Micromorphic(length=ELL)
    cm.Parameters.plastic_regularisation = reg
    u_opt = _solve(stokes)
    chi_opt = np.asarray(uw.function.global_evaluate(reg.chi.sym, _probe()))

    assert [c.field for c in stokes._coupled_fields] == [reg.chi]
    assert np.abs(chi_hand).max() > 0.1                 # chi is driven
    assert np.abs(u_opt - u_hand).max() < 1.0e-12 * np.abs(u_hand).max()
    assert np.abs(chi_opt - chi_hand).max() < 1.0e-12 * np.abs(chi_hand).max()
    assert stokes.snes.getIterationNumber() == its_hand


def test_no_regularisation_adds_no_field():
    mesh, stokes, tau0 = _shear_box()
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    stokes.constitutive_model.Parameters.yield_stress = tau0
    assert stokes.constitutive_model.Parameters.plastic_regularisation is None
    stokes._register_model_coupled_fields()
    assert stokes._coupled_fields == []


def test_a_stated_modulus_and_the_refusals():
    mesh, stokes, tau0 = _shear_box()
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.yield_stress = tau0
    reg = Micromorphic(length=ELL, modulus=0.5)
    cm.Parameters.plastic_regularisation = reg
    assert float(sympy.sympify(reg.modulus.sym)) == 0.5

    # one object per model: chi belongs to its model
    other = uw.constitutive_models.ViscoPlasticFlowModel(stokes.Unknowns)
    with pytest.raises(ValueError, match="already belongs"):
        other.Parameters.plastic_regularisation = reg
    with pytest.raises(TypeError, match="Micromorphic"):
        cm.Parameters.plastic_regularisation = 0.1

    # coupled problems are Newton only
    stokes.consistent_jacobian = False
    with pytest.raises(ValueError, match="consistent"):
        stokes.solve()


def test_each_yielding_family_takes_the_shift():
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25)
    stokes = uw.systems.Stokes(mesh)
    for family, kwargs in ((uw.constitutive_models.ViscoElasticPlasticFlowModel, {}),
                           (uw.constitutive_models.TransverseIsotropicVEPFlowModel, {})):
        cm = family(stokes.Unknowns, **kwargs)
        cm.Parameters.yield_stress = 1.0
        if family is uw.constitutive_models.TransverseIsotropicVEPFlowModel:
            cm.Parameters.director = sympy.Matrix([0.0, 1.0])
        before = cm._regularised_yield_stress(cm.Parameters.yield_stress)
        reg = Micromorphic(length=ELL)
        cm.Parameters.plastic_regularisation = reg
        after = cm._regularised_yield_stress(cm.Parameters.yield_stress)
        assert before == cm.Parameters.yield_stress
        assert sympy.sympify(after - before).has(reg.chi.sym[0])
        (field, F0, F1), = cm._coupled_field_requests()
        assert field is reg.chi


def test_multimaterial_masks_each_source_by_its_material():
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2, qdegree=2, regular=True)
    swarm = uw.swarm.Swarm(mesh)
    material = uw.swarm.IndexSwarmVariable("Mmm", swarm, indices=2, proxy_degree=1)
    swarm.populate(fill_param=2)
    stokes = uw.systems.Stokes(mesh)
    weak = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns, material_name="weak")
    strong = uw.constitutive_models.ViscoPlasticFlowModel(stokes.Unknowns, material_name="strong")
    strong.Parameters.yield_stress = 1.0
    reg = Micromorphic(length=ELL)
    strong.Parameters.plastic_regularisation = reg
    stokes.constitutive_model = uw.MultiMaterialConstitutiveModel(
        stokes.Unknowns, material, [weak, strong])
    stokes._register_model_coupled_fields()
    assert [c.field for c in stokes._coupled_fields] == [reg.chi]
    assert reg.mask == material.sym[1]


def test_describe_reports_the_regularisation():
    mesh, stokes, tau0 = _shear_box()
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.yield_stress = tau0
    cm.Parameters.plastic_regularisation = Micromorphic(length=ELL)
    d = cm.describe()
    assert "Micromorphic" in d["facts"]["plastic regularisation"]
    names = {t["name"] for t in d["terms"]}
    assert {"plastic_regularisation.length", "plastic_regularisation.modulus"} <= names


def test_a_snapshot_carries_chi(tmp_path):
    # chi is part of the state: restored with the flow, the next solve starts converged
    mesh, stokes, tau0 = _shear_box()
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.yield_stress = tau0
    reg = Micromorphic(length=ELL)
    cm.Parameters.plastic_regularisation = reg
    _solve(stokes)
    saved = np.array(reg.chi.data, copy=True)
    path = str(tmp_path / "micromorphic.snap.h5")
    orchestration_model = uw.get_default_model()
    orchestration_model.save_state(file=path)
    reg.chi.data[...] = 0.0
    stokes.Unknowns.u.data[...] = 0.0
    orchestration_model.load_state(path)
    assert np.array_equal(np.asarray(reg.chi.data), saved)
    stokes.solve(zero_init_guess=False)
    assert stokes.snes.getIterationNumber() == 0


def test_the_regularisation_can_be_replaced_and_removed():
    # Changing the slot after a solve changes what is solved: the old field leaves the
    # solver, the new one joins it, and with none the solve is the plain yield law.
    mesh, stokes, tau0 = _shear_box()
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.yield_stress = tau0
    first = Micromorphic(length=ELL)
    cm.Parameters.plastic_regularisation = first
    _solve(stokes)

    second = Micromorphic(length=2 * ELL)
    cm.Parameters.plastic_regularisation = second
    _solve(stokes)
    assert [c.field for c in stokes._coupled_fields] == [second.chi]

    cm.Parameters.plastic_regularisation = None
    u_removed = _solve(stokes)
    assert stokes._coupled_fields == []

    mesh, plain, tau0 = _shear_box()
    plain.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    plain.constitutive_model.Parameters.yield_stress = tau0
    u_plain = _solve(plain)
    assert np.abs(u_removed - u_plain).max() < 1.0e-8 * np.abs(u_plain).max()
