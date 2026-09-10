"""Materials tied to the mesh rather than to particles.

``MaterialRegions`` is the other distribution: same declaration, same handoff
to a solver, but the level sets come from mesh labels or from geometry, so
there are no particles to populate or advect. A registry can be shared between
the two.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

ETA_TOP, H = 1.0e3, 0.5


def _box(cell_size=0.1, regular=True):
    return uw.meshing.UnstructuredSimplexBox(
        cellSize=cell_size, qdegree=2, regular=regular
    )


def _couette(mesh, tag, materials, h=H):
    v = uw.discretisation.MeshVariable(f"v{tag}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"p{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.materials = materials
    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1e-8
    stokes.solve()
    A = 1.0 / (h + (1.0 - h) / ETA_TOP)
    X = np.asarray(v.coords)
    exact = np.where(X[:, 1] < h, A * X[:, 1], A * h + A / ETA_TOP * (X[:, 1] - h))
    return np.sqrt(np.mean((np.asarray(v.data[:, 0]) - exact) ** 2))


def test_regions_need_no_particles():
    """The same model as the swarm case, with the geometry on the mesh."""
    mesh = _box()
    materials = uw.MaterialRegions(mesh, name="Rc")
    materials.add("lower", shear_viscosity_0=1.0, density=3300)
    materials.add("upper", shear_viscosity_0=ETA_TOP, density=3400)
    materials["upper"] = mesh.X[1] > H

    assert _couette(mesh, "rc", materials) < 1e-5
    assert abs(
        float(uw.maths.Integral(mesh, materials.shear_viscosity_0).evaluate())
        - (1.0 * H + ETA_TOP * (1.0 - H))
    ) < 1e-8


def test_region_masks_are_a_sharp_partition_of_unity():
    mesh = _box(cell_size=0.15, regular=False)
    materials = uw.MaterialRegions(mesh, name="Rp")
    materials.add("a", shear_viscosity_0=1.0)
    materials.add("b", shear_viscosity_0=2.0)
    materials.add("c", shear_viscosity_0=3.0)
    materials["b"] = mesh.X[1] > 0.33
    materials["c"] = mesh.X[1] > 0.66

    materials._ensure_built()
    U = np.column_stack(
        [np.asarray(v.data[:, 0]) for v in materials._level_sets()]
    )
    assert np.allclose(U.sum(axis=1), 1.0, atol=1e-12), U.sum(axis=1)
    assert set(np.unique(U).tolist()) <= {0.0, 1.0}      # no fractions, ever
    assert abs(float(uw.maths.Integral(mesh, materials["c"].mask).evaluate())
               - (1 - 0.66)) < 0.02


def test_a_region_can_come_from_a_mesh_label():
    """The gmsh-physical-group route, exercised with a label built by hand."""
    mesh = _box(cell_size=0.2)
    dm = mesh.dm
    c0, c1 = dm.getHeightStratum(0)
    centroids = np.array([dm.computeCellGeometryFVM(c)[1] for c in range(c0, c1)])

    dm.createLabel("TopHalf")
    upper_cells = [c for c in range(c0, c1) if centroids[c - c0][1] > H]
    for c in upper_cells:
        dm.setLabelValue("TopHalf", c, 1)

    materials = uw.MaterialRegions(mesh, name="Rl")
    materials.add("lower", shear_viscosity_0=1.0)
    materials.add("upper", shear_viscosity_0=ETA_TOP)
    materials["upper"] = "TopHalf"

    area = float(uw.maths.Integral(mesh, materials["upper"].mask).evaluate())
    assert abs(area - (1 - H)) < 0.05, area

    with pytest.raises(KeyError, match="no label"):
        materials["upper"] = "NotALabel"


def test_a_registry_is_shared_between_distributions():
    """Definitions live in the registry, so two distributions can use one set
    of materials — and a property change reaches both."""
    mesh = _box(cell_size=0.2)
    rocks = uw.MaterialRegistry()
    rocks.add("lower", shear_viscosity_0=1.0)
    rocks.add("upper", shear_viscosity_0=ETA_TOP)

    regions = uw.MaterialRegions(mesh, registry=rocks, name="Rsh")
    regions["upper"] = mesh.X[1] > H
    swarm = uw.swarm.MaterialSwarm(mesh, registry=rocks, fill_param=3, name="Ssh")
    swarm["upper"] = mesh.X[1] > H

    exact = 1.0 * H + ETA_TOP * (1.0 - H)
    for distribution in (regions, swarm):
        got = float(
            uw.maths.Integral(mesh, distribution.shear_viscosity_0).evaluate()
        )
        assert abs(got - exact) < 1e-8, (distribution, got)

    rocks["upper"].set(shear_viscosity_0=7.0)
    exact = 1.0 * H + 7.0 * (1.0 - H)
    for distribution in (regions, swarm):
        got = float(
            uw.maths.Integral(mesh, distribution.shear_viscosity_0).evaluate()
        )
        assert abs(got - exact) < 1e-8, (distribution, got)


def test_a_registry_stands_alone_and_round_trips():
    """No mesh, no swarm: a material library is just definitions."""
    rocks = uw.MaterialRegistry()
    rocks.add("mantle", shear_viscosity_0=1.0, density=3300,
              description="upper mantle", reference="T&S (2014)")
    rocks.add("slab", shear_viscosity_0=1.0e3, density=3400)

    assert rocks.list_materials() == ["mantle", "slab"]
    assert rocks["mantle"].index == 0 and rocks["slab"].index == 1
    assert rocks.declared_properties() == {"shear_viscosity_0", "density"}

    config = rocks.export_config()
    clone = uw.MaterialRegistry().import_config(config)
    assert clone.list_materials() == rocks.list_materials()
    assert clone["mantle"].description == "upper mantle"
    assert float(clone["slab"].get_property("shear_viscosity_0")) == 1.0e3

    # the enum spelling reaches the same place as the keyword
    rocks["slab"].set_property(uw.MaterialProperty.DENSITY, 3500)
    assert float(rocks["slab"].get_property("density")) == 3500


def test_materials_must_be_declared_before_the_level_sets_are_built():
    mesh = _box(cell_size=0.25)
    materials = uw.MaterialRegions(mesh, name="Rd")
    materials.add("a", shear_viscosity_0=1.0)
    materials["a"] = mesh.X[1] > H
    materials._ensure_built()
    with pytest.raises(RuntimeError, match="already built"):
        materials.add("b", shear_viscosity_0=2.0)


def test_a_property_can_be_a_law_for_regions_too():
    mesh = _box(cell_size=0.2)
    T = uw.discretisation.MeshVariable("Trl", mesh, 1, degree=1)
    with uw.synchronised_array_update():
        T.data[:, 0] = np.asarray(T.coords)[:, 1]

    materials = uw.MaterialRegions(mesh, name="Rlaw")
    materials.add("a", shear_viscosity_0=1.0)
    materials.add("b", shear_viscosity_0=10 * sympy.exp(-T.sym[0]))
    materials["b"] = mesh.X[1] > H

    exact = 0.5 + 10.0 * (np.exp(-0.5) - np.exp(-1.0))
    got = float(uw.maths.Integral(mesh, materials.shear_viscosity_0).evaluate())
    assert abs(got - exact) < 2e-3, got
