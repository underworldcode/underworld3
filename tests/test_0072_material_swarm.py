"""Materials named on a swarm, with no level sets in sight.

``MaterialSwarm`` is the user-facing material interface: declare the
materials and their properties, say where each one is, hand the swarm to the
solver. The index variable, the level sets and the blend are machinery.
"""

import warnings

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


def _layered(mesh, tag, fill=3, **kwargs):
    materials = uw.swarm.MaterialSwarm(mesh, fill_param=fill, name=tag, **kwargs)
    materials.add("lower", shear_viscosity_0=1.0, density=3300)
    materials.add("upper", shear_viscosity_0=ETA_TOP, density=3400)
    materials["upper"] = mesh.X[1] > H
    return materials


def _couette(mesh, tag, materials):
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
    A = 1.0 / (H + (1.0 - H) / ETA_TOP)
    X = np.asarray(v.coords)
    exact = np.where(X[:, 1] < H, A * X[:, 1], A * H + A / ETA_TOP * (X[:, 1] - H))
    return np.sqrt(np.mean((np.asarray(v.data[:, 0]) - exact) ** 2))


def test_a_model_names_materials_and_never_writes_a_mask():
    """The whole interface: two named materials, a region, one assignment."""
    mesh = _box()
    materials = _layered(mesh, "Ma")

    assert _couette(mesh, "a", materials) < 1e-5
    # and the blend is exactly what the assembler integrated
    assert abs(
        float(uw.maths.Integral(mesh, materials.shear_viscosity_0).evaluate())
        - (1.0 * H + ETA_TOP * (1.0 - H))
    ) < 1e-8
    assert abs(
        float(uw.maths.Integral(mesh, materials["upper"].mask).evaluate()) - (1 - H)
    ) < 1e-8


def test_a_property_may_be_a_law_not_a_number():
    """The reason the partition of unity exists: a material whose viscosity is
    an expression cannot be stored as a value at the integration points, but it
    blends symbolically."""
    mesh = _box(cell_size=0.2)
    T = uw.discretisation.MeshVariable("Tlaw", mesh, 1, degree=1)
    with uw.synchronised_array_update():
        T.data[:, 0] = np.asarray(T.coords)[:, 1]

    materials = uw.swarm.MaterialSwarm(mesh, fill_param=3, name="Mlaw")
    materials.add("a", shear_viscosity_0=1.0)
    materials.add("b", shear_viscosity_0=10 * sympy.exp(-T.sym[0]))
    materials["b"] = mesh.X[1] > H

    blended = materials.shear_viscosity_0
    assert T.sym[0] in blended.atoms(type(T.sym[0]))            # the law survived
    # exact: 0.5 * 1 + integral over the top half of 10 exp(-y)
    exact = 0.5 + 10.0 * (np.exp(-0.5) - np.exp(-1.0))
    assert abs(float(uw.maths.Integral(mesh, blended).evaluate()) - exact) < 2e-3


@pytest.mark.parametrize("kind", ["symbolic", "combined", "array", "callable"])
def test_regions_can_be_written_four_ways(kind):
    mesh = _box(cell_size=0.2)
    materials = uw.swarm.MaterialSwarm(mesh, fill_param=3, name=f"Mr{kind[:4]}")
    materials.add("bg", shear_viscosity_0=1.0)
    materials.add("box", shear_viscosity_0=2.0)
    materials.populate()

    x, y = mesh.X
    if kind == "symbolic":
        region, area = y > H, 0.5
    elif kind == "combined":
        region, area = (y > H) & (x < 0.5), 0.25
    elif kind == "array":
        coords = np.asarray(materials._particle_coordinates.data)
        region, area = coords[:, 1] > H, 0.5
    else:
        region, area = (lambda c: c[:, 1] > H), 0.5

    materials["box"] = region
    assert abs(
        float(uw.maths.Integral(mesh, materials["box"].mask).evaluate()) - area
    ) < 0.02


def test_only_recognised_properties_reach_the_constitutive_model():
    """shear_viscosity_0 is pushed; density is not a viscous-model parameter
    and stays a symbol for the model script to use in the body force."""
    mesh = _box(cell_size=0.2)
    materials = _layered(mesh, "Mp", fill=2)
    v = uw.discretisation.MeshVariable("vp", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("pp", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.materials = materials

    pushed = stokes.constitutive_model.Parameters.shear_viscosity_0
    assert abs(
        float(uw.maths.Integral(mesh, pushed.sym if hasattr(pushed, "sym") else pushed
                                ).evaluate()) - (1.0 * H + ETA_TOP * (1.0 - H))
    ) < 1e-8

    assert not hasattr(stokes.constitutive_model.Parameters, "density")
    assert abs(
        float(uw.maths.Integral(mesh, materials.density).evaluate())
        - (3300 * H + 3400 * (1 - H))
    ) < 1e-8


def test_a_property_nothing_uses_is_reported():
    """A misspelled viscosity is silently the default one, so say so."""
    mesh = _box(cell_size=0.25)
    materials = uw.swarm.MaterialSwarm(mesh, fill_param=2, name="Mw")
    materials.add("a", shear_viscosity_0=1.0, viscocity=99.0)
    materials.add("b", shear_viscosity_0=2.0, viscocity=99.0)
    materials["b"] = mesh.X[1] > H

    v = uw.discretisation.MeshVariable("vw", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("pw", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.materials = materials
    stokes.add_dirichlet_bc((0.0, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stokes._build()
    assert any("viscocity" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]


def test_a_property_missing_from_one_material_is_an_error():
    mesh = _box(cell_size=0.25)
    materials = uw.swarm.MaterialSwarm(mesh, fill_param=2, name="Mm")
    materials.add("a", shear_viscosity_0=1.0, density=3300)
    materials.add("b", shear_viscosity_0=2.0)               # no density
    with pytest.raises(KeyError, match="density"):
        materials.density


def test_materials_must_be_declared_before_the_swarm_is_populated():
    mesh = _box(cell_size=0.25)
    materials = uw.swarm.MaterialSwarm(mesh, fill_param=2, name="Md")
    materials.add("a", shear_viscosity_0=1.0)
    materials.populate()
    with pytest.raises(RuntimeError, match="already populated"):
        materials.add("b", shear_viscosity_0=2.0)


def test_a_property_changed_after_attachment_reaches_the_solver():
    mesh = _box(cell_size=0.2)
    materials = _layered(mesh, "Mc", fill=2)
    v = uw.discretisation.MeshVariable("vc", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("pc", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.materials = materials

    materials["upper"].set(shear_viscosity_0=7.0)
    pushed = stokes.constitutive_model.Parameters.shear_viscosity_0
    expr = pushed.sym if hasattr(pushed, "sym") else pushed
    assert abs(
        float(uw.maths.Integral(mesh, expr).evaluate()) - (1.0 * H + 7.0 * (1 - H))
    ) < 1e-8


def test_the_mixing_rule_only_matters_for_fractional_masks():
    """With the default sampling exactly one mask is 1, so arithmetic and
    harmonic blending give the same integral; with the share they do not."""
    mesh = _box(cell_size=0.15, regular=False)
    sharp = _layered(mesh, "Msh", fill=4)
    both = [
        float(uw.maths.Integral(mesh, sharp.blend("shear_viscosity_0", m)).evaluate())
        for m in ("arithmetic", "harmonic")
    ]
    assert abs(both[0] - both[1]) < 1e-6 * abs(both[0])

    shared = _layered(mesh, "Msr", fill=4, proxy_sampling="share")
    both = [
        float(uw.maths.Integral(mesh, shared.blend("shear_viscosity_0", m)).evaluate())
        for m in ("arithmetic", "harmonic")
    ]
    assert abs(both[0] - both[1]) > 1.0


def test_it_is_a_swarm():
    """Advection, population control and extra state variables all still work:
    a MaterialSwarm is a Swarm."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5), cellSize=0.15, qdegree=2
    )
    materials = uw.swarm.MaterialSwarm(mesh, fill_param=3, name="Mad")
    materials.add("bg", shear_viscosity_0=1.0)
    materials.add("layer", shear_viscosity_0=10.0)
    strain = uw.swarm.SwarmVariable(
        "eps", materials, 1, proxy_location="integration_points",
        proxy_sampling="share",
    )
    materials["layer"] = sympy.Abs(mesh.X[1]) < 0.2
    materials.population_control = dict(min_per_cell=6)

    area0 = float(uw.maths.Integral(mesh, materials["layer"].mask).evaluate())
    x, y = mesh.X
    for _ in range(6):
        materials.advection(sympy.Matrix([[x, -y]]), 0.05, order=2)

    labels = np.unique(np.asarray(materials.index.data).reshape(-1))
    assert set(labels.tolist()) <= {0, 1}                 # still labels
    area1 = float(uw.maths.Integral(mesh, materials["layer"].mask).evaluate())
    assert area1 < area0                                  # the layer thinned
    assert area1 > 0.3 * area0                            # but did not fall apart
    assert np.asarray(strain.data).shape[0] == materials.local_size
