"""Live-rampable JIT constants — issue #302.

A constant UWexpression listed in ``solver.constants_manifest`` must be
readable from its ``constants[]`` slot in the compiled kernels, never baked
as a C literal. The old two-phase lowering substituted constants[] slots
with a TOP-LEVEL xreplace before unwrapping, so any constant nested inside
another UWexpression — which includes EVERY ``Parameters.*`` value, since
the setter template-wraps it — was silently folded to a literal: ramping
its ``.sym`` between solves had no effect until a full rebuild.
"""

import numpy as np
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _build():
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
    T = uw.discretisation.MeshVariable("T302", mesh, 1, degree=2)

    k_direct = uw.expression(r"k_d", 1.0, "constant at parameter top level")
    k_nested = uw.expression(r"k_n302", 1.0, "constant nested in a wrapper")
    # Non-constant wrapper: the collection recurses into it and manifests
    # k_nested, but a top-level substitution cannot see inside it — the
    # baked-constant topology from issue #302.
    wrapper = uw.expression(
        r"\eta_{w302}", k_nested * 2.0 + 0.05 * T.sym[0] ** 2, "wrapper")

    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = (
        k_direct * (1.0 + 0.1 * T.sym[0] ** 2) + wrapper)
    poisson.f = 1.0
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.add_dirichlet_bc(0.0, "Bottom")
    return poisson, T, k_direct, k_nested


def _mean(poisson, T):
    # COLD every time. The diffusivity depends on T, so this is a nonlinear solve and
    # its answer is only pinned to the SNES tolerance — a warm start stops at a
    # different point in that same tolerance ball. This test compares solutions at
    # rtol 1e-12 to check that a constant was read from its slot rather than baked as a
    # literal, so it must hold the initial guess fixed or it measures the warm-start
    # policy instead of the thing it is about.
    poisson.solve(zero_init_guess=True)
    return float(np.asarray(T.data)[:, 0].mean())


def test_manifested_constants_ramp_without_rebuild():
    poisson, T, k_direct, k_nested = _build()
    base = _mean(poisson, T)

    names = [e.name for _i, e in poisson.constants_manifest]
    assert "k_d" in names and "k_n302" in names

    # Ramp each constant via .sym alone — REAL solves, no rebuild between.
    k_direct.sym = 10.0
    direct_ramped = _mean(poisson, T)
    k_direct.sym = 1.0

    k_nested.sym = 10.0
    nested_ramped = _mean(poisson, T)
    k_nested.sym = 1.0

    assert abs(direct_ramped - base) > 1e-8, (
        "top-level constant did not ramp without a rebuild")
    assert abs(nested_ramped - base) > 1e-8, (
        "NESTED constant did not ramp without a rebuild (issue #302)")

    # Ramping back restores the base solution exactly — no hidden state.
    assert np.isclose(_mean(poisson, T), base, rtol=1e-12)


def test_ramped_solution_matches_rebuilt_solution():
    # The live-ramped answer must equal the answer from a full rebuild at
    # the same parameter value — the manifest and the C source agree.
    poisson, T, k_direct, k_nested = _build()
    _mean(poisson, T)
    k_nested.sym = 10.0
    ramped = _mean(poisson, T)

    poisson2, T2, _kd2, k_nested2 = _build()
    k_nested2.sym = 10.0
    rebuilt = _mean(poisson2, T2)

    assert np.isclose(ramped, rebuilt, rtol=1e-10), (
        f"ramped {ramped} != rebuilt {rebuilt}")


def test_two_constants_with_the_same_name_do_not_collapse():
    """Two constants may legitimately share a symbol name — every
    ViscousFlowModel calls its viscosity \\eta — and they must reach the
    compiled kernel as two different ``constants[]`` slots.

    The JIT placeholder is a plain ``sympy.Symbol`` subclass, so a placeholder
    named for the expression alone made the two the SAME symbol whatever their
    index. A two-material Stokes solve then assembled
    ``(phi_0 + phi_1) * constants[k]`` — one uniform viscosity — and returned
    exactly the linear-shear answer while the manifest and the symbolic
    expression both looked correct.

    NB the two constants have to be built the way a solver builds them (through
    ``Parameters``, which tags each one). Two BARE expressions of the same name
    are the same symbol to sympy by design — ``2*a + 3*b`` is ``5*\\eta`` — so
    they never reach the JIT as two things in the first place.
    """
    import sympy
    from underworld3.utilities._jitextension import _extract_constants

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.5, qdegree=2)
    v = uw.discretisation.MeshVariable("vjc", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("pjc", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

    lower = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns, material_name="lo")
    lower.Parameters.shear_viscosity_0 = 1.0
    upper = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns, material_name="up")
    upper.Parameters.shear_viscosity_0 = 1000.0
    a = lower.Parameters.shear_viscosity_0
    b = upper.Parameters.shear_viscosity_0
    assert a != b, "the Parameters route no longer distinguishes same-named constants"

    phi0, phi1, xs = sympy.symbols("phi0 phi1 xs")
    manifest, subs_map = _extract_constants((a * phi0 * xs + b * phi1 * xs,), mesh)

    assert len(manifest) == 2, manifest
    placeholders = {subs_map[a], subs_map[b]}
    assert len(placeholders) == 2, "the two placeholders are the same symbol"

    lowered = sympy.expand((a * phi0 * xs + b * phi1 * xs).xreplace(subs_map))
    assert len(lowered.free_symbols & placeholders) == 2, lowered


def test_same_named_placeholders_order_deterministically():
    """The ORDERING half of the same-name problem, pinned without MPI.

    ``_hashable_content`` gives two placeholders separate identities but does
    nothing for ``Symbol.sort_key()``, which comes from the name. Two
    placeholders that sort equal leave term order inside an ``Add`` to hash
    order, which is randomised per process — so the generated C differs
    between MPI ranks and the cross-rank hash check aborts the run, on roughly
    half of launches. This test is the deterministic proxy: distinct sort keys,
    and a sum that canonicalises the same however it is written.
    """
    import sympy
    from underworld3.utilities._jitextension import _JITConstant

    c0 = _JITConstant(0, name=r"\eta")
    c1 = _JITConstant(1, name=r"\eta")

    assert c0.sort_key() != c1.sort_key(), (
        "same-named placeholders sort equal; Add term order will follow the "
        "hash seed and the generated C will differ between ranks")
    assert c0 != c1 and c0 is not c1              # the identity half

    p0, p1 = sympy.symbols("p0 p1")
    written_one_way = sympy.printing.ccode(p0 / c0 + p1 / c1)
    written_the_other = sympy.printing.ccode(p1 / c1 + p0 / c0)
    assert written_one_way == written_the_other, (
        written_one_way, written_the_other)
    assert "constants[0]" in written_one_way and "constants[1]" in written_one_way


def test_the_manifest_order_does_not_move_when_a_value_changes():
    """Slot assignment follows creation order, not value.

    ``_stable_sort_key`` falls through to ``str(expr)``, which for a
    UWexpression is its CURRENT VALUE, so tie-breaking two same-named
    constants on it permutes their ``constants[]`` slots the moment one is
    ramped past the other lexically. Ramping 1000 -&gt; 0.5 does exactly that.
    Slots that move for a reason unrelated to the model invalidate the JIT
    cache needlessly, and would swap the two values outright for anything that
    keyed on the manifest rather than on the generated source.

    NB the values matter: 1000 -&gt; 1e-6 does NOT expose it, because
    ``"1.00000000000000e-6"`` sorts after ``"1.00000000000000"`` as a prefix.
    """
    import sympy
    from underworld3.utilities._jitextension import _extract_constants

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.5, qdegree=2)
    v = uw.discretisation.MeshVariable("vmo", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("pmo", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

    lower = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns, material_name="lo")
    lower.Parameters.shear_viscosity_0 = 1.0
    upper = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns, material_name="up")
    upper.Parameters.shear_viscosity_0 = 1000.0
    a = lower.Parameters.shear_viscosity_0
    b = upper.Parameters.shear_viscosity_0
    assert a != b and a.name == b.name, "expected two distinct same-named constants"

    x = sympy.Symbol("xmo")
    slot_of = lambda manifest: {id(e): i for i, e in manifest}

    before = slot_of(_extract_constants((a * x + b * x,), mesh)[0])
    assert len(before) == 2, before

    b.sym = 0.5                       # in place: the same object, a new value
    after = slot_of(_extract_constants((a * x + b * x,), mesh)[0])

    assert before == after, (
        "ramping a value permuted the constants[] slots", before, after)


def test_two_materials_solve_the_layered_problem_not_the_uniform_one():
    """The end-to-end form of the same defect.

    Two ViscousFlowModels composed into a layered viscosity must give the
    layered answer. Before the fix this returned exactly linear shear (L2
    2.819e-1 against the layered profile, 1.502e-10 against uniform shear)
    because the two \\eta constants shared a constants[] slot. The check that
    does not depend on the mesh or the level-set representation is that the
    composed model and the equivalent single-model blend now agree.
    """
    import sympy

    eta_top, h = 1.0e3, 0.5
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, qdegree=2, regular=True)
    swarm = uw.swarm.Swarm(mesh)
    material = uw.swarm.IndexSwarmVariable("Mjc", swarm, indices=2, proxy_degree=1)
    swarm.populate(fill_param=3)
    X = np.asarray(swarm._particle_coordinates.data)
    with uw.synchronised_array_update():
        material.data[:, 0] = (X[:, 1] > h).astype(int)

    def _solve(tag, build_model):
        v = uw.discretisation.MeshVariable(f"vjc{tag}", mesh, mesh.dim, degree=2)
        p = uw.discretisation.MeshVariable(f"pjc{tag}", mesh, 1, degree=1)
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        build_model(stokes)
        stokes.add_dirichlet_bc((1.0, 0.0), "Top")
        stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
        stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
        stokes.tolerance = 1e-8
        stokes.solve()
        return np.asarray(v.data[:, 0]), np.asarray(v.coords)

    def _composed(stokes):
        lower = uw.constitutive_models.ViscousFlowModel(
            stokes.Unknowns, material_name="lower")
        lower.Parameters.shear_viscosity_0 = 1.0
        upper = uw.constitutive_models.ViscousFlowModel(
            stokes.Unknowns, material_name="upper")
        upper.Parameters.shear_viscosity_0 = eta_top
        stokes.constitutive_model = uw.MultiMaterialConstitutiveModel(
            stokes.Unknowns, material, [lower, upper])

    def _blended(stokes):
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = (
            material.createMask([1.0, eta_top]))

    composed, coords = _solve("c", _composed)
    blended, _ = _solve("b", _blended)

    # the two routes are the same problem, and must give the same answer
    assert np.sqrt(np.mean((composed - blended) ** 2)) < 1e-9

    # and it is NOT the uniform-viscosity answer (which is linear shear)
    assert np.sqrt(np.mean((composed - coords[:, 1]) ** 2)) > 1e-2
