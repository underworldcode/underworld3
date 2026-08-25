"""Rotated strong free-slip BC (add_rotated_freeslip_bc) through the real solver API.

Increment 1 validation: on an axis-aligned box, rotated free-slip on all four walls
must reproduce the native essential free-slip solve, enforce zero wall-normal flow,
and (annulus) give per-node radial free-slip with machine-zero leakage.
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw
from underworld3 import analytic as A
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

_EPS = np.finfo(float).eps


def _wrap(dm, m0):
    return uw.discretisation.Mesh(
        dm.clone(), simplex=True,
        coordinate_system_type=m0.CoordinateSystem.coordinate_type,
        qdegree=3, boundaries=m0.boundaries)


def _solcx_essential(mesh, sol):
    v = uw.discretisation.MeshVariable("vE", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pE", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    s.add_essential_bc((sympy.oo, 0.0), "Top")
    s.add_essential_bc((sympy.oo, 0.0), "Bottom")
    s.add_essential_bc((0.0, sympy.oo), "Left")
    s.add_essential_bc((0.0, sympy.oo), "Right")
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()
    return v


def test_rotated_freeslip_box_reproduces_essential():
    """Box: rotated free-slip on all 4 axis-aligned walls == native essential free-slip."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 16), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)
    vE = _solcx_essential(mesh, sol)

    v = uw.discretisation.MeshVariable("vR", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pR", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    rel = np.linalg.norm(v.data - vE.data) / np.linalg.norm(vE.data)
    assert rel < 1e-4, f"rotated free-slip differs from essential by {rel:.2e}"
    # exact analytic accuracy
    assert sol.velocity_error(v) < 1e-3


def test_rotated_linear_workspace_reuses_unchanged_operator():
    """Repeated linear solves reuse the rotated workspace across solves.

    RHS-only changes ride the iteration-0 fast path (no Jacobian assembly, no
    ptap, no PCSetUp — ``workspace_reused``); an operator-coefficient FIELD
    change is detected by the state-counter key and refreshes the operator
    values IN PLACE on the same objects (same Q/Ahat/KSP handles — the
    structure tier); an explicit ``time=`` solve vetoes the fast path."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3
    )
    temperature = uw.discretisation.MeshVariable(
        "Tcache", mesh, 1, degree=1, continuous=True
    )
    viscosity = uw.discretisation.MeshVariable(
        "Etacache", mesh, 1, degree=1, continuous=True
    )
    velocity = uw.discretisation.MeshVariable(
        "Vcache", mesh, mesh.dim, degree=2, continuous=True
    )
    pressure = uw.discretisation.MeshVariable(
        "Pcache", mesh, 1, degree=1, continuous=False
    )
    temperature.data[:, 0] = 1.0 + temperature.coords[:, 0]
    viscosity.data[:, 0] = 1.0

    stokes = uw.systems.Stokes(
        mesh, velocityField=velocity, pressureField=pressure
    )
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity.sym[0]
    stokes.bodyforce = sympy.Matrix([[0.0, -temperature.sym[0]]])
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.tolerance = 1.0e-8

    stokes.solve()
    velocity_1 = velocity.data.copy()
    cache_1 = stokes._rotated_linear_cache
    assert cache_1 is not None, "no workspace cached after a linear rotated solve"
    handles_1 = (
        cache_1["Q"].handle,
        cache_1["Ahat"].handle,
        cache_1["ctx"]["ksp"].handle,
    )
    assert not stokes._rotated_freeslip_info["workspace_reused"]
    assert not stokes._rotated_freeslip_info["rotation_reused"]

    # RHS-only change: fast path (same handles, no reassembly), and the
    # linear-in-forcing solution exactly doubles.
    temperature.data[:, 0] *= 2.0
    stokes.solve(zero_init_guess=False)

    cache_2 = stokes._rotated_linear_cache
    handles_2 = (
        cache_2["Q"].handle,
        cache_2["Ahat"].handle,
        cache_2["ctx"]["ksp"].handle,
    )
    assert handles_2 == handles_1
    assert stokes._rotated_freeslip_info["workspace_reused"]
    assert stokes._rotated_freeslip_info["rotation_reused"]
    relative_scaling_error = (
        np.linalg.norm(velocity.data - 2.0 * velocity_1)
        / np.linalg.norm(2.0 * velocity_1)
    )
    assert relative_scaling_error < 1.0e-6

    # Operator-coefficient FIELD change: the state-counter key catches it, the
    # operator values are refreshed in place (same objects, no fast path).
    velocity_2 = velocity.data.copy()
    viscosity.data[:, 0] *= 2.0
    stokes.solve(zero_init_guess=False)

    assert not stokes._rotated_freeslip_info["workspace_reused"]
    assert stokes._rotated_freeslip_info["rotation_reused"]
    assert stokes._rotated_linear_cache["ctx"]["ksp"].handle == handles_1[2]
    viscosity_scaling_error = (
        np.linalg.norm(velocity.data - 0.5 * velocity_2)
        / np.linalg.norm(0.5 * velocity_2)
    )
    assert viscosity_scaling_error < 1.0e-6

    # An explicit time= solve must veto the fast path (petsc_t reaches the
    # kernels outside any state counter), while the workspace objects persist.
    refreshed_velocity = velocity.data.copy()
    stokes.solve(zero_init_guess=False, time=0.5)
    assert not stokes._rotated_freeslip_info["workspace_reused"]
    assert stokes._rotated_linear_cache["ctx"]["ksp"].handle == handles_1[2]
    time_refresh_error = (
        np.linalg.norm(velocity.data - refreshed_velocity)
        / np.linalg.norm(refreshed_velocity)
    )
    assert time_refresh_error < 1.0e-6


def _solve_report(info):
    """The rotated solve's own verdict, for failure messages: an assertion
    that says only "the answer moved" cannot tell a stale operator from a
    linear solve that stopped early, or either of those from a null-space
    gauge that one solve removed and the other did not."""
    rows = info.get("normal_rows")
    gauge = info.get("rotation_gauge") or {}
    modes = "; ".join(
        f"[{i}] viol={m['viol']:.3e} op_viol="
        f"{'n/a' if m['op_viol'] is None else format(m['op_viol'], '.3e')} "
        f"accepted={m['accepted']}"
        f"{'' if m['accepted'] else ' (' + str(m['rejected_by']) + ')'}"
        for i, m in enumerate(gauge.get("modes", [])))
    return (f"converged={info.get('converged')} "
            f"ksp_reason={info.get('ksp_reason')} "
            f"ksp_its={info.get('ksp_its')} "
            f"newton_its={info.get('nonlinear_iterations')} "
            f"|r|={info.get('rnorm')} |r0|={info.get('rnorm0')} "
            f"rotation_reused={info.get('rotation_reused')} "
            f"workspace_reused={info.get('workspace_reused')} "
            f"constrained_rows={None if rows is None else len(rows)} "
            f"distinct_rows={None if rows is None else len(set(rows))} "
            f"boundaries={info.get('boundaries')} "
            f"gauge_offered={gauge.get('offered')} "
            f"gauge_orthonormal={gauge.get('orthonormal')} "
            f"gauge_removed={info.get('rotation_gauge_removed')} "
            f"modes=({modes})")


class _LocatorTally:
    """Count point-location work, how much came back ``-1``, and how much of
    that the #551 rejection radius is responsible for.

    Every call is answered twice — once as shipped, once with
    ``_local_cell_reach`` set aside so the walk cannot reject anything early —
    and the cells are compared. ``radius_changed=0`` means the rejection radius
    did not alter a single answer, which is the measurement that decides
    whether the locator is implicated in a downstream disagreement. It is not
    an argument, it is a count, and it is taken on whatever mesh the machine
    running the test actually built.
    """

    def __init__(self):
        self.calls = 0
        self.points = 0
        self.rejected = 0
        self.radius_changed = 0
        self._mesh_cls = uw.discretisation.discretisation_mesh.Mesh
        self._wrapped = self._mesh_cls._get_closest_local_cells_internal

    def __enter__(self):
        outer = self

        def counted(mesh_self, coords, **kwargs):
            got = outer._wrapped(mesh_self, coords, **kwargs)
            saved = getattr(mesh_self, "_local_cell_reach", None)
            try:
                mesh_self._local_cell_reach = None      # no early rejection
                reference = outer._wrapped(mesh_self, coords, **kwargs)
            finally:
                mesh_self._local_cell_reach = saved
            got_a = np.asarray(got).reshape(-1)
            outer.calls += 1
            outer.points += len(coords)
            outer.rejected += int(np.count_nonzero(got_a < 0))
            outer.radius_changed += int(np.count_nonzero(
                got_a != np.asarray(reference).reshape(-1)))
            return got

        self._mesh_cls._get_closest_local_cells_internal = counted
        return self

    def __exit__(self, *exc):
        self._mesh_cls._get_closest_local_cells_internal = self._wrapped
        return False

    def __str__(self):
        return (f"calls={self.calls} points={self.points} "
                f"returned_-1={self.rejected} "
                f"radius_changed={self.radius_changed}")


def _rigid_body_decomposition(coords, diff):
    """Split a velocity difference field into rigid-body modes and the rest.

    Two solves that both drive the residual to machine zero on a system with
    the same initial residual can only differ in the null space, so the
    question "is the difference a rigid-body mode?" has a yes/no answer. The
    modes are the ones the solver itself considers (``_rigid_rotation_modes``:
    ``(-y, x)`` in 2-D, ``e_k x r`` in 3-D) plus the translations, built here
    from the nodal coordinates so this is independent of the solver's own
    machinery. Nodal (not PETSc-global) inner products — exact in serial,
    which is where this test runs.

    Returns ``(|d|, per-direction shares, |d| off the rigid-body span)``. The
    modes are NOT mutually orthogonal — the rotation about the origin has a
    large translation component on a box in the first quadrant — so they are
    Gram-Schmidt'd in order, exactly as the solver does before projecting, and
    the per-direction shares are therefore shares on the k-th ORTHOGONALISED
    direction, not on the named mode. The number that answers the question is
    the off-span norm: a pure rigid rotation gives 4e-18 of it, a random field
    gives 0.9985 of it (both measured).
    """
    coords = np.asarray(coords, dtype=np.float64)
    diff = np.asarray(diff, dtype=np.float64)
    dim = coords.shape[1]
    modes = {}
    for axis in range(dim):
        e = np.zeros_like(coords)
        e[:, axis] = 1.0
        modes[f"translation_{'xyz'[axis]}"] = e
    if dim == 2:
        modes["rotation_z"] = np.column_stack([-coords[:, 1], coords[:, 0]])
    else:
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        zero = np.zeros_like(x)
        modes["rotation_x"] = np.column_stack([zero, -z, y])
        modes["rotation_y"] = np.column_stack([z, zero, -x])
        modes["rotation_z"] = np.column_stack([-y, x, zero])

    total = np.linalg.norm(diff)
    residual = diff.copy()
    shares = {}
    basis = []
    for name, mode in modes.items():
        w = mode.copy()
        for q in basis:                      # Gram-Schmidt, as the solver does
            w -= np.vdot(q, w) * q
        n = np.linalg.norm(w)
        if n <= 1.0e-14:
            continue
        w /= n
        basis.append(w)
        component = np.vdot(w, diff)
        shares[name] = abs(component) / (total + 1.0e-300)
        residual -= component * w
    return total, shares, np.linalg.norm(residual)


def _rampable_rotated_stokes(mesh, k_expr, tag, forcing=None):
    """Rotated free-slip Stokes with viscosity given by a rampable
    UWexpression constant (the #416 idiom used by every continuation
    driver). ``forcing`` optionally supplies a scalar field for the body
    force (an RHS-only mesh variable); default is a fixed analytic load."""
    v = uw.discretisation.MeshVariable(f"v{tag}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"p{tag}", mesh, 1, degree=1,
                                       continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = k_expr
    x, y = mesh.X
    load = forcing.sym[0] if forcing is not None \
        else sympy.sin(sympy.pi * x) * sympy.cos(sympy.pi * y)
    s.bodyforce = sympy.Matrix([[0.0, load]])
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    return s, v


def test_rotated_workspace_constant_ramp_invalidates():
    """THE blind-spot regression (PR #418 review, unresolved finding): a
    rampable UWexpression constant changes value with NO state-counter bump
    (#416 contract). The workspace key must see it through the packed
    constants[] signature: the fast path must NOT fire, and the second
    solution must match a fresh-solver control at the ramped viscosity.

    Fail-before validated: on the state-counter-only key (the ported
    original), solve 2 reports workspace_reused=True — the flag lies about a
    stale operator (the answer is still rescued by the loop's structural
    safety net, unlike the original one-shot path which returned a
    bit-identical stale solution)."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(12, 12), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    T = uw.discretisation.MeshVariable("Trmp", mesh, 1, degree=1)
    xy = T.coords
    T.data[:, 0] = np.sin(np.pi * xy[:, 0]) * np.cos(np.pi * xy[:, 1])

    k = uw.function.expression(r"k_\eta", 1.0, "rampable viscosity")
    s1, v1 = _rampable_rotated_stokes(mesh, k, "Rmp", forcing=T)
    s1.solve()

    # negative control for the test itself: an RHS-only field change must
    # ride the fast path, proving the skip is ARMED before we assert the
    # ramp defeats it.
    T.data[:, 0] *= 2.0
    s1.solve(zero_init_guess=False)
    assert s1._rotated_freeslip_info["workspace_reused"], (
        "fast path did not fire on an RHS-only change — the ramp assertion "
        "below would pass vacuously")
    u1 = v1.data.copy()

    # THE RAMP: value change only — no .sym rebuild, no state bump anywhere
    k.sym = 2.0
    s1.solve(zero_init_guess=False)
    info = s1._rotated_freeslip_info
    assert not info["workspace_reused"], (
        "constant ramp rode the fast path — the operator key is blind to "
        "rampable constants again (#416 / PR #418 review finding)")
    assert info["rotation_reused"], "geometry tier should survive a ramp"
    u2 = v1.data.copy()

    # fresh-solver control at the ramped viscosity, same forcing
    k_c = uw.function.expression(r"k_c", 2.0, "control viscosity")
    s_c, v_c = _rampable_rotated_stokes(mesh, k_c, "Ctl", forcing=T)
    s_c.solve()
    err = np.linalg.norm(u2 - v_c.data) / np.linalg.norm(v_c.data)
    assert err < 1e-6, f"ramped solve differs from fresh control by {err:.2e}"
    # and the linear model's exact halved-velocity scaling
    half = np.linalg.norm(u2 - 0.5 * u1) / np.linalg.norm(0.5 * u1)
    assert half < 1e-6, f"ramped solve is not the halved velocity ({half:.2e})"


def test_rotated_workspace_deform_invalidates():
    """mesh.deform between solves: the geometry changed, so the whole workspace
    must be rebuilt and the answer must match a fresh solver on the deformed
    mesh — in every direction the deformed system actually determines.

    Two claims, deliberately separated, because only one of them is well posed:

    **(a) the workspace was invalidated.** ``rotation_reused`` and
    ``workspace_reused`` both False, the locator's rejection radius followed
    the deform, and both solves converged. This is the contract #543 wrote the
    test for and it is asserted hard.

    **(b) the two answers agree.** This claim WAS ill posed. Rotated free-slip
    on a CURVED boundary used to lose the constant-pressure gauge (#560),
    leaving one direction the operator did not pin: a coordinate change of two
    machine epsilons moved the velocity by 1.3e-01 and the move did not scale
    with the perturbation, so a plain comparison passed only where the two
    assemblies happened to agree bitwise — green on macOS (err exactly 0.0 in
    81 consecutive runs) and intermittently red on CI (err 7e-2 to 1.2e-1, the
    size of the unpinned component rather than a drift). The test then compared
    the two solutions with that one direction projected out.

    #560 is fixed and the same probe now measures 9.9e-15, so the projection
    and the probe solve are gone and the two solutions are compared directly,
    at the same 1e-6 tolerance as before.

    The instrumentation below (gauge decisions, constrained-row counts, locator
    tallies, rigid-body decomposition) stays: it is what turned an unreadable
    CI failure into #560, and it is the diagnostic for the next one."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(10, 10), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)

    k = uw.function.expression(r"k_d", 1.0, "viscosity")
    s1, v1 = _rampable_rotated_stokes(mesh, k, "Dfm")
    s1.solve()
    assert s1._rotated_linear_cache is not None
    info_1 = dict(s1._rotated_freeslip_info)
    reach_before = mesh._local_cell_reach

    # bump the top boundary (the free-surface pattern). The deform is where the
    # point locator is actually exercised — measured, the solves themselves make
    # no location calls at all — so the tally goes here.
    coords = mesh.X.coords.copy()
    coords[:, 1] += 0.02 * coords[:, 1] * np.sin(np.pi * coords[:, 0])
    with _LocatorTally() as tally_deform:
        mesh.deform(coords)

    # Mesh-side invariant: the kd-tree index and the point locator's rejection
    # radius (#551) are measured together in _build_kd_tree_index, which
    # deform() drops and rebuilds eagerly. A stale SMALL radius is the one way
    # point location silently loses points, and this is the only test in the
    # suite that deforms a mesh between two solves, so it is the one that
    # would see it.
    reach_after = mesh._local_cell_reach
    assert reach_after != reach_before, (
        f"the locator reach did not follow the deform "
        f"({reach_before:.8g} -> {reach_after:.8g}) — it is measured with the "
        f"kd-tree index and must be rebuilt with it")

    # NEGATIVE CONTROL for that tally: squeeze the reach and the comparison
    # must see answers change. Otherwise a "radius_changed=0" report is the
    # instrument failing to fire, not the radius being inert.
    probe = np.ascontiguousarray(
        np.random.default_rng(5).uniform(0.02, 0.98, size=(400, 2)))
    with _LocatorTally() as tally_tight:
        mesh._LOCATOR_REACH_MARGIN = 0.05
        try:
            mesh._get_closest_local_cells_internal(
                probe, tol=mesh._EVAL_FACE_TOL)
        finally:
            del mesh._LOCATOR_REACH_MARGIN
    assert tally_tight.radius_changed > 0, (
        "a reach margin of 0.05 changed no locator answer, so the "
        "radius_changed counts reported below cannot tell an inert rejection "
        "radius from an instrument that is not looking")

    with _LocatorTally() as tally_deformed:
        s1.solve()
    info = s1._rotated_freeslip_info
    assert not info["rotation_reused"], (
        "workspace survived a mesh.deform — stale rotation Q in use")
    assert not info["workspace_reused"]

    k_c = uw.function.expression(r"k_dc", 1.0, "control viscosity")
    with _LocatorTally() as tally_control:
        s_c, v_c = _rampable_rotated_stokes(mesh, k_c, "DfC")
        s_c.solve()
    info_c = s_c._rotated_freeslip_info

    # Build every diagnostic EAGERLY, not inside the assertion messages: an
    # instrument that only runs when the test fails is an instrument that has
    # never been run.
    report_1 = _solve_report(info_1)
    report_deformed = _solve_report(info)
    report_control = _solve_report(info_c)
    diff = np.asarray(v1.data) - np.asarray(v_c.data)
    err = np.linalg.norm(diff) / np.linalg.norm(v_c.data)
    total, shares, off_mode = _rigid_body_decomposition(v1.coords, diff)
    share_text = " ".join(f"{n}={f:.4f}" for n, f in shares.items())
    off_fraction = off_mode / (total + 1.0e-300)
    rigid_fraction = np.sqrt(max(0.0, 1.0 - off_fraction ** 2))

    # NEGATIVE CONTROL for the decomposition, run on the mesh CI actually
    # built: a pure rigid rotation must come out entirely inside the span, a
    # random field almost entirely outside it. Without this the off-span
    # fraction reported above would be a number nobody had checked.
    node_coords = np.asarray(v1.coords, dtype=np.float64)
    pure = np.column_stack([-node_coords[:, 1], node_coords[:, 0]])
    _, _, pure_off = _rigid_body_decomposition(v1.coords, pure)
    assert pure_off / np.linalg.norm(pure) < 1e-10, (
        f"the rigid-body decomposition does not recognise a pure rotation "
        f"({pure_off / np.linalg.norm(pure):.3e} of it off the span)")
    noise = np.random.default_rng(0).normal(size=pure.shape)
    _, _, noise_off = _rigid_body_decomposition(v1.coords, noise)
    assert noise_off / np.linalg.norm(noise) > 0.9, (
        f"the rigid-body decomposition absorbs a random field "
        f"({noise_off / np.linalg.norm(noise):.3f} of it off the span), so a "
        f"small off-span fraction above would prove nothing")

    assert info["converged"], (
        f"the post-deform solve did not converge: {report_deformed}")
    assert info_c["converged"], (
        f"the fresh control solve did not converge: {report_control}")

    # ------------------------------------------------------------------
    # (b) the answers agree, in the directions the system actually determines
    # ------------------------------------------------------------------
    # This comparison used to be ill posed, and is not any more. Rotated
    # free-slip on a CURVED boundary lost the constant-pressure gauge (#560),
    # leaving one direction the operator did not pin: a coordinate change of
    # two machine epsilons moved the velocity by 1.33e-01 and the size of the
    # move did not track the size of the perturbation, so "the two solves
    # agree" was false as stated. The test passed only where the two
    # assemblies happened to agree bitwise — green on macOS/arm64, red on CI
    # at err 7e-2 to 1.2e-1, the SIZE of the unpinned component rather than a
    # drift. It compared the two solutions with the unpinned direction
    # projected out, measured by an extra perturbed solve.
    #
    # #560 is fixed (the nodal normal is measure-weighted, so the constant
    # pressure is a null vector of the constrained operator again) and that
    # probe now measures 9.9e-15 rather than 1.3e-01. The projection and the
    # probe solve are therefore gone and the solutions are compared directly,
    # at the original 1e-6 tolerance.
    constrained_err = err

    # NEGATIVE CONTROL: inject a discrepancy and check the comparison sees it,
    # or "constrained_err is small" would be true of anything.
    injected = np.random.default_rng(6).normal(size=diff.shape).ravel()
    injected *= 1.0e-3 * np.linalg.norm(v_c.data) / np.linalg.norm(injected)
    poisoned_err = np.linalg.norm(diff.ravel() + injected) / np.linalg.norm(v_c.data)
    assert poisoned_err > 1.0e-4, (
        f"a deliberate 1e-3 discrepancy shows up as only {poisoned_err:.3e}, so "
        f"the comparison below would not notice a real disagreement")

    assert constrained_err < 1e-6, (
        f"post-deform solve differs from fresh control by {constrained_err:.2e}\n"
        f"  first solve   : {report_1}\n"
        f"  post-deform   : {report_deformed}\n"
        f"  fresh control : {report_control}\n"
        f"  locator reach : {reach_before:.8g} -> {reach_after:.8g}\n"
        f"  locator work  : deform {tally_deform} | post-deform solve "
        f"{tally_deformed} | control {tally_control}\n"
        f"  difference    : |d|={total:.6e}\n"
        f"                  IN the rigid-body span: {rigid_fraction:.6f} of |d|\n"
        f"                  OFF it:                 {off_fraction:.6f} of |d| "
        f"({off_mode:.6e})\n"
        f"                  per orthogonalised direction (NOT per named mode, "
        f"they are not orthogonal): {share_text}\n"
        f"This is the assertion that survives #560: the two solves must agree "
        f"in every direction the operator determines. A failure here is NOT "
        f"the pressure gauge. If the constrained row counts or the locator "
        f"tallies differ between the two solvers, point location is "
        f"implicated; if they match, the two OPERATORS differ and something "
        f"survived the deform.")


@pytest.mark.level_2
def test_rotated_freeslip_spherical_shell_3d():
    """3D spherical shell, free-slip inner+outer (the Zhong #248 configuration):
    all THREE rigid rotations must be recognised as nullspace modes and projected
    out, the self-contained Schur solve must converge in a bounded iteration
    count (the 1/mu-mass Schur preconditioner — ~30 its with selfp), and the
    radial leak must be machine-zero on both boundaries."""
    RI, RO = 0.55, 1.0
    mesh = uw.meshing.SphericalShell(radiusInner=RI, radiusOuter=RO,
                                     cellSize=0.25, qdegree=2)
    x, y, z = mesh.X
    r = sympy.sqrt(x**2 + y**2 + z**2)
    v = uw.discretisation.MeshVariable("Vs", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Ps", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    # Y_20-like radial internal load, zero on the boundaries
    ylm = (3 * (z / r) ** 2 - 1) / 2
    g = (r - RI) * (RO - r) * 20.0
    s.bodyforce = ylm * g / r * sympy.Matrix([[x, y, z]])
    nhat = sympy.Matrix([[x / r, y / r, z / r]])
    s.add_rotated_freeslip_bc(0, "Lower", normal=nhat)
    s.add_rotated_freeslip_bc(0, "Upper", normal=nhat)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.tolerance = 1e-7
    s.solve()

    info = s._rotated_freeslip_info
    assert info["ksp_reason"] > 0, f"rotated KSP diverged: {info['ksp_reason']}"
    assert max(info["ksp_its"]) <= 25, (
        f"Schur iteration blow-out: {info['ksp_its']} outer its per increment "
        "(1/mu-mass Schur preconditioning regressed?)")
    assert info["rotation_gauge_removed"], "3D rotation gauge not detected/removed"

    vc = v.coords
    rr = np.linalg.norm(vc, axis=1)
    rhat = vc / rr[:, None]
    vr = np.einsum("ij,ij->i", v.data, rhat)
    vmax = np.linalg.norm(v.data, axis=1).max() + 1e-30
    for lab, mask in [("inner", np.abs(rr - RI) < 1e-3), ("outer", np.abs(rr - RO) < 1e-3)]:
        leak = np.abs(vr[mask]).max() / vmax
        assert leak < 1e-10, f"{lab} radial leakage {leak:.2e} not machine-zero"
    # all three rigid-rotation gauges removed (nodal check, serial test)
    for k, t in enumerate([
            np.column_stack([np.zeros(len(vc)), -vc[:, 2], vc[:, 1]]),
            np.column_stack([vc[:, 2], np.zeros(len(vc)), -vc[:, 0]]),
            np.column_stack([-vc[:, 1], vc[:, 0], np.zeros(len(vc))])]):
        rotfrac = abs(np.sum(v.data * t)) / (np.linalg.norm(t) * np.linalg.norm(v.data) + 1e-30)
        assert rotfrac < 1e-8, f"rotation mode {k} gauge {rotfrac:.2e} not removed"


def _spherical3d_reaction_topography(cell_size=0.13):
    """Zhong l=2 topography recovered directly from rotated constraint reactions."""

    radius_inner = 0.55
    radius_outer = 1.0
    radius_internal = 0.775
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusOuter=radius_outer,
        radiusInternal=radius_internal,
        radiusInner=radius_inner,
        cellSize=cell_size,
        qdegree=2,
        degree=1,
    )
    velocity = uw.discretisation.MeshVariable(
        "U_topography_3d", mesh, mesh.dim, degree=2, continuous=True
    )
    pressure = uw.discretisation.MeshVariable(
        "P_topography_3d", mesh, 1, degree=1, continuous=True
    )
    stokes = uw.systems.Stokes(
        mesh, velocityField=velocity, pressureField=pressure
    )
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    theta = mesh.CoordinateSystem.xR[1]
    unit_r = mesh.CoordinateSystem.unit_e_0
    harmonic = sympy.assoc_legendre(2, 0, sympy.cos(theta))
    stokes.add_natural_bc(harmonic * unit_r, "Internal")
    stokes.add_rotated_freeslip_bc(0, "Upper", normal=unit_r)
    stokes.add_rotated_freeslip_bc(0, "Lower", normal=-unit_r)
    stokes.petsc_use_pressure_nullspace = True
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.tolerance = 1.0e-5
    stokes.solve()

    dm = mesh.dm
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, mesh.dim)
    vertex_start, vertex_end = dm.getDepthStratum(0)
    local_vertex_keys = {
        tuple(np.round(cvec[csec.getOffset(point) // mesh.dim], 12))
        for point in range(vertex_start, vertex_end)
    }
    vertex_keys = set()
    for rank_keys in uw.mpi.comm.allgather(local_vertex_keys):
        vertex_keys.update(rank_keys)

    def harmonic_coefficients(boundary, response_sign):
        xs, sigma_nn = stokes.boundary_normal_traction(boundary)
        local = {
            tuple(np.round(x, 12)): -float(value)
            for x, value in zip(xs, sigma_nn)
        }
        samples = {}
        for rank_values in uw.mpi.comm.allgather(local):
            samples.update(rank_values)
        coords = np.asarray(list(samples))
        topography = np.asarray(list(samples.values()))
        radii = np.linalg.norm(coords, axis=1)
        harmonic_values = 0.5 * (3.0 * (coords[:, 2] / radii) ** 2 - 1.0)
        is_vertex = np.array([key in vertex_keys for key in samples], dtype=bool)

        def fit(mask):
            return float(
                response_sign
                * np.dot(topography[mask], harmonic_values[mask])
                / np.dot(harmonic_values[mask], harmonic_values[mask])
            )

        return fit(np.ones(len(coords), dtype=bool)), fit(is_vertex), fit(~is_vertex)

    return (
        *harmonic_coefficients("Upper", 1.0),
        *harmonic_coefficients("Lower", -1.0),
    )


def test_p2_triangle_mass_has_zero_vertex_row_sums():
    """WHY mass='auto' is P1-projected on a 3D P2 trace, and not the consistent solve.

    The P2 TRIANGLE mass has vertex rows summing to exactly zero, so it is singular
    on constants along those rows and M^-1 amplifies any perturbation of the nodal
    load at VERTICES without bound -- by O(1), and independently of h (#633: the
    vertex error is flat at ~0.28 over a 3.2x node-count range, while the same
    measurement on a 2-D annulus converges away ~O(h^2)). The 2-D P2 LINE mass has
    positive vertex row sums and needs none of this, which is why 2-D never showed
    the defect. Guarding the identity here keeps the reason attached to the choice.
    """
    from underworld3.utilities.boundary_flux import _P2_TRIANGLE_MASS

    row_sums = _P2_TRIANGLE_MASS.sum(axis=1)
    # Row sum i IS the integral of basis function i: sum_j INT(phi_i phi_j) =
    # INT(phi_i sum_j phi_j) = INT(phi_i), the basis being a partition of unity.
    # So this asserts INT(phi_vertex) = 0 — the P2 triangle vertex basis has ZERO
    # MEAN, which is why its DOF carries no mass and why 'auto' reconstructs the
    # vertices from the midpoints instead of recovering them.
    assert np.allclose(row_sums[:3], 0.0), (
        f"P2 triangle vertex row sums {row_sums[:3]} are no longer zero — the "
        "reason mass='auto' avoids the consistent solve has changed")
    assert np.all(row_sums[3:] > 0.0), "P2 triangle midpoint rows should be positive"


@pytest.mark.level_2
def test_rotated_freeslip_spherical3d_reaction_topography():
    """3D reaction loads must be divided by boundary mass to recover pointwise stress.

    Every node class is asserted, because the failure this guards is confined to ONE
    of them: under the consistent P2 surface mass the VERTEX values were 7.6% low with
    no penalty at all and 28% low at penalty=10, while the midpoints stayed within 1%
    and the facet-integrated value stayed correct (#633). An assertion on the
    aggregate alone passes straight through that.

    Tolerances are the measured discretisation error at this resolution with ~2x
    headroom, not round numbers: at cellSize=0.13 the midpoint-reconstructed recovery
    lands within 0.26% on the surface and 2.4% at the CMB. They were 0.10/0.12 at
    cellSize=0.25, loose enough to pass a recovery that was 7.6% wrong.
    """

    surface, surface_vertices, surface_midpoints, cmb, cmb_vertices, cmb_midpoints = (
        _spherical3d_reaction_topography()
    )
    assert np.isclose(surface, 0.41920, rtol=0.01)
    assert np.isclose(surface_vertices, 0.41920, rtol=0.01)
    assert np.isclose(surface_midpoints, 0.41920, rtol=0.01)
    assert np.isclose(cmb, 0.77060, rtol=0.05)
    assert np.isclose(cmb_vertices, 0.77060, rtol=0.05)
    assert np.isclose(cmb_midpoints, 0.77060, rtol=0.05)


def test_rotated_freeslip_annulus_zero_leakage():
    """Annulus: per-node radial free-slip on both arcs → machine-zero v_r leakage,
    with the analytic radial normal."""
    RI, RO = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.1, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    th = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable("Va", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pa", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.bodyforce = sympy.Matrix([[x / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0,
                                 y / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0]])
    nhat = sympy.Matrix([[x / r, y / r]])
    s.add_rotated_freeslip_bc(0, "Lower", normal=nhat)
    s.add_rotated_freeslip_bc(0, "Upper", normal=nhat)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    vc = v.coords
    rr = np.hypot(vc[:, 0], vc[:, 1])
    rhat = vc / rr[:, None]
    vr = np.einsum("ij,ij->i", v.data, rhat)
    vmax = np.linalg.norm(v.data, axis=1).max() + 1e-30
    for lab, mask in [("inner", np.abs(rr - RI) < 1e-4), ("outer", np.abs(rr - RO) < 1e-4)]:
        leak = np.abs(vr[mask]).max() / vmax
        assert leak < 1e-10, f"{lab} radial leakage {leak:.2e} not machine-zero"
    # rigid-rotation gauge removed
    t = np.column_stack([-vc[:, 1], vc[:, 0]])
    rotfrac = abs(np.sum(v.data * t) / np.sum(t * t)) * np.sqrt(np.sum(t * t)) / (np.linalg.norm(v.data) + 1e-30)
    assert rotfrac < 1e-8, f"rotation gauge {rotfrac:.2e} not removed"


def test_rotated_freeslip_geometric_fmg_velocity_block():
    """The rotated free-slip velocity block is driven by GEOMETRIC FMG on a custom
    prolongation (set_custom_fmg) — no direct solve — and still reproduces the
    essential free-slip solution, with the wall-normal velocity exact."""
    # a small 2-level nested hierarchy (one refinement) keeps the cost down while
    # still driving the velocity block by geometric FMG on the custom prolongation.
    m0 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.2, regular=True, qdegree=3)
    coarse = [_wrap(m0.dm, m0)]
    fine = _wrap(m0.dm.refine(), m0)
    sol = A.SolCx(fine, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)

    vE = _solcx_essential(fine, sol)

    v = uw.discretisation.MeshVariable("vF", fine, fine.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pF", fine, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(fine, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.saddle_preconditioner = 1.0 / sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    custom_mg.set_custom_fmg(s, coarse, builder="barycentric", field_id=0)
    s.solve()

    # geometric MG on the velocity block converged, and matches essential
    info = s._rotated_freeslip_info
    assert info["ksp_reason"] > 0
    rel = np.linalg.norm(v.data - vE.data) / np.linalg.norm(vE.data)
    assert rel < 5e-3, f"FMG rotated free-slip differs from essential by {rel:.2e}"

    # The velocity block really IS multigrid (PETSc's own view of the sub-PC, not our
    # bookkeeping), preconditioned by the 1/mu pressure mass.
    assert info["velocity_pc"] == "custom-FMG"
    assert info["schur_pre"] == "1/mu-mass"
    assert info["velocity_pc_type"] == "mg", (
        f"velocity sub-PC is {info['velocity_pc_type']!r}, not multigrid — the custom "
        f"FMG install did not take")

    # PCFieldSplit applies the Schur complement through the velocity sub-KSP, so that
    # KSP must converge to a tolerance. A `preonly` single multigrid cycle hands the
    # pressure Krylov a different system, which stagnates above its tolerance and
    # leaves the outer Krylov to make up the difference. Measured on this
    # configuration: FGMRES-wrapped FMG = 1 outer iteration, `preonly` = 6.
    assert max(info["ksp_its"]) <= 3, (
        f"rotated FMG outer iteration blow-out: {info['ksp_its']} "
        f"(inexact Schur application — is the velocity sub-KSP `preonly` again?)")


def test_rotated_freeslip_boundary_normal_traction_solcx():
    """sigma_nn recovered by boundary_normal_traction on the top boundary reproduces
    the exact SolCx analytic sigma_yy (mean-removed) to a few percent. Exercises the
    Cartesian-reaction + n_hat projection (corner-correct) + consistent P2 line mass.

    Run at a modest resolution: SolCx breakage is catastrophic (corr collapses toward
    0, relL2 blows up to O(1)), not gradual, so a coarse mesh still catches it — the
    thresholds carry margin over the res-24 values (corr 0.998, relL2 0.056)."""
    res = 24
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)
    v = uw.discretisation.MeshVariable("vT", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pT", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    xs, sig = s.boundary_normal_traction("Top")
    top = np.asarray(xs)
    syy = np.asarray(sol.evaluate_stress(top))[:, 1]
    syy = syy - syy.mean()
    sig = np.asarray(sig)
    corr = np.dot(sig, syy) / (np.linalg.norm(sig) * np.linalg.norm(syy))
    sig = sig if corr >= 0 else -sig                      # sigma = -R sign convention
    relL2 = np.linalg.norm(sig - syy) / np.linalg.norm(syy)
    assert abs(corr) > 0.97, f"sigma_nn shape corr {corr:.3f} too low"
    assert relL2 < 0.10, f"sigma_nn relL2 vs analytic sigma_yy {relL2:.3f} too large"


def test_rotated_freeslip_sigma_nn_lumped_no_overshoot():
    """The default (lumped) sigma_nn de-smear is MONOTONE at the SolCx viscosity jump:
    its total variation matches the analytic (no Gibbs overshoot), whereas the
    consistent-mass de-smear adds spurious variation. This is the property that matters
    for driving a free surface (an overshoot injects a spurious surface-velocity pulse)."""
    res = 24
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)
    v = uw.discretisation.MeshVariable("vL", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pL", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    def total_variation(xs, sig):
        o = np.argsort(np.asarray(xs)[:, 0])
        return float(np.sum(np.abs(np.diff(np.asarray(sig)[o]))))

    xs, sig_l = s.boundary_normal_traction("Top", mass="lumped")     # default
    _, sig_c = s.boundary_normal_traction("Top", mass="consistent")
    syy = np.asarray(sol.evaluate_stress(np.asarray(xs)))[:, 1]
    tv_ref = total_variation(xs, syy - syy.mean())
    tv_l = total_variation(xs, sig_l)
    tv_c = total_variation(xs, sig_c)
    # lumped adds essentially no spurious variation over the analytic; consistent does
    assert tv_l < 1.1 * tv_ref, f"lumped TV {tv_l:.3f} exceeds analytic {tv_ref:.3f}"
    assert tv_l < tv_c, f"lumped TV {tv_l:.3f} not smoother than consistent {tv_c:.3f}"


def _powerlaw_stokes(mesh, prefix, amp=2.0, nexp=3.0, cj=None):
    """A genuinely NONLINEAR Stokes: power-law viscosity eta = eps_II^(1/n - 1)
    (smooth, so Newton/Picard iterates robustly), driven by a horizontally-varying
    vertical body force so there is real shear. ``cj`` sets ``consistent_jacobian``
    (None=default frozen/Picard, True=consistent Newton, "continuation"=staged)."""
    x, y = mesh.X
    v = uw.discretisation.MeshVariable(prefix + "v", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable(prefix + "p", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    g = sympy.Matrix([[v.sym[0].diff(x), v.sym[0].diff(y)],
                      [v.sym[1].diff(x), v.sym[1].diff(y)]])
    e = 0.5 * (g + g.T)
    eII = sympy.sqrt(0.5 * (e[0, 0] ** 2 + e[1, 1] ** 2) + e[0, 1] ** 2 + 1.0e-12)
    s.constitutive_model.Parameters.shear_viscosity_0 = eII ** (1.0 / nexp - 1.0)
    s.bodyforce = sympy.Matrix([[0.0, -amp * sympy.cos(sympy.pi * x)]])
    s.penalty = 0.0
    s.tolerance = 1e-7
    s.petsc_use_pressure_nullspace = True
    if cj is not None:
        s.consistent_jacobian = cj
    return s, v, p


def _powerlaw_essential(mesh, prefix, cj=None):
    s, v, p = _powerlaw_stokes(mesh, prefix, cj=cj)
    s.add_essential_bc((sympy.oo, 0.0), "Top")
    s.add_essential_bc((sympy.oo, 0.0), "Bottom")
    s.add_essential_bc((0.0, sympy.oo), "Left")
    s.add_essential_bc((0.0, sympy.oo), "Right")
    s.solve()
    return v, s.snes.getIterationNumber()


@pytest.fixture(scope="module")
def plaw_box_ref():
    """A small power-law box + its native ESSENTIAL nonlinear free-slip solution,
    solved ONCE (Newton tangent) and shared across the rotated-vs-essential box
    tests. The converged solution is tangent-independent, so this one reference
    serves the Picard / Newton / continuation tests. Returns (mesh, vE_data,
    essential_newton_iters)."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    vE, its = _powerlaw_essential(mesh, "refE", cj=True)
    return mesh, np.copy(vE.data), its


def test_rotated_freeslip_nonlinear_matches_essential(plaw_box_ref):
    """Default (frozen/Picard) tangent through the rotated path: genuinely iterates
    and converges to the native essential nonlinear free-slip answer (both impose
    v_n=0 — identical discrete problem), with machine-zero wall-normal flow on every
    wall. Exercises the rotated constraint INSIDE the nonlinear iteration."""
    mesh, vE, _ = plaw_box_ref
    s, vR, pR = _powerlaw_stokes(mesh, "nlP")           # default (Picard) tangent
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.solve()
    info = s._rotated_freeslip_info
    assert info["nonlinear_iterations"] > 1, "rotated solve did not genuinely iterate"
    # the reported count is the number of Newton increments solved — one linear
    # solve per increment (guards the loop-index off-by-one in the result dict)
    assert info["nonlinear_iterations"] == len(info["ksp_its"])
    assert np.linalg.norm(vR.data - vE) / np.linalg.norm(vE) < 1e-3
    vc = vR.coords
    for lab, msk, comp in [("Top", np.abs(vc[:, 1] - 1) < 1e-6, 1),
                           ("Bottom", np.abs(vc[:, 1]) < 1e-6, 1),
                           ("Left", np.abs(vc[:, 0]) < 1e-6, 0),
                           ("Right", np.abs(vc[:, 0] - 1) < 1e-6, 0)]:
        leak = np.abs(vR.data[msk, comp]).max() if msk.any() else 0.0
        assert leak < 1e-10, f"{lab} wall-normal velocity {leak:.2e} not machine-zero"


def test_rotated_freeslip_newton_tangent(plaw_box_ref):
    """consistent_jacobian=True is genuine Newton through the rotated path: it
    converges in about the same small iteration count as the native essential Newton
    solve (NOT the ~4-5x larger Picard count) and to the same answer at (near) machine
    precision — the rotated constraint does not degrade the consistent tangent."""
    mesh, vE, ess_its = plaw_box_ref
    s, vR, pR = _powerlaw_stokes(mesh, "ntR", cj=True)
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.solve()
    its = s._rotated_freeslip_info["nonlinear_iterations"]
    assert its <= 2 * ess_its + 2, (
        f"rotated Newton took {its} iters vs essential Newton {ess_its} "
        f"— tangent likely not the consistent one")
    assert its < 20, f"rotated Newton not converging at Newton rate ({its} iters)"
    assert np.linalg.norm(vR.data - vE) / np.linalg.norm(vE) < 1e-6


def test_rotated_freeslip_continuation_tangent(plaw_box_ref):
    """consistent_jacobian='continuation' works through the rotated path: a staged
    Picard→Newton solve (α=0 to the loose newton_switch_rtol, then α=1) that switches
    tangents and converges to the essential Newton answer. picard=N extends the α=0
    phase (so the wrapper's picard forwarding and the staging are both exercised)."""
    mesh, vE, _ = plaw_box_ref
    s, vR, pR = _powerlaw_stokes(mesh, "ctR", cj="continuation")
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.solve()
    info = s._rotated_freeslip_info
    assert info["continuation_switched"], "continuation never switched Picard→Newton"
    assert 1 < info["nonlinear_iterations"] < 36, "continuation not staging as expected"
    assert np.linalg.norm(vR.data - vE) / np.linalg.norm(vE) < 1e-6

    # picard=N holds the α=0 (Picard) phase for >= N iterations before switching
    s2, _, _ = _powerlaw_stokes(mesh, "ctR2", cj="continuation")
    for wall in ("Top", "Bottom", "Left", "Right"):
        s2.add_rotated_freeslip_bc(0, wall)
    s2.solve(picard=25)
    assert s2._rotated_freeslip_info["nonlinear_iterations"] >= 25, (
        "picard did not extend the continuation Picard phase")


def test_rotated_freeslip_picard_newton_unsupported_raises(plaw_box_ref):
    """picard>0 with the pure consistent-Newton tangent has no frozen warmup tangent
    to form, so the rotated path raises a clear NotImplementedError pointing to
    'continuation' rather than silently ignoring it. Also confirms the SNES_Stokes
    wrapper forwards picard to the plain-Stokes solve at all."""
    mesh, _, _ = plaw_box_ref
    s, v, p = _powerlaw_stokes(mesh, "prN", cj=True)
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    with pytest.raises(NotImplementedError, match="continuation"):
        s.solve(picard=3)


def test_rotated_freeslip_nonlinear_warm_start(plaw_box_ref):
    """Warm-start (zero_init_guess=False) through the nonlinear rotated path: a 2-step
    'time loop' re-solving from the previous converged state stays correct and
    converges in few iterations (the step-norm exit avoids chasing machine noise)."""
    mesh, vE, _ = plaw_box_ref
    s, vR, pR = _powerlaw_stokes(mesh, "wsR", cj=True)     # Newton (fast)
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.solve()                              # cold
    s.solve(zero_init_guess=False)         # warm (step 2)
    assert np.linalg.norm(vR.data - vE) / np.linalg.norm(vE) < 1e-6
    # warm-start from an already-converged state converges in few iterations
    assert s._rotated_freeslip_info["nonlinear_iterations"] <= 3


def test_rotated_freeslip_nonlinear_geometric_fmg():
    """The NONLINEAR rotated free-slip velocity block is driven by geometric FMG on
    the custom prolongation (set_custom_fmg) — the rotated prolongation is built once
    and reused across Newton iterations — and still converges to the essential
    nonlinear free-slip solution."""
    # a small 2-level nested hierarchy (one refinement) + the Newton tangent keep the
    # cost down while still exercising the rotated custom-FMG prolongation reuse.
    m0 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.4, regular=True, qdegree=3)
    coarse = [_wrap(m0.dm, m0)]
    fine = _wrap(m0.dm.refine(), m0)
    vE, _ = _powerlaw_essential(fine, "nlfE", cj=True)

    s, vR, pR = _powerlaw_stokes(fine, "nlfR", cj=True)
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    custom_mg.set_custom_fmg(s, coarse, builder="barycentric", field_id=0)
    s.solve()

    info = s._rotated_freeslip_info
    assert info["nonlinear_iterations"] > 1
    assert info["ksp_reason"] > 0                # geometric MG increment converged
    rel = np.linalg.norm(vR.data - vE.data) / np.linalg.norm(vE.data)
    assert rel < 5e-3, f"nonlinear FMG rotated free-slip differs from essential by {rel:.2e}"


def test_rotated_freeslip_nonlinear_annulus_zero_leakage():
    """Genuinely-rotated frame under nonlinear iteration: a power-law annulus with
    per-node radial free-slip on both arcs genuinely iterates, gives machine-zero
    radial leakage, and the rigid-rotation gauge is removed."""
    RI, RO = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.3, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    th = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable("Vnla", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pnla", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    g = sympy.Matrix([[v.sym[0].diff(x), v.sym[0].diff(y)],
                      [v.sym[1].diff(x), v.sym[1].diff(y)]])
    e = 0.5 * (g + g.T)
    eII = sympy.sqrt(0.5 * (e[0, 0] ** 2 + e[1, 1] ** 2) + e[0, 1] ** 2 + 1.0e-12)
    s.constitutive_model.Parameters.shear_viscosity_0 = eII ** (1.0 / 3.0 - 1.0)
    s.bodyforce = sympy.Matrix([[x / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0,
                                 y / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0]])
    nhat = sympy.Matrix([[x / r, y / r]])
    s.consistent_jacobian = True                 # Newton tangent (few iterations)
    s.add_rotated_freeslip_bc(0, "Lower", normal=nhat)
    s.add_rotated_freeslip_bc(0, "Upper", normal=nhat)
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1e-7
    s.solve()

    info = s._rotated_freeslip_info
    assert info["nonlinear_iterations"] > 1, "annulus rotated solve did not genuinely iterate"
    assert info["rotation_gauge_removed"]
    vc = v.coords
    rr = np.hypot(vc[:, 0], vc[:, 1])
    rhat = vc / rr[:, None]
    vr = np.einsum("ij,ij->i", v.data, rhat)
    vmax = np.linalg.norm(v.data, axis=1).max() + 1e-30
    for lab, mask in [("inner", np.abs(rr - RI) < 1e-4), ("outer", np.abs(rr - RO) < 1e-4)]:
        leak = np.abs(vr[mask]).max() / vmax
        assert leak < 1e-10, f"{lab} radial leakage {leak:.2e} not machine-zero"


def test_rotated_freeslip_dynamic_topography_field():
    """dynamic_topography writes h = -(sigma_nn - mean)/(rho g) onto a scalar surface
    MeshVariable (the free-surface hand-off): the field at the top vertices reproduces
    the analytic SolCx topography, and the field is usable symbolically (BdIntegral)."""
    res = 24
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)
    v = uw.discretisation.MeshVariable("vD", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("pD", mesh, 1, degree=1, continuous=False)
    hf = uw.discretisation.MeshVariable("hD", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.solve()

    ret = s.dynamic_topography("Top", hf, buoyancy_scale=1.0)
    assert ret is hf
    # field at the top vertices reproduces the analytic topography (mean-removed)
    hc = hf.coords
    top = np.abs(hc[:, 1] - 1.0) < 1e-6
    tt = np.asarray(sol.topography_top(hc[top])); tt = tt - tt.mean()
    hv = hf.data[top, 0]
    corr = np.dot(hv, tt) / (np.linalg.norm(hv) * np.linalg.norm(tt))
    hv = hv if corr >= 0 else -hv
    relL2 = np.linalg.norm(hv - tt) / np.linalg.norm(tt)
    assert abs(corr) > 0.97, f"topography field corr {corr:.3f} too low"
    assert relL2 < 0.12, f"topography field relL2 {relL2:.3f} too large"
    # symbolically usable (the free-surface integrator reads it via BdIntegral)
    bdl2 = float(np.sqrt(uw.maths.BdIntegral(
        mesh=mesh, fn=hf.sym[0] ** 2, boundary="Top").evaluate()))
    assert bdl2 > 0.0


# --- Prescribed non-zero wall-normal datum (u.n = ũ_n) ---------------------------
# The rotated constraint imposes u.n = datum strongly, via the value-first conds
# argument: conds=0 is pure free-slip (the held lid); a non-zero conds is the
# "consistent" material-surface velocity. Both share the same rotated matrix and
# differ only in the constraint RHS, so an explicit zero datum must reproduce plain
# free-slip exactly.

def _annulus_datum_solve(mode, tag):
    RI, RO = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.1, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    nhat = sympy.Matrix([[x / r, y / r]])
    v = uw.discretisation.MeshVariable("Vd" + tag, mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pd" + tag, mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    blob = sympy.exp(-(((x - 0.75) ** 2 + y ** 2) / 0.05))
    s.bodyforce = sympy.Matrix([[50.0 * blob * x / r, 50.0 * blob * y / r]])
    s.add_essential_bc((0.0, 0.0), "Lower")          # no-slip inner (pins rotation gauge)
    datum = {"plain": 0, "zero": 0.0, "cos": x / r}[mode]   # cos: mean-zero => ∮u.n=0
    s.add_rotated_freeslip_bc(datum, "Upper", normal=nhat)
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    s.tolerance = 1e-9
    s.solve()
    return mesh, v


def test_rotated_freeslip_datum_zero_matches_freeslip():
    """datum={'Upper': 0} (the prescribed-datum path with a zero datum) reproduces plain
    rotated free-slip to iterative round-off — the held / free-slip case is untouched.
    (A zero datum takes the same free-slip RHS path; the residual is the iterative
    solver's run-to-run round-off, ~1e-15, not a code-path difference.)"""
    _, v0 = _annulus_datum_solve("plain", "0")
    _, vz = _annulus_datum_solve("zero", "z")
    d = np.abs(vz.data - v0.data).max()
    assert d < 1e-10, f"zero datum perturbs the free-slip solution by {d:.2e} (> round-off)"


def test_rotated_freeslip_prescribed_normal_datum():
    """A prescribed non-zero wall-normal datum u.n = cos(theta) is imposed to the SAME
    (machine) precision as u.n = 0 at the constrained surface nodes."""
    RO = 1.0
    mesh, v = _annulus_datum_solve("cos", "c")
    vc = v.coords
    rr = np.hypot(vc[:, 0], vc[:, 1])
    outer = np.abs(rr - RO) < 2.0e-2                    # outer velocity nodes (incl. edge mids)
    rhat = vc[outer] / rr[outer, None]
    vn = np.einsum("ij,ij->i", v.data[outer], rhat)
    target = vc[outer, 0] / rr[outer]                   # cos(theta) at the same node coords
    err = np.abs(vn - target).max()
    assert err < 1e-8, f"prescribed u.n=cos(theta) not imposed: max nodal error {err:.2e}"
    assert vn.max() > 0.9 and vn.min() < -0.9, "prescribed normal velocity magnitude wrong"


def test_rotated_freeslip_nonlinear_prescribed_normal_datum():
    """The prescribed wall-normal datum u.n = cos(theta) through the NONLINEAR rotated
    path (power-law annulus): the loop genuinely iterates and every iterate is kept
    feasible, so the converged solution carries the datum to the same (machine)
    precision as the linear path — the constraint is exact independent of where the
    nonlinear iteration stops."""
    RI, RO = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.2, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    nhat = sympy.Matrix([[x / r, y / r]])
    v = uw.discretisation.MeshVariable("Vdn", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable("Pdn", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    g = sympy.Matrix([[v.sym[0].diff(x), v.sym[0].diff(y)],
                      [v.sym[1].diff(x), v.sym[1].diff(y)]])
    e = 0.5 * (g + g.T)
    eII = sympy.sqrt(0.5 * (e[0, 0] ** 2 + e[1, 1] ** 2) + e[0, 1] ** 2 + 1.0e-12)
    s.constitutive_model.Parameters.shear_viscosity_0 = eII ** (1.0 / 3.0 - 1.0)
    blob = sympy.exp(-(((x - 0.75) ** 2 + y ** 2) / 0.05))
    s.bodyforce = sympy.Matrix([[50.0 * blob * x / r, 50.0 * blob * y / r]])
    s.add_essential_bc((0.0, 0.0), "Lower")          # no-slip inner (pins rotation gauge)
    s.add_rotated_freeslip_bc(x / r, "Upper", normal=nhat)   # u.n = cos(theta), mean-zero flux
    s.consistent_jacobian = True                     # Newton tangent (few iterations)
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1e-7
    s.solve()

    info = s._rotated_freeslip_info
    assert info["nonlinear_iterations"] > 1, "nonlinear datum solve did not genuinely iterate"
    assert info["converged"], "nonlinear datum solve did not converge"
    vc = v.coords
    rr = np.hypot(vc[:, 0], vc[:, 1])
    outer = np.abs(rr - RO) < 4.0e-2                 # outer velocity nodes (incl. edge mids)
    rhat = vc[outer] / rr[outer, None]
    vn = np.einsum("ij,ij->i", v.data[outer], rhat)
    target = vc[outer, 0] / rr[outer]                # cos(theta) at the same node coords
    err = np.abs(vn - target).max()
    assert err < 1e-8, f"nonlinear u.n=cos(theta) not imposed: max nodal error {err:.2e}"
    assert vn.max() > 0.9 and vn.min() < -0.9, "prescribed normal velocity magnitude wrong"


def test_rotated_solve_fields_carry_inhomogeneous_dirichlet_walls():
    """The copy-back gap: essential DOFs are absent from the global vector, so
    the rotated path's field scatter left them at ZERO wherever the Dirichlet
    datum g != 0 — the solve was right, every field-based diagnostic
    (projection, integral, evaluate) read a garbage boundary strip. Caught by
    the split-fault work (far-field stress off by 20%); fixed by the
    DMPlexInsertBoundaryValues shim in the copy-back. Homogeneous walls hid
    this from every earlier rotated test — zero happens to be their datum.
    """
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("vIB", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("pIB", mesh, 1, degree=1,
                                       continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.tolerance = 1e-8
    s.petsc_use_pressure_nullspace = True
    # Inhomogeneous Dirichlet lid and floor, rotated free-slip sides: the
    # combination that exposes the gap.
    s.add_dirichlet_bc((y - 0.5, 0.0), "Top")
    s.add_dirichlet_bc((y - 0.5, 0.0), "Bottom")
    s.add_rotated_freeslip_bc(0, "Left")
    s.add_rotated_freeslip_bc(0, "Right")
    s.solve()

    vc = np.asarray(v.coords)
    vd = np.asarray(v.data)
    for name, mask, target in (
            ("Top", vc[:, 1] > 1 - 1e-9, +0.5),
            ("Bottom", vc[:, 1] < 1e-9, -0.5)):
        assert mask.sum() > 0
        err = np.abs(vd[mask, 0] - target).max()
        assert err < 1e-10, (
            f"{name} wall u_x in the FIELD is off by {err:.2e}; the rotated "
            "copy-back dropped the inhomogeneous essential values")
        assert np.abs(vd[mask, 1]).max() < 1e-10


@pytest.mark.parametrize("use_lu", [False, True])
def test_the_rotated_solve_reports_its_own_convergence(use_lu):
    """The converged reason must describe the ROTATED solve, not a linear step.

    Two defects made this false, and both reported "unconverged" for a solve
    that had converged:

    - the direct path pins one pressure DOF out of the linear system
      (`_naive_pressure_pin`) but the residual kept counting that row, so the
      loop could never meet its tolerance and ran to `max_it` against a floor
      it had defined as unreachable. Measured on a fault-contact problem: the
      velocity residual reached 6e-12 at the first increment and the whole of
      the remaining |F| was the pinned DOF, bit-identical for eight further
      no-op iterations;
    - the loop never published a reason on the SNES, so
      `snes.getConvergedReason()` returned 0 even on a clean exit — and the
      generic solver's own convergence check reads exactly that.

    Both paths are covered because the pin exists only in the direct one.
    """
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(12, 12), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)

    v = uw.discretisation.MeshVariable(
        f"v_reason_{int(use_lu)}", mesh, mesh.dim, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable(
        f"p_reason_{int(use_lu)}", mesh, 1, degree=1, continuous=False)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.penalty = 0.0
    s.tolerance = 1e-9
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s._rotated_use_lu = use_lu
    s.solve()

    reason = int(s.snes.getConvergedReason())
    assert reason > 0, (
        f"the rotated solve reports reason {reason}; a converged solve must "
        "say so, because the generic path decides convergence from this")

    # The answer has to be right as well as reported right — a reason set
    # unconditionally would pass the assertion above and mean nothing.
    leak = np.abs(np.asarray(v.data)[:, 1]).max()
    assert np.isfinite(leak)
