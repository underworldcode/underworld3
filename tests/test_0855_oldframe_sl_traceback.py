"""Regression tests for the old-frame semi-Lagrangian reach-back.

On a moving mesh the standard ALE trace-back samples the CARRY'd SL
history on the NEW geometry and subtracts ``v_mesh = Δx/dt`` to
compensate the node motion. That fold is lossy at a disequilibrium free
surface and blows up high-Ra free-surface convection. The cure
(``DuDt.old_frame_traceback = True``) computes the departure foot from
the physical velocity and samples ``psi_star`` on the mesh EPHEMERALLY
restored to the previous-step (old) geometry — where the foot is always
representable.

These lock the *mechanism* (Stage 0 of the lagged-clone design):

* no-op when the mesh does not move (bit-identical to the standard path);
* the working mesh geometry is restored after each step (the old-geometry
  sampling is genuinely ephemeral) and the one-step stash is consumed;
* the history is sampled on the OLD geometry, not the new one;
* the field stays bounded on a moving mesh (with the monotone limiter);
* BDF2 (order-2, two history levels) runs and stays bounded.

The full high-Ra free-surface convection cure (T∈[0,1] at Ra=1e5) is
validated end-to-end by a heavier driver script
(``~/+Simulations/fs_convection_goal4/oldframe_release_validation.py``);
it is too slow for the unit suite.

See ``docs/developer/design/lagged-clone-sl-history.md``.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

_OMEGA = 0.6  # solid-body rotation rate for the (divergence-free) velocity


def _build(old_frame, order=1, res=8, kappa=1.0e-2, monotone_mode="clamp",
           flux_theta=0.5):
    """A small annulus carrying a rotation velocity and an advected scalar.

    No Stokes solve — the velocity is a prescribed solid-body rotation
    (divergence-free, tangential to the annulus boundaries) so the tests
    are fast and isolate the SL trace-back.
    """
    mesh = uw.meshing.Annulus(
        radiusInner=0.5, radiusOuter=1.0, cellSize=1.0 / res, qdegree=3)

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)

    # Solid-body rotation v = Ω (−y, x): divergence-free, v·n̂ = 0 on the
    # circular arcs, so it neither compresses T nor pushes it across the
    # boundary — T stays bounded under exact advection.
    vc = np.asarray(v.coords)
    v.data[:, 0] = -_OMEGA * vc[:, 1]
    v.data[:, 1] = _OMEGA * vc[:, 0]

    # Smooth initial scalar bump, strictly inside [0, 1].
    tc = np.asarray(T.coords)
    r2 = (tc[:, 0] - 0.72) ** 2 + tc[:, 1] ** 2
    T.data[:, 0] = np.exp(-r2 / 0.05)

    # Always build the DuDt explicitly so the ONLY difference between the
    # old_frame on/off cases is the flag (the default solver path hardcodes
    # order 1 and uses different projection bcs, which would confound the
    # bit-identical comparison). An explicit DuDt also lets BDF2 genuinely
    # allocate two history levels.
    #
    # The advective DuDt carries the BDF order; the diffusive DFDt is kept
    # as the order-1 Adams-Moulton θ-method flux (so the two scheme knobs
    # are paired explicitly, not coupled). SLCN = order 1 + flux_theta 0.5;
    # SL-BDF2 = order 2 + flux_theta 1.0 (flux implicit at n+1 — a BDF2
    # stencil with a CN flux is inconsistent).
    duDt = uw.systems.ddt.SemiLagrangian(
        mesh, T.sym, v.sym,
        vtype=uw.VarType.SCALAR, degree=T.degree,
        continuous=T.continuous, varsymbol=T.symbol,
        bcs=[], order=order, smoothing=0.0,
        monotone_mode=monotone_mode, theta=0.5,
        old_frame_traceback=old_frame,
    )
    adv_diff = uw.systems.AdvDiffusionSLCN(
        mesh, u_Field=T, V_fn=v.sym, DuDt=duDt, order=1)
    adv_diff.DFDt.theta = flux_theta

    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = kappa
    return mesh, T, v, adv_diff


def _wobble(X0, amp, phase=0.0):
    """A smooth mode-3 radial perturbation of the annulus INTERIOR.

    Computed from a FIXED reference geometry ``X0`` (not the current,
    already-moved coordinates) so repeated calls do not compound into a
    tangled mesh. ``phase`` rotates the mode-3 pattern so successive
    steps genuinely move the nodes. Tapered to zero at BOTH boundaries
    (peak at mid-annulus), so the arcs stay put and the tangential
    rotation velocity keeps v·n̂ = 0 — no throughflow. Moves only
    interior nodes: also the interior-node (mmpde/OT) adaptation regime
    of Task 2, and the cleanest setting to isolate the SL trace-back
    from boundary BC effects.
    """
    c = np.asarray(X0)
    r = np.sqrt((c ** 2).sum(1))
    th = np.arctan2(c[:, 1], c[:, 0])
    s = np.clip((r - 0.5) / 0.5, 0.0, 1.0)  # 0 at inner, 1 at outer
    bump = np.sin(np.pi * s)                # 0 at both boundaries
    rad = r + amp * bump * np.cos(3.0 * th - phase)
    out = c.copy()
    nz = r > 1.0e-12
    out[nz, 0] = rad[nz] * np.cos(th[nz])
    out[nz, 1] = rad[nz] * np.sin(th[nz])
    return out


def _gmin_gmax(field_col):
    """Mesh-global (parallel-safe) min/max of a 1-D nodal column."""
    comm = uw.mpi.comm
    field_col = np.asarray(field_col)
    lo = field_col.min() if field_col.size else np.inf
    hi = field_col.max() if field_col.size else -np.inf
    return (comm.allreduce(lo, op=uw.mpi._MPI.MIN),
            comm.allreduce(hi, op=uw.mpi._MPI.MAX))


def test_oldframe_noop_on_static_mesh_is_bit_identical():
    """With no mesh motion, old_frame never fires (no on_remesh) so the
    trajectory is bit-identical to the standard trace-back."""
    dt = 0.01

    _, T_off, _, ad_off = _build(old_frame=False)
    for _ in range(4):
        ad_off.solve(timestep=dt)
    ref = T_off.data.copy()

    _, T_on, _, ad_on = _build(old_frame=True)
    for _ in range(4):
        ad_on.solve(timestep=dt)
    got = T_on.data.copy()

    assert np.array_equal(ref, got), (
        "old_frame must be a no-op on a non-moving mesh "
        f"(max|Δ|={np.abs(ref - got).max():.3e})")


def test_oldframe_restores_geometry_and_consumes_stash():
    """The old-geometry sampling is ephemeral: after a deform+solve the
    mesh sits at the committed (new) geometry and the one-step stash is
    cleared."""
    dt = 0.01
    mesh, T, v, adv_diff = _build(old_frame=True)

    target = _wobble(np.asarray(mesh.X.coords), amp=0.03)
    mesh.deform(target, dt=dt)
    assert adv_diff.DuDt._oldframe_X is not None, (
        "on_remesh should stash the old geometry for old_frame")

    adv_diff.solve(timestep=dt)

    after = np.asarray(mesh.X.coords)
    assert np.allclose(after, target, atol=1.0e-12), (
        "ephemeral old-geometry sampling must restore the committed "
        f"geometry (max|Δ|={np.abs(after - target).max():.3e})")
    assert adv_diff.DuDt._oldframe_X is None, (
        "the one-step old-geometry stash must be consumed by update_pre_solve")


def test_oldframe_samples_on_old_geometry_not_new():
    """The history is sampled on the previous-step geometry.

    Drive ``update_pre_solve`` directly with ``store_result=False`` (so
    the history is advected, not re-recorded — isolating the trace-back
    sample) at ``dt=0`` (foot == node coords). The resulting ``psi_star``
    must match an independent OLD-geometry sample of the (carried)
    history at the foot, and must differ from the NEW-geometry sample —
    proving the ephemeral restore samples the previous-step geometry.
    """
    mesh, T, v, adv_diff = _build(old_frame=True, kappa=0.0)
    duDt = adv_diff.DuDt
    duDt.initialise_history()  # psi_star[0] := T on the initial (old) geometry

    old_X = np.asarray(mesh.X.coords).copy()  # geometry the history belongs to
    new_X = _wobble(old_X, amp=0.05)
    mesh.deform(new_X, dt=1.0)  # on_remesh stashes old_X; psi_star CARRY'd
    assert duDt._oldframe_X is not None

    psi0 = duDt.psi_star[0]
    foot = np.asarray(psi0.coords).copy()  # dt=0 → departure foot == nodes

    # Independent references: sample the (unmodified) history at `foot` on
    # each geometry, BEFORE update_pre_solve overwrites psi_star[0].
    with mesh.ephemeral_coords():
        mesh._deform_mesh(old_X)
        ref_old = np.asarray(
            uw.function.global_evaluate(psi0.sym[0], foot, monotone="clamp")
        ).flatten()
    ref_new = np.asarray(
        uw.function.global_evaluate(psi0.sym[0], foot, monotone="clamp")
    ).flatten()

    duDt.update_pre_solve(0.0, store_result=False)
    got = np.asarray(psi0.data[:, 0]).copy()

    assert np.allclose(got, ref_old, atol=1.0e-9), (
        "old_frame psi_star must match the OLD-geometry sample "
        f"(max|Δ|={np.abs(got - ref_old).max():.3e})")
    # The two geometries genuinely differ where the interior moved, so
    # the old/new samples must not be identical (else the test proves
    # nothing). Check GLOBALLY — under MPI the moved region may live on a
    # single rank, so a per-rank check would spuriously coincide.
    local_diff = float(np.abs(ref_old - ref_new).max()) if ref_old.size else 0.0
    global_diff = uw.mpi.comm.allreduce(local_diff, op=uw.mpi._MPI.MAX)
    assert global_diff > 1.0e-6, (
        "test setup error: old and new geometry samples coincide globally")


def test_oldframe_keeps_field_bounded_on_moving_mesh():
    """Advect-and-deform for several steps; with the monotone limiter the
    old-frame trace-back keeps the scalar inside its physical range."""
    dt = 0.01
    mesh, T, v, adv_diff = _build(old_frame=True, monotone_mode="clamp")

    X0 = np.asarray(mesh.X.coords).copy()
    lo0, hi0 = _gmin_gmax(T.data[:, 0])
    for s in range(6):
        mesh.deform(_wobble(X0, amp=0.03, phase=0.6 * s), dt=dt)
        adv_diff.solve(timestep=dt)
        lo, hi = _gmin_gmax(T.data[:, 0])
        # The decisive property is boundedness: the standard ALE fold
        # diverges to O(100) on the violent high-Ra free surface, while
        # old-frame stays in range. Allow modest SLCN/CN ringing.
        assert lo > -1.0e-2, f"step {s}: undershoot {lo:.3e} (divergence?)"
        assert hi < hi0 + 1.0e-2, f"step {s}: overshoot {hi:.3e} (init {hi0:.3e})"


def test_oldframe_sl_bdf2_runs_and_stays_bounded():
    """Proper SL-BDF2 (BDF2 time-difference + flux implicit at n+1, θ=1):
    order-2 allocates two history levels; the 'one clone, any order'
    reach-back keeps the field bounded."""
    dt = 0.01
    mesh, T, v, adv_diff = _build(old_frame=True, order=2, monotone_mode="clamp",
                                  flux_theta=1.0)  # SL-BDF2 flux centring
    assert len(adv_diff.DuDt.psi_star) == 2, "BDF2 must allocate two history levels"
    assert adv_diff.DFDt.theta == 1.0, "SL-BDF2 flux must be implicit at n+1 (θ=1)"

    X0 = np.asarray(mesh.X.coords).copy()
    _, hi0 = _gmin_gmax(T.data[:, 0])
    for s in range(6):
        mesh.deform(_wobble(X0, amp=0.03, phase=0.6 * s), dt=dt)
        adv_diff.solve(timestep=dt)
        lo, hi = _gmin_gmax(T.data[:, 0])
        assert lo > -1.0e-2, f"BDF2 step {s}: undershoot {lo:.3e} (divergence?)"
        assert hi < hi0 + 1.0e-2, f"BDF2 step {s}: overshoot {hi:.3e}"
