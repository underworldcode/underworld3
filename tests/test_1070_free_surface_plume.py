"""FreeSurface manager on the Crameri Case-2 plume (uw.systems.FreeSurface).

A buoyant plume under a stiff lid raises a true free surface. This exercises the
whole manager — the derived held (rotated free-slip, sigma_nn -> h_inf) and
consistent solves, the exponential surface step, the interior carry + deform, and
the level-set composition transport with enclosed-area conservation — and checks
the physical signature at coarse resolution: the surface rises, the relaxation is
bounded (no overshoot), and the enclosed buoyancy is conserved.

The values are validated against the reference driver
(``~/+Simulations/FreeSurface/crameri_study/scripts/crameri_box.py``): at res 48 the
per-step surface amplitude agrees to < 0.2 %.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _case2(res):
    """Coarse Crameri Case-2: box, plume level set, smooth lid/plume viscosity, and a
    stress-free-top Stokes with rotated free-slip walls. Returns (fs, buoyancy)."""
    L, H = 1.0, 0.25
    lid_frac, lid_eta, plume_eta = 0.143, 100.0, 0.1
    plume_r, px0, py0, drho, rho_g = 0.0179, 0.5, 0.107, 30.3, 1000.0

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(L, H), cellSize=L / res, qdegree=3
    )
    x, y = mesh.X
    yhat = sympy.Matrix([[0, 1]])

    phi = uw.discretisation.MeshVariable(
        "phi", mesh, vtype=uw.VarType.SCALAR, degree=3, continuous=True
    )
    phi.array[:, 0, 0] = np.asarray(
        uw.function.evaluate(
            sympy.sqrt((x - px0) ** 2 + (y - py0) ** 2) - plume_r, phi.coords
        )
    ).flatten()

    eps = 0.7 * (L / res)
    buoyancy = 0.5 * (1.0 - sympy.tanh(phi.sym[0] / eps))
    w = 0.5 * (L / res)
    s_lid = 0.5 * (1.0 + sympy.tanh((y - H * (1.0 - lid_frac)) / w))
    s_plume = 0.5 * (1.0 + sympy.tanh((sympy.Min(sympy.Max(buoyancy, 0.0), 1.0) - 0.5) / 0.10))
    eta = sympy.exp(s_lid * sympy.log(lid_eta) + s_plume * sympy.log(plume_eta))

    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
    stokes.bodyforce = (drho * buoyancy - rho_g) * yhat
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_rotated_freeslip_bc(0.0, "Left")
    stokes.add_rotated_freeslip_bc(0.0, "Right")
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.tolerance = 1.0e-5

    fs = uw.systems.FreeSurface(
        stokes, "Top", buoyancy_scale=rho_g, composition=phi, conserve=buoyancy,
        driving_buoyancy=drho * buoyancy * yhat,
    )
    return fs, buoyancy


def _surface_amplitude(fs):
    top = fs.mesh.X.coords[fs._surf_rows, -1]
    return float(np.abs(top - top.mean()).max()) if top.size else 0.0


def test_free_surface_plume_rises_and_conserves():
    """The surface rises under the plume, the relaxation does not overshoot, and the
    enclosed buoyancy is conserved."""
    fs, buoyancy = _case2(res=40)
    enclosed = uw.maths.Integral(fs.mesh, buoyancy)
    area0 = float(enclosed.evaluate())

    amplitude, increment = [], []
    prev = _surface_amplitude(fs)
    for _ in range(2):
        fs.solve()
        fs.advance(1.0)
        now = _surface_amplitude(fs)
        amplitude.append(now)
        increment.append(now - prev)
        prev = now

    assert amplitude[0] > 0.0, "surface did not rise under the plume"
    assert amplitude[1] > amplitude[0], "surface amplitude not growing (early Case-2)"
    # Bounded relaxation: the per-step rise stays finite and does not run away.
    assert 0.0 < increment[1] < increment[0] * 5.0, "surface step is not bounded"

    area = float(enclosed.evaluate())
    assert abs(area - area0) / area0 < 0.02, "enclosed buoyancy not conserved"


def _powerlaw(v, x):
    """Power-law viscosity eta = eps_II^(1/3 - 1), written from v's own velocity."""
    g = sympy.Matrix([[v.sym[0].diff(x[0]), v.sym[0].diff(x[1])],
                      [v.sym[1].diff(x[0]), v.sym[1].diff(x[1])]])
    e = 0.5 * (g + g.T)
    eII = sympy.sqrt(0.5 * (e[0, 0] ** 2 + e[1, 1] ** 2) + e[0, 1] ** 2 + 1.0e-12)
    return eII ** (1.0 / 3.0 - 1.0)


def test_free_surface_nonlinear_viscosity_self_consistent():
    """With a strain-rate-dependent (power-law) viscosity, the derived held solve must
    use its OWN velocity, not the free solution. Guards the constitutive-model copy:
    the viscosity is written from the free solve's velocity-gradient symbols, which
    must be rebound onto each derived solve. The FreeSurface held solve must match an
    independently built held solve whose viscosity uses its own velocity."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 0.5), cellSize=1.0 / 20, qdegree=3
    )
    x = mesh.X
    driving = 50.0 * sympy.exp(-(((x[0] - 0.5) ** 2 + (x[1] - 0.25) ** 2) / 0.02)) \
        * sympy.Matrix([[0, 1]])

    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = _powerlaw(stokes.u, x)
    stokes.bodyforce = driving
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_rotated_freeslip_bc(0.0, "Left")
    stokes.add_rotated_freeslip_bc(0.0, "Right")
    stokes.consistent_jacobian = True
    stokes.tolerance = 1.0e-8

    fs = uw.systems.FreeSurface(stokes, "Top", buoyancy_scale=50.0)
    fs.solve()

    ref = uw.systems.Stokes(mesh)
    ref.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ref.constitutive_model.Parameters.shear_viscosity_0 = _powerlaw(ref.u, x)
    ref.bodyforce = driving
    ref.add_essential_bc((0.0, 0.0), "Bottom")
    ref.add_rotated_freeslip_bc(0.0, "Left")
    ref.add_rotated_freeslip_bc(0.0, "Right")
    ref.add_rotated_freeslip_bc(0.0, "Top")
    ref.petsc_use_pressure_nullspace = True
    ref.consistent_jacobian = True
    ref.tolerance = 1.0e-8
    ref.solve(zero_init_guess=True)

    a = np.asarray(fs.held.u.array).reshape(-1)
    b = np.asarray(ref.u.array).reshape(-1)
    assert np.linalg.norm(b) > 1.0e-3, "reference held solve produced no flow"
    rel = np.linalg.norm(a - b) / np.linalg.norm(b)
    assert rel < 1.0e-6, f"held viscosity not rebound to own velocity (rel diff {rel:.2e})"


@pytest.mark.mpi(min_size=2)
def test_free_surface_plume_parallel():
    """The manager runs on a distributed mesh (mesh.deform every step) and produces a
    physically consistent rising surface — the surface step is not serial-only."""
    fs, _ = _case2(res=40)
    for _ in range(2):
        fs.solve()
        fs.advance(1.0)
    amplitude = _surface_amplitude(fs)
    # coarse res-40 reference is O(1e-4); assert the right sign and magnitude band.
    assert 1.0e-5 < amplitude < 1.0e-3, f"parallel surface amplitude off: {amplitude}"


@pytest.mark.level_1
def test_freesurface_annulus_ring_filter_no_seam():
    """The theta-ordered global-ring surface filter must not seam at the along-surface
    coordinate extremes (theta=0, theta=pi) the way the old extract_surface KDTree mapping
    did (106 ring nodes -> 104 submesh DOFs collided there -> a 5 km jump). Filter a smooth
    ring field and check the max node-to-node jump is uniform around the ring; and the
    gather/scatter round-trip is the identity."""
    mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0, cellSize=0.06, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    rhat = sympy.Matrix([[x / r, y / r]])
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = rhat
    stokes.add_essential_bc((0.0, 0.0), "Lower")
    fs = uw.systems.FreeSurface(stokes, "Upper", buoyancy_scale=1.0, normal=rhat,
                                tangent_advect="shape", surface_filter=20)

    rc = fs._ring_coords                         # local ring nodes, x-sorted order
    th = np.arctan2(rc[:, 1], rc[:, 0])
    field = np.cos(2.0 * th) + 0.1 * np.cos(9.0 * th)     # smooth swell + a little noise

    # gather -> scatter is the identity (serial: full ring on the one rank)
    assert np.allclose(fs._ring_scatter(fs._ring_gather(field)), field, atol=1e-12)

    filt = fs._filter_surface(field)
    o = np.argsort(th)
    ff = filt[o]
    jump = np.abs(np.diff(np.r_[ff, ff[0]]))     # node-to-node around the closed ring
    thf = th[o]
    at_ext = ((np.abs(thf) < 0.12) | (np.abs(np.abs(thf) - np.pi) < 0.12))
    assert jump[at_ext].max() < 4.0 * np.median(jump), \
        f"filter seams at theta=0/pi: extreme jump {jump[at_ext].max():.2e} vs median {np.median(jump):.2e}"


def _annulus_freesurface(constraint):
    """Isoviscous annulus convection cell driven by a mode-4 thermal perturbation, with
    the free surface managed by ``uw.systems.FreeSurface``."""
    r_in, r_out = 0.5, 1.0
    mesh = uw.meshing.Annulus(radiusInner=r_in, radiusOuter=r_out, cellSize=0.1, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    rhat = sympy.Matrix([[x / r, y / r]])
    theta = sympy.atan2(y, x)
    T = uw.discretisation.MeshVariable(f"Tfs{constraint}", mesh, 1, degree=2, continuous=True)
    profile = ((r_out - r) / (r_out - r_in)
               + 0.1 * sympy.sin(4 * theta) * sympy.sin(sympy.pi * (r - r_in) / (r_out - r_in)))
    T.array[:, 0, 0] = np.clip(
        np.asarray(uw.function.evaluate(profile, T.coords)).flatten(), 0.0, 1.0)

    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = (3.0e4 / (r_out - r_in) ** 3) * T.sym[0] * rhat
    stokes.add_essential_bc((0.0, 0.0), "Lower")
    stokes.tolerance = 1.0e-5
    fs = uw.systems.FreeSurface(stokes, "Upper", buoyancy_scale=5.0e5, normal=rhat,
                                composition=T, tangent_advect="shape", surface_filter=20,
                                smooth_length=0.2, consistent_constraint=constraint)
    fs._comp_adv.constitutive_model.Parameters.diffusivity = 1.0
    fs._comp_adv.add_dirichlet_bc(1.0, "Lower")
    fs._comp_adv.add_dirichlet_bc(0.0, "Upper")
    return mesh, fs, rhat


@pytest.mark.level_2
def test_freesurface_prescribed_rate_is_flux_free():
    r"""The rate handed to the consistent solve must carry no NET flux.

    An incompressible interior over a closed no-slip base can neither gain nor lose
    volume, so the prescribed :math:`\tilde u_n` must satisfy :math:`\oint \tilde u_n
    \,\mathrm{d}s = 0`. Surface nodes are unevenly spaced, so removing the *nodal* mean
    does not achieve this — it leaves ~1% of net flux, which a strong rotated constraint
    cannot absorb (it is then asking for a flow that does not exist). ``_surface_mean``
    weights the demean by arc length instead. Measured with exact FE boundary quadrature,
    not the nodal sum the manager itself uses, so it cannot pass by construction."""
    mesh, fs, rhat = _annulus_freesurface("strong")
    net = uw.maths.BdIntegral(mesh=mesh, fn=fs._un_target.sym[0], boundary="Upper")
    gross = uw.maths.BdIntegral(mesh=mesh, fn=sympy.Abs(fs._un_target.sym[0]),
                                boundary="Upper")
    for _ in range(3):
        fs.solve()
        fs.advance(fs.estimate_dt(advect_scale=10.0))
    ratio = abs(float(net.evaluate())) / abs(float(gross.evaluate()))
    assert ratio < 5.0e-3, f"prescribed rate carries {ratio:.2e} of net flux (nodal demean?)"


@pytest.mark.level_2
def test_freesurface_strong_constraint_beats_penalty():
    r"""With a flux-free datum the STRONG rotated constraint holds the surface as a
    material boundary far better than the weak penalty, and does not leak volume.

    This is the reason ``consistent_constraint`` defaults to ``"strong"``: the penalty
    both misses the prescribed rate and passes a net volume flux through the surface,
    and material crossing the surface is what puts semi-Lagrangian departure points
    outside the domain."""
    errors, leaks = {}, {}
    for constraint in ("strong", "penalty"):
        mesh, fs, rhat = _annulus_freesurface(constraint)
        net = uw.maths.BdIntegral(mesh=mesh, fn=fs._adv_velocity.sym.dot(rhat),
                                  boundary="Upper")
        gross = uw.maths.BdIntegral(mesh=mesh, fn=sympy.Abs(fs._adv_velocity.sym.dot(rhat)),
                                    boundary="Upper")
        for _ in range(4):
            fs.solve()
            fs.advance(fs.estimate_dt(advect_scale=10.0))
        coords = fs._ring_coords                       # LIVE surface nodes, ring order
        normal = fs._normal_direction(coords)
        realised = sum(
            np.asarray(uw.function.evaluate(fs._adv_velocity.sym[i], coords)).flatten()
            * normal[:, i] for i in range(mesh.dim))
        target = np.asarray(fs._un_target.array[fs._un_target_rows, 0, 0]).flatten()
        errors[constraint] = np.abs(realised - target).max() / np.abs(target).max()
        leaks[constraint] = abs(float(net.evaluate())) / abs(float(gross.evaluate()))

    assert errors["strong"] < 0.5 * errors["penalty"], (
        f"strong constraint not better: {errors['strong']:.2e} vs "
        f"penalty {errors['penalty']:.2e}")
    assert leaks["strong"] < 1.0e-3, \
        f"strong constraint leaks net volume flux {leaks['strong']:.2e}"


@pytest.mark.level_1
def test_freesurface_ring_quadrature_is_exact_in_parallel():
    """The arc-length datum gauge must be partition-independent.

    Runs on any rank count: the ring is gathered globally, so the weights and the weighted
    mean must be identical to serial. Surface nodes on a partition seam appear on both
    ranks, and the trapezoidal rule must split their weight rather than double-count it —
    hence the exact circumference check. This isolates the half of the datum path that IS
    parallel-correct, so a future fix to the surface-array/field round trip (see
    _PARALLEL_DATUM_DEFECT) does not have to re-establish it."""
    r_out = 1.0
    mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=r_out, cellSize=0.1, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    rhat = sympy.Matrix([[x / r, y / r]])
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = rhat
    stokes.add_essential_bc((0.0, 0.0), "Lower")
    fs = uw.systems.FreeSurface(stokes, "Upper", buoyancy_scale=1.0, normal=rhat)

    weights = fs._ring_weights()
    # The gauge must match the FE boundary quadrature EXACTLY: the datum's
    # flux-free condition lives in the discrete space, so the weight total is the
    # FE measure of the (polygonal) boundary — NOT the ideal-circle arc length,
    # which differs at O(h^2) and was the strong-datum compatibility floor
    # (block-split evidence: the whole stalled residual sat in the pressure rows).
    # A seam double-count breaks this equality at the first shared node.
    fe_len = float(uw.maths.BdIntegral(mesh=mesh, fn=sympy.S.One,
                                       boundary="Upper").evaluate())
    assert abs(float(weights.sum()) - fe_len) < 1.0e-9, \
        f"ring weights sum to {weights.sum():.10f}, FE boundary measure is {fe_len:.10f} " \
        "(seam double-count?)"
    # sanity: the polygonal measure approximates the circle at O(h^2)
    assert abs(fe_len - 2.0 * np.pi * r_out) < 1.0e-2

    coords = fs._ring_coords
    theta = np.arctan2(coords[:, 1], coords[:, 0])
    assert abs(fs._surface_mean(np.ones(coords.shape[0])) - 1.0) < 1.0e-12
    for k in (1, 2, 3):
        mean = fs._surface_mean(np.cos(k * theta))
        assert abs(mean) < 1.0e-10, f"weighted mean of cos({k}*theta) = {mean:.2e}, not zero"
