"""3D FreeSurface on a spherical shell: the end-to-end loop must run and produce
physically sensible topography (guards the dimension-general surface machinery:
owned-facet trace-mass gauge, P1-projected sigma_nn recovery, radial deform).

Pinned here: construction, one solve/advance cycle, a finite and genuinely
volume-preserving h_inf, the DIRECTION of the surface response, the explicit
refusal of the 2D-only features, and the relaxation RATE of an initial Y20
topography against the half-space Cathles rate.

The rate test closes the second half of #496. It is a REGRESSION pin at one
fixed resolution: the ratio measured 0.528 at cell 0.35 and 0.516 at cell 0.25,
so it is close to converged, but two coarse resolutions do not establish that.
The decay lands on a floor -- 16% of the initial amplitude here, 11% at cell
0.25 -- which is the resolution-convergent #431-class bias in the recovered
h_inf, not part of the physics. Because the decay is onto a floor rather than to
zero, the FIT PROTOCOL is part of the specification: the same record yields
rates of 0.064, 0.110 or 0.142 depending on how the floor is treated. It is
fixed in _fit_relaxation and must not be changed casually.

Measurements, negative controls and the ordering bug that had to be fixed first:
~/+Simulations/spherical_relaxation_rate_496/README.md.
"""
import numpy as np
from mpi4py import MPI
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _shell_stokes(cell=0.35):
    mesh = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.547,
                                     cellSize=cell, qdegree=3)
    x, y, z = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2 + z ** 2)
    rhat = sympy.Matrix([[x / r, y / r, z / r]])
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    blob = sympy.exp(-(((x - 0.75) ** 2 + y ** 2 + z ** 2) / 0.05))
    stokes.bodyforce = 50.0 * blob * rhat.T
    stokes.add_essential_bc((0.0, 0.0, 0.0), "Lower")
    stokes.tolerance = 1.0e-5
    return mesh, stokes, rhat


def _surface_carrier(fs, mesh, name):
    """A P1 field on the surface plus the rows that hold a surface array.

    Ordering trap, and the reason the rows come from ``_surf_coords``: every
    surface ARRAY (``_current_shape``, ``_h_inf``, the increment ``_carry_and_deform``
    consumes) is indexed in ``_surf_rows`` order, which is ``_surf_coords`` sorted
    by REFERENCE x. ``_field_rows`` re-sorts whatever coordinates it is handed, so
    matching against ``_ring_coords`` -- documented at free_surface.py:211 as LIVE,
    deformed geometry -- sorts by DEFORMED x instead and silently permutes the fill.
    """
    field = uw.discretisation.MeshVariable(name, mesh, 1, degree=1, continuous=True)
    rows, _ = fs._field_rows(field, fs._surf_coords)
    return field, rows


def test_freesurface_spherical_shell_end_to_end():
    """Construction + one full solve/advance on the shell; h_inf finite and
    volume-preserving; the surface responds toward equilibrium (|h| grows from
    flat under the one-sided load and stays bounded by |h_inf|)."""
    mesh, stokes, rhat = _shell_stokes()
    fs = uw.systems.FreeSurface(stokes, "Upper", buoyancy_scale=50.0, normal=rhat)
    fs.solve()
    h_inf = np.asarray(fs._h_inf)
    assert np.isfinite(h_inf).all(), "3D h_inf recovery produced non-finite values"
    scale = uw.mpi.comm.allreduce(
        float(np.abs(h_inf).max()) if h_inf.size else 0.0, op=MPI.MAX)
    assert scale > 1.0e-4, "no topographic response to the load"

    # The datum must be VOLUME-preserving, measured by an integrator that does not
    # share the gauge's own weight vector. Asserting _surface_mean(h_inf) instead
    # is vacuous: free_surface.py:901 builds h_inf as _demean(-_demean(h)) and
    # _demean IS subtraction of _surface_mean, so the quantity is zero by
    # construction -- measured 1e-18 relative against a 1e-8 threshold, and it
    # stays zero even if the trace-mass weights themselves are wrong, because both
    # sides use the same weights. The FE boundary integral of the P1 carrier is an
    # independent implementation of the same integral and does catch that.
    h_field, h_rows = _surface_carrier(fs, mesh, "hinfP1")
    h_field.array[...] = 0.0
    h_field.array[h_rows, 0, 0] = h_inf
    area = uw.maths.BdIntegral(mesh=mesh, fn=sympy.sympify(1), boundary="Upper")
    flux = uw.maths.BdIntegral(mesh=mesh, fn=h_field.sym[0], boundary="Upper")
    datum = float(flux.evaluate()) / float(area.evaluate())
    assert abs(datum) < 1.0e-12 * scale, (
        f"h_inf datum is not volume-preserving: <h_inf> = {datum:.3e} over the "
        f"boundary, {abs(datum) / scale:.2e} of max|h_inf|")

    # NEGATIVE CONTROL for the line above: the trace-mass weighting must actually
    # do work here, or "the weighted mean vanishes" says nothing. The UNWEIGHTED
    # nodal mean of the same data is not zero (measured 2.5e-4 of max|h_inf| at
    # this resolution) -- surface nodes are not equally spaced, so a node-count
    # gauge would not be volume-preserving.
    nodal = (uw.mpi.comm.allreduce(float(h_inf.sum()) if h_inf.size else 0.0)
             / max(uw.mpi.comm.allreduce(int(h_inf.size)), 1))
    assert abs(nodal) > 1.0e-6 * scale, (
        f"the unweighted nodal mean is also ~zero ({abs(nodal) / scale:.2e} of "
        "max|h_inf|), so the volume-preservation check above has no content on "
        "this mesh -- it cannot distinguish the trace-mass gauge from node counting")

    fs.advance(fs.estimate_dt(advect_scale=10.0))
    shape = np.asarray(fs._current_shape())
    assert np.isfinite(shape).all()
    shape_max = uw.mpi.comm.allreduce(
        float(np.abs(shape).max()) if shape.size else 0.0, op=MPI.MAX)
    assert 0.0 < shape_max <= 1.5 * scale, \
        "surface did not move toward (or overshot) equilibrium"

    # DIRECTION, not just magnitude. The bound above is symmetric in sign: flip
    # the 3-D recovery and |shape| is unchanged, so it passes while the surface
    # moves the wrong way (#496). Starting from flat, the displacement IS the
    # shape, so it must correlate POSITIVELY with the equilibrium it is moving
    # toward. Measured +0.995 here; a sign error gives about -0.995.
    interesting = np.abs(h_inf) > 1.0e-12
    direction = float(np.corrcoef(shape.ravel()[interesting.ravel()],
                                  np.asarray(h_inf).ravel()[interesting.ravel()])[0, 1])
    assert direction > 0.9, (
        f"surface moved away from equilibrium (corr {direction:+.3f}) — the "
        "magnitude bound above cannot see a sign error")


def _fit_relaxation(t, A):
    r"""Fit :math:`A(t) = A_\infty + A_1 e^{-\lambda t}` and return
    ``(lambda, A_inf, residual_rms)``.

    The floor :math:`A_\infty` is a FREE parameter, and that is the whole protocol.
    The decay does not run to zero -- it lands on the Y20 bias in the recovered
    h_inf -- so the extracted rate depends entirely on how the floor is treated:
    on this same record, fitting with no floor gives 0.064, floor free 0.108, and
    floor pinned to the last sample 0.142. Any of the three can be called "the
    fitted decay rate"; only one of them is a stable number, so the test fixes it.
    """
    from scipy.optimize import curve_fit

    def model(tt, A_inf, A_1, lam):
        return A_inf + A_1 * np.exp(-lam * tt)

    p, _ = curve_fit(model, t, A, p0=[A[-1] * 0.5, A[0], 0.1], maxfev=200000)
    return p[2], p[0], float((A - model(t, *p)).std())


def test_freesurface_spherical_relaxation_rate():
    """An initial Y20 topography must relax at the shell rate (#496, second half).

    Every other free-surface test starts FLAT and is driven by a load, so nothing
    in the suite exercises the decay of an imposed topography and a rate
    regression is invisible. Here a constant-density shell under radial gravity
    carries a degree-2 bump and nothing else: the topographic self-load is the
    only driver, so the modal amplitude must decay exponentially at a rate set by
    the Stokes solve.
    """
    r_out, r_in, eps, dt, nsteps = 1.0, 0.547, 0.01, 1.5, 8
    rho_g = eta = 1.0

    mesh = uw.meshing.SphericalShell(radiusOuter=r_out, radiusInner=r_in,
                                     cellSize=0.35, qdegree=3)
    x, y, z = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2 + z ** 2)
    rhat = sympy.Matrix([[x / r, y / r, z / r]])
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
    stokes.bodyforce = -rho_g * rhat.T
    stokes.add_essential_bc((0.0, 0.0, 0.0), "Lower")
    stokes.tolerance = 1.0e-6
    fs = uw.systems.FreeSurface(stokes, "Upper", buoyancy_scale=rho_g,
                                normal=rhat, background_buoyancy="analytic")

    h_field, h_rows = _surface_carrier(fs, mesh, "hY20")
    Y20 = 0.5 * (3.0 * (z / r) ** 2 - 1.0)
    num = uw.maths.BdIntegral(mesh=mesh, fn=h_field.sym[0] * Y20, boundary="Upper")
    den = uw.maths.BdIntegral(mesh=mesh, fn=Y20 ** 2, boundary="Upper")

    def modal_amplitude():
        h_field.array[...] = 0.0
        h_field.array[h_rows, 0, 0] = fs._current_shape()
        return float(num.evaluate()) / float(den.evaluate())

    # impose the bump through the manager's own carrier, so the initial state is
    # exactly a valid FS state (smooth interior decay + radial deform)
    sc = fs._ring_coords
    y20 = 0.5 * (3.0 * (sc[:, 2] / np.linalg.norm(sc, axis=1)) ** 2 - 1.0)
    fs._carry_and_deform(eps * y20, dt=0.0)

    ts, As = [0.0], [modal_amplitude()]
    assert As[0] > 0.5 * eps, \
        f"the imposed Y20 bump did not register on the surface (A0 = {As[0]:.3e})"
    for _ in range(nsteps):
        fs.solve()
        fs.advance(dt)              # FIXED step: estimate_dt is CFL-based and is
        ts.append(ts[-1] + dt)      # ~10 tau here, which hops the whole decay
        As.append(modal_amplitude())
    t, A = np.array(ts), np.array(As)

    rate, floor, resid = _fit_relaxation(t, A)

    # gmsh triangulates differently across platforms (serial_reference.py:202
    # records e.g. 1417 vs 1395 cells for a spherical shell), so a band failure
    # has to name the mesh it was measured on or it is not diagnosable.
    # Reference: 585 cells, macOS/arm64.
    ncells = uw.mpi.comm.allreduce(mesh.dm.getStratumSize("depth", mesh.dim))
    where = (f"[{ncells} cells, A0 = {A[0]:.4e}, "
             f"floor {floor / A[0] * 100:.1f}% of A0]")

    # A rate is only meaningful if the decay IS an exponential; a fit rammed
    # through a non-exponential record would otherwise report a number and pass.
    assert resid < 5.0e-3 * A[0], (
        f"the decay is not a clean exponential (residual rms {resid:.2e} = "
        f"{resid / A[0] * 100:.2f}% of A0) — the fitted rate is meaningless")
    assert 0.0 < floor < 0.4 * A[0], (
        f"h_inf Y20 bias floor {floor / A[0] * 100:.1f}% of A0 is out of band; "
        "this is the #431-class recovery defect and should shrink with resolution")

    # The shell over a no-slip base must relax SLOWER than a half-space (finite
    # depth, fixed bottom) but on the same order -- an O(1) correction below 1.
    cathles = rho_g / (2.0 * eta * np.sqrt(2.0 * 3.0) / r_out)   # k = sqrt(l(l+1))/R
    ratio = rate / cathles
    assert 0.45 < ratio < 0.62, (
        f"Y20 relaxation rate {rate:.4f} is {ratio:.3f} of the half-space Cathles "
        f"rate, on this mesh {where}.\n"
        f"rate {cathles:.4f}; expected an O(1) shell correction below 1 (finite "
        "depth over a no-slip base relaxes slower than a half-space).\n"
        "Measured: 0.528 at cell 0.35 and 0.516 at cell 0.25 (so the ratio is "
        "nearly resolution-converged, unlike the floor); 0.518 to 0.552 as the "
        "fitted record grows from 5 to 20 steps; np=1 and np=2 agree to 5 digits.\n"
        "What this band catches: a sign error (growth, not decay), an order-of-"
        "magnitude error, and the half-space value itself (1.0) being returned. "
        "What it does NOT catch: a 2x error in buoyancy_scale, measured at 0.476 "
        "-- the exponential update is only weakly sensitive to it, and a band "
        "tight enough to see it would sit within a few percent of the resolution "
        "spread and flake.")


def test_freesurface_spherical_refuses_2d_only_features():
    """The 2D-only features fail loudly at construction in 3D, not silently."""
    mesh, stokes, rhat = _shell_stokes()
    with pytest.raises(NotImplementedError, match="tangential"):
        uw.systems.FreeSurface(stokes, "Upper", normal=rhat, tangent_advect="shape")
    with pytest.raises(NotImplementedError, match="filter"):
        uw.systems.FreeSurface(stokes, "Upper", normal=rhat, surface_filter=10)
