"""Shared helpers for the VE/VEP analytical benchmark suite.

Three benchmark cases share this module:

* ``bench_ve_harmonic.py``   — Maxwell shear under :math:`V_{top}(t) = V_0 \\sin(\\omega t)`
* ``bench_ve_square.py``     — Maxwell shear under square-wave :math:`V_{top}`
* ``bench_vep_square.py``    — same square-wave forcing with Min-mode plasticity

Common setup
------------
* Mesh: ``StructuredQuadBox`` 16×8 over ``(±1, ±0.5)``.
* Velocity at top/bottom: ``±V_top(t)``, free at left/right.
* Pure shear with strain rate ``γ̇ = 2·V_top/H = V_top``.
* Centre-point stress sample.
* Scaling: ``η = μ = 1``, so Maxwell relaxation time ``t_r = 1`` and the
  steady-state VE stress under sustained shear is ``η·γ̇``.

Logging
-------
Each run writes a self-contained ``.npz`` to ``output/benchmarks/<name>.npz``
holding the simulation trace, the analytical reference, the parameter
dict, and metadata. Plotting is decoupled — see ``plot_benchmarks.py``.
"""

import os
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import expression


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
OUTPUT_DIR = os.path.join(_REPO_ROOT, "output", "benchmarks")
FIG_DIR = os.path.join(_REPO_ROOT, "docs", "advanced", "figures")


# ---------------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = dict(
    eta=1.0,         # shear viscosity
    mu=1.0,          # shear modulus
    H=1.0,           # box height (top–bottom)
    W=2.0,           # box width  (left–right)
    elementRes=(16, 8),
    velocity_degree=2,
    pressure_degree=1,
    bdf_order=2,
)


def t_relax(params):
    return params["eta"] / params["mu"]


# ---------------------------------------------------------------------------
# Analytical solutions
# ---------------------------------------------------------------------------

def maxwell_oscillatory(t, eta, mu, gamma_dot_0, omega):
    r"""Closed-form Maxwell shear stress under sinusoidal forcing
    :math:`\dot\gamma(t) = \dot\gamma_0 \sin(\omega t)`.

    Solving :math:`\dot\sigma + \sigma/t_r = \mu\dot\gamma` with
    :math:`\sigma(0) = 0` gives

    .. math::
        \sigma(t) = \frac{\eta\dot\gamma_0}{1+\mathrm{De}^2}
        \left[\sin(\omega t) - \mathrm{De}\cos(\omega t) + \mathrm{De}\,e^{-t/t_r}\right]

    where :math:`\mathrm{De} = \omega t_r` is the Deborah number. After
    transient decay (:math:`t \gg t_r`) the steady response has amplitude
    :math:`\eta\dot\gamma_0/\sqrt{1+\mathrm{De}^2}` and phase lag
    :math:`\varphi = \arctan(\mathrm{De})`.
    """
    t_r = eta / mu
    De = omega * t_r
    pre = eta * gamma_dot_0 / (1.0 + De**2)
    return pre * (np.sin(omega * t) - De * np.cos(omega * t) + De * np.exp(-t / t_r))


def maxwell_square_wave(t, eta, mu, gamma_dot_0, half_period):
    r"""Closed-form Maxwell shear stress under square-wave forcing.

    Within each half-period the stress relaxes exponentially toward the
    steady-state value :math:`\pm\eta\dot\gamma_0` from the value at the
    period boundary:

    .. math::
        \sigma(t) = s_n\sigma_{\mathrm{ss}} + (\sigma_{0,n} - s_n\sigma_{\mathrm{ss}})\,
                    e^{-(t - t_n)/t_r}

    where :math:`s_n = (-1)^n` is the sign in half-period :math:`n` and
    :math:`\sigma_{0,n}` is the stress at the start of that half-period.
    """
    t_r = eta / mu
    sigma_ss = eta * gamma_dot_0
    out = np.zeros_like(np.asarray(t, dtype=float))
    sigma_start = 0.0
    for i, ti in enumerate(np.asarray(t, dtype=float)):
        n = int(ti / half_period)
        t_local = ti - n * half_period
        # Replay periods 0..n-1 to find sigma at start of period n
        sigma_n = 0.0
        for j in range(n):
            sign = 1.0 if j % 2 == 0 else -1.0
            target = sign * sigma_ss
            sigma_n = target + (sigma_n - target) * np.exp(-half_period / t_r)
        sign = 1.0 if n % 2 == 0 else -1.0
        target = sign * sigma_ss
        out[i] = target + (sigma_n - target) * np.exp(-t_local / t_r)
    return out


def vep_square_wave(t, eta, mu, gamma_dot_0, tau_y, half_period):
    r"""Closed-form VEP shear stress under square-wave forcing with
    Min-mode plasticity.

    Within each half-period, the stress evolves under Maxwell:

    .. math::
        \sigma(t) = s_n\sigma_{\mathrm{ss}} + (\sigma_{0,n} - s_n\sigma_{\mathrm{ss}})\,
                    e^{-(t - t_n)/t_r}

    until :math:`|\sigma| = \tau_y`, after which the plastic flow holds
    :math:`\sigma = \pm\tau_y`. The next half-period starts from the
    *clipped* value (``±τ_y`` if the previous period yielded; otherwise
    the unclipped end value).

    When :math:`\eta\dot\gamma_0 \le \tau_y` the solution coincides with
    the unclipped Maxwell square-wave.
    """
    t_arr = np.asarray(t, dtype=float)
    t_r = eta / mu
    sigma_ss = eta * gamma_dot_0
    out = np.zeros_like(t_arr)

    # Pre-compute σ at the start of each half-period (including clipping)
    n_half_max = int(np.ceil(t_arr[-1] / half_period)) + 2
    sigma_at_start = [0.0]
    for n in range(n_half_max):
        sign = 1.0 if n % 2 == 0 else -1.0
        target = sign * sigma_ss
        sigma_0 = sigma_at_start[-1]
        sigma_end = target + (sigma_0 - target) * np.exp(-half_period / t_r)
        # Clip to ±τ_y at the period boundary if the unclipped value would
        # have exceeded the yield surface
        sigma_end_clipped = np.clip(sigma_end, -tau_y, tau_y)
        sigma_at_start.append(float(sigma_end_clipped))

    # Evaluate at each requested t
    for i, ti in enumerate(t_arr):
        n = int(ti / half_period)
        t_local = ti - n * half_period
        sign = 1.0 if n % 2 == 0 else -1.0
        target = sign * sigma_ss
        sigma_0 = sigma_at_start[n]
        sigma_unclipped = target + (sigma_0 - target) * np.exp(-t_local / t_r)
        out[i] = np.clip(sigma_unclipped, -tau_y, tau_y)
    return out


# ---------------------------------------------------------------------------
# Stokes problem builder
# ---------------------------------------------------------------------------

def build_stokes(label, params, yield_stress=None, yield_mode="min", yield_softness=0.0):
    """Construct a VE_Stokes problem with the standard mesh / BCs.

    Parameters
    ----------
    label : str
        Used to namespace the mesh variable names so multiple problems can
        coexist in one Python session.
    params : dict
        Material parameters (see DEFAULT_PARAMS).
    yield_stress : float or None
        If ``None``, pure VE (yield_stress is set to a large finite value).
        Otherwise enables VEP with the given yield stress.
    yield_mode : str
        Passed to ``constitutive_model._yield_mode``. ``"min"`` (the unified
        δ-parameterised soft-min) or ``"harmonic"``. ``"softmin"`` is accepted
        as a legacy alias for ``"min"``.
    yield_softness : float
        The soft-min δ (``constitutive_model._yield_softness``). ``0.0`` (default)
        is exact Min; ``> 0`` is a controlled smooth-min.

    Returns
    -------
    mesh, stokes, V_top, params
        ``V_top`` is the user-facing UWexpression for the top BC velocity.
    """
    p = dict(params)
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=p["elementRes"],
        minCoords=(-p["W"] / 2.0, -p["H"] / 2.0),
        maxCoords=(p["W"] / 2.0, p["H"] / 2.0),
    )
    v = uw.discretisation.MeshVariable(f"U_{label}", mesh, mesh.dim, degree=p["velocity_degree"])
    pp = uw.discretisation.MeshVariable(f"P_{label}", mesh, 1, degree=p["pressure_degree"])
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=pp)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=p["bdf_order"],
    )
    stokes.constitutive_model.Parameters.shear_viscosity_0 = p["eta"]
    stokes.constitutive_model.Parameters.shear_modulus = p["mu"]
    stokes.constitutive_model.Parameters.yield_stress = (
        yield_stress if yield_stress is not None else 1.0e6
    )
    stokes.constitutive_model.Parameters.strainrate_inv_II_min = 1.0e-6
    stokes.constitutive_model._yield_mode = "min" if yield_mode == "softmin" else yield_mode
    stokes.constitutive_model._yield_softness = yield_softness

    V_top = expression(rf"V_{{top,{label}}}", sympy.Float(0.0), "Top V")
    stokes.add_dirichlet_bc((V_top, 0.0), "Top")
    stokes.add_dirichlet_bc((-V_top, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_force_iteration"] = True

    return mesh, stokes, V_top, p


# ---------------------------------------------------------------------------
# Per-step probe
# ---------------------------------------------------------------------------

def probe_centre(stokes, c=np.array([[0.0, 0.0]])):
    return float(uw.function.evaluate(stokes.tau.sym[0, 1], c).flatten()[0])


# ---------------------------------------------------------------------------
# Self-contained npz logger
# ---------------------------------------------------------------------------

def save_run(name, *, params, params_extra=None, **arrays):
    """Save a benchmark run to ``output/benchmarks/<name>.npz``.

    Parameters
    ----------
    name : str
        Output filename stem (no extension).
    params : dict
        Material/numerical parameters used for the run.  Stored as a
        single ``params`` field for re-creation/replotting.
    params_extra : dict or None
        Per-benchmark scalar metadata (omega, half_period, tau_y, …).
    **arrays
        Per-step arrays: times, sigma, sigma_ana, dt, gamma_dot, etc.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = f"{OUTPUT_DIR}/{name}.npz"
    payload = {f"arr_{k}": np.asarray(v) for k, v in arrays.items()}
    payload["__params__"] = np.asarray(repr(dict(params)), dtype=object)
    payload["__params_extra__"] = np.asarray(repr(dict(params_extra or {})), dtype=object)
    payload["__keys__"] = np.asarray(list(arrays.keys()), dtype=object)
    payload["__name__"] = np.asarray(name, dtype=object)
    np.savez(path, **payload)
    return path


def load_run(name):
    """Reverse of :func:`save_run`. Returns ``(arrays, params, extra)``."""
    path = f"{OUTPUT_DIR}/{name}.npz"
    with np.load(path, allow_pickle=True) as f:
        keys = list(f["__keys__"])
        arrays = {k: f[f"arr_{k}"] for k in keys}
        params = eval(str(f["__params__"]))
        extra = eval(str(f["__params_extra__"]))
    return arrays, params, extra


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def error_metrics(sigma, sigma_ana):
    """Standard error report: max and rms absolute error."""
    diff = sigma - sigma_ana
    return dict(
        max_abs=float(np.max(np.abs(diff))),
        rms=float(np.sqrt(np.mean(diff**2))),
        rel_max=float(np.max(np.abs(diff)) / (np.max(np.abs(sigma_ana)) + 1e-30)),
    )


def fit_amp_phase(t, sigma, omega):
    """Least-squares fit of ``A·sin(ωt − φ)`` to ``sigma``.

    Returns ``(A, phi)``.  Drops the first ``2*t_r`` to skip the
    transient (assumes ``t_r = 1`` and that the array is long enough).
    """
    mask = t > 4.0  # skip ~4 t_r of transient
    if mask.sum() < 8:
        mask = np.ones_like(t, dtype=bool)
    ts = t[mask]
    ss = sigma[mask]
    # σ ≈ a·sin(ωt) + b·cos(ωt) — fit (a, b) by linear least squares
    M = np.column_stack([np.sin(omega * ts), np.cos(omega * ts)])
    coeffs, *_ = np.linalg.lstsq(M, ss, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    A = np.sqrt(a**2 + b**2)
    # σ = A·sin(ωt − φ) → A·(cos(φ)sin(ωt) − sin(φ)cos(ωt)) = a·sin + b·cos
    # so a = A cos(φ), b = −A sin(φ).  Hence φ = atan2(−b, a).
    phi = float(np.arctan2(-b, a))
    return A, phi
