"""Frictionless contact on a split-node fault
(:mod:`underworld3.utilities.fault_contact`).

The perfectly slippery fault: no opening ([v]·n̂ = 0, strong) and zero shear
traction, with the tangential slip EMERGING from the far-field load — the
stress-driven crack, where the kinematic fault of the split-node gate
prescribed both sides. The analytic mode-II crack in an incompressible medium
under remote shear Δτ gives an elliptical velocity jump

    V(x) = (Δτ/η)·sqrt(a² − x²),  peak V = Δτ·a/η,

so the solved slip must be one-signed, elliptical in shape, vanish at the
unsplit tips, and peak below-but-near Δτ·a/η (a finite box suppresses the
far field and with it the peak). What is asserted, and what each would catch:

- **the pairing is complete** — 2m−1 velocity-carrying pairs for a chain of m
  facets (m−1 duplicated vertices + m doubled facets carrying the P2
  midpoint DOFs). A missed pair is a welded node.
- **leak is machine zero** — the strong jump-normal constraint, read through
  the pairing (a coordinate query cannot distinguish the sides).
- **slip profile** — sign, tip decay, elliptical fit, peak against the crack
  value. A wrong pair block (sign, normalisation, row placement) distorts
  the profile long before it breaks conformity or convergence.
- **the solve genuinely converged** — the rotated loop's own verdict.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.fault_contact import (add_coulomb_fault_bc,
                                                 add_frictionless_fault_bc,
                                                 add_viscous_fault_bc,
                                                 fault_normal_traction,
                                                 fault_slip, solve_with_fault)
from underworld3.utilities.fault_split import split_fault
from underworld3.utilities.line_cut import cut_along_lines, pull_vertex_onto

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

TIP_A = np.array([0.30, 0.45])
TIP_B = np.array([0.70, 0.55])
HALF = 0.5 * np.linalg.norm(TIP_B - TIP_A)


def _split_box(cell_size=1 / 24, name="Flt"):
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=cell_size, regular=False, qdegree=2)
    boundaries = base._boundaries_with(name)
    dm = pull_vertex_onto(base.dm, np.vstack([TIP_A, TIP_B]))
    dm, _info = cut_along_lines(dm, [np.vstack([TIP_A, TIP_B])],
                                label=name,
                                label_value=boundaries[name].value)
    cut = uw.discretisation.Mesh(
        dm, simplex=True,
        coordinate_system_type=base.CoordinateSystem.coordinate_type,
        qdegree=base.qdegree, boundaries=boundaries, verbose=False)
    cut.parent = base
    cut._relationship_kind = "refinement"
    cut._refine_dofs_coincide = False
    cut.regions = base.regions
    cut._parent_mesh_version = base._mesh_version
    return split_fault(cut, name)


def test_frictionless_fault_slips_like_a_crack():
    mesh = _split_box()
    x, y = mesh.X

    v = uw.discretisation.MeshVariable("vFC", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("pFC", mesh, 1, degree=0,
                                       continuous=False)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.tolerance = 1e-6
    stokes.petsc_use_pressure_nullspace = True
    for side in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), side)

    add_frictionless_fault_bc(stokes, "Flt")
    info = solve_with_fault(stokes)
    assert info["converged"], "the rotated fault-contact solve did not converge"

    # Pairing completeness: m facets -> (m-1) vertex pairs + m facet pairs.
    m = mesh.dm.getLabel("FltPlus").getStratumSize(
        mesh.boundaries["FltPlus"].value)
    s, V, leak = fault_slip(stokes, "Flt", info)
    assert len(s) == 2 * m - 1

    # No opening, to machine precision, at every pair.
    assert np.abs(leak).max() < 1e-13

    # The slip: one-signed, tip-decaying, elliptical, near the crack peak.
    Vmag = np.abs(V)
    sign = np.sign(V[np.argmax(Vmag)])
    assert ((sign * V) >= -1e-12 * Vmag.max()).all(), "slip changes sign"
    assert Vmag.max() > 0.1 * HALF, "the fault barely slips — welded somewhere"
    # ends (nearest pairs to the unsplit tips) carry much less than the peak
    assert Vmag[0] < 0.5 * Vmag.max() and Vmag[-1] < 0.5 * Vmag.max()
    # elliptical profile: V ~ Vmax * sqrt(1 - u^2) on the tip-to-tip span
    u = 2.0 * (s - s.min()) / (s.max() - s.min() + 1e-30) - 1.0
    # the span s covers pair nodes only; the true tips sit half a facet
    # beyond each end, so stretch u accordingly before the fit
    du = (s[1] - s[0]) / (s.max() - s.min() + 1e-30)
    u = u / (1.0 + du)
    ellipse = np.sqrt(np.clip(1.0 - u ** 2, 0.0, None))
    fit = (sign * V) / Vmag.max()
    misfit = np.sqrt(np.mean((fit - ellipse * fit.max()) ** 2))
    assert misfit < 0.15, f"slip profile is not elliptical (rms {misfit:.3f})"
    # peak against the analytic crack Δτ·a/η = HALF (Δτ = η = 1): below it
    # (finite box), but the right size.
    assert 0.4 * HALF < Vmag.max() < 1.05 * HALF, (
        f"peak slip {Vmag.max():.4f} vs crack value {HALF:.4f}")


def test_viscous_fault_bridges_welded_to_free():
    """The linear interface law spans its two exact limits monotonically.

    tau = eta_f * V: at eta_f = 0 the frictionless crack must be reproduced
    bit-for-bit in character (same peak); at eta_f >> eta/a the fault welds
    (slip collapses, the box shears as if uncut); in between the slip drops
    monotonically through the compliance scale eta/a. The dashpot occupies
    exactly the tangent slot a friction law will use, so this family is the
    coupling machinery's calibration curve.
    """
    mesh = _split_box()
    x, y = mesh.X

    peaks = {}
    for tag, eta_f in (("free", 0.0), ("mid", 1.0 / HALF), ("hard", 5.0e3)):
        v = uw.discretisation.MeshVariable(f"v{tag}", mesh, 2, degree=2)
        p = uw.discretisation.MeshVariable(f"p{tag}", mesh, 1, degree=0,
                                           continuous=False)
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        stokes.tolerance = 1e-6
        stokes.petsc_use_pressure_nullspace = True
        for side in ("Top", "Bottom", "Left", "Right"):
            stokes.add_dirichlet_bc((y - 0.5, 0.0), side)
        add_viscous_fault_bc(stokes, eta_f, "Flt")
        info = solve_with_fault(stokes)
        assert info["converged"], f"eta_f={eta_f}: solve did not converge"
        _s, V, leak = fault_slip(stokes, "Flt", info)
        assert np.abs(leak).max() < 1e-13, (
            f"eta_f={eta_f}: the no-opening constraint leaked")
        peaks[tag] = float(np.abs(V).max())

    assert peaks["free"] > 0.1 * HALF, "the eta_f=0 member does not slip"
    # strict monotone bridging, with a genuinely intermediate middle member
    assert peaks["free"] > peaks["mid"] > peaks["hard"]
    assert 0.15 * peaks["free"] < peaks["mid"] < 0.9 * peaks["free"], (
        f"eta_f = eta/a should sit mid-family: {peaks}")
    assert peaks["hard"] < 0.02 * peaks["free"], (
        f"a stiff dashpot should weld the fault: {peaks}")


def test_coulomb_fault_slides_and_sticks():
    """The first nonlinear law, against its two analytic regimes.

    The fault plane here is ~14 deg to x, so the resolved driving shear is
    tau_inf = cos(2*theta) ~ 0.88 of the unit far-field shear. A regularised
    Coulomb fault with strength mu*sigma_n BELOW tau_inf slides at constant
    stress drop: by superposition its slip is the frictionless crack's
    scaled by (tau_inf - strength)/tau_inf. Strength ABOVE tau_inf sticks:
    the slip collapses to the regularisation creep ~ V0, orders of
    magnitude below the crack. Both regimes must come out of the SAME
    consistent-Newton machinery — the tangent is sympy-derived from the
    arctan law, and the loop must genuinely converge in each.
    """
    mesh = _split_box()
    x, y = mesh.X
    d = (TIP_B - TIP_A) / np.linalg.norm(TIP_B - TIP_A)
    tau_inf = float(d[0] ** 2 - d[1] ** 2)      # cos(2*theta) resolved shear

    def solve_case(tag, register):
        v = uw.discretisation.MeshVariable(f"v{tag}", mesh, 2, degree=2)
        p = uw.discretisation.MeshVariable(f"p{tag}", mesh, 1, degree=0,
                                           continuous=False)
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        stokes.tolerance = 1e-6
        stokes.petsc_use_pressure_nullspace = True
        for side in ("Top", "Bottom", "Left", "Right"):
            stokes.add_dirichlet_bc((y - 0.5, 0.0), side)
        register(stokes)
        info = solve_with_fault(stokes)
        assert info["converged"], f"{tag}: did not converge"
        _s, V, leak = fault_slip(stokes, "Flt", info)
        assert np.abs(leak).max() < 1e-12
        return float(np.abs(V).max()), info["nonlinear_iterations"]

    V_free, _ = solve_case(
        "cf", lambda st: add_frictionless_fault_bc(st, "Flt"))

    # Sliding: strength = half the resolved driving shear.
    strength = 0.5 * tau_inf
    V_slide, its_slide = solve_case(
        "cs", lambda st: add_coulomb_fault_bc(
            st, 1.0, "Flt", sigma_n=strength, V0=1e-4 * V_free))
    ratio = V_slide / V_free
    assert 0.3 < ratio < 0.7, (
        f"sliding slip ratio {ratio:.3f}; constant-stress-drop scaling "
        f"predicts ~{(tau_inf - strength) / tau_inf:.2f}")
    assert its_slide < 30, f"sliding solve took {its_slide} Newton iterations"

    # Stuck: strength safely above the resolved driving shear.
    V0_stick = 1e-6
    V_stick, its_stick = solve_case(
        "ck", lambda st: add_coulomb_fault_bc(
            st, 1.0, "Flt", sigma_n=2.0, V0=V0_stick))
    assert V_stick < 20 * V0_stick, (
        f"stuck fault creeps at {V_stick:.2e}, far above the "
        f"regularisation velocity {V0_stick:.0e}")
    assert its_stick < 40, f"stick solve took {its_stick} Newton iterations"


def test_reaction_fed_normal_stress():
    """sigma_n from the no-opening constraint's reaction, not prescribed.

    Two claims. (1) The recovery itself: on the frictionless fault under
    unit far-field shear, the transmitted normal traction should match the
    resolved background sigma_nn = 2 n_x n_y (mode-II slip perturbs the
    normal traction antisymmetrically, so the MEAN along the fault stays at
    the background value). (2) The law coupling: Coulomb with
    sigma_n="reaction" must land between frictionless and a strong
    prescribed fault, sliding at the strength the recovered compression
    actually provides — with the whole sigma feedback Picard-lagged inside
    the same Newton loop.
    """
    mesh = _split_box()
    x, y = mesh.X
    d = (TIP_B - TIP_A) / np.linalg.norm(TIP_B - TIP_A)
    nrm = np.array([-d[1], d[0]])
    sigma_bg = 2.0 * nrm[0] * nrm[1] * d[0] * d[1] / (d[0] * d[1])  # sign via components
    sigma_bg = 2.0 * (-d[1]) * d[0]          # n_x n_y * 2 with n = (-dy, dx)
    tau_inf = float(d[0] ** 2 - d[1] ** 2)

    def solve_case(tag, register):
        v = uw.discretisation.MeshVariable(f"v{tag}", mesh, 2, degree=2)
        p = uw.discretisation.MeshVariable(f"p{tag}", mesh, 1, degree=0,
                                           continuous=False)
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        stokes.tolerance = 1e-6
        stokes.petsc_use_pressure_nullspace = True
        for side in ("Top", "Bottom", "Left", "Right"):
            stokes.add_dirichlet_bc((y - 0.5, 0.0), side)
        register(stokes)
        info = solve_with_fault(stokes)
        assert info["converged"], f"{tag}: did not converge"
        _s, V, leak = fault_slip(stokes, "Flt", info)
        assert np.abs(leak).max() < 1e-12
        return stokes, info, float(np.abs(V).max())

    # (1) traction recovery against the resolved background
    st_free, info_free, V_free = solve_case(
        "rf", lambda st: add_frictionless_fault_bc(st, "Flt"))
    _s, sig = fault_normal_traction(st_free, "Flt", info_free)
    mean_sig = float(np.mean(sig))
    assert np.sign(mean_sig) == np.sign(sigma_bg)
    assert abs(mean_sig - sigma_bg) < 0.35 * abs(sigma_bg), (
        f"recovered mean sigma_nn {mean_sig:.3f} vs background "
        f"{sigma_bg:.3f}")

    # (2) the reaction-fed Coulomb law
    mu = 1.0
    _st, info_rc, V_rc = solve_case(
        "rc", lambda st: add_coulomb_fault_bc(
            st, mu, "Flt", sigma_n="reaction", V0=1e-4 * V_free))
    strength = mu * abs(sigma_bg)
    predicted = max(tau_inf - strength, 0.0) / tau_inf
    ratio = V_rc / V_free
    assert 0.5 * predicted < ratio < min(1.6 * predicted + 0.05, 0.95), (
        f"reaction-fed slip ratio {ratio:.3f} vs constant-stress-drop "
        f"prediction {predicted:.3f} (strength {strength:.3f} from the "
        f"recovered compression)")
    assert info_rc["nonlinear_iterations"] < 40


def test_registration_refuses_an_unsplit_mesh():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 8,
        regular=False, qdegree=2)
    v = uw.discretisation.MeshVariable("vNC", base, 2, degree=2)
    p = uw.discretisation.MeshVariable("pNC", base, 1, degree=0,
                                       continuous=False)
    stokes = uw.systems.Stokes(base, velocityField=v, pressureField=p)
    with pytest.raises(ValueError, match="no split-fault pairing"):
        add_frictionless_fault_bc(stokes, "Flt")
    with pytest.raises(RuntimeError, match="no fault contact registered"):
        solve_with_fault(stokes)
