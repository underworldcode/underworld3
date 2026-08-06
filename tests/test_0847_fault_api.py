"""The deployed fault interface: ``Mesh.add_fault`` + ``solver.add_fault_bc``
+ plain ``solve()``.

The user-facing pipeline over the split-node backend, asserted end to end:

- one call faults the mesh (tips placed, cut, split, pairing recorded) from
  either a ``(name, points)`` pair or a ``uw.meshing.Surface``;
- ``add_fault_bc(0, name)`` is the frictionless contact, ``add_fault_bc
  (eta_f, name)`` the viscous interface, and an ORDINARY ``solve()``
  dispatches to the rotated fault path — no special driver in user code;
- a two-segment offset network (the J0 junction pattern of the deployment
  design) works in one ``add_fault`` call: both faults slip, no opening
  anywhere;
- the guard/difficulty probes refuse loudly on the fault path, exactly as
  they do for rotated free-slip.

The physics itself (crack profile, compliance family) is validated in
``test_0846_fault_contact.py``; here the assertions are that the INTERFACE
delivers the same behaviour.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.fault_contact import fault_slip

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

TIP_A = np.array([0.30, 0.45])
TIP_B = np.array([0.70, 0.55])
HALF = 0.5 * np.linalg.norm(TIP_B - TIP_A)


def _box(cell_size=1 / 24):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=cell_size, regular=False, qdegree=2)


def _shear_stokes(mesh, tag):
    x, y = mesh.X
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
    return stokes


def test_one_call_fault_and_plain_solve():
    """(name, points) -> add_fault -> add_fault_bc(0) -> solve()."""
    split = _box().add_fault(("Flt", np.vstack([TIP_A, TIP_B])))
    assert "FltPlus" in [b.name for b in split.boundaries]
    assert "Flt" in split._fault_point_pairs

    stokes = _shear_stokes(split, "A")
    stokes.add_fault_bc(0, "Flt")
    stokes.solve()

    info = stokes._rotated_freeslip_info
    assert info is not None and info["converged"]
    _s, V, leak = fault_slip(stokes, "Flt", info)
    assert np.abs(leak).max() < 1e-13
    # the frictionless crack, as in test_0846
    assert 0.4 * HALF < np.abs(V).max() < 1.05 * HALF

    # the probes that cannot reach the rotated path refuse loudly
    with pytest.raises(NotImplementedError, match="fault-contact"):
        stokes.guard(wall_per_step=10.0)
    with pytest.raises(NotImplementedError, match="fault-contact"):
        stokes.estimate_difficulty(2)


def test_viscous_law_through_the_solver_method():
    split = _box().add_fault(("Flt", np.vstack([TIP_A, TIP_B])))

    peaks = {}
    for tag, eta_f in (("free", 0.0), ("visc", 1.0 / HALF)):
        stokes = _shear_stokes(split, tag)
        stokes.add_fault_bc(eta_f, "Flt")
        stokes.solve()
        _s, V, _leak = fault_slip(stokes, "Flt",
                                  stokes._rotated_freeslip_info)
        peaks[tag] = float(np.abs(V).max())
    # eta_f = eta/a sits mid-family (the measured compliance curve)
    assert 0.15 * peaks["free"] < peaks["visc"] < 0.9 * peaks["free"]


def test_surface_object_input():
    base = _box()
    surface = uw.meshing.Surface("Flt", base, np.vstack([TIP_A, TIP_B]))
    split = base.add_fault(surface)
    assert "FltMinus" in [b.name for b in split.boundaries]
    assert len(split._fault_point_pairs["Flt"]) > 0


def test_offset_network_in_one_call():
    """The J0 junction pattern: two en-echelon segments, one add_fault."""
    A = ("FltA", np.array([[0.20, 0.42], [0.48, 0.49]]))
    B = ("FltB", np.array([[0.55, 0.52], [0.82, 0.59]]))
    split = _box().add_fault([A, B])
    for name in ("FltA", "FltB"):
        assert f"{name}Plus" in [b.name for b in split.boundaries]
        assert len(split._fault_point_pairs[name]) > 0

    stokes = _shear_stokes(split, "N")
    stokes.add_fault_bc(0, "FltA")
    stokes.add_fault_bc(0, "FltB")
    stokes.solve()
    info = stokes._rotated_freeslip_info
    assert info["converged"]
    for name in ("FltA", "FltB"):
        _s, V, leak = fault_slip(stokes, name, info)
        assert np.abs(leak).max() < 1e-13, f"{name} opened"
        assert np.abs(V).max() > 0.02, f"{name} does not slip"


def test_analytic_normal_smooths_a_sampled_curve():
    """``add_fault_bc(normal=...)`` on a polyline-sampled arc.

    The facet-AVERAGED per-node normal zig-zags at the sampling kinks, so
    the no-opening constraint forbids smooth slip past each kink — a
    normal-traction sawtooth locked to the kink positions (the negative
    control: it MUST be there, or this test can't see the fix). The
    analytic arc normal on the SAME kinked mesh removes it. Measured at
    h = 0.02 the ratio is ~7x; asserted at 3x. The exact-normal case runs
    on the SECOND mesh of the test, which is the regression for the
    unwrap coordinate re-tagging (issue #501's fault-side twin).
    """
    import sympy

    from underworld3.utilities.fault_contact import fault_normal_traction

    R, half_chord = 0.35, 0.22
    alpha = np.arcsin(half_chord / R)
    centre = np.array([0.5, 0.5 - R * np.cos(alpha)])
    ang = np.linspace(np.pi / 2 + alpha, np.pi / 2 - alpha, 9)  # 8 segments
    pts = centre + R * np.column_stack([np.cos(ang), np.sin(ang)])

    rough = {}
    for tag in ("avg", "exact", "trace"):
        split = _box(0.02).add_fault(("Arc", pts))
        stokes = _shear_stokes(split, f"an_{tag}")
        if tag == "exact":
            x, y = split.X
            stokes.add_fault_bc(0, boundary="Arc", normal=sympy.Matrix(
                [[x - centre[0], y - centre[1]]]))
        elif tag == "trace":
            # the smoothed normal built from the polyline itself — the
            # analytic-formula-free route for digitized traces
            stokes.add_fault_bc(0, boundary="Arc", normal="trace")
        else:
            stokes.add_fault_bc(0, boundary="Arc")
        stokes.solve()
        info = stokes._rotated_freeslip_info
        _s, _V, leak = fault_slip(stokes, "Arc", info)
        assert np.abs(leak).max() < 1e-10, f"{tag}: fault opened"
        s_n, sig = fault_normal_traction(stokes, "Arc", info)
        mid = (s_n > 0.08) & (s_n < s_n.max() - 0.08)
        rough[tag] = float(np.sqrt(np.mean(np.diff(sig[mid], 2) ** 2)))

    assert rough["avg"] > 3.0 * rough["exact"], (
        f"analytic normal did not smooth the sampled curve: "
        f"avg {rough['avg']:.4f} vs exact {rough['exact']:.4f}")
    assert rough["avg"] > 3.0 * rough["trace"], (
        f"trace-smoothed normal did not smooth the sampled curve: "
        f"avg {rough['avg']:.4f} vs trace {rough['trace']:.4f}")

    # a genuinely foreign symbol is still refused, and refusal must not
    # corrupt the already-registered override
    with pytest.raises(ValueError, match="not mesh coordinates"):
        stokes.add_fault_bc(0, boundary="Arc", normal=sympy.Matrix(
            [[sympy.Symbol("q_foreign"), 1]]))
    assert "Arc" in stokes._fault_normal_overrides
