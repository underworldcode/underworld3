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
from underworld3.utilities.fault_contact import (add_frictionless_fault_bc,
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
