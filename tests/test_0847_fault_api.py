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
