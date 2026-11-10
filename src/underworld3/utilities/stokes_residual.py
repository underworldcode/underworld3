"""Fresh no-solve residual of saved fields in a prepared native Stokes model.

The caller must reconstruct the original linear equations, auxiliary data,
loads, homogeneous essential conditions, field layout and quadrature before
calling. This verifies the original coefficient equations; it does not supply
missing historical solver reasons or certify physical discretisation error.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from underworld3.systems.solvers import SNES_Stokes


def evaluate_linear_stokes_residual(stokes: SNES_Stokes, *, rtol: float) -> dict:
    """Return ||F(saved)||/||F(0)|| without solving or changing saved fields.

    Uses the same native build/auxiliary/field-packing recipe as
    ``compute_volume_residual_fields`` but evaluates the original SNES callback
    globally, including boundary terms and an already-installed linear sheet
    RHS. It does not substitute a volume-only residual, project the residual,
    attach gauges, change solver options, or manufacture convergence reasons.
    This private helper is restricted to linear homogeneous-Dirichlet models;
    the caller is responsible for establishing solution-independent equations.
    """
    rtol = float(rtol)
    if not math.isfinite(rtol) or rtol <= 0:
        raise ValueError('rtol must be finite and positive')
    if getattr(stokes, "_rotated_freeslip_bcs", None):
        raise ValueError("Rotated free-slip equations require their constrained residual diagnostic")
    for bc in stokes.essential_bcs:
        if any(sp.sympify(value).is_zero is not True
               for value in bc.fn if value is not None):
            raise ValueError('Saved native residual verification needs homogeneous essential data')
    stokes._build(False, False, None)
    mesh = stokes.mesh
    mesh.update_lvec()
    stokes.dm.setAuxiliaryVec(mesh.lvec, None)
    mesh._verify_integration_rule(getattr(stokes, 'petsc_fe_u', None))
    stokes._update_constants()
    saved = stokes.dm.createGlobalVec()
    zero, residual, reference = (saved.duplicate() for _ in range(3))
    try:
        saved.set(0.)
        zero.set(0.)
        for name, var in stokes.fields.items():
            iset, subdm = stokes._subdict[name]
            subvec = saved.getSubVector(iset)
            try:
                subdm.localToGlobal(var.vec, subvec)
            finally:
                saved.restoreSubVector(iset, subvec)
        stokes.snes.computeFunction(saved, residual)
        stokes.snes.computeFunction(zero, reference)
        actual, scale = float(residual.norm()), float(reference.norm())
        if not math.isfinite(actual) or not math.isfinite(scale):
            raise ValueError('Nonfinite original native residual')
        if scale <= 0:
            raise ValueError('Zero reference residual: relative verification is undefined')
        relative = actual/scale
        return dict(
            scope='fresh full original constrained native equations',
            residual_norm=actual, reference_residual_norm=scale,
            relative_residual=relative, relative_tolerance=rtol,
            passed=bool(relative <= rtol), global_size=int(saved.getSize()),
            outer_iterations_performed=0,
            historical_positive_reason_recovered=False,
            pressure_gauge_projection_applied=False,
        )
    finally:
        for vector in (saved, zero, residual, reference):
            vector.destroy()
