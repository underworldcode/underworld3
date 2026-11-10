"""Public snapshots and verification of finalized native Stokes equations."""
from dataclasses import dataclass
from copy import deepcopy
import hashlib

import numpy as np
from mpi4py import MPI
from underworld3.utilities.stokes_residual import evaluate_linear_stokes_residual


@dataclass(frozen=True)
class RotatedStokesDiagnostics:
    settings: dict[str, object]
    pressure: dict[str, object]
    preconditioner: dict[str, object]
    residual: dict[str, object]
    iterations: tuple[int, ...]


def solve_report_info(report, *, reason_namespace: str) -> dict[str, object]:
    """Read actual solver reasons, preserving the KSP/SNES namespace distinction."""
    if reason_namespace not in ('PETSc.KSP.ConvergedReason', 'PETSc.SNES.ConvergedReason'):
        raise ValueError('Use the PETSc KSP or SNES reason namespace')
    scope = ('last rotated KSP increment' if reason_namespace == 'PETSc.KSP.ConvergedReason'
             else 'native SNES solve')
    return dict(reason=int(report.reason), reason_name=str(report.reason_str),
                reason_namespace=reason_namespace, reason_scope=scope,
                converged=bool(report.converged))


def rotated_stokes_diagnostics(stokes, *, rtol: float) -> RotatedStokesDiagnostics:
    """Snapshot native rotated-solver data and verify its finalized full residual.

    This reads the original constrained-equation norms computed by the native
    solver after finalization. It never substitutes the projected residual for
    the full residual or assumes requested constant pressure is a null mode.
    """
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError('rtol must be finite and positive')
    info = stokes._rotated_freeslip_info
    settings = ('velocity_ksp_type', 'pressure_ksp_type', 'ksp_max_it')
    pressure = ('requested','verified','projection_active','attached','checks')
    preconditioner = ('outer_pc_type','retained_base_pc_type','pressure_coarse_correction_active','pressure_coarse_correction_metrics')
    required = set(settings+preconditioner+('ksp_its','fresh_residual_scope','fresh_unprojected_rnorm','fresh_projected_rnorm','fresh_reference_rnorm','fresh_unprojected_reference_rnorm'))
    required.update('pressure_nullspace_'+name for name in pressure)
    missing = required-info.keys()
    if missing:
        raise RuntimeError(f'Rotated Stokes diagnostics are incomplete: {sorted(missing)}')
    if info['fresh_residual_scope'] != 'normal rows excluded':
        raise RuntimeError('Rotated residual must describe original equations with only wall-normal rows constrained')
    actual, projected, reference = (float(info[name]) for name in (
        'fresh_unprojected_rnorm','fresh_projected_rnorm','fresh_unprojected_reference_rnorm'))
    if not np.all(np.isfinite([actual,projected,reference])) or reference <= 0:
        raise ArithmeticError('Native Stokes residuals require finite norms and positive forcing reference')
    relative = actual/reference
    if actual > rtol*reference:
        raise ArithmeticError(f'Fresh original native residual {relative} exceeds {rtol}')
    return RotatedStokesDiagnostics(
        settings={name:deepcopy(info[name]) for name in settings},
        pressure={name:deepcopy(info['pressure_nullspace_'+name]) for name in pressure},
        preconditioner={name:deepcopy(info[name]) for name in preconditioner},
        residual=dict(scope='full constrained native equations', evaluation='fresh after finalization, original constrained equations',
            projected_norm=projected, full_norm=actual, reference_norm=reference,
            projected_reference_norm=float(info['fresh_reference_rnorm']),
            pressure_gauge_norm=float(stokes._rotated_pressure_gauge_residual),
            relative=relative, passed=True),
        iterations=tuple(info['ksp_its']),
    )


def verify_linear_stokes_residual(stokes, *, rtol: float) -> dict:
    """Verify the original native callback and field-byte preservation on all ranks."""
    comm = stokes.mesh.dm.getComm().tompi4py()
    before = tuple(hashlib.sha256(np.asarray(var.array).tobytes()).hexdigest() for var in stokes.fields.values())
    result = evaluate_linear_stokes_residual(stokes, rtol=rtol)
    after = tuple(hashlib.sha256(np.asarray(var.array).tobytes()).hexdigest() for var in stokes.fields.values())
    if comm.allreduce(before != after, op=MPI.LOR):
        raise ArithmeticError('Original residual verification changed the solved fields')
    if not result['passed']:
        raise ArithmeticError(f'Fresh original linear Stokes residual fails: {result}')
    return result
