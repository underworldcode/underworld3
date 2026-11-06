"""Public numerical diagnostics and opt-in spherical Stokes observations."""
from .runtime import collective_file_sha256, runtime_info
from .observer import (
    SphericalSolveState, SphericalStokesObserver, observe_spherical_stokes,
    solve_with_observer, array_sha256, assign_spherical_fields,
    prepare_spherical_replay_fields,
)
from .spherical import QuadratureOrders, spherical_stokes_errors, spherical_stokes_field_differences
from .spherical_reactions import HarmonicTarget, spherical_stokes_reactions
from .stokes import rotated_stokes_diagnostics, verify_linear_stokes_residual, solve_report_info
