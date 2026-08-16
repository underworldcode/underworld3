r"""
Post-processing helpers for Underworld3 models.

The post-processing package contains derived diagnostics that are useful across
multiple model scripts but do not belong in a solver implementation.
"""

from . import geoid
from . import topography
from .geoid import spherical_shell_dynamic_response

__all__ = ["geoid", "topography", "spherical_shell_dynamic_response"]
