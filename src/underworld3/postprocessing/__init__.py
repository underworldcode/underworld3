r"""
Post-processing helpers for Underworld3 models.

The post-processing package contains derived diagnostics that are useful across
multiple model scripts but do not belong in a solver implementation.
"""

from . import geoid

__all__ = ["geoid"]
