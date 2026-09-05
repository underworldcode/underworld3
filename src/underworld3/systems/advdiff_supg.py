"""Compatibility imports for pre-unification SUPG users.

The only implementation lives in advection_diffusion_eulerian.
"""

from .advection_diffusion_eulerian import (
    AdvDiffusionSUPGState,
    SNES_AdvectionDiffusion_SUPG as SNES_AdvectionDiffusionSUPG,
)

__all__ = ["AdvDiffusionSUPGState", "SNES_AdvectionDiffusionSUPG"]
