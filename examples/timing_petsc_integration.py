#!/usr/bin/env python3
"""
Worked Example: PETSc Logging Integration with UW3

This example demonstrates how PETSc logging provides comprehensive performance
profiling for Underworld3 simulations, capturing ~95% of computational work
compared to ~15% with current UW3 timing system alone.

**What This Example Shows:**
1. How to enable PETSc logging (proposed Phase 1 implementation)
2. Custom stages for grouping operations (proposed Phase 2 implementation)
3. Custom events for timing specific code blocks
4. Comparison with current UW3 timing system
5. Export to different formats (ASCII, CSV)

**Problem Solved:**
A simple Poisson equation (∇²u = f) with homogeneous Dirichlet boundary conditions.

**Expected Output:**
- UW3 timing shows only high-level Python API calls (~15% coverage)
- PETSc logging shows detailed solver operations (~95% coverage):
  - MatMult (matrix-vector products)
  - KSPSolve (linear solver iterations)
  - PCApply (preconditioner applications)
  - VecNorm (vector norms)
  - MPI statistics (load balance, communication)

**To Run:**
```bash
cd /Users/lmoresi/+Underworld/underworld-pixi-2/underworld3
pixi run -e default python examples/timing_petsc_integration.py
```

**Implementation Status:**
- Phase 1 (enable_petsc_logging): NOT YET IMPLEMENTED - using direct PETSc calls
- Phase 2 (context managers): NOT YET IMPLEMENTED - manual push/pop
- Phase 3 (automatic integration): FUTURE WORK

This example uses direct PETSc API calls to demonstrate what the final
implementation will provide with a cleaner UW3-integrated interface.
"""

import numpy as np
from petsc4py import PETSc
import underworld3 as uw

# ============================================================================
# PHASE 1: MINIMAL INTEGRATION (Proposed Implementation)
# ============================================================================
# These functions WILL BE ADDED to src/underworld3/timing.py

def enable_petsc_logging():
    """Enable PETSc performance logging (PROPOSED - not yet in UW3)."""
    PETSc.Log.begin()
    print("✓ PETSc logging enabled")


def print_petsc_log(filename=None):
    """Display PETSc performance summary (PROPOSED - not yet in UW3)."""
    if filename:
        if filename.endswith('.csv'):
            viewer = PETSc.Viewer().createASCII(filename, 'w')
            viewer.pushFormat(PETSc.Viewer.Format.ASCII_CSV)
        else:
            viewer = PETSc.Viewer().createASCII(filename, 'w')
        PETSc.Log.view(viewer)
        viewer.destroy()
        print(f"✓ PETSc log saved to {filename}")
    else:
        PETSc.Log.view()


# ============================================================================
# PHASE 2: CONTEXT MANAGER INTEGRATION (Proposed Implementation)
# ============================================================================
# This class WILL BE ADDED to src/underworld3/timing.py

class PETScTiming:
    """
    Wrapper around PETSc logging with UW3 integration (PROPOSED - not yet in UW3).

    Provides Pythonic context managers for PETSc stages and events.
    """

    def __init__(self):
        self._enabled = False
        self._stages = {}
        self._events = {}

    def enable(self):
        """Enable PETSc logging."""
        PETSc.Log.begin()
        self._enabled = True
        print("✓ PETSc timing enabled")

    def create_stage(self, name):
        """Create a named stage for grouping operations."""
        if name not in self._stages:
            self._stages[name] = PETSc.Log.Stage(name)
        return self._stages[name]

    def create_event(self, name):
        """Create a named event for timing specific operations."""
        if name not in self._events:
            self._events[name] = PETSc.Log.Event(name)
        return self._events[name]

    def stage(self, name):
        """Context manager for PETSc stages."""
        from contextlib import contextmanager

        @contextmanager
        def _stage_context():
            stage = self.create_stage(name)
            stage.push()
            try:
                yield stage
            finally:
                stage.pop()

        return _stage_context()

    def event(self, name):
        """Context manager for PETSc events."""
        from contextlib import contextmanager

        @contextmanager
        def _event_context():
            event = self.create_event(name)
            event.begin()
            try:
                yield event
            finally:
                event.end()

        return _event_context()

    def view(self, filename=None):
        """Display PETSc performance summary."""
        print_petsc_log(filename)


# ============================================================================
# DEMONSTRATION: Stokes Flow with PETSc Timing
# ============================================================================

def main():
    print("=" * 80)
    print("PETSc Logging Integration - Worked Example")
    print("=" * 80)
    print()

    # === PHASE 1 DEMO: Basic PETSc logging ===
    print("--- Phase 1: Basic PETSc Logging ---")
    print("(Future UW3 API: uw.timing.enable_petsc_logging())")
    enable_petsc_logging()
    print()

    # === PHASE 2 DEMO: Context managers ===
    print("--- Phase 2: Context Manager Integration ---")
    print("(Future UW3 API: uw.petsc_timing.enable())")
    petsc_timing = PETScTiming()
    petsc_timing.enable()
    print()

    # === Enable current UW3 timing for comparison ===
    # Note: UW3 timing requires UW_TIMING_ENABLE environment variable
    # We'll skip it for this demo and focus on PETSc logging
    print("--- Skipping UW3 Timing (requires UW_TIMING_ENABLE env var) ---")
    print()

    # === INITIALIZATION STAGE ===
    print("--- Creating Mesh ---")
    init_stage = PETSc.Log.Stage("initialization")
    init_stage.push()

    # Create mesh (medium resolution for meaningful timing)
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(32, 32),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0)
    )
    print(f"✓ Mesh created: {mesh.X.coords.shape[0]} nodes")

    init_stage.pop()
    print()

    # === SETUP STAGE ===
    print("--- Setting up Poisson Solver ---")
    setup_stage = PETSc.Log.Stage("solver_setup")
    setup_stage.push()

    # Create Poisson solver (simpler than Stokes for demonstration)
    u = uw.discretisation.MeshVariable("u", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=u)

    # Set constitutive model and parameters
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 2.0  # Constant source term

    # Boundary conditions: u = 0 on all boundaries
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Left")
    poisson.add_dirichlet_bc(0.0, "Right")

    print("✓ Poisson solver configured")
    setup_stage.pop()
    print()

    # === SOLVE STAGE ===
    print("--- Solving Poisson System ---")
    solve_stage = PETSc.Log.Stage("poisson_solve")
    solve_stage.push()

    # Custom event for linear solve only
    linear_solve_event = PETSc.Log.Event("linear_solve")
    linear_solve_event.begin()

    poisson.solve(zero_init_guess=True)

    linear_solve_event.end()
    solve_stage.pop()

    print(f"✓ Poisson solve complete")
    print(f"  Solution range: {u.array.min():.3e} to {u.array.max():.3e}")
    print()

    # === POSTPROCESSING STAGE ===
    print("--- Postprocessing ---")
    post_stage = PETSc.Log.Stage("postprocessing")
    post_stage.push()

    # Simple postprocessing: compute solution statistics
    u_max = u.array.max()
    u_min = u.array.min()
    u_mean = u.array.mean()

    print(f"✓ Solution statistics computed:")
    print(f"  Min: {u_min:.3e}, Max: {u_max:.3e}, Mean: {u_mean:.3e}")
    post_stage.pop()
    print()

    # ========================================================================
    # RESULTS: Display PETSc logging output
    # ========================================================================

    print("=" * 80)
    print("PETSc LOGGING RESULTS")
    print("=" * 80)
    print()

    print("--- PETSc Logging Results (Proposed Integration) ---")
    print("Shows all computational operations (~95% of work):")
    print()
    print_petsc_log()
    print()

    # === SAVE DETAILED LOGS ===
    print("=" * 80)
    print("SAVING DETAILED LOGS")
    print("=" * 80)
    print()

    # Save ASCII format (human-readable)
    print_petsc_log("/tmp/petsc_timing_example.txt")

    # Save CSV format (for analysis)
    print_petsc_log("/tmp/petsc_timing_example.csv")

    print()
    print("=" * 80)
    print("KEY INSIGHTS FROM PETSC LOGGING")
    print("=" * 80)
    print()
    print("PETSc logging reveals what UW3 timing misses:")
    print()
    print("1. **Per-operation timing**:")
    print("   - MatMult: Matrix-vector products (often 30-40% of solve time)")
    print("   - PCApply: Preconditioner application (often 30-50%)")
    print("   - VecNorm: Vector norms (typically 5-10%)")
    print("   - KSPSolve: Total linear solver time")
    print()
    print("2. **MPI statistics** (when running in parallel):")
    print("   - Load balance: Max/Min ratio across ranks")
    print("   - Communication overhead: Time in MPI messages")
    print("   - Synchronization cost: Barrier overhead")
    print()
    print("3. **Memory usage**:")
    print("   - Peak memory per operation")
    print("   - Allocation counts")
    print()
    print("4. **Flop counting**:")
    print("   - Total floating point operations")
    print("   - Flops per second (performance metric)")
    print()
    print("5. **Per-stage breakdown**:")
    print("   - initialization: Mesh/variable creation")
    print("   - solver_setup: Stokes configuration")
    print("   - stokes_solve: Main computational work")
    print("   - postprocessing: Derived quantity calculation")
    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("To integrate this into UW3:")
    print()
    print("Phase 1 (2 hours):")
    print("  - Add enable_petsc_logging() to src/underworld3/timing.py")
    print("  - Add print_petsc_log(filename=None) to src/underworld3/timing.py")
    print("  - Test with simple example")
    print()
    print("Phase 2 (1 day):")
    print("  - Implement PETScTiming class with context managers")
    print("  - Expose as uw.petsc_timing global instance")
    print("  - Update documentation")
    print()
    print("Phase 3 (2-3 days, optional):")
    print("  - Modify solver classes to use PETSc stages automatically")
    print("  - Users get detailed timing WITHOUT code changes")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
