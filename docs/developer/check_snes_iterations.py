#!/usr/bin/env python3
"""
Check SNES Iterations and Convergence

Research script to demonstrate how to detect if SNES took zero iterations
due to absolute tolerance issues with large physical values.
"""

import underworld3 as uw
import sympy
import numpy as np

def test_snes_diagnostics(use_scaling=False, problem_type="lid_driven"):
    """
    Test SNES iteration counting and convergence detection.

    Parameters:
    -----------
    use_scaling : bool
        Whether to use geological scaling
    problem_type : str
        "lid_driven" for guaranteed flow, "thermal" for buoyancy
    """

    if use_scaling:
        print(f"\n=== {problem_type.upper()} PROBLEM WITH SCALING ===")

        # Set geological scaling
        coeffs = uw.scaling.get_coefficients()
        coeffs["[length]"] = 1000 * uw.scaling.units.km
        coeffs["[time]"] = 1e6 * uw.scaling.units.year
        coeffs["[mass]"] = 1e24 * uw.scaling.units.kg

        # Reasonable physical parameters
        viscosity_physical = 1e21 * uw.scaling.units.Pa * uw.scaling.units.s
        velocity_physical = 5 * uw.scaling.units.cm / uw.scaling.units.year

        viscosity_nd = uw.scaling.non_dimensionalise(viscosity_physical)
        velocity_nd = uw.scaling.non_dimensionalise(velocity_physical)

        print(f"Scaled parameters:")
        print(f"  Viscosity: {viscosity_nd:.3e}")
        print(f"  Velocity: {velocity_nd:.3e}")

    else:
        print(f"\n=== {problem_type.upper()} PROBLEM WITHOUT SCALING ===")

        # Reset to SI defaults
        coeffs = uw.scaling.get_coefficients()
        coeffs["[length]"] = 1.0 * uw.scaling.units.meter
        coeffs["[time]"] = 1.0 * uw.scaling.units.second
        coeffs["[mass]"] = 1.0 * uw.scaling.units.kilogram

        # Large SI values
        viscosity_nd = 1e21  # Pa⋅s - Very large!
        velocity_nd = 1.58e-9  # m/s - Very small!

        print(f"SI parameters:")
        print(f"  Viscosity: {viscosity_nd:.0e} Pa⋅s")
        print(f"  Velocity: {velocity_nd:.2e} m/s")

    # Create mesh and variables
    mesh = uw.meshing.StructuredQuadBox(elementRes=(6, 6))
    x, y = mesh.X

    u = uw.discretisation.MeshVariable("velocity", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("pressure", mesh, 1, degree=1)

    # Create Stokes system
    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_nd

    if problem_type == "lid_driven":
        # Simple lid-driven cavity
        stokes.bodyforce = sympy.Matrix([0, 0])
        stokes.add_dirichlet_bc((velocity_nd, 0.0), "Top")
        stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
        stokes.add_dirichlet_bc((0.0, 0.0), "Left")
        stokes.add_dirichlet_bc((0.0, 0.0), "Right")

    elif problem_type == "thermal":
        # Thermal buoyancy
        temperature = uw.discretisation.MeshVariable("temperature", mesh, 1, degree=1)

        # Set temperature field
        with uw.synchronised_array_update():
            coords = mesh.data
            y_coords = coords[:, 1]
            T_bottom, T_top = 1600.0, 300.0
            T_dimensional = T_bottom - (T_bottom - T_top) * y_coords
            T_pint = T_dimensional * uw.scaling.units.kelvin
            T_nd = uw.scaling.non_dimensionalise(T_pint)
            temperature.array[:, 0, 0] = T_nd

        # Buoyancy parameters
        if use_scaling:
            rho0_physical = 3300.0 * uw.scaling.units.kg / uw.scaling.units.m**3
            alpha_physical = 3e-5 / uw.scaling.units.kelvin
            g_physical = 9.81 * uw.scaling.units.meter / uw.scaling.units.second**2

            rho0_nd = uw.scaling.non_dimensionalise(rho0_physical)
            alpha_nd = uw.scaling.non_dimensionalise(alpha_physical)
            g_nd = uw.scaling.non_dimensionalise(g_physical)
        else:
            rho0_nd = 3300.0  # kg/m³
            alpha_nd = 3e-5   # 1/K
            g_nd = 9.81       # m/s²

        density_expr = rho0_nd * (1 - alpha_nd * temperature.sym[0])
        buoyancy_force = -density_expr * g_nd * sympy.Matrix([0, 1])
        stokes.bodyforce = buoyancy_force

        # Free slip boundaries to allow flow
        stokes.add_dirichlet_bc((None, 0.0), "Bottom")
        stokes.add_dirichlet_bc((0.0, None), "Left")
        stokes.add_dirichlet_bc((0.0, None), "Right")
        stokes.add_dirichlet_bc((None, 0.0), "Top")

    # Solver settings - enable monitoring to see iterations
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["pc_type"] = "lu"

    # Enable SNES monitoring to see what happens
    stokes.petsc_options["snes_monitor"] = None  # Shows SNES iterations
    stokes.petsc_options["ksp_monitor"] = None   # Shows KSP iterations

    print("Solving...")

    try:
        stokes.solve()

        # ========== KEY DIAGNOSTICS ==========

        # Check convergence reason
        converged_reason = stokes.snes.getConvergedReason()
        converged = converged_reason > 0

        # Check iteration counts
        snes_iterations = stokes.snes.getIterationNumber()  # or stokes.snes.its
        linear_iterations = stokes.snes.getLinearSolveIterations()

        # Get tolerances
        rtol, atol, stol, maxit = stokes.snes.getTolerances()

        print(f"\n=== SNES DIAGNOSTICS ===")
        print(f"Converged: {converged}")
        print(f"Convergence reason: {converged_reason}")
        print(f"SNES iterations: {snes_iterations}")
        print(f"Linear iterations: {linear_iterations}")
        print(f"Tolerances - rtol: {rtol:.1e}, atol: {atol:.1e}, stol: {stol:.1e}")
        print(f"Max iterations: {maxit}")

        # Interpret convergence reason
        if converged_reason == 2:
            print("✓ Converged: Residual norm decreased by rtol")
        elif converged_reason == 3:
            print("✓ Converged: Residual norm less than atol")
        elif converged_reason == 4:
            print("✓ Converged: Step size less than stol")
        elif converged_reason <= 0:
            print(f"✗ Did not converge: reason {converged_reason}")

        # Check for zero iterations (the problem we're investigating)
        if snes_iterations == 0:
            print("⚠️  WARNING: SNES took ZERO iterations!")
            print("   This suggests the initial guess was already 'converged'")
            print("   due to absolute tolerance being too large for the problem scale")
        elif snes_iterations == 1:
            print("⚠️  SUSPICIOUS: SNES took only 1 iteration")
            print("   Check if absolute tolerance is appropriate for problem scale")
        else:
            print(f"✓ SNES took {snes_iterations} iterations - normal behavior")

        # Analyze solution
        u_magnitude = np.sqrt(u.array[:, 0, 0]**2 + u.array[:, 0, 1]**2)
        u_max = np.max(u_magnitude)
        u_rms = np.sqrt(np.mean(u_magnitude**2))

        print(f"\nSolution analysis:")
        print(f"  Max velocity (ND): {u_max:.3e}")
        print(f"  RMS velocity (ND): {u_rms:.3e}")

        if u_max == 0.0:
            print("⚠️  Zero velocity solution - check problem setup")

        return {
            'success': True,
            'converged': converged,
            'convergence_reason': converged_reason,
            'snes_iterations': snes_iterations,
            'linear_iterations': linear_iterations,
            'tolerances': {'rtol': rtol, 'atol': atol, 'stol': stol},
            'max_velocity': u_max,
            'scaling': 'geological' if use_scaling else 'default',
            'problem_type': problem_type
        }

    except Exception as e:
        print(f"✗ Solution failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'scaling': 'geological' if use_scaling else 'default',
            'problem_type': problem_type
        }

def main():
    """Test SNES diagnostics with different scaling approaches"""

    print("SNES Iteration and Convergence Diagnostics")
    print("=" * 60)

    results = {}

    # Test 1: Lid-driven cavity without scaling (should work)
    results['lid_default'] = test_snes_diagnostics(use_scaling=False, problem_type="lid_driven")

    # Test 2: Lid-driven cavity with scaling (should work better)
    results['lid_scaled'] = test_snes_diagnostics(use_scaling=True, problem_type="lid_driven")

    # Test 3: Thermal buoyancy without scaling (may have issues)
    results['thermal_default'] = test_snes_diagnostics(use_scaling=False, problem_type="thermal")

    # Test 4: Thermal buoyancy with scaling (should be better)
    results['thermal_scaled'] = test_snes_diagnostics(use_scaling=True, problem_type="thermal")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for name, result in results.items():
        if result['success']:
            its = result['snes_iterations']
            status = "⚠️  ZERO ITER" if its == 0 else f"✓ {its} iter"
            vel = result['max_velocity']
            print(f"{name:15}: {status:12} | vel={vel:.2e} | converged={result['converged']}")
        else:
            print(f"{name:15}: ✗ FAILED     | {result['error']}")

    return results

if __name__ == "__main__":
    results = main()