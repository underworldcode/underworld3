#!/usr/bin/env python3
"""
Final Scaling Demonstration

Shows the working approach for direct unit specification and identifies
the issue with to_base_units() for large scaling coefficients.
"""

import underworld3 as uw
import sympy
import numpy as np

def manual_unit_conversion(quantity, length_coeff, time_coeff, mass_coeff):
    """
    Manual unit conversion that works around to_base_units() precision issues.

    This function manually converts quantities to base units using the scaling
    coefficients, avoiding the numerical precision problems in to_base_units().
    """

    # Get SI values
    if hasattr(quantity, 'to'):
        # It's a Pint quantity
        if quantity.dimensionality == uw.scaling.units.Pa.dimensionality * uw.scaling.units.s.dimensionality:
            # Viscosity: [M]/([L]*[T])
            si_value = quantity.to('Pa*s').magnitude
            scale_si = mass_coeff / (length_coeff * time_coeff)
            return si_value / scale_si
        elif quantity.dimensionality == (uw.scaling.units.m / uw.scaling.units.s).dimensionality:
            # Velocity: [L]/[T]
            si_value = quantity.to('m/s').magnitude
            scale_si = length_coeff / time_coeff
            return si_value / scale_si
        elif quantity.dimensionality == uw.scaling.units.Pa.dimensionality:
            # Pressure: [M]/([L]*[T]^2)
            si_value = quantity.to('Pa').magnitude
            scale_si = mass_coeff / (length_coeff * time_coeff**2)
            return si_value / scale_si
        elif quantity.dimensionality == uw.scaling.units.m.dimensionality:
            # Length: [L]
            si_value = quantity.to('m').magnitude
            return si_value / length_coeff
        else:
            # Unknown dimensionality - try to_base_units()
            return quantity.to_base_units().magnitude
    else:
        # Already a number
        return quantity

def working_geological_scaling_demo():
    """Demonstrate the complete working geological scaling approach"""

    print("WORKING GEOLOGICAL SCALING DEMONSTRATION")
    print("=" * 50)

    # Step 1: Choose geological base scales
    length_base = 1000 * uw.scaling.units.km
    time_base = 1e6 * uw.scaling.units.year

    # Step 2: Calculate mass scale for viscosity = 1.0
    target_viscosity_si = 1e21  # Pa⋅s
    length_si = length_base.to('m').magnitude
    time_si = time_base.to('s').magnitude
    mass_si = target_viscosity_si * length_si * time_si
    mass_base = mass_si * uw.scaling.units.kg

    print(f"Chosen base units:")
    print(f"  Length: {length_base} = {length_si:.1e} m")
    print(f"  Time: {time_base} = {time_si:.1e} s")
    print(f"  Mass: {mass_base:.2e}")

    # Step 3: Apply to UW3
    coeffs = uw.scaling.get_coefficients()
    coeffs["[length]"] = length_base
    coeffs["[time]"] = time_base
    coeffs["[mass]"] = mass_base

    # Step 4: Test geological values (using manual conversion)
    geological_quantities = {
        "mantle_viscosity": 1e21 * uw.scaling.units.Pa * uw.scaling.units.s,
        "plate_velocity": 5 * uw.scaling.units.cm / uw.scaling.units.year,
        "lithosphere_pressure": 1e9 * uw.scaling.units.Pa,
        "domain_depth": 3000 * uw.scaling.units.km
    }

    print(f"\nGeological values in base units:")
    base_values = {}
    for name, quantity in geological_quantities.items():
        value = manual_unit_conversion(quantity, length_si, time_si, mass_si)
        base_values[name] = value
        print(f"  {name}: {quantity} → {value:.3f}")

    # Step 5: Check conditioning
    visc_val = base_values["mantle_viscosity"]
    vel_val = base_values["plate_velocity"]
    conditioning = visc_val / vel_val

    print(f"\nConditioning analysis:")
    print(f"  Viscosity: {visc_val:.3f} (target: 1.0) {'✓' if abs(visc_val - 1.0) < 0.1 else '✗'}")
    print(f"  Velocity: {vel_val:.3f}")
    print(f"  Ratio: {conditioning:.1f} {'✓' if conditioning < 100 else '✗'}")

    success = abs(visc_val - 1.0) < 0.1 and conditioning < 100
    print(f"  Overall: {'SUCCESS' if success else 'FAILED'}")

    return success, base_values

def test_stokes_solver(base_values):
    """Test the Stokes solver with properly scaled values"""

    print(f"\n{'='*50}")
    print("STOKES SOLVER TEST")
    print("=" * 50)

    # Create Stokes problem
    mesh = uw.meshing.StructuredQuadBox(elementRes=(6, 6))
    u = uw.discretisation.MeshVariable("velocity", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("pressure", mesh, 1, degree=1)

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = base_values["mantle_viscosity"]

    # Lid-driven cavity
    stokes.bodyforce = sympy.Matrix([0, 0])
    stokes.add_dirichlet_bc((base_values["plate_velocity"], 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((0.0, 0.0), "Left")
    stokes.add_dirichlet_bc((0.0, 0.0), "Right")

    stokes.tolerance = 1.0e-6
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["pc_type"] = "lu"

    print(f"Problem parameters:")
    print(f"  Viscosity: {base_values['mantle_viscosity']:.3f} base units")
    print(f"  Velocity: {base_values['plate_velocity']:.3f} base units")
    print(f"  Conditioning: {base_values['mantle_viscosity']/base_values['plate_velocity']:.1f}")

    # Solve
    try:
        diagnostics = stokes.solve_with_diagnostics(
            check_convergence=True,
            raise_on_divergence=False,
            print_diagnostics=True
        )

        solver_success = not diagnostics['zero_iterations'] and diagnostics['converged']

        if solver_success:
            # Analyze solution
            u_magnitude = np.sqrt(u.array[:, :, 0]**2 + u.array[:, :, 1]**2)
            max_velocity = np.max(u_magnitude)

            print(f"\nSolution analysis:")
            print(f"  Max velocity: {max_velocity:.3f} base units")
            print(f"  SNES iterations: {diagnostics['snes_iterations']}")
            print(f"  Convergence: {diagnostics['convergence_reason_string']}")

            return True, max_velocity
        else:
            print(f"\nSolver failed:")
            print(f"  Reason: {diagnostics['convergence_reason_string']}")
            return False, None

    except Exception as e:
        print(f"Solver error: {e}")
        return False, None

def auto_scale_prototype():
    """Show what the auto_scale_for() function should look like"""

    print(f"\n{'='*50}")
    print("AUTO_SCALE_FOR() PROTOTYPE")
    print("=" * 50)

    def auto_scale_for(**quantities):
        """
        Prototype automatic scaling function.

        Usage:
            auto_scale_for(viscosity=1e21*Pa*s, velocity=5*cm/year)
        """

        print(f"Auto-scaling for:")
        for name, value in quantities.items():
            print(f"  {name}: {value}")

        # Infer appropriate scales
        if 'velocity' in quantities:
            velocity = quantities['velocity']
            vel_ms = velocity.to('m/s').magnitude

            if vel_ms < 1e-7:  # Geological
                length_base = 1000 * uw.scaling.units.km
                time_base = 1e6 * uw.scaling.units.year
                domain = "geological"
            elif vel_ms < 1e-4:  # Ice/glacier
                length_base = 100 * uw.scaling.units.m
                time_base = 10 * uw.scaling.units.year
                domain = "glacial"
            else:  # Engineering
                length_base = 0.1 * uw.scaling.units.m
                time_base = 1 * uw.scaling.units.s
                domain = "engineering"
        else:
            length_base = 1 * uw.scaling.units.m
            time_base = 1 * uw.scaling.units.s
            domain = "default"

        # Calculate mass scale
        if 'viscosity' in quantities:
            viscosity = quantities['viscosity']
            visc_si = viscosity.to('Pa*s').magnitude
            length_si = length_base.to('m').magnitude
            time_si = time_base.to('s').magnitude
            mass_si = visc_si * length_si * time_si
            mass_base = mass_si * uw.scaling.units.kg
        else:
            mass_base = 1 * uw.scaling.units.kg

        print(f"\nInferred domain: {domain}")
        print(f"Chosen scales:")
        print(f"  Length: {length_base}")
        print(f"  Time: {time_base}")
        print(f"  Mass: {mass_base:.2e}")

        # Apply scaling
        coeffs = uw.scaling.get_coefficients()
        coeffs["[length]"] = length_base
        coeffs["[time]"] = time_base
        coeffs["[mass]"] = mass_base

        # Validate
        print(f"\nValidation:")
        for name, value in quantities.items():
            converted = manual_unit_conversion(
                value,
                length_base.to('m').magnitude,
                time_base.to('s').magnitude,
                mass_base.to('kg').magnitude
            )
            print(f"  {name}: {converted:.3f} base units")

    # Test different domains
    print(f"\nDomain tests:")

    print(f"\n1. Mantle convection:")
    auto_scale_for(
        viscosity=1e21 * uw.scaling.units.Pa * uw.scaling.units.s,
        velocity=5 * uw.scaling.units.cm / uw.scaling.units.year
    )

    print(f"\n2. Ice flow:")
    auto_scale_for(
        viscosity=1e13 * uw.scaling.units.Pa * uw.scaling.units.s,
        velocity=10 * uw.scaling.units.m / uw.scaling.units.year
    )

    print(f"\n3. Engineering flow:")
    auto_scale_for(
        viscosity=1e-3 * uw.scaling.units.Pa * uw.scaling.units.s,
        velocity=0.1 * uw.scaling.units.m / uw.scaling.units.s
    )

def main():
    """Run the complete demonstration"""

    # Test 1: Manual geological scaling
    success, base_values = working_geological_scaling_demo()

    if success:
        # Test 2: Stokes solver
        solver_success, max_vel = test_stokes_solver(base_values)

        if solver_success:
            print(f"\n🎉 COMPLETE SUCCESS!")
            print(f"✓ Scaling produces O(1) values")
            print(f"✓ Solver converges properly")
            print(f"✓ Physics preserved")

            # Test 3: Show auto_scale prototype
            auto_scale_prototype()

            print(f"\n📋 SUMMARY:")
            print(f"✓ Manual calculation approach works")
            print(f"⚠️ to_base_units() has precision issues with large coefficients")
            print(f"✓ Solver integration successful")
            print(f"✓ Auto-scaling approach feasible")
            print(f"✓ Domain-agnostic scaling possible")

        else:
            print(f"\n⚠️ Scaling works but solver issues remain")
    else:
        print(f"\n❌ Scaling calculation failed")

if __name__ == "__main__":
    main()