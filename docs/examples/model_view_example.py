#!/usr/bin/env python3
"""
Example demonstrating the Model.view() method functionality.

This example shows how to use the view() method to get concise summaries
of model contents with different levels of detail.
"""

import underworld3 as uw
import sympy as sp

def model_view_example():
    """Demonstrate Model.view() with different verbosity levels"""

    # Create a comprehensive model
    uw.reset_default_model()
    model = uw.get_default_model()

    print("Creating a thermal convection model...")

    # Create mesh
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1
    )

    # Create variables
    temperature = uw.discretisation.MeshVariable("temperature", mesh, 1, degree=2)
    velocity = uw.discretisation.MeshVariable("velocity", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("pressure", mesh, 1, degree=1)

    # Add materials with SymPy expressions
    x, y, Ra = sp.symbols('x y Ra')
    model.set_material('mantle', {
        'viscosity': 1e21,
        'density': 3300,
        'rayleigh_number': Ra,
        'temperature_profile': sp.sin(sp.pi * x) * sp.cos(sp.pi * y)
    })

    model.set_material('lithosphere', {
        'viscosity': 1e24,
        'density': 2800,
        'thermal_conductivity': 3.0
    })

    # Configure PETSc solver options
    model.set_petsc_option('ksp_type', 'fgmres')
    model.set_petsc_option('pc_type', 'fieldsplit')
    model.set_petsc_option('ksp_rtol', '1e-6')

    # Add metadata
    model.metadata['experiment'] = 'thermal_convection_benchmark'
    model.metadata['author'] = 'researcher'
    model.metadata['ra_target'] = 1e6

    print("\n" + "="*60)
    print("BASIC MODEL SUMMARY (verbose=0)")
    print("="*60)
    model.view()

    print("\n" + "="*60)
    print("DETAILED MODEL SUMMARY (verbose=1)")
    print("="*60)
    model.view(verbose=1)

    print("\n" + "="*60)
    print("COMPLETE MODEL SUMMARY (verbose=2, show_petsc=True)")
    print("="*60)
    model.view(verbose=2, show_petsc=True)

    return model

def specialized_config_example():
    """Demonstrate view() with ThermalConvectionConfig"""

    print("\n" + "="*60)
    print("SPECIALIZED CONFIGURATION MODEL")
    print("="*60)

    # Create specialized thermal convection configuration
    config = uw.ThermalConvectionConfig(
        rayleigh_number=1e5,
        cellsize=0.05,
        stokes_tolerance=1e-8,
        temperature_top=273.15,
        temperature_bottom=1873.15,
        stokes_solver_type='pcdksp'
    )

    # Create model from configuration
    model = uw.create_thermal_convection_model(config, "convection_benchmark")

    print("Model created from ThermalConvectionConfig:")
    model.view(verbose=1, show_petsc=True)

    return model

if __name__ == '__main__':
    print("Model.view() Method Examples")
    print("="*60)

    # Example 1: Comprehensive model
    model1 = model_view_example()

    # Example 2: Specialized configuration
    model2 = specialized_config_example()

    print("\n" + "="*60)
    print("USAGE SUMMARY")
    print("="*60)
    print("• model.view()                    - Basic summary")
    print("• model.view(verbose=1)           - Variable and material details")
    print("• model.view(verbose=2)           - Include solvers and metadata")
    print("• model.view(show_petsc=True)     - Show PETSc options")
    print("• model.view(show_materials=False) - Hide materials section")
    print("\nIn Jupyter notebooks, output will be formatted as rich Markdown.")
    print("In terminal/scripts, output falls back to clean text format.")
    print("="*60)