"""
Underworld3 Model Architecture

This module provides the Model class that serves as a central orchestrator for 
simulation components and object lifecycle management.

The Model object:
- Eliminates circular references by becoming the central authority for mesh/swarms
- Enables mesh swapping with automatic notification to dependent objects
- Provides simple container for organizing simulation components
- Supports serialization for model reuse and sharing

Design Philosophy:
- Simple orchestration, not complex validation
- Use existing sympy expressions, not separate parameter system
- PETSc handles command-line options
- Materials are just dictionaries of expressions/values
"""

import weakref
import json
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path
import sympy
from datetime import datetime

# Pydantic imports for enhanced functionality
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
import yaml

# Import the Pint-native implementation
import os
import sys
sys.path.append(os.path.dirname(__file__))

try:
    from pint_model_implementation import PintNativeModelMixin
except ImportError:
    # Create a minimal version if import fails
    class PintNativeModelMixin:
        pass


class ModelState(Enum):
    """Model lifecycle states"""
    INITIALIZING = "initializing"
    CONFIGURED = "configured" 
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class Model(PintNativeModelMixin, BaseModel):
    """
    Central orchestrator for Underworld3 simulations.

    Enhanced with Pydantic for validation, serialization, and configuration management
    while preserving the original design philosophy of simple orchestration.

    The Model object manages:
    - Mesh and coordinate system lifecycle
    - Swarm registration and migration
    - Variable dependencies and updates
    - Parameter validation and propagation
    - Material definitions and assignments
    - Solver coordination

    Benefits:
    - Eliminates circular references between components
    - Enables mesh swapping and dynamic reconfiguration
    - Provides single point for parameter management
    - Supports model composition and reuse
    - Enhanced serialization with YAML/JSON support
    - Optional validation with Pydantic

    Example:
        >>> model = uw.Model()
        >>> model.set_mesh(mesh)
        >>> swarm = model.create_swarm()
        >>> temperature = model.add_variable("temperature", mesh, uw.VarType.SCALAR)
    """

    # Pydantic fields with validation
    name: str = Field(default_factory=lambda: f"Model_{id(object())}")
    state: ModelState = ModelState.CONFIGURED
    materials: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=0, description="Model version for change tracking")

    # NEW: Optional PETSc state for complete reproducibility
    petsc_state: Optional[Dict[str, str]] = None

    model_config = ConfigDict(
        # Allow complex objects like PETSc objects, weak references
        arbitrary_types_allowed=True,
        # Validate on assignment for immediate error catching
        validate_assignment=True,
        # Allow extra attributes for extensibility
        extra="allow"
    )

    # Declare private attributes for Pydantic v2
    _meshes: Any = PrivateAttr(default_factory=dict)
    _primary_mesh_id: Optional[int] = PrivateAttr(default=None)
    _swarms: Any = PrivateAttr(default_factory=dict)
    _variables: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _solvers: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, name: Optional[str] = None, **kwargs):
        """
        Initialize a new Model instance.

        Parameters:
        -----------
        name : str, optional
            Human-readable name for this model instance
        **kwargs : dict
            Additional arguments for Pydantic BaseModel
        """
        # Handle name generation before calling super().__init__
        if name is not None:
            kwargs['name'] = name

        super().__init__(**kwargs)

        # Set initial state if not provided
        if self.state == ModelState.CONFIGURED:
            # Transition through initializing to configured
            self.state = ModelState.INITIALIZING
            self.state = ModelState.CONFIGURED
    
    @property
    def mesh(self):
        """The primary mesh for this model"""
        if self._primary_mesh_id is None:
            return None
        return self._meshes.get(self._primary_mesh_id)

    def _register_mesh(self, mesh):
        """
        Internal method to register a mesh with this model.
        Called automatically from Mesh.__init__

        Parameters:
        -----------
        mesh : uw.discretisation.Mesh
            Mesh instance to register
        """
        mesh_id = id(mesh)
        self._meshes[mesh_id] = mesh

        # Set as primary mesh if no primary mesh exists
        if self._primary_mesh_id is None:
            self._primary_mesh_id = mesh_id

        self.version += 1

    def list_meshes(self) -> Dict[int, Any]:
        """List all meshes registered with this model"""
        return dict(self._meshes)

    def get_mesh(self, mesh_id: int):
        """Get a mesh by ID from the model registry"""
        return self._meshes.get(mesh_id)

    def set_primary_mesh(self, mesh):
        """
        Set a mesh as the primary mesh for this model.

        Parameters:
        -----------
        mesh : uw.discretisation.Mesh
            Mesh to set as primary (must already be registered)
        """
        mesh_id = id(mesh)
        if mesh_id not in self._meshes:
            # Register the mesh first
            self._register_mesh(mesh)
        else:
            # Just update the primary mesh pointer
            self._primary_mesh_id = mesh_id
            self.version += 1
    
    def _register_swarm(self, swarm):
        """
        Internal method to register a swarm with this model.
        Called automatically from Swarm.__init__
        
        Parameters:
        -----------
        swarm : uw.swarm.Swarm
            Swarm instance to register
        """
        swarm_id = id(swarm)
        self._swarms[swarm_id] = swarm
    
    def _register_variable(self, name, variable):
        """
        Internal method to register a variable with this model.
        Called automatically from MeshVariable/SwarmVariable.__init__

        Handles namespace collision by creating fully qualified names for variables
        on different meshes/swarms but with the same symbolic name.

        Parameters:
        -----------
        name : str
            Variable name (may not be unique across meshes/swarms)
        variable : MeshVariable or SwarmVariable
            Variable instance to register
        """
        # For SwarmVariables, ensure we keep strong reference to swarm to prevent garbage collection
        from .swarm import SwarmVariable
        if isinstance(variable, SwarmVariable):
            swarm = variable.swarm  # This will raise if swarm already garbage collected
            swarm_id = id(swarm)
            # Ensure swarm is registered with strong reference
            if swarm_id not in self._swarms:
                self._register_swarm(swarm)

        # Check for namespace collision
        if name in self._variables:
            existing_var = self._variables[name]
            # If same variable object, nothing to do
            if existing_var is variable:
                return

            # If different variable with same name, create qualified names
            # Use attribute detection instead of isinstance for wrapper compatibility

            # Determine variable types by available attributes
            existing_is_mesh = hasattr(existing_var, 'mesh') and not hasattr(existing_var, 'swarm')
            existing_is_swarm = hasattr(existing_var, 'swarm')
            variable_is_mesh = hasattr(variable, 'mesh') and not hasattr(variable, 'swarm')
            variable_is_swarm = hasattr(variable, 'swarm')

            if existing_is_mesh and variable_is_mesh:
                # Both mesh variables - qualify by mesh ID
                existing_qualified = f"{name}_mesh{id(existing_var.mesh)}"
                new_qualified = f"{name}_mesh{id(variable.mesh)}"

                # Move existing variable to qualified name
                self._variables[existing_qualified] = existing_var
                self._variables[new_qualified] = variable

                # Keep original name pointing to most recent (backward compatibility)
                self._variables[name] = variable

            elif existing_is_swarm and variable_is_swarm:
                # Both swarm variables - qualify by swarm ID
                existing_qualified = f"{name}_swarm{id(existing_var.swarm)}"
                new_qualified = f"{name}_swarm{id(variable.swarm)}"

                # Move existing variable to qualified name
                self._variables[existing_qualified] = existing_var
                self._variables[new_qualified] = variable

                # Keep original name pointing to most recent (backward compatibility)
                self._variables[name] = variable

                # Ensure both swarms are strongly referenced
                if existing_is_swarm:
                    existing_swarm = existing_var.swarm
                    if id(existing_swarm) not in self._swarms:
                        self._register_swarm(existing_swarm)

            else:
                # Mixed types (mesh vs swarm) - qualify by type and container ID
                if existing_is_mesh:
                    existing_qualified = f"{name}_mesh{id(existing_var.mesh)}"
                else:
                    existing_qualified = f"{name}_swarm{id(existing_var.swarm)}"
                    # Ensure existing swarm is strongly referenced
                    existing_swarm = existing_var.swarm
                    if id(existing_swarm) not in self._swarms:
                        self._register_swarm(existing_swarm)

                if variable_is_mesh:
                    new_qualified = f"{name}_mesh{id(variable.mesh)}"
                else:
                    new_qualified = f"{name}_swarm{id(variable.swarm)}"
                    # Swarm already handled above for SwarmVariable

                # Move existing variable to qualified name
                self._variables[existing_qualified] = existing_var
                self._variables[new_qualified] = variable

                # Keep original name pointing to most recent (backward compatibility)
                self._variables[name] = variable
        else:
            # No collision, simple registration
            self._variables[name] = variable
    
    def _register_solver(self, name, solver):
        """
        Internal method to register a solver with this model.
        Called automatically from solver.__init__
        
        Parameters:
        -----------
        name : str
            Solver name/identifier
        solver : Solver instance
            Solver to register
        """
        self._solvers[name] = solver
    
    def set_mesh(self, mesh):
        """
        Set or replace the primary mesh for this model.

        This method handles:
        - Registering mesh if not already registered
        - Setting as primary mesh
        - Incrementing version for change tracking

        Parameters:
        -----------
        mesh : uw.discretisation.Mesh
            The new mesh to use as primary for this model
        """
        old_primary_id = self._primary_mesh_id
        self.set_primary_mesh(mesh)

        if old_primary_id != self._primary_mesh_id:
            print(f"Model '{self.name}': Replaced mesh (version {self.version})")

        # TODO: Implement mesh change notifications
        # - Notify swarms of mesh change
        # - Update variable dependencies
        # - Invalidate solver caches

    def _update_mesh_for_swarm(self, swarm, new_mesh):
        """
        Internal method to coordinate mesh handover for a specific swarm.

        This method handles the model-level coordination when a swarm
        is assigned to a new mesh via swarm.mesh = new_mesh.

        Parameters
        ----------
        swarm : uw.swarm.Swarm
            Swarm being reassigned to new mesh
        new_mesh : uw.discretisation.Mesh
            New mesh to assign

        Notes
        -----
        This is called from swarm.mesh setter and handles:
        - Unregistering swarm from old mesh
        - Updating model's mesh reference
        - Registering swarm with new mesh
        """
        old_mesh = self.mesh

        # Unregister swarm from old mesh
        if old_mesh is not None:
            try:
                old_mesh.unregister_swarm(swarm)
            except (AttributeError, ValueError):
                # Mesh may not support swarm registration or swarm not registered
                pass

        # Update model's mesh reference
        self._register_mesh(new_mesh)

        # Register swarm with new mesh
        new_mesh.register_swarm(swarm)

        print(f"Model '{self.name}': Swarm reassigned to new mesh")

    def get_variable(self, name: str, mesh=None, swarm=None):
        """
        Get a variable by name from the model registry.

        Parameters:
        -----------
        name : str
            Variable name to look up
        mesh : Mesh, optional
            If provided, look for variable specifically on this mesh
        swarm : Swarm, optional
            If provided, look for variable specifically on this swarm

        Returns:
        --------
        Variable or None
        """
        # Try exact name first
        if name in self._variables:
            variable = self._variables[name]

            # If mesh/swarm specified, verify it matches
            if mesh is not None:
                from .discretisation import MeshVariable
                if isinstance(variable, MeshVariable) and variable.mesh is mesh:
                    return variable
                # Look for qualified name
                qualified_name = f"{name}_mesh{id(mesh)}"
                return self._variables.get(qualified_name)

            if swarm is not None:
                from .swarm import SwarmVariable
                if isinstance(variable, SwarmVariable) and variable.swarm is swarm:
                    return variable
                # Look for qualified name
                qualified_name = f"{name}_swarm{id(swarm)}"
                return self._variables.get(qualified_name)

            # No specific container requested, return whatever we found
            return variable

        # Try qualified names if exact name not found
        if mesh is not None:
            qualified_name = f"{name}_mesh{id(mesh)}"
            return self._variables.get(qualified_name)

        if swarm is not None:
            qualified_name = f"{name}_swarm{id(swarm)}"
            return self._variables.get(qualified_name)

        return None

    def get_qualified_name(self, variable):
        """
        Get the fully qualified name for a variable.

        Returns the name that uniquely identifies this variable in the model registry,
        which may include mesh/swarm ID qualifiers to resolve namespace conflicts.

        Parameters:
        -----------
        variable : MeshVariable or SwarmVariable
            Variable to get qualified name for

        Returns:
        --------
        str or None : Qualified name if found, None otherwise
        """
        # Search through all registered names
        for name, registered_var in self._variables.items():
            if registered_var is variable:
                return name
        return None

    def transfer_variable_data(self, source_var, target_var):
        """
        Transfer data from source variable to target variable using global_evaluate.

        This enables persistent variable identity across mesh changes by transferring
        data from variables on old meshes to variables on new meshes.

        Parameters:
        -----------
        source_var : MeshVariable or SwarmVariable
            Source variable to transfer data from
        target_var : MeshVariable or SwarmVariable
            Target variable to transfer data to

        Returns:
        --------
        bool : True if transfer successful, False otherwise
        """
        try:
            # Import global_evaluate for mesh-to-mesh transfer
            import underworld3 as uw

            # Get target coordinates based on variable type
            from .discretisation import MeshVariable
            from .swarm import SwarmVariable

            # Use attribute detection for wrapper compatibility
            if hasattr(target_var, 'coords') and hasattr(target_var, 'mesh'):
                # Mesh variable (direct or wrapped)
                target_coords = target_var.coords
            elif hasattr(target_var, 'swarm'):
                # Swarm variable
                target_coords = target_var.swarm.points
            else:
                raise ValueError(f"Unsupported target variable type: {type(target_var)}")

            # Perform the transfer using global_evaluate
            # This handles arbitrary mesh decompositions and refinement
            with uw.synchronised_array_update("persistent variable transfer"):
                target_var.array[...] = uw.function.evaluate(
                    source_var.sym, target_coords
                )

            return True

        except Exception as e:
            print(f"Warning: Failed to transfer data from {source_var.name} to {target_var.name}: {e}")
            return False

    def list_variables(self) -> Dict[str, Any]:
        """List all variables registered with this model"""
        return dict(self._variables)
    
    def list_swarms(self) -> Dict[str, Any]:
        """List all swarms registered with this model"""
        return dict(self._swarms)
    
    def add_solver(self, name: str, solver):
        """Register a solver with this model"""
        self._solvers[name] = solver
        return solver
    
    def get_solver(self, name: str):
        """Get a solver by name from the model registry"""
        return self._solvers.get(name)
    
    def define_parameter(self, name: str, ptype=None, **kwargs):
        """
        Define a new parameter with validation rules.
        
        NOTE: Parameter system not yet implemented. Use model.materials dict directly.
        
        Parameters:
        -----------
        name : str
            Parameter path (e.g., 'material.viscosity', 'solver.tolerance')
        ptype : ParameterType, optional
            Parameter type for validation (not used yet)
        **kwargs : dict
            Additional arguments
        """
        # TODO: Implement when parameter system is ready
        raise NotImplementedError("Parameter system not yet implemented. Use model.materials dict directly.")
    
    def set_material(self, name: str, properties: Dict[str, Any]):
        """
        Set material properties (simple dictionary).
        
        Parameters:
        -----------
        name : str
            Material name (e.g., 'mantle', 'crust', 'plume')
        properties : dict
            Dictionary of property_name -> value/expression
            
        Example:
        --------
        >>> model.set_material('mantle', {'viscosity': 1e21, 'density': 3300})
        """
        self.materials[name] = properties
        print(f"Model '{self.name}': Set material '{name}' with {len(properties)} properties")
    
    def get_material(self, name: str):
        """Get material properties by name"""
        return self.materials.get(name, {})

    def set_reference_quantities(self, **quantities):
        """
        Set reference quantities for automatic units scaling.

        This method enables the hybrid SymPy+Pint units workflow by allowing users
        to specify their problem in natural units, from which the system derives
        optimal scaling for numerical conditioning.

        Parameters:
        -----------
        **quantities : dict
            Named reference quantities using Pint units, e.g.:
            - mantle_viscosity=1e21*uw.units.Pa*uw.units.s
            - plate_velocity=5*uw.units.cm/uw.units.year
            - domain_depth=3000*uw.units.km

        Example:
        --------
        >>> model.set_reference_quantities(
        ...     mantle_viscosity=1e21*uw.units.Pa*uw.units.s,
        ...     plate_velocity=5*uw.units.cm/uw.units.year,
        ...     domain_depth=3000*uw.units.km
        ... )

        Notes:
        ------
        This method creates a Pint-native registry with model-specific constants
        using the _constants pattern for optimal numerical conditioning.
        """
        # Store reference quantities
        self._reference_quantities = quantities.copy()

        print(f"Model '{self.name}': Set {len(quantities)} reference quantities")
        for name, qty in quantities.items():
            print(f"  {name}: {qty}")

        # Perform dimensional analysis to derive fundamental scales
        self._fundamental_scales = self._derive_fundamental_scales()

        # Create Pint registry with model-specific _constants
        self._create_pint_registry()

        # Store metadata for serialization
        if 'reference_quantities' not in self.metadata:
            self.metadata['reference_quantities'] = {}

        for name, qty in quantities.items():
            self.metadata['reference_quantities'][name] = {
                'value': str(qty),
                'magnitude': float(qty.magnitude) if hasattr(qty, 'magnitude') else qty,
                'units': str(qty.units) if hasattr(qty, 'units') else 'dimensionless'
            }

        # Mark that this model has units scaling enabled
        self.metadata['units_enabled'] = True
        self.metadata['units_backend'] = 'pint_native'

        # Set default scaling mode
        if not hasattr(self, '_scaling_mode'):
            self._scaling_mode = 'exact'

    def get_reference_quantities(self):
        """Get the reference quantities for this model."""
        return self.metadata.get('reference_quantities', {})

    def has_units(self):
        """Check if this model has units scaling enabled."""
        return self.metadata.get('units_enabled', False)

    def _derive_fundamental_scales(self) -> dict:
        """
        Derive fundamental scales from reference quantities using dimensional analysis.

        Same logic as existing implementation, returns scales that become Pint _constants.
        """
        if not hasattr(self, '_reference_quantities'):
            return {}

        # Use existing dimensional analysis from enhanced model
        try:
            from .utilities.units_mixin import DimensionalAnalysis
            analyzer = DimensionalAnalysis()

            # Convert reference quantities to format expected by analyzer
            ref_for_analysis = {}
            for name, qty in self._reference_quantities.items():
                ref_for_analysis[name] = analyzer._units_backend.create_quantity(
                    qty.magnitude, str(qty.units)
                )

            # Use existing derivation logic
            return analyzer.derive_fundamental_scales(ref_for_analysis)

        except ImportError:
            # Fallback: simplified dimensional analysis for basic cases
            return self._simple_dimensional_analysis()

    def _simple_dimensional_analysis(self) -> dict:
        """
        Pure mathematical dimensional analysis using linear algebra.

        Uses physics (dimensional structure) not linguistics (names) to derive
        fundamental scales. Works with any user terminology.
        """
        return self._comprehensive_dimensional_analysis()

    def _comprehensive_dimensional_analysis(self) -> dict:
        """
        Comprehensive dimensional analysis using linear algebra and Pint.

        Analyzes dimensional coverage, provides intelligent error handling,
        and uses human-friendly formatting. Zero dependency on naming conventions.
        """
        import pint
        import numpy as np

        # Initialize analysis tools
        ureg = pint.UnitRegistry()
        fundamental_dims = ['[length]', '[time]', '[mass]', '[temperature]']

        # Build dimensional matrix from pure physics
        matrix = []
        magnitudes = []
        names = []

        for name, qty in self._reference_quantities.items():
            dims = dict(qty.dimensionality)
            row = [dims.get(dim, 0) for dim in fundamental_dims]
            matrix.append(row)
            magnitudes.append(qty.to_base_units().magnitude)
            names.append(name)

        matrix = np.array(matrix)

        # Analyze system properties
        rank = np.linalg.matrix_rank(matrix)
        n_dims = len(fundamental_dims)
        n_quantities = len(names)

        print(f"\n=== Dimensional Analysis ===")
        print(f"Matrix rank: {rank}/{n_dims} (need {n_dims} for complete system)")
        print(f"Quantities: {n_quantities}")

        # Handle different system types
        if rank < n_dims:
            return self._handle_under_determined_system(matrix, names, fundamental_dims, ureg)
        elif rank == n_dims:
            return self._solve_complete_system(matrix, magnitudes, names, fundamental_dims, ureg)
        else:
            print("⚠️  Unexpected matrix rank - proceeding with best approximation")
            return self._solve_complete_system(matrix, magnitudes, names, fundamental_dims, ureg)

    def _handle_under_determined_system(self, matrix, names, fundamental_dims, ureg):
        """Handle incomplete dimensional coverage with helpful suggestions."""
        import numpy as np

        # Identify missing dimensions
        covered_dims = []
        for i, dim in enumerate(fundamental_dims):
            if np.any(matrix[:, i] != 0):
                covered_dims.append(dim.strip('[]'))

        missing_dims = [dim.strip('[]') for dim in fundamental_dims if dim not in ['[' + d + ']' for d in covered_dims]]

        print(f"❌ Incomplete dimensional coverage")
        print(f"Covered: {covered_dims}")
        print(f"Missing: {missing_dims}")

        print("\n💡 Suggestions to complete the system:")
        suggestions = {
            'mass': ["Add density: material_density=3000*uw.units.kg/uw.units.m**3",
                    "Add viscosity: fluid_viscosity=1e-3*uw.units.Pa*uw.units.s",
                    "Add pressure: reference_pressure=1*uw.units.bar"],
            'time': ["Add velocity with existing length scale",
                    "Add time: process_time=1*uw.units.hour",
                    "Add frequency: oscillation_freq=10*uw.units.Hz"],
            'length': ["Add length: characteristic_length=1*uw.units.meter",
                      "Add area: cross_section=1*uw.units.cm**2",
                      "Add volume: chamber_volume=1*uw.units.liter"],
            'temperature': ["Add temperature: reference_temperature=300*uw.units.K"]
        }

        for dim in missing_dims:
            if dim in suggestions:
                print(f"\nFor {dim} dimension:")
                for suggestion in suggestions[dim][:2]:
                    print(f"  • {suggestion}")

        # Return partial scales where possible
        partial_scales = {}
        scale_sources = {}

        for i, dim_name in enumerate(['length', 'time', 'mass', 'temperature']):
            if np.any(matrix[:, i] != 0):
                # Find first quantity that contributes to this dimension
                for j, name in enumerate(names):
                    if matrix[j, i] != 0:
                        # For direct dimensions (power = 1), use the quantity directly
                        if matrix[j, i] == 1 and np.sum(np.abs(matrix[j, :])) == 1:
                            partial_scales[dim_name] = self._reference_quantities[name]
                            scale_sources[dim_name] = name
                            break

        self._scale_sources = scale_sources
        return partial_scales

    def _solve_complete_system(self, matrix, magnitudes, names, fundamental_dims, ureg):
        """Solve complete dimensional system using linear algebra."""
        import numpy as np

        try:
            # Solve the linear system: matrix @ scales = log(magnitudes)
            log_mags = np.log10(magnitudes)
            scales_log = np.linalg.lstsq(matrix, log_mags, rcond=None)[0]
            fundamental_scales_values = 10**scales_log

            # Create Pint quantities for the fundamental scales
            fundamental_scales = {}
            scale_sources = {}

            base_units = [ureg.meter, ureg.second, ureg.kilogram, ureg.kelvin]
            dim_names = ['length', 'time', 'mass', 'temperature']

            for i, (dim_name, base_unit, scale_value) in enumerate(zip(dim_names, base_units, fundamental_scales_values)):
                fundamental_scales[dim_name] = scale_value * base_unit

                # Find which original quantity contributed most to this dimension
                contributors = []
                for j, name in enumerate(names):
                    if matrix[j, i] != 0:
                        contribution = abs(matrix[j, i])
                        contributors.append((contribution, name))

                if contributors:
                    # Sort by contribution magnitude
                    contributors.sort(reverse=True)
                    main_contributor = contributors[0][1]
                    if len(contributors) == 1 and matrix[j, i] == 1:
                        scale_sources[dim_name] = main_contributor
                    else:
                        scale_sources[dim_name] = f"derived from dimensional analysis"
                else:
                    scale_sources[dim_name] = "derived"

            # Verification
            print("\n✅ Complete dimensional system solved")
            print("Verification (should be ≈ 1.0):")
            for j, name in enumerate(names):
                predicted_mag = np.prod([fundamental_scales_values[i]**matrix[j,i]
                                       for i in range(len(fundamental_scales_values))])
                actual_mag = magnitudes[j]
                ratio = actual_mag / predicted_mag
                print(f"  {name}: ratio = {ratio:.3f}")

            self._scale_sources = scale_sources
            return fundamental_scales

        except Exception as e:
            print(f"❌ Linear system solving failed: {e}")
            print("Falling back to partial analysis...")
            return self._handle_under_determined_system(matrix, names, fundamental_dims, ureg)

    def _create_pint_registry(self):
        """
        Create Pint registry with model-specific _constants.

        Uses Pint's _constants pattern like _100km for model units.
        """
        import pint

        # Create model-specific registry
        self._pint_registry = pint.UnitRegistry()

        # Store constant definitions for inspection
        self._model_constants = {}

        # Define multiple aliases for fundamental dimensions
        # ASCII aliases: Easy to type
        # Unicode aliases: Beautiful for display
        self._dimension_aliases = {
            'length': {
                'ascii': ['L_ref', 'L_scale', 'L_model', 'length_scale'],
                'unicode': 'ℒ',      # U+2112, SCRIPT CAPITAL L
                'display': 'ℒ'       # Preferred for display
            },
            'time': {
                'ascii': ['T_ref', 'T_scale', 'T_model', 'time_scale'],
                'unicode': '𝒯',       # U+1D4AF, MATHEMATICAL SCRIPT CAPITAL T
                'display': '𝒯'
            },
            'mass': {
                'ascii': ['M_ref', 'M_scale', 'M_model', 'mass_scale'],
                'unicode': 'ℳ',       # U+2133, SCRIPT CAPITAL M
                'display': 'ℳ'
            },
            'temperature': {
                'ascii': ['Temp_ref', 'Theta_ref', 'T_thermal', 'temp_scale'],
                'unicode': 'Θ',       # U+0398, GREEK CAPITAL THETA
                'display': 'Θ'
            },
            'current': {
                'ascii': ['I_ref', 'I_scale', 'current_scale'],
                'unicode': 'ℐ',       # U+2110, SCRIPT CAPITAL I
                'display': 'ℐ'
            },
            'amount': {
                'ascii': ['N_ref', 'amount_scale', 'mol_scale'],
                'unicode': 'ℕ',       # U+2115, DOUBLE-STRUCK CAPITAL N
                'display': 'ℕ'
            },
            'luminosity': {
                'ascii': ['Lum_ref', 'light_scale'],
                'unicode': 'ℒᵤₘ',     # Custom for luminous intensity
                'display': 'ℒᵤₘ'
            }
        }

        # Define _constants for each fundamental scale
        for dimension, scale_qty in self._fundamental_scales.items():
            # Convert to base SI units for clean definitions
            si_qty = scale_qty.to_base_units()
            magnitude = si_qty.magnitude
            si_units = str(si_qty.units)

            # Create _constant name following Pint pattern
            const_name = self._generate_constant_name(magnitude, si_units, dimension)

            # Get aliases for this dimension
            aliases_info = self._dimension_aliases.get(dimension, {
                'ascii': [f"Scale_{dimension}"],
                'unicode': f"Scale_{dimension}",
                'display': f"Scale_{dimension}"
            })

            # Define the base technical constant
            definition = f"{const_name} = {magnitude} * {si_units}"
            self._pint_registry.define(definition)

            # Define all ASCII aliases
            ascii_aliases = aliases_info.get('ascii', [])
            for alias in ascii_aliases:
                alias_definition = f"{alias} = {const_name}"
                self._pint_registry.define(alias_definition)

            # Define Unicode alias
            unicode_alias = aliases_info.get('unicode')
            if unicode_alias:
                unicode_definition = f"{unicode_alias} = {const_name}"
                self._pint_registry.define(unicode_definition)

            # Store for later access
            self._model_constants[dimension] = {
                'constant_name': const_name,
                'ascii_aliases': ascii_aliases,
                'unicode_alias': unicode_alias,
                'display_alias': aliases_info.get('display', ascii_aliases[0] if ascii_aliases else const_name),
                'definition': definition,
                'original_scale': scale_qty,
                'magnitude': magnitude,
                'si_units': si_units
            }

    def _substitute_display_aliases(self, units_str: str) -> str:
        """
        Replace raw constant names with elegant display aliases in unit strings.

        Examples:
        "_6p31e41kg" → "ℳ"
        "_1000km" → "ℒ"
        "_631152000000000s" → "𝒯"
        """
        display_str = units_str

        # Replace each constant with its display alias
        for dimension, const_info in self._model_constants.items():
            const_name = const_info['constant_name']
            display_alias = const_info['display_alias']

            # Replace the constant name with display alias
            display_str = display_str.replace(const_name, display_alias)

        return display_str

    def get_unit_aliases(self) -> dict:
        """
        Get all available unit aliases for user reference.

        Returns
        -------
        dict
            Dictionary mapping dimensions to their available aliases.

        Examples
        --------
        >>> model.get_unit_aliases()
        {
            'length': {
                'ascii': ['L_ref', 'L_scale', 'L_model', 'length_scale'],
                'unicode': 'ℒ',
                'display': 'ℒ',
                'technical': '_1000km'
            },
            ...
        }
        """
        aliases = {}
        for dimension, const_info in self._model_constants.items():
            aliases[dimension] = {
                'ascii': const_info['ascii_aliases'],
                'unicode': const_info['unicode_alias'],
                'display': const_info['display_alias'],
                'technical': const_info['constant_name']
            }
        return aliases

    def _generate_constant_name(self, magnitude: float, si_units: str, dimension: str) -> str:
        """Generate Pint _constant names following _100km pattern."""
        # Simplify common SI units
        unit_shortcuts = {
            'meter': 'm',
            'kilogram': 'kg',
            'second': 's',
            'kelvin': 'K',
            'pascal * second': 'Pas',
            'pascal': 'Pa'
        }

        unit_short = unit_shortcuts.get(si_units, si_units.replace(' ', '').replace('*', ''))

        # Format magnitude for readability - ensure valid Python identifiers
        if abs(magnitude) >= 1e15 or abs(magnitude) <= 1e-6:
            # Very large or very small: use scientific notation with safe formatting
            exp_str = f"{magnitude:.2e}"
            # Replace problematic characters: . → p, + → pos, - → neg, e → E
            formatted = (exp_str.replace('.', 'p')
                              .replace('e+', 'E')
                              .replace('e-', 'Eneg')
                              .replace('e', 'E'))
        elif magnitude >= 1000 and si_units == 'meter' and magnitude % 1000 == 0:
            # Convert meters to km for readability
            return f"_{int(magnitude//1000)}km"
        elif magnitude >= 1000:
            # Large round numbers
            formatted = f"{int(magnitude)}" if magnitude == int(magnitude) else f"{magnitude:.0f}"
        else:
            # Small numbers or decimals - ensure valid identifier
            formatted = f"{magnitude}".replace('.', 'p').replace('-', 'neg')

        return f"_{formatted}{unit_short}"

    def _convert_to_user_time_unit(self, time_base_units, velocity_quantity):
        """
        Infer appropriate time unit from velocity reference quantity.

        This implements Stage 2 of the two-stage simplification:
        After rationalizing to SI base units, convert to user-friendly time units
        by inferring the appropriate scale from the velocity's unit system.

        Parameters
        ----------
        time_base_units : pint.Quantity
            Time in SI base units (seconds)
        velocity_quantity : pint.Quantity
            The velocity quantity used to derive time (contains unit hints)

        Returns
        -------
        pint.Quantity
            Time in user-appropriate units (year, megayear, day, etc.)

        Examples
        --------
        >>> # If velocity is in cm/year, time should be in years or megayears
        >>> time_sec = 1.26e14 * ureg.second
        >>> velocity = 5 * ureg.cm / ureg.year
        >>> result = model._convert_to_user_time_unit(time_sec, velocity)
        >>> # result: 40.0 megayear (not 1.26e14 second)
        """
        from .scaling import units as u

        velocity_units_str = str(velocity_quantity.units)

        # Infer time unit from velocity's time component
        if 'year' in velocity_units_str.lower():
            # Geological time scale
            time_years = time_base_units.to('year').magnitude
            if abs(time_years) > 1e6:
                return time_base_units.to('megayear')
            elif abs(time_years) > 1e3:
                return time_base_units.to('kiloyear')
            else:
                return time_base_units.to('year')
        elif 'day' in velocity_units_str.lower():
            # Daily time scale
            time_days = time_base_units.to('day').magnitude
            if abs(time_days) > 365:
                return time_base_units.to('year')
            else:
                return time_base_units.to('day')
        elif 'hour' in velocity_units_str.lower() or 'hr' in velocity_units_str.lower():
            # Hourly time scale
            return time_base_units.to('hour')
        elif 'minute' in velocity_units_str.lower() or 'min' in velocity_units_str.lower():
            # Minute time scale
            return time_base_units.to('minute')
        else:
            # Default: keep as seconds or convert to most appropriate
            time_seconds = time_base_units.magnitude
            if abs(time_seconds) > 31557600e6:  # > 1 Myr
                return time_base_units.to('megayear')
            elif abs(time_seconds) > 31557600:  # > 1 year
                return time_base_units.to('year')
            elif abs(time_seconds) > 86400:  # > 1 day
                return time_base_units.to('day')
            elif abs(time_seconds) > 3600:  # > 1 hour
                return time_base_units.to('hour')
            else:
                return time_base_units  # Keep as seconds

    def _choose_display_units(self, quantity, dimension_name='[unknown]'):
        """
        Choose magnitude-appropriate units for display.

        This implements the display-time unit selection strategy: choose units
        that give values between 0.1 and 1000 for better readability.

        Parameters
        ----------
        quantity : pint.Quantity
            Quantity to display in appropriate units
        dimension_name : str, optional
            Dimension name for context (e.g., '[length]', '[time]', '[mass]')

        Returns
        -------
        pint.Quantity
            Same quantity in magnitude-appropriate units

        Examples
        --------
        >>> # Length: 2e6 m → 2000 km
        >>> length = 2e6 * ureg.meter
        >>> display_length = model._choose_display_units(length, '[length]')
        >>> # display_length: 2000 kilometer

        >>> # Time: 1.26e14 s → 40 Myr
        >>> time = 1.26e14 * ureg.second
        >>> display_time = model._choose_display_units(time, '[time]')
        >>> # display_time: 40.0 megayear
        """
        from .scaling import units as u

        # Get base magnitude for comparison
        base_qty = quantity.to_base_units()
        base_magnitude = abs(base_qty.magnitude)

        # Define unit options for each dimension (ordered from small to large)
        unit_options = {
            '[length]': ['nanometer', 'micrometer', 'millimeter', 'centimeter', 'meter', 'kilometer', 'megameter'],
            '[time]': ['microsecond', 'millisecond', 'second', 'minute', 'hour', 'day', 'year', 'kiloyear', 'megayear'],
            '[mass]': ['microgram', 'milligram', 'gram', 'kilogram', 'tonne', 'kilotonne', 'megatonne'],
            '[temperature]': ['kelvin'],  # Temperature doesn't need magnitude scaling
            '[current]': ['microampere', 'milliampere', 'ampere', 'kiloampere'],
            '[substance]': ['micromole', 'millimole', 'mole', 'kilomole'],
            '[luminosity]': ['millicandela', 'candela', 'kilocandela']
        }

        # Get unit options for this dimension
        options = unit_options.get(dimension_name, [])

        if not options:
            # If no options defined, return as-is
            return quantity

        # Try each unit option and find the one that gives good magnitude
        best_unit = None
        best_magnitude = None
        best_score = float('inf')

        for unit_name in options:
            try:
                converted = quantity.to(unit_name)
                magnitude = abs(converted.magnitude)

                # Score based on how far from ideal range [1, 1000]
                if 1 <= magnitude <= 1000:
                    # In ideal range - score by distance from geometric mean (31.6)
                    import math
                    score = abs(math.log10(magnitude) - 1.5)  # log10(31.6) ≈ 1.5
                elif magnitude < 1:
                    # Too small - penalize more for being further below 1
                    score = 10 * (1 - magnitude)
                else:
                    # Too large - penalize for being above 1000
                    score = 10 + math.log10(magnitude / 1000)

                if score < best_score:
                    best_score = score
                    best_unit = unit_name
                    best_magnitude = magnitude
            except:
                # Unit conversion failed, skip this option
                continue

        # Return in best unit if found, otherwise return original
        if best_unit is not None:
            return quantity.to(best_unit)
        else:
            return quantity

    def derive_fundamental_scalings(self):
        """
        Derive fundamental scaling units (length, time, mass, temperature) from reference quantities.

        This method analyzes the dimensional structure of reference quantities to automatically
        determine optimal fundamental scalings for the problem domain.

        Returns
        -------
        dict
            Dictionary of fundamental scalings with keys like '[length]', '[time]', '[mass]', '[temperature]'

        Examples
        --------
        >>> model.set_reference_quantities(
        ...     mantle_viscosity=1e21*uw.units.Pa*uw.units.s,
        ...     plate_velocity=5*uw.units.cm/uw.units.year,
        ...     mantle_temperature=1500*uw.units.K
        ... )
        >>> scalings = model.derive_fundamental_scalings()
        >>> scalings['[length]']  # Derived from plate_velocity
        >>> scalings['[time]']    # Derived from plate_velocity
        >>> scalings['[temperature]']  # Direct from mantle_temperature
        """
        ref_qty = self.get_reference_quantities()
        if not ref_qty:
            return {}

        # Check for over-determined systems before proceeding
        conflict_analysis = self._detect_scaling_conflicts(ref_qty)
        if conflict_analysis['has_conflicts']:
            import warnings

            conflict_msg = "Over-determined dimensional system detected:\n\n"
            for i, conflict in enumerate(conflict_analysis['conflicts'], 1):
                conflict_msg += f"{i}. {conflict}\n"

            conflict_msg += "\nSuggested resolutions:\n"
            for i, resolution in enumerate(conflict_analysis['resolutions'], 1):
                conflict_msg += f"{i}. {resolution}\n"

            if conflict_analysis['redundant_quantities']:
                conflict_msg += f"\nRedundant quantities to consider removing: {', '.join(conflict_analysis['redundant_quantities'])}\n"

            conflict_msg += "\nProceeding with derivation using the first quantity found for each dimension, but results may be inconsistent."

            warnings.warn(conflict_msg, UserWarning, stacklevel=2)

        # Import units backend for dimensional analysis
        try:
            from .scaling import units as u
        except ImportError:
            from .utilities.units_mixin import PintBackend
            backend = PintBackend()
            u = backend.registry

        scalings = {}
        derived_info = {}  # Track how each scaling was derived

        # Direct mappings for pure dimensions
        for name, info in ref_qty.items():
            try:
                # Recreate the quantity for dimensional analysis
                qty = u.Quantity(info['magnitude'], info['units'])
                dimensionality = qty.dimensionality

                # Direct temperature mapping
                if dimensionality == u.kelvin.dimensionality:
                    scalings['[temperature]'] = qty
                    derived_info['[temperature]'] = f"from {name}"

                # Direct length mapping
                elif dimensionality == u.meter.dimensionality:
                    scalings['[length]'] = qty
                    derived_info['[length]'] = f"from {name}"

                # Direct time mapping
                elif dimensionality == u.second.dimensionality:
                    scalings['[time]'] = qty
                    derived_info['[time]'] = f"from {name}"

                # Direct mass mapping
                elif dimensionality == u.kilogram.dimensionality:
                    scalings['[mass]'] = qty
                    derived_info['[mass]'] = f"from {name}"

                # Direct current mapping
                elif dimensionality == u.ampere.dimensionality:
                    scalings['[current]'] = qty
                    derived_info['[current]'] = f"from {name}"

                # Direct substance mapping
                elif dimensionality == u.mole.dimensionality:
                    scalings['[substance]'] = qty
                    derived_info['[substance]'] = f"from {name}"

                # Direct luminosity mapping
                elif dimensionality == u.candela.dimensionality:
                    scalings['[luminosity]'] = qty
                    derived_info['[luminosity]'] = f"from {name}"

            except Exception:
                continue

        # Derive composite scalings from compound quantities
        # Step 1: First pass - derive time from velocity if we have length
        for name, info in ref_qty.items():
            try:
                qty = u.Quantity(info['magnitude'], info['units'])
                dimensionality = qty.dimensionality

                # Velocity: length/time -> can derive time if we have length
                if dimensionality == (u.meter / u.second).dimensionality:
                    if '[time]' not in scalings and '[length]' in scalings:
                        # Derive time from length / velocity
                        time_raw = scalings['[length]'] / qty
                        # TWO-STAGE SIMPLIFICATION:
                        # Stage 1: Rationalize to SI base units to eliminate compound nonsense
                        time_base = time_raw.to_base_units()
                        # Stage 2: Convert to user-appropriate time unit (infer from velocity)
                        time_user = self._convert_to_user_time_unit(time_base, qty)
                        scalings['[time]'] = time_user
                        derived_info['[time]'] = f"from length_scale ÷ {name}"
                        break  # Only need one time derivation

            except Exception:
                continue

        # Step 2: Second pass - derive mass from viscosity or density if we have the necessary scales
        for name, info in ref_qty.items():
            try:
                qty = u.Quantity(info['magnitude'], info['units'])
                dimensionality = qty.dimensionality

                # Viscosity: pressure*time = mass/(length*time)
                if dimensionality == (u.pascal * u.second).dimensionality:
                    if '[length]' in scalings and '[time]' in scalings and '[mass]' not in scalings:
                        # Derive mass from viscosity * length * time
                        mass_raw = qty * scalings['[length]'] * scalings['[time]']
                        # TWO-STAGE SIMPLIFICATION:
                        # Stage 1: Rationalize to SI base units
                        mass_base = mass_raw.to_base_units()
                        # Stage 2: Keep in kilogram (standard mass unit)
                        scalings['[mass]'] = mass_base
                        derived_info['[mass]'] = f"from {name} × length_scale × time_scale"
                        break  # Only need one mass derivation

                # Density: mass/volume = mass/length^3
                # So mass = density * length^3
                elif dimensionality == (u.kilogram / u.meter**3).dimensionality:
                    if '[length]' in scalings and '[mass]' not in scalings:
                        # Derive mass from density * length^3
                        mass_raw = qty * scalings['[length]']**3
                        # Simplify to base units (kilogram)
                        mass_base = mass_raw.to_base_units()
                        scalings['[mass]'] = mass_base
                        derived_info['[mass]'] = f"from {name} × length_scale³"
                        break  # Only need one mass derivation

            except Exception:
                continue

        # Step 3: Third pass - derive current, substance, and luminosity from compound quantities
        for name, info in ref_qty.items():
            try:
                qty = u.Quantity(info['magnitude'], info['units'])
                dimensionality = qty.dimensionality

                # Electric field: voltage/length = mass*length/(time^3*current*length) = mass/(time^3*current)
                # So current = mass/(time^3*electric_field)
                if dimensionality == (u.volt / u.meter).dimensionality:
                    if '[mass]' in scalings and '[time]' in scalings and '[current]' not in scalings:
                        # Derive current from mass / (time^3 * electric_field)
                        current_raw = scalings['[mass]'] / (scalings['[time]']**3 * qty)
                        # Simplify to base units (ampere)
                        current_base = current_raw.to_base_units()
                        scalings['[current]'] = current_base
                        derived_info['[current]'] = f"from mass_scale ÷ (time_scale³ × {name})"
                        break  # Only need one current derivation

                # Concentration: substance/volume = substance/length^3
                # So substance = concentration * length^3
                elif dimensionality == (u.mole / u.meter**3).dimensionality:
                    if '[length]' in scalings and '[substance]' not in scalings:
                        # Derive substance from concentration * length^3
                        substance_raw = qty * scalings['[length]']**3
                        # Simplify to base units (mole)
                        substance_base = substance_raw.to_base_units()
                        scalings['[substance]'] = substance_base
                        derived_info['[substance]'] = f"from {name} × length_scale³"
                        break  # Only need one substance derivation

                # Luminous flux density: luminosity/area = luminosity/length^2
                # So luminosity = flux_density * length^2
                elif dimensionality == (u.candela / u.meter**2).dimensionality:
                    if '[length]' in scalings and '[luminosity]' not in scalings:
                        # Derive luminosity from flux_density * length^2
                        luminosity_raw = qty * scalings['[length]']**2
                        # Simplify to base units (candela)
                        luminosity_base = luminosity_raw.to_base_units()
                        scalings['[luminosity]'] = luminosity_base
                        derived_info['[luminosity]'] = f"from {name} × length_scale²"
                        break  # Only need one luminosity derivation

            except Exception:
                continue

        # Apply scale optimization for readability if requested
        if scalings and self.get_scaling_mode() == 'readable':
            # Apply optimization to get nice round numbers
            original_scalings = scalings.copy()
            scalings = self._optimize_scales_for_readability(scalings)

            # Update derivation info to note optimization
            optimized_info = derived_info.copy()
            for dim_name in scalings:
                if dim_name in original_scalings and dim_name in optimized_info:
                    original_source = optimized_info[dim_name]
                    original_val = original_scalings[dim_name]
                    optimized_val = scalings[dim_name]
                    if hasattr(original_val, 'magnitude') and hasattr(optimized_val, 'magnitude'):
                        if abs(original_val.magnitude - optimized_val.magnitude) > 1e-10:
                            optimized_info[dim_name] = f"{original_source} (optimized: {original_val.magnitude:.3g} → {optimized_val.magnitude:.3g})"

            derived_info = optimized_info

        # Store derivation information for display
        if scalings:
            self.metadata['derived_scalings'] = {
                'fundamental_scalings': {k: str(v) for k, v in scalings.items()},
                'derivation_info': derived_info,
                'scaling_mode': self.get_scaling_mode()
            }

        return scalings

    def get_fundamental_scales(self) -> dict:
        """
        Get fundamental scales in user-friendly format.

        Returns the derived fundamental scaling quantities (length, time, mass, temperature)
        that the model uses for dimensional analysis. These are the base scales from which
        all other quantities are converted to model units.

        Returns
        -------
        dict
            Dictionary mapping dimension names to their scaling quantities:
            - 'length': Length scale quantity (e.g., 2900 km)
            - 'time': Time scale quantity (e.g., 580 km*year/cm)
            - 'mass': Mass scale quantity (e.g., 1.68e27 kg equivalent)
            - 'temperature': Temperature scale quantity (e.g., 1500 K)
            - Plus any additional dimensions detected (current, substance, etc.)

        Examples
        --------
        >>> model.set_reference_quantities(
        ...     mantle_depth=2900*uw.units.km,
        ...     plate_velocity=5*uw.units.cm/uw.units.year,
        ...     mantle_viscosity=1e21*uw.units.Pa*uw.units.s,
        ...     mantle_temperature=1500*uw.units.K
        ... )
        >>> scales = model.get_fundamental_scales()
        >>> scales['length']  # 2900 kilometer
        >>> scales['time']    # 580 kilometer * year / centimeter
        """
        scalings = self.derive_fundamental_scalings()

        # Convert internal scaling format to user-friendly names
        result = {}
        dimension_map = {
            '[length]': 'length',
            '[time]': 'time',
            '[mass]': 'mass',
            '[temperature]': 'temperature',
            '[current]': 'current',
            '[substance]': 'substance',
            '[luminosity]': 'luminosity'
        }

        for internal_name, scale_qty in scalings.items():
            if internal_name in dimension_map:
                friendly_name = dimension_map[internal_name]
                # Apply magnitude-based unit selection for display
                display_qty = self._choose_display_units(scale_qty, internal_name)
                result[friendly_name] = display_qty
            else:
                # Handle any unexpected dimensions gracefully
                clean_name = internal_name.strip('[]')
                # Try to apply display unit selection even for unexpected dimensions
                display_qty = self._choose_display_units(scale_qty, internal_name)
                result[clean_name] = display_qty

        return result

    def get_model_base_units(self) -> dict:
        """
        Get model base units showing what "1.0" represents in each dimension.

        Returns a dictionary mapping dimension names to their model base unit definitions,
        showing both the derived units and SI base units for clarity.

        Returns
        -------
        dict
            Dictionary with dimension names as keys and formatted unit strings as values

        Examples
        --------
        >>> model.set_reference_quantities(
        ...     mantle_depth=2900*uw.units.km,
        ...     plate_velocity=5*uw.units.cm/uw.units.year,
        ...     mantle_viscosity=1e21*uw.units.Pa*uw.units.s,
        ...     mantle_temperature=1500*uw.units.K
        ... )
        >>> base_units = model.get_model_base_units()
        >>> base_units['mass']  # "1.0 model mass = 1.68e+27 kg"
        """
        scales = self.get_fundamental_scales()
        base_units = {}

        for dim_name, scale_qty in scales.items():
            if hasattr(scale_qty, 'magnitude') and hasattr(scale_qty, 'units'):
                magnitude = scale_qty.magnitude
                units_str = str(scale_qty.units)

                # Format the model base unit definition
                if magnitude >= 1e6 or magnitude <= 1e-6:
                    model_def = f"1.0 model {dim_name} = {magnitude:.2e} {units_str}"
                else:
                    model_def = f"1.0 model {dim_name} = {magnitude:.3g} {units_str}"

                # Add SI base units if different
                try:
                    base_units_qty = scale_qty.to_base_units()
                    base_magnitude = base_units_qty.magnitude
                    base_units_str = str(base_units_qty.units)

                    if str(scale_qty.units) != base_units_str:
                        if base_magnitude >= 1e6 or base_magnitude <= 1e-6:
                            model_def += f" (≡ {base_magnitude:.2e} {base_units_str})"
                        else:
                            model_def += f" (≡ {base_magnitude:.3g} {base_units_str})"

                except:
                    pass  # Keep original if conversion fails

                base_units[dim_name] = model_def
            else:
                base_units[dim_name] = f"1.0 model {dim_name} = {scale_qty}"

        return base_units

    def _format_scale_representations(self, scale_qty) -> str:
        """
        Format a scale quantity showing three representations:
        1. Derivation form (how it was derived)
        2. SI base units (simplified)
        3. Model base units (what 1.0 represents in the model)

        Parameters
        ----------
        scale_qty : pint.Quantity
            The scale quantity to format

        Returns
        -------
        str
            Formatted string with all three representations
        """
        if not hasattr(scale_qty, 'magnitude') or not hasattr(scale_qty, 'units'):
            return str(scale_qty)

        # 1. Derivation form (current complex units)
        magnitude = scale_qty.magnitude
        if magnitude >= 1e6 or magnitude <= 1e-6:
            derivation_str = f"{magnitude:.2e} {scale_qty.units}"
        else:
            derivation_str = f"{magnitude:.3g} {scale_qty.units}"

        # 2. SI base units (simplified)
        try:
            base_units_qty = scale_qty.to_base_units()
            base_magnitude = base_units_qty.magnitude
            base_units = str(base_units_qty.units)

            if base_magnitude >= 1e6 or base_magnitude <= 1e-6:
                si_str = f"{base_magnitude:.2e} {base_units}"
            else:
                si_str = f"{base_magnitude:.3g} {base_units}"
        except:
            si_str = "cannot simplify"

        # 3. Model base units (what 1.0 represents)
        try:
            # The scale quantity itself represents what "1.0" means in model units
            # So we format it in a clear way to show this
            if magnitude >= 1e6 or magnitude <= 1e-6:
                model_unit_str = f"1.0 model unit = {magnitude:.2e} {scale_qty.units}"
            else:
                model_unit_str = f"1.0 model unit = {magnitude:.3g} {scale_qty.units}"

            # If we can convert to base units, show that too
            if si_str != "cannot simplify":
                base_units_qty = scale_qty.to_base_units()
                base_magnitude = base_units_qty.magnitude
                base_units_str = str(base_units_qty.units)

                if base_magnitude >= 1e6 or base_magnitude <= 1e-6:
                    model_unit_str += f" (≡ {base_magnitude:.2e} {base_units_str})"
                else:
                    model_unit_str += f" (≡ {base_magnitude:.3g} {base_units_str})"

        except:
            model_unit_str = derivation_str

        # Format all three representations
        if si_str != "cannot simplify" and si_str != derivation_str:
            return f"{derivation_str}\n    → SI: {si_str}\n    → Model: {model_unit_str}"
        else:
            return f"{derivation_str}\n    → Model: {model_unit_str}"

    def get_scale_summary(self) -> str:
        """
        Get a human-readable summary of all fundamental scales.

        Returns a formatted string showing the fundamental scales derived from
        reference quantities, including how each scale was derived and what
        reference quantities it makes close to unity.

        Returns
        -------
        str
            Multi-line string with formatted scale summary

        Examples
        --------
        >>> model.set_reference_quantities(
        ...     mantle_depth=2900*uw.units.km,
        ...     plate_velocity=5*uw.units.cm/uw.units.year,
        ...     mantle_viscosity=1e21*uw.units.Pa*uw.units.s,
        ...     mantle_temperature=1500*uw.units.K
        ... )
        >>> print(model.get_scale_summary())
        Fundamental Scales Summary:

        Length Scale: 2900 kilometer
          - From: mantle_depth
          - Makes: mantle_depth ≈ 1.0 in model units

        Time Scale: 580 kilometer·year/centimeter
          - From: length_scale ÷ plate_velocity
          - Makes: plate_velocity ≈ 1.0 in model units

        Mass Scale: 1.68e+27 kg (equivalent)
          - From: mantle_viscosity × length_scale × time_scale
          - Makes: mantle_viscosity ≈ 1.0 in model units

        Temperature Scale: 1500 kelvin
          - From: mantle_temperature
          - Makes: mantle_temperature ≈ 1.0 in model units
        """
        scales = self.get_fundamental_scales()
        if not scales:
            return "No fundamental scales derived. Set reference quantities first."

        lines = ["Fundamental Scales Summary:", ""]

        # Define display order and nice names
        scale_order = ['length', 'time', 'mass', 'temperature', 'current', 'substance', 'luminosity']
        scale_titles = {
            'length': 'Length Scale',
            'time': 'Time Scale',
            'mass': 'Mass Scale',
            'temperature': 'Temperature Scale',
            'current': 'Current Scale',
            'substance': 'Substance Scale',
            'luminosity': 'Luminosity Scale'
        }

        for scale_name in scale_order:
            if scale_name in scales:
                scale_qty = scales[scale_name]
                title = scale_titles[scale_name]

                # Use Pint's smart formatting capabilities
                compact_scale = scale_qty.to_compact()
                si_scale = scale_qty.to_base_units()
                friendly_scale = self._get_domain_friendly_scale(scale_qty)

                lines.append(f"{title}: {compact_scale}")
                if friendly_scale != compact_scale:
                    lines.append(f"    → Friendly: {friendly_scale}")
                lines.append(f"    → SI: {si_scale}")
                lines.append(f"    → Model: 1.0 model unit = {compact_scale} (≡ {si_scale})")

                # Show derivation source with user's terminology
                if hasattr(self, '_scale_sources') and scale_name in self._scale_sources:
                    source_info = self._scale_sources[scale_name]
                    lines.append(f"  - From: {source_info}")

                    # Show what this makes ~1.0 in model units
                    if not source_info.startswith("derived") and not source_info.startswith("from"):
                        lines.append(f"  - Makes: {source_info} ≈ 1.0 in model units")
                    else:
                        lines.append(f"  - {source_info.title()}")
                else:
                    lines.append("  - From: dimensional analysis")

                lines.append("")

        # Add any additional scales not in the standard order
        for scale_name, scale_qty in scales.items():
            if scale_name not in scale_order:
                compact_scale = scale_qty.to_compact()
                lines.append(f"{scale_name.title()} Scale: {compact_scale}")
                lines.append("")

        return "\n".join(lines).rstrip()

    def _get_domain_friendly_scale(self, qty):
        """Choose domain-appropriate units based on magnitude using Pint's capabilities."""
        magnitude = qty.to_base_units().magnitude
        dimensionality = qty.dimensionality

        # Use Pint's unit registry for comparisons
        import pint
        ureg = pint.UnitRegistry()

        try:
            if dimensionality == ureg.meter.dimensionality:
                # Length scales - choose appropriate geological/engineering units
                if magnitude > 1e6:      # > 1000 km
                    return qty.to('megameter')
                elif magnitude > 1e3:    # > 1 km
                    return qty.to('kilometer')
                elif magnitude > 1:      # > 1 m
                    return qty.to('meter')
                elif magnitude > 1e-3:   # > 1 mm
                    return qty.to('millimeter')
                else:
                    return qty.to('micrometer')

            elif dimensionality == ureg.second.dimensionality:
                # Time scales - choose appropriate geological/engineering units
                if magnitude > 3.15e13:  # > 1 Myr
                    return qty.to('megayear')
                elif magnitude > 3.15e7: # > 1 year
                    return qty.to('year')
                elif magnitude > 86400:  # > 1 day
                    return qty.to('day')
                elif magnitude > 3600:   # > 1 hour
                    return qty.to('hour')
                elif magnitude > 60:     # > 1 minute
                    return qty.to('minute')
                else:
                    return qty.to('second')

            elif dimensionality == ureg.kilogram.dimensionality:
                # Mass scales
                if magnitude > 1e15:     # Very large masses
                    return qty.to('petagram')
                elif magnitude > 1e9:    # Large masses
                    return qty.to('gigagram')
                elif magnitude > 1e3:    # Engineering masses
                    return qty.to('megagram')
                else:
                    return qty.to('kilogram')

            elif dimensionality == ureg.kelvin.dimensionality:
                # Temperature - usually fine as is
                return qty.to('kelvin')

        except Exception:
            # Fallback to Pint's automatic compact formatting
            pass

        return qty.to_compact()

    def list_derived_scales(self) -> dict:
        """
        List which scales were derived vs. directly specified.

        Returns a dictionary categorizing fundamental scales by how they were obtained:
        either directly from reference quantities or derived from compound quantities.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'direct': List of (dimension, source) tuples for directly specified scales
            - 'derived': List of (dimension, derivation) tuples for derived scales
            - 'missing': List of standard dimensions that couldn't be determined

        Examples
        --------
        >>> model.set_reference_quantities(
        ...     mantle_depth=2900*uw.units.km,
        ...     plate_velocity=5*uw.units.cm/uw.units.year,
        ...     mantle_viscosity=1e21*uw.units.Pa*uw.units.s,
        ...     mantle_temperature=1500*uw.units.K
        ... )
        >>> derivation = model.list_derived_scales()
        >>> derivation['direct']    # [('length', 'mantle_depth'), ('temperature', 'mantle_temperature')]
        >>> derivation['derived']   # [('time', 'length_scale ÷ plate_velocity'), ('mass', 'mantle_viscosity × length_scale × time_scale')]
        >>> derivation['missing']   # []
        """
        scales = self.get_fundamental_scales()
        derivation_info = self.metadata.get('derived_scalings', {}).get('derivation_info', {})

        # Standard dimensions we expect to see
        standard_dimensions = ['length', 'time', 'mass', 'temperature']
        extended_dimensions = ['current', 'substance', 'luminosity']

        direct = []
        derived = []
        present_dimensions = set()

        # Categorize each scale found
        for dimension_name, scale_qty in scales.items():
            present_dimensions.add(dimension_name)
            internal_name = f"[{dimension_name}]"

            if internal_name in derivation_info:
                source = derivation_info[internal_name]

                # Check if it's a direct mapping (starts with "from " and no operators)
                if source.startswith("from ") and " × " not in source and " ÷ " not in source:
                    ref_quantity_name = source.replace("from ", "")
                    direct.append((dimension_name, ref_quantity_name))
                else:
                    # It's derived from compound quantities
                    derived.append((dimension_name, source))
            else:
                # No derivation info available (shouldn't happen with current implementation)
                direct.append((dimension_name, "unknown source"))

        # Find missing standard dimensions
        missing = [dim for dim in standard_dimensions if dim not in present_dimensions]

        # Find any additional dimensions beyond the standard set
        additional = [dim for dim in present_dimensions
                     if dim not in standard_dimensions and dim not in extended_dimensions]

        result = {
            'direct': direct,
            'derived': derived,
            'missing': missing
        }

        # Add additional dimensions if any were found
        if additional:
            result['additional'] = additional

        return result

    def validate_dimensional_completeness(self, required_dimensions=None) -> dict:
        """
        Validate if reference quantities provide complete dimensional coverage.

        Checks whether the current reference quantities span enough dimensions
        to derive fundamental scales for all required dimensions, and provides
        suggestions for completing under-determined systems.

        Parameters
        ----------
        required_dimensions : list, optional
            List of dimensions required (e.g., ['length', 'time', 'mass', 'temperature']).
            If None, uses the standard 4-dimension set.

        Returns
        -------
        dict
            Validation result containing:
            - 'status': 'complete', 'under_determined', or 'no_reference_quantities'
            - 'missing_dimensions': List of missing dimensions (if any)
            - 'suggestions': List of suggested reference quantities to add
            - 'derivable_dimensions': List of dimensions that can be derived
            - 'analysis': Human-readable analysis string

        Examples
        --------
        >>> # Under-determined system
        >>> model.set_reference_quantities(mantle_depth=2900*uw.units.km)
        >>> result = model.validate_dimensional_completeness()
        >>> result['status']  # 'under_determined'
        >>> result['missing_dimensions']  # ['time', 'mass', 'temperature']

        >>> # Complete system
        >>> model.set_reference_quantities(
        ...     mantle_depth=2900*uw.units.km,
        ...     plate_velocity=5*uw.units.cm/uw.units.year,
        ...     mantle_viscosity=1e21*uw.units.Pa*uw.units.s,
        ...     mantle_temperature=1500*uw.units.K
        ... )
        >>> result = model.validate_dimensional_completeness()
        >>> result['status']  # 'complete'
        """
        if required_dimensions is None:
            required_dimensions = ['length', 'time', 'mass', 'temperature']

        ref_qty = self.get_reference_quantities()
        if not ref_qty:
            return {
                'status': 'no_reference_quantities',
                'missing_dimensions': required_dimensions,
                'suggestions': self._suggest_reference_quantities(required_dimensions),
                'derivable_dimensions': [],
                'analysis': "No reference quantities set. Please add reference quantities to enable dimensional analysis."
            }

        # Get what dimensions we can actually derive
        scales = self.get_fundamental_scales()
        derivable_dimensions = list(scales.keys())

        # Find missing dimensions
        missing_dimensions = [dim for dim in required_dimensions if dim not in derivable_dimensions]

        if not missing_dimensions:
            status = 'complete'
            analysis = f"✅ Dimensional system is complete. Can derive scales for: {', '.join(derivable_dimensions)}"
            suggestions = []
        else:
            status = 'under_determined'
            analysis = (f"⚠️ System is under-determined. "
                       f"Can derive: {', '.join(derivable_dimensions)}. "
                       f"Missing: {', '.join(missing_dimensions)}")
            suggestions = self._suggest_reference_quantities(missing_dimensions)

        return {
            'status': status,
            'missing_dimensions': missing_dimensions,
            'suggestions': suggestions,
            'derivable_dimensions': derivable_dimensions,
            'analysis': analysis
        }

    def _suggest_reference_quantities(self, missing_dimensions) -> list:
        """
        Suggest specific reference quantities to complete missing dimensions.

        Parameters
        ----------
        missing_dimensions : list
            List of dimension names that are missing

        Returns
        -------
        list
            List of suggestion strings for each missing dimension
        """
        suggestions = {
            'length': [
                "Add a length scale: domain_size=1000*uw.units.km",
                "Or characteristic_length=100*uw.units.m",
                "Or layer_thickness=50*uw.units.km"
            ],
            'time': [
                "Add a time scale: characteristic_time=1*uw.units.Myr",
                "Or add velocity: plate_velocity=5*uw.units.cm/uw.units.year (with length scale)",
                "Or process_timescale=1000*uw.units.year"
            ],
            'mass': [
                "Add mass-related quantity: density=3300*uw.units.kg/uw.units.m**3",
                "Or viscosity: viscosity=1e21*uw.units.Pa*uw.units.s (with length & time)",
                "Or stress: stress=100*uw.units.MPa",
                "Or characteristic_mass=1e24*uw.units.kg"
            ],
            'temperature': [
                "Add temperature scale: characteristic_temperature=1500*uw.units.K",
                "Or temperature_difference=1000*uw.units.K",
                "Or melting_temperature=1600*uw.units.K"
            ],
            'current': [
                "Add electrical quantity: conductivity=1e-3*uw.units.S/uw.units.m",
                "Or current_density=1*uw.units.A/uw.units.m**2",
                "Or magnetic_field=1e-4*uw.units.T (with mass, length, time)"
            ],
            'substance': [
                "Add chemical quantity: concentration=1*uw.units.mol/uw.units.L",
                "Or molar_mass=0.1*uw.units.kg/uw.units.mol",
                "Or reaction_rate=1e-6*uw.units.mol/(uw.units.m**3*uw.units.s)"
            ],
            'luminosity': [
                "Add light quantity: luminous_flux=1000*uw.units.lm",
                "Or illuminance=500*uw.units.lx",
                "Note: Luminosity rarely needed for geophysical problems"
            ]
        }

        result = []
        for dimension in missing_dimensions:
            if dimension in suggestions:
                dim_suggestions = suggestions[dimension]
                result.append(f"For {dimension} dimension:")
                result.extend([f"  - {suggestion}" for suggestion in dim_suggestions])
            else:
                result.append(f"For {dimension} dimension: Add a quantity with {dimension} units")

        return result

    def _handle_conversion_failure(self, qty, qty_dimensionality):
        """
        Handle conversion failure with diagnostic error messages and suggestions.

        Parameters
        ----------
        qty : pint.Quantity
            The quantity that failed to convert
        qty_dimensionality : pint.util.UnitsContainer
            The dimensionality of the failed quantity

        Returns
        -------
        UWQuantity
            Dimensionless quantity with original magnitude (fallback)

        Raises
        ------
        ValueError
            With diagnostic information when strict error handling is enabled
        """
        from .function import quantity as create_quantity

        # Analyze what dimensions are needed vs. available
        needed_dimensions = set()
        if hasattr(qty_dimensionality, '_d'):
            for base_dim in qty_dimensionality._d.keys():
                # Convert Pint internal dimension to our naming
                if base_dim == '[length]':
                    needed_dimensions.add('length')
                elif base_dim == '[time]':
                    needed_dimensions.add('time')
                elif base_dim == '[mass]':
                    needed_dimensions.add('mass')
                elif base_dim == '[temperature]':
                    needed_dimensions.add('temperature')
                elif base_dim == '[current]':
                    needed_dimensions.add('current')
                elif base_dim == '[substance]':
                    needed_dimensions.add('substance')
                elif base_dim == '[luminosity]':
                    needed_dimensions.add('luminosity')

        # Get validation results
        validation = self.validate_dimensional_completeness(list(needed_dimensions))

        # Build diagnostic message
        missing_dims = validation.get('missing_dimensions', [])
        if missing_dims:
            error_msg = (
                f"Cannot convert {qty} to model units. "
                f"Missing fundamental scales for: {', '.join(missing_dims)}.\n\n"
                f"The quantity has dimensionality {qty_dimensionality} which requires "
                f"scales for: {', '.join(needed_dimensions)}.\n"
                f"Currently available: {', '.join(validation.get('derivable_dimensions', []))}.\n\n"
                f"Suggestions:\n" + "\n".join(validation.get('suggestions', []))
            )

            # For now, issue a warning instead of raising an error for backward compatibility
            import warnings
            warnings.warn(
                f"Dimensional analysis incomplete: {error_msg}\n"
                f"Returning dimensionless quantity with original magnitude {qty.magnitude}.",
                UserWarning,
                stacklevel=3
            )

        # Return dimensionless fallback
        return create_quantity(float(qty.magnitude))

    def set_scaling_mode(self, mode='exact'):
        """
        Set the scaling mode for fundamental scales.

        Parameters
        ----------
        mode : {'exact', 'readable'}
            Scaling mode to use:
            - 'exact': Reference quantities scale to exactly 1.0 (default)
            - 'readable': Reference quantities scale to O(1) with nice round numbers

        Examples
        --------
        >>> # Default exact mode: reference quantities become exactly 1.0
        >>> model.set_scaling_mode('exact')
        >>> model.set_reference_quantities(mantle_depth=2900*uw.units.km)
        >>> model.to_model_units(2900*uw.units.km)  # → UWQuantity(1.0, 'model_length_units')

        >>> # Readable mode: reference quantities become O(1) with nice scales
        >>> model.set_scaling_mode('readable')
        >>> model.set_reference_quantities(mantle_depth=2900*uw.units.km)
        >>> model.to_model_units(2900*uw.units.km)  # → UWQuantity(2.9, 'model_length_units')
        >>> # Internal scale becomes 1000 km instead of 2900 km
        """
        if mode not in ['exact', 'readable']:
            raise ValueError(f"Scaling mode must be 'exact' or 'readable', got '{mode}'")

        self.metadata['scaling_mode'] = mode

        # Clear any cached scalings to force recomputation with new mode
        if 'derived_scalings' in self.metadata:
            del self.metadata['derived_scalings']

    def get_scaling_mode(self) -> str:
        """
        Get the current scaling mode.

        Returns
        -------
        str
            Current scaling mode: 'exact' or 'readable'
        """
        return self.metadata.get('scaling_mode', 'exact')

    def _optimize_scales_for_readability(self, scalings):
        """
        Optimize scales to use nice round numbers while keeping reference quantities O(1).

        Parameters
        ----------
        scalings : dict
            Dictionary of fundamental scalings from derive_fundamental_scalings()

        Returns
        -------
        dict
            Optimized scalings with nice round numbers
        """
        optimized = {}

        for dim_name, scale_qty in scalings.items():
            if hasattr(scale_qty, 'magnitude') and hasattr(scale_qty, 'units'):
                # Find nearest "nice" scale
                nice_magnitude = self._find_nice_scale(scale_qty.magnitude)

                # Import units backend to create new quantity
                try:
                    from .scaling import units as u
                    nice_scale = u.Quantity(nice_magnitude, str(scale_qty.units))
                    optimized[dim_name] = nice_scale
                except:
                    # Fallback to original scale if optimization fails
                    optimized[dim_name] = scale_qty
            else:
                # Non-standard scale format, keep as-is
                optimized[dim_name] = scale_qty

        return optimized

    def _find_nice_scale(self, value):
        """
        Find nearest 'nice' number (powers of 10 times 1, 2, or 5).

        Parameters
        ----------
        value : float
            Original scale value

        Returns
        -------
        float
            Nearest nice scale value

        Examples
        --------
        >>> model._find_nice_scale(2900)   # → 1000 or 5000
        >>> model._find_nice_scale(0.38)   # → 0.5 or 0.2
        >>> model._find_nice_scale(15.7)   # → 10 or 20
        """
        import math

        if value == 0:
            return value

        # Get the order of magnitude
        log_val = math.log10(abs(value))
        power = math.floor(log_val)
        mantissa = 10**(log_val - power)

        # Choose from nice mantissas: 1, 2, 5
        nice_mantissas = [1, 2, 5, 10]  # Include 10 to handle edge cases

        # Find the closest nice mantissa
        best_mantissa = min(nice_mantissas, key=lambda x: abs(x - mantissa))

        # Handle the case where best choice is 10 (bump to next power)
        if best_mantissa == 10:
            best_mantissa = 1
            power += 1

        nice_value = best_mantissa * (10 ** power)

        # Preserve the sign
        return nice_value if value >= 0 else -nice_value

    def _detect_scaling_conflicts(self, ref_qty):
        """
        Detect conflicts in over-determined dimensional systems.

        Parameters
        ----------
        ref_qty : dict
            Reference quantities dictionary

        Returns
        -------
        dict
            Dictionary with keys:
            - 'has_conflicts': bool indicating if conflicts were found
            - 'conflicts': list of conflict descriptions
            - 'resolutions': list of suggested resolutions
            - 'redundant_quantities': list of quantity names that could be removed
        """
        # Import units for dimensional analysis
        try:
            from .scaling import units as u
        except ImportError:
            from .utilities.units_mixin import PintBackend
            backend = PintBackend()
            u = backend.registry

        conflicts = []
        resolutions = []
        redundant_quantities = []

        # Build a map of what dimensions each quantity provides
        quantity_dimensions = {}
        dimension_providers = {}

        for name, info in ref_qty.items():
            try:
                qty = u.Quantity(info['magnitude'], info['units'])
                dimensionality = qty.dimensionality
                quantity_dimensions[name] = dimensionality

                # Track which quantities can provide each base dimension
                if dimensionality == u.kelvin.dimensionality:
                    dimension_providers.setdefault('[temperature]', []).append(name)
                elif dimensionality == u.meter.dimensionality:
                    dimension_providers.setdefault('[length]', []).append(name)
                elif dimensionality == u.second.dimensionality:
                    dimension_providers.setdefault('[time]', []).append(name)
                elif dimensionality == u.kilogram.dimensionality:
                    dimension_providers.setdefault('[mass]', []).append(name)
                elif dimensionality == u.ampere.dimensionality:
                    dimension_providers.setdefault('[current]', []).append(name)
                elif dimensionality == u.mole.dimensionality:
                    dimension_providers.setdefault('[substance]', []).append(name)
                elif dimensionality == u.candela.dimensionality:
                    dimension_providers.setdefault('[luminosity]', []).append(name)

            except Exception:
                continue

        # Check for direct conflicts (multiple quantities of same dimension)
        for dimension, providers in dimension_providers.items():
            if len(providers) > 1:
                conflicts.append(
                    f"Multiple {dimension.strip('[]')} scales provided: {', '.join(providers)}"
                )
                resolutions.append(
                    f"Remove all but one of: {', '.join(providers)}"
                )
                redundant_quantities.extend(providers[1:])  # Keep first, mark others as redundant

        # Check for composite conflicts (e.g., length + time + velocity)
        # Look for velocity conflicts
        velocity_quantities = []
        length_quantities = []
        time_quantities = []

        for name, dimensionality in quantity_dimensions.items():
            if dimensionality == (u.meter / u.second).dimensionality:
                velocity_quantities.append(name)
            elif dimensionality == u.meter.dimensionality:
                length_quantities.append(name)
            elif dimensionality == u.second.dimensionality:
                time_quantities.append(name)

        # If we have length, time, AND velocity, that's over-determined
        if length_quantities and time_quantities and velocity_quantities:
            conflicts.append(
                f"Over-determined length/time/velocity system: length ({', '.join(length_quantities)}), "
                f"time ({', '.join(time_quantities)}), velocity ({', '.join(velocity_quantities)}). "
                f"Since velocity = length/time, providing all three over-determines the system."
            )
            resolutions.append(
                "Remove one of the three categories (length, time, or velocity) and let it be derived automatically."
            )
            # Suggest removing velocity as it's usually the most convenient to derive
            redundant_quantities.extend(velocity_quantities)

        # Check for viscosity conflicts (mass + length + time + viscosity)
        viscosity_quantities = []
        mass_quantities = []

        for name, dimensionality in quantity_dimensions.items():
            if dimensionality == (u.pascal * u.second).dimensionality:
                viscosity_quantities.append(name)
            elif dimensionality == u.kilogram.dimensionality:
                mass_quantities.append(name)

        if mass_quantities and length_quantities and time_quantities and viscosity_quantities:
            conflicts.append(
                f"Over-determined mass/viscosity system: mass ({', '.join(mass_quantities)}), "
                f"viscosity ({', '.join(viscosity_quantities)}), plus length and time scales. "
                f"Since viscosity involves mass/length/time, providing all over-determines the system."
            )
            resolutions.append(
                "Remove either the direct mass scale or the viscosity, and let mass be derived automatically."
            )
            # Suggest removing direct mass as viscosity derivation is more common in geophysics
            redundant_quantities.extend(mass_quantities)

        return {
            'has_conflicts': len(conflicts) > 0,
            'conflicts': conflicts,
            'resolutions': resolutions,
            'redundant_quantities': list(set(redundant_quantities))  # Remove duplicates
        }

    def show_optimal_units(self):
        """
        Display the optimal units implied by this model's reference quantities.

        Shows fundamental scalings (length, time, mass, temperature) derived from
        reference quantities and suggests optimal units for various physical quantities.
        """
        scalings = self.derive_fundamental_scalings()

        if not scalings:
            try:
                from IPython.display import Markdown, display
                display(Markdown("⚠️ **No reference quantities set**\n\nSet reference quantities first to see optimal units."))
            except ImportError:
                import underworld3 as uw
                uw.pprint("No reference quantities set. Use model.set_reference_quantities() first.")
            return

        try:
            from IPython.display import Markdown, display

            content = [f"## Optimal Units for Model: {self.name}"]
            content.append("*Derived from your reference quantities*")

            # Show fundamental scalings
            content.append("\n### Fundamental Scalings")
            derivation_info = self.metadata.get('derived_scalings', {}).get('derivation_info', {})

            fundamental_order = ['[length]', '[time]', '[mass]', '[temperature]']
            for dim in fundamental_order:
                if dim in scalings:
                    value = scalings[dim]
                    source = derivation_info.get(dim, 'direct')
                    content.append(f"- **{dim.strip('[]').title()}**: `{value}` _{source}_")

            # Derive optimal units for common quantities
            content.append("\n### Recommended Units for This Problem")
            content.append("*Designed to make your characteristic scales close to 1*")

            recommendations = []

            # Helper function to find optimal scale factor
            def find_optimal_scale_factor(quantity, base_unit_name, target_range=(1, 10)):
                """Find scale factor that brings quantity magnitude into target range."""
                try:
                    # Convert to base unit first
                    in_base = quantity.to(base_unit_name)
                    magnitude = in_base.magnitude

                    # Find power of 10 that brings magnitude into target range
                    import math
                    if magnitude == 0:
                        return 1, base_unit_name, magnitude

                    log_mag = math.log10(abs(magnitude))

                    # Find the power of 10 that brings us into target range
                    target_log = math.log10(target_range[0])  # Aim for lower end of range
                    power_adjustment = math.floor(log_mag - target_log)

                    scale_factor = 10 ** power_adjustment
                    scaled_magnitude = magnitude / scale_factor

                    # Format the optimal unit
                    if scale_factor == 1:
                        optimal_unit = base_unit_name
                    elif scale_factor == 1000:
                        # Common case: 1000*km becomes "1000*km"
                        optimal_unit = f"1000*{base_unit_name}"
                    elif scale_factor == 1e6:
                        optimal_unit = f"1e6*{base_unit_name}"
                    elif scale_factor == 0.01:
                        optimal_unit = f"0.01*{base_unit_name}"
                    else:
                        optimal_unit = f"{scale_factor}*{base_unit_name}"

                    return scale_factor, optimal_unit, scaled_magnitude
                except:
                    return 1, base_unit_name, quantity.magnitude

            # Length-based quantities with optimal scaling
            if '[length]' in scalings:
                length_scale = scalings['[length]']
                scale_factor, optimal_unit, scaled_mag = find_optimal_scale_factor(length_scale, 'km')

                recommendations.append((
                    "**Length/Distance**",
                    f"`{optimal_unit}`",
                    f"Your scale: {length_scale} → {scaled_mag:.1f} in optimal units"
                ))

                # Show what this means for mesh dimensions
                mesh_example = f"*Mesh coordinates: 0 to {scaled_mag:.1f} (instead of 0 to {length_scale.to('km').magnitude:.0f} km)*"
                recommendations.append(("", "", mesh_example))

            # Time-based quantities with optimal scaling
            if '[time]' in scalings:
                time_scale = scalings['[time]']
                # Try different time units to find best scaling
                try:
                    time_years = time_scale.to('year')
                    if time_years.magnitude >= 1000:
                        # Use Ma (million years) for geological time
                        scale_factor, optimal_unit, scaled_mag = find_optimal_scale_factor(time_years, 'Ma')
                        recommendations.append((
                            "**Time**",
                            f"`{optimal_unit}`",
                            f"Your scale: {time_scale} → {scaled_mag:.1f} Ma"
                        ))
                    else:
                        scale_factor, optimal_unit, scaled_mag = find_optimal_scale_factor(time_years, 'year')
                        recommendations.append((
                            "**Time**",
                            f"`{optimal_unit}`",
                            f"Your scale: {time_scale} → {scaled_mag:.1f} years"
                        ))
                except:
                    # Fallback to seconds
                    scale_factor, optimal_unit, scaled_mag = find_optimal_scale_factor(time_scale, 's')
                    recommendations.append((
                        "**Time**",
                        f"`{optimal_unit}`",
                        f"Your scale: {time_scale} → {scaled_mag:.1f} seconds"
                    ))

            # Velocity with optimal scaling
            if '[length]' in scalings and '[time]' in scalings:
                vel_scale = scalings['[length]'] / scalings['[time]']
                try:
                    vel_cm_year = vel_scale.to('cm/year')
                    scale_factor, optimal_unit, scaled_mag = find_optimal_scale_factor(vel_cm_year, 'cm/year')
                    recommendations.append((
                        "**Velocity**",
                        f"`{optimal_unit}`",
                        f"Your scale: {vel_scale.to('cm/year'):.2f} cm/year → {scaled_mag:.1f} in optimal units"
                    ))
                except:
                    # Fallback to m/s
                    vel_ms = vel_scale.to('m/s')
                    scale_factor, optimal_unit, scaled_mag = find_optimal_scale_factor(vel_ms, 'm/s')
                    recommendations.append((
                        "**Velocity**",
                        f"`{optimal_unit}`",
                        f"Your scale: {vel_ms:.2e} m/s → {scaled_mag:.1f} in optimal units"
                    ))

            # Temperature (usually doesn't need scaling, but check)
            if '[temperature]' in scalings:
                temp_scale = scalings['[temperature]']
                temp_k = temp_scale.to('K')
                if 100 <= temp_k.magnitude <= 10000:  # Reasonable range, no scaling needed
                    recommendations.append((
                        "**Temperature**",
                        "`K`",
                        f"Your scale: {temp_k:.0f} (good as-is)"
                    ))
                else:
                    scale_factor, optimal_unit, scaled_mag = find_optimal_scale_factor(temp_k, 'K')
                    recommendations.append((
                        "**Temperature**",
                        f"`{optimal_unit}`",
                        f"Your scale: {temp_k:.0f} K → {scaled_mag:.1f} in optimal units"
                    ))

            # Pressure/Stress with optimal scaling
            if '[mass]' in scalings and '[length]' in scalings and '[time]' in scalings:
                pressure_scale = scalings['[mass]'] / (scalings['[length]'] * scalings['[time]']**2)
                pressure_pa = pressure_scale.to('Pa')

                # Try GPa first for geological problems
                if pressure_pa.magnitude >= 1e6:
                    pressure_gpa = pressure_pa.to('GPa')
                    scale_factor, optimal_unit, scaled_mag = find_optimal_scale_factor(pressure_gpa, 'GPa')
                    recommendations.append((
                        "**Pressure/Stress**",
                        f"`{optimal_unit}`",
                        f"Your scale: {pressure_pa:.2e} Pa → {scaled_mag:.1f} GPa"
                    ))
                else:
                    scale_factor, optimal_unit, scaled_mag = find_optimal_scale_factor(pressure_pa, 'Pa')
                    recommendations.append((
                        "**Pressure/Stress**",
                        f"`{optimal_unit}`",
                        f"Your scale: {pressure_pa:.2e} Pa → {scaled_mag:.1f} in optimal units"
                    ))

            # Viscosity with optimal scaling
            if '[mass]' in scalings and '[length]' in scalings and '[time]' in scalings:
                visc_scale = scalings['[mass]'] / (scalings['[length]'] * scalings['[time]'])
                visc_pas = visc_scale.to('Pa*s')
                scale_factor, optimal_unit, scaled_mag = find_optimal_scale_factor(visc_pas, 'Pa*s')
                recommendations.append((
                    "**Viscosity**",
                    f"`{optimal_unit}`",
                    f"Your scale: {visc_pas:.2e} Pa*s → {scaled_mag:.1f} in optimal units"
                ))

            for quantity, unit, scale_info in recommendations:
                content.append(f"- {quantity}: {unit} _{scale_info}_")

            content.append(f"\n### Usage Example with Optimal Scaling")
            content.append("```python")
            content.append("# Create mesh with order-1 dimensions using optimal units:")

            # Find length and time recommendations for the example
            length_rec = None
            time_rec = None
            for quantity, unit, scale_info in recommendations:
                if "Length" in quantity:
                    length_rec = (unit.strip('`'), scale_info)
                elif "Time" in quantity:
                    time_rec = (unit.strip('`'), scale_info)

            if length_rec:
                optimal_length_unit, length_info = length_rec
                # Extract the scaled magnitude from the info
                if "→" in length_info:
                    scaled_mag = length_info.split("→")[1].split()[0]
                    content.append(f"# With {optimal_length_unit}, your domain becomes {scaled_mag}")
                    content.append("mesh = uw.meshing.UnstructuredSimplexBox(")
                    content.append("    minCoords=(0.0, 0.0),")
                    content.append(f"    maxCoords=({scaled_mag}, {scaled_mag}),  # Order-1 dimensions!")
                    content.append("    cellSize=0.1,  # Natural resolution")
                    content.append("    qdegree=2")
                    content.append(")")
                    content.append("")

            content.append("# Create variables with optimal units:")
            if recommendations:
                shown_count = 0
                for quantity, unit, scale_info in recommendations:
                    if quantity and "**" in quantity and shown_count < 3:
                        clean_quantity = quantity.replace("**", "").replace("*", "").lower()
                        clean_unit = unit.strip('`')
                        if clean_quantity == "length/distance":
                            content.append(f'position = uw.discretisation.MeshVariable("pos", mesh, 2, units="{clean_unit}")')
                        elif clean_quantity == "velocity":
                            content.append(f'velocity = uw.discretisation.MeshVariable("v", mesh, 2, units="{clean_unit}")')
                        elif clean_quantity == "temperature":
                            content.append(f'temperature = uw.discretisation.MeshVariable("T", mesh, 1, units="{clean_unit}")')
                        shown_count += 1

            content.append("")
            content.append("# Initialize with natural order-1 values:")
            content.append("# No more awkward large numbers!")
            content.append("```")

            # Add explanation of benefits
            content.append(f"\n### Why This Scaling Matters")
            content.append("- **Order-1 mesh coordinates**: Easy to understand and work with")
            content.append("- **Natural numerical values**: Better for visualization and debugging")
            content.append("- **Optimal conditioning**: Better for solvers and numerical stability")
            content.append("- **Clear physical intuition**: Dimensionless numbers close to 1 are easier to interpret")

            display(Markdown("\n".join(content)))

        except ImportError:
            # Fallback for non-Jupyter environments
            import underworld3 as uw

            uw.pprint(f"Optimal Units for Model: {self.name}")
            uw.pprint("=" * 50)

            uw.pprint("Fundamental Scalings:")
            derivation_info = self.metadata.get('derived_scalings', {}).get('derivation_info', {})
            for dim in ['[length]', '[time]', '[mass]', '[temperature]']:
                if dim in scalings:
                    value = scalings[dim]
                    source = derivation_info.get(dim, 'direct')
                    uw.pprint(f"  {dim.strip('[]').title()}: {value} ({source})")

            uw.pprint("\nRecommended units based on your reference quantities:")
            uw.pprint("  Use model.show_optimal_units() in Jupyter for detailed recommendations")

    def scale_to_physical(self, coordinates, dimension='length'):
        """
        Convert dimensionless model coordinates to physical units.

        This method converts model-unit coordinate arrays (where the model domain
        is scaled to ~1.0) back to physical units using the model's fundamental scales.

        Parameters
        ----------
        coordinates : array-like
            Dimensionless coordinate array in model units
        dimension : str, default 'length'
            Fundamental dimension to use for scaling ('length', 'time', 'mass', 'temperature')

        Returns
        -------
        UWQuantity
            Coordinates in physical units with appropriate units

        Examples
        --------
        >>> model.set_reference_quantities(domain_length=1000*uw.units.km, ...)
        >>> mesh = uw.meshing.StructuredQuadBox(minCoords=(-3,-3), maxCoords=(3,3), ...)
        >>> # mesh.points are in model units (dimensionless, domain spans -3 to 3)
        >>> physical_coords = model.scale_to_physical(mesh.points)
        >>> # Result: coordinates in kilometers, spanning -3000 to 3000 km

        Raises
        ------
        ValueError
            If no reference quantities set or dimension not available
        """
        if not hasattr(self, '_pint_registry'):
            raise ValueError("No reference quantities set. Use model.set_reference_quantities() first.")

        # Get fundamental scales
        scales = self.get_fundamental_scales()
        if dimension not in scales:
            raise ValueError(f"Dimension '{dimension}' not available. Available: {list(scales.keys())}")

        # Get the scale for this dimension
        scale_qty = scales[dimension]

        # Import here to avoid circular imports
        from .function import quantity as create_quantity
        import numpy as np

        # Convert coordinates to numpy array for consistency
        coords = np.asarray(coordinates)

        # Scale: model_coordinates * scale = physical_coordinates
        physical_magnitude = coords * scale_qty.to_base_units().magnitude

        # Create UWQuantity with proper units
        scale_units = str(scale_qty.to_base_units().units)
        return create_quantity(physical_magnitude, units=scale_units)

    def to_model_units(self, quantity):
        """
        Safely coerce any quantity to model units using smart protocol pattern.

        This method is designed to be safe to call repeatedly and handles edge cases:
        1. Does nothing if model has no units (no reference quantities set)
        2. Does nothing if the quantity is already in model units
        3. Does nothing if the quantity is dimensionless
        4. Uses protocol pattern for extensible conversion

        Parameters
        ----------
        quantity : Any
            Quantity to convert (UWQuantity, Pint quantity, numeric, etc.)

        Returns
        -------
        UWQuantity or original quantity
            Converted quantity in model units, or original if no conversion needed
        """
        # 1) Do nothing if there are no units (no reference quantities set)
        if not hasattr(self, '_pint_registry'):
            return quantity

        # 2) Do nothing if quantity is already in model units
        # Check if it's a UWQuantity with model_units=True flag
        if hasattr(quantity, '_is_model_units') and getattr(quantity, '_is_model_units', False):
            return quantity

        # 3) Early check for dimensionless quantities - do nothing
        try:
            # Quick dimensionality check without full conversion
            if hasattr(quantity, 'dimensionality'):
                dim_dict = dict(quantity.dimensionality)
                if not dim_dict:  # Empty = dimensionless
                    return quantity
            elif hasattr(quantity, '_has_pint_qty') and quantity._has_pint_qty:
                if hasattr(quantity, '_pint_qty'):
                    dim_dict = dict(quantity._pint_qty.dimensionality)
                    if not dim_dict:  # Empty = dimensionless
                        return quantity
        except Exception:
            pass  # Continue with full conversion attempt

        # Protocol pattern: Try hidden method first
        if hasattr(quantity, '_to_model_units_'):
            try:
                result = quantity._to_model_units_(self)
                if result is not None:
                    return result
                # If None returned, fall through to general approach
            except Exception:
                # If hidden method fails, fall through to general approach
                pass

        # General approach for any object
        result = self._convert_to_model_units_general(quantity)

        # If conversion failed (returns None), return original quantity
        if result is None:
            return quantity

        return result

    def _convert_to_model_units_general(self, quantity):
        """
        Convert any quantity to model units using dimensional analysis.

        Returns None for dimensionless quantities gracefully.
        """
        from .function import quantity as create_quantity

        # Convert input to base SI units for dimension analysis if possible
        if hasattr(quantity, 'to_base_units'):
            si_qty = quantity.to_base_units()
        else:
            si_qty = quantity

        # Check if dimensionless first (early return with None)
        try:
            # Get dimensionality using robust approach
            if hasattr(si_qty, '_has_pint_qty') and si_qty._has_pint_qty and hasattr(si_qty, '_pint_qty'):
                dimensionality = si_qty._pint_qty.dimensionality
                magnitude = si_qty._pint_qty.magnitude
            elif hasattr(si_qty, 'dimensionality') and si_qty.dimensionality is not None:
                dimensionality = si_qty.dimensionality
                # For Pint quantities, use .magnitude directly (don't try float conversion)
                if hasattr(si_qty, 'magnitude'):
                    magnitude = si_qty.magnitude
                elif hasattr(si_qty, 'value'):
                    magnitude = si_qty.value
                else:
                    magnitude = float(si_qty)  # Only as last resort
            else:
                # Assume it's a plain number (dimensionless)
                return None

            # Check if dimensionless
            dim_dict = dict(dimensionality)
            if not dim_dict:  # Empty dimensionality dict = dimensionless
                return None

        except Exception:
            # If we can't get dimensionality, assume dimensionless
            return None

        # GENERALIZED DIMENSIONAL APPROACH: Use dimensional analysis to compute scaling
        try:
            model_magnitude = magnitude  # Use the magnitude we extracted above
            model_units_parts = []

            for base_dim, power in dim_dict.items():
                dim_name = str(base_dim).strip('[]')

                if dim_name in self._model_constants:
                    const_info = self._model_constants[dim_name]
                    const_name = const_info['constant_name']
                    const_magnitude = const_info['magnitude']

                    # Apply dimensional scaling: value / (scale^power)
                    model_magnitude /= (const_magnitude ** power)

                    # Build unit expression
                    if power == 1:
                        model_units_parts.append(const_name)
                    elif power == -1:
                        model_units_parts.append(f"{const_name}**-1")
                    else:
                        model_units_parts.append(f"{const_name}**{power}")
                else:
                    # Missing fundamental dimension - cannot convert
                    raise ValueError(f"Missing fundamental dimension: {dim_name}")

            # Construct unit string
            if model_units_parts:
                model_units = "*".join(model_units_parts)
                return create_quantity(model_magnitude, _custom_units=model_units, _model_registry=self._pint_registry, _model_instance=self, _is_model_units=True)
            else:
                # Dimensionless quantity (shouldn't reach here due to early check, but handle gracefully)
                return None

        except Exception:
            # Any conversion failure - return None gracefully
            return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Export model configuration to dictionary for serialization.

        Provides enhanced serialization with SymPy expression conversion
        and PETSc state capture for complete reproducibility.

        Returns
        -------
        dict
            Model configuration dictionary suitable for JSON/YAML export
        """
        # Convert materials with SymPy expression handling
        serialized_materials = {}
        for mat_name, properties in self.materials.items():
            serialized_props = {}
            for prop_name, value in properties.items():
                if isinstance(value, sympy.Basic):
                    serialized_props[prop_name] = {
                        'type': 'sympy_expression',
                        'value': str(value),
                        'latex': sympy.latex(value) if hasattr(sympy, 'latex') else None
                    }
                else:
                    serialized_props[prop_name] = value
            serialized_materials[mat_name] = serialized_props

        config = {
            'model_name': self.name,
            'model_version': self.version,
            'state': self.state.value,
            'mesh_type': type(self.mesh).__name__ if self.mesh else None,
            'mesh_count': len(self._meshes),
            'variables': list(self._variables.keys()),
            'swarm_count': len(self._swarms),
            'solver_count': len(self._solvers),
            'materials': serialized_materials,
            'metadata': self.metadata,
            'petsc_state': self.petsc_state,
            'export_timestamp': datetime.now().isoformat(),
        }

        return config
    
    def export_configuration(self) -> Dict[str, Any]:
        """Alias for to_dict() for backward compatibility"""
        return self.to_dict()
    
    def from_dict(self, config: Dict[str, Any]):
        """
        Import model configuration from dictionary.

        Handles enhanced serialization format with SymPy expression reconstruction.
        Note: Only imports materials and metadata for now.
        Mesh/variables/swarms must be recreated.

        Parameters
        ----------
        config : dict
            Configuration dictionary from to_dict() or YAML export
        """
        # Import materials with SymPy expression reconstruction
        if 'materials' in config:
            reconstructed_materials = {}
            for mat_name, properties in config['materials'].items():
                reconstructed_props = {}
                for prop_name, value in properties.items():
                    if isinstance(value, dict) and value.get('type') == 'sympy_expression':
                        # Reconstruct SymPy expression from string
                        try:
                            reconstructed_props[prop_name] = sympy.sympify(value['value'])
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Could not reconstruct SymPy expression '{value['value']}': {e}")
                            reconstructed_props[prop_name] = value['value']  # Fall back to string
                    else:
                        reconstructed_props[prop_name] = value
                reconstructed_materials[mat_name] = reconstructed_props
            self.materials = reconstructed_materials

        if 'metadata' in config:
            self.metadata = config['metadata']

        if 'petsc_state' in config and config['petsc_state']:
            self.petsc_state = config['petsc_state']

        print(f"Model '{self.name}': Imported configuration")

    def to_yaml(self, file_path: Optional[str] = None) -> str:
        """
        Export model configuration to YAML format.

        Parameters
        ----------
        file_path : str, optional
            If provided, write YAML to this file path

        Returns
        -------
        str
            YAML string representation of model configuration
        """
        config = self.to_dict()
        yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)

        if file_path:
            Path(file_path).write_text(yaml_str)
            print(f"Model '{self.name}': Exported to {file_path}")

        return yaml_str

    def from_yaml(self, yaml_content: str = None, file_path: str = None):
        """
        Import model configuration from YAML.

        Parameters
        ----------
        yaml_content : str, optional
            YAML string to parse
        file_path : str, optional
            Path to YAML file to load
        """
        if file_path:
            yaml_content = Path(file_path).read_text()
        elif yaml_content is None:
            raise ValueError("Must provide either yaml_content or file_path")

        config = yaml.safe_load(yaml_content)
        self.from_dict(config)

        if file_path:
            print(f"Model '{self.name}': Imported from {file_path}")

    @classmethod
    def from_yaml_file(cls, file_path: str) -> 'Model':
        """
        Create a new Model instance from YAML file.

        Parameters
        ----------
        file_path : str
            Path to YAML configuration file

        Returns
        -------
        Model
            New model instance with loaded configuration
        """
        config = yaml.safe_load(Path(file_path).read_text())

        # Extract model-specific fields
        model_data = {
            'name': config.get('model_name', 'imported_model'),
            'materials': config.get('materials', {}),
            'metadata': config.get('metadata', {})
        }

        model = cls(**model_data)
        print(f"Model '{model.name}': Created from {file_path}")
        return model

    def capture_petsc_state(self) -> Dict[str, str]:
        """
        Capture current PETSc options database state.

        Returns all PETSc options currently set, enabling complete
        reproducibility of simulation parameters.

        Returns
        -------
        dict
            Dictionary of PETSc option names and values
        """
        try:
            import petsc4py.PETSc as PETSc

            # Get all currently set PETSc options
            options = {}
            opts = PETSc.Options()

            # PETSc doesn't provide a direct way to list all options,
            # but we can capture commonly used ones and any that were set
            # This is a simplified implementation - full implementation would
            # require more sophisticated PETSc option introspection

            common_options = [
                'ksp_type', 'pc_type', 'ksp_rtol', 'ksp_atol', 'ksp_max_it',
                'snes_type', 'snes_rtol', 'snes_atol', 'snes_max_it',
                'ts_type', 'ts_dt', 'ts_max_time', 'ts_max_steps',
                'mat_type', 'vec_type', 'dm_plex_box_faces'
            ]

            for option in common_options:
                try:
                    value = opts.getString(option, None)
                    if value is not None:
                        options[option] = value
                except:
                    pass  # Option not set or not accessible

            self.petsc_state = options
            print(f"Model '{self.name}': Captured {len(options)} PETSc options")
            return options

        except ImportError:
            print("Warning: petsc4py not available, cannot capture PETSc state")
            return {}

    def restore_petsc_state(self, petsc_options: Optional[Dict[str, str]] = None):
        """
        Restore PETSc options database from captured state.

        Parameters
        ----------
        petsc_options : dict, optional
            PETSc options to restore. If None, uses self.petsc_state
        """
        if petsc_options is None:
            petsc_options = self.petsc_state or {}

        if not petsc_options:
            print(f"Model '{self.name}': No PETSc state to restore")
            return

        try:
            import petsc4py.PETSc as PETSc
            opts = PETSc.Options()

            for option, value in petsc_options.items():
                opts[option] = value

            print(f"Model '{self.name}': Restored {len(petsc_options)} PETSc options")

        except ImportError:
            print("Warning: petsc4py not available, cannot restore PETSc state")

    def set_petsc_option(self, option: str, value: str):
        """
        Set a PETSc option and track it in model state.

        Parameters
        ----------
        option : str
            PETSc option name (without leading -)
        value : str
            Option value
        """
        try:
            import petsc4py.PETSc as PETSc
            opts = PETSc.Options()
            opts[option] = value

            # Track in model state
            if self.petsc_state is None:
                self.petsc_state = {}
            self.petsc_state[option] = value

            print(f"Model '{self.name}': Set PETSc option {option} = {value}")

        except ImportError:
            print("Warning: petsc4py not available, cannot set PETSc option")
    
    def view(self, verbose: int = 0, show_materials: bool = True, show_petsc: bool = False):
        """
        Display a concise summary of the model contents.

        Parameters
        ----------
        verbose : int, default 0
            Verbosity level:
            0 = Basic summary
            1 = Include variable details and material properties
            2 = Include solver information and metadata
        show_materials : bool, default True
            Whether to show materials summary
        show_petsc : bool, default False
            Whether to show PETSc options (can be lengthy)

        Example
        -------
        >>> model.view()                    # Basic summary
        >>> model.view(verbose=1)           # Detailed view
        >>> model.view(verbose=2, show_petsc=True)  # Full details
        """
        import textwrap

        # Build markdown content
        lines = []
        lines.append(f"# Model: {self.name}")
        lines.append(f"**Status:** {self.state.value} (version {self.version})")
        lines.append("")

        # Mesh information
        if self.mesh:
            mesh_type = type(self.mesh).__name__
            try:
                mesh_desc = f"{mesh_type}"
                if hasattr(self.mesh, 'dm') and self.mesh.dm:
                    # Try to get mesh statistics
                    try:
                        coords = self.mesh.dm.getCoordinates()
                        if coords:
                            node_count = coords.getSize()
                            mesh_desc += f" ({node_count:,} nodes)"
                    except:
                        pass
                lines.append(f"**Mesh:** {mesh_desc}")
            except:
                lines.append(f"**Mesh:** {mesh_type}")
        else:
            lines.append("**Mesh:** *No mesh assigned*")
        lines.append("")

        # Variables summary
        var_count = len(self._variables)
        lines.append(f"**Variables:** {var_count} registered")
        if var_count > 0 and verbose >= 1:
            for name, var in self._variables.items():
                try:
                    var_type = type(var).__name__
                    if hasattr(var, 'num_components'):
                        components = var.num_components
                        if components == 1:
                            var_desc = f"scalar"
                        elif components in [2, 3]:
                            var_desc = f"vector ({components}D)"
                        else:
                            var_desc = f"tensor ({components} components)"
                    else:
                        var_desc = "unknown type"
                    lines.append(f"  - `{name}`: {var_desc}")
                except:
                    lines.append(f"  - `{name}`: {type(var).__name__}")
        elif var_count > 0:
            var_names = list(self._variables.keys())
            if len(var_names) <= 3:
                lines.append(f"  - {', '.join(f'`{name}`' for name in var_names)}")
            else:
                lines.append(f"  - {', '.join(f'`{name}`' for name in var_names[:3])}, ...")
        lines.append("")

        # Swarms summary
        swarm_count = len(self._swarms)
        lines.append(f"**Swarms:** {swarm_count} registered")
        if swarm_count > 0 and verbose >= 1:
            for swarm_id, swarm in self._swarms.items():
                try:
                    if hasattr(swarm, 'data') and swarm.data is not None:
                        particle_count = len(swarm.data)
                        lines.append(f"  - Swarm {swarm_id}: {particle_count:,} particles")
                    else:
                        lines.append(f"  - Swarm {swarm_id}: unknown size")
                except:
                    lines.append(f"  - Swarm {swarm_id}: {type(swarm).__name__}")
        lines.append("")

        # Materials summary
        if show_materials and self.materials:
            mat_count = len(self.materials)
            lines.append(f"**Materials:** {mat_count} defined")
            if verbose >= 1:
                for mat_name, properties in self.materials.items():
                    prop_count = len(properties)
                    if prop_count <= 3:
                        prop_names = list(properties.keys())
                        lines.append(f"  - `{mat_name}`: {', '.join(prop_names)}")
                    else:
                        prop_names = list(properties.keys())[:3]
                        lines.append(f"  - `{mat_name}`: {', '.join(prop_names)}, ... ({prop_count} total)")
            else:
                mat_names = list(self.materials.keys())
                if len(mat_names) <= 3:
                    lines.append(f"  - {', '.join(f'`{name}`' for name in mat_names)}")
                else:
                    lines.append(f"  - {', '.join(f'`{name}`' for name in mat_names[:3])}, ...")
            lines.append("")

        # Solvers summary
        if verbose >= 2:
            solver_count = len(self._solvers)
            lines.append(f"**Solvers:** {solver_count} registered")
            if solver_count > 0:
                for name, solver in self._solvers.items():
                    lines.append(f"  - `{name}`: {type(solver).__name__}")
            lines.append("")

        # PETSc options
        if show_petsc and self.petsc_state:
            lines.append(f"**PETSc Options:** {len(self.petsc_state)} set")
            if verbose >= 1:
                for option, value in self.petsc_state.items():
                    lines.append(f"  - `{option}`: {value}")
            lines.append("")

        # Metadata
        if verbose >= 2 and self.metadata:
            lines.append(f"**Metadata:** {len(self.metadata)} entries")
            for key, value in self.metadata.items():
                if isinstance(value, dict):
                    lines.append(f"  - `{key}`: dict with {len(value)} items")
                elif isinstance(value, (list, tuple)):
                    lines.append(f"  - `{key}`: {type(value).__name__} with {len(value)} items")
                else:
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                    lines.append(f"  - `{key}`: {value_str}")
            lines.append("")

        # Usage hints
        lines.append("---")
        lines.append("**Usage hints:**")
        lines.append("- `model.view(verbose=1)` - Show variable and material details")
        lines.append("- `model.view(verbose=2)` - Show all components including solvers")
        lines.append("- `model.to_dict()` - Export complete configuration")
        lines.append("- `model.to_yaml()` - Export as YAML file")
        if self._variables:
            lines.append("- `model.get_variable('name')` - Access specific variables")
        if self.materials:
            lines.append("- `model.get_material('name')` - Access material properties")

        # Display as markdown
        content = "\n".join(lines)
        try:
            from IPython.display import Markdown, display
            display(Markdown(content))
        except (ImportError, NameError):
            # Fallback to plain text if not in Jupyter
            print("=" * 60)
            # Convert markdown to plain text
            plain_text = content.replace("# ", "").replace("**", "").replace("`", "'")
            print(plain_text)
            print("=" * 60)

    def __repr__(self):
        """Override Pydantic's __repr__ for better user experience."""
        mesh_info = f"mesh={type(self.mesh).__name__}" if self.mesh else f"meshes={len(self._meshes)}"
        var_count = len(self._variables)
        swarm_count = len(self._swarms)

        # Add units information
        ref_qty = self.get_reference_quantities()
        units_info = f"units={'set' if ref_qty else 'not_set'}"

        return (f"Model('{self.name}', {mesh_info}, "
                f"{var_count} variables, {swarm_count} swarms, {units_info})")

    def __str__(self):
        """String representation for print() calls."""
        return self.__repr__()

    def view(self):
        """
        Display comprehensive model information following the established view() pattern.

        Shows model configuration, units setup, registered components, and provides
        guidance for setting up units if not configured.
        """
        try:
            from IPython.display import Markdown, display

            # Build markdown content
            content = [f"## Model: {self.name}"]

            # Model state and basic info
            content.append(f"**State**: {self.state.value}")
            content.append(f"**Version**: {self.version}")

            # Mesh information
            if self.mesh:
                content.append(f"\n### Primary Mesh")
                content.append(f"- **Type**: {type(self.mesh).__name__}")
                content.append(f"- **Dimension**: {self.mesh.dim if hasattr(self.mesh, 'dim') else 'Unknown'}")

            total_meshes = len(self._meshes)
            if total_meshes > 1:
                content.append(f"- **Total meshes**: {total_meshes}")
            elif total_meshes == 0:
                content.append(f"\n### Meshes")
                content.append("⚠️ No meshes registered")

            # Variables and swarms
            var_count = len(self._variables)
            swarm_count = len(self._swarms)

            content.append(f"\n### Components")
            content.append(f"- **Variables**: {var_count}")
            content.append(f"- **Swarms**: {swarm_count}")
            content.append(f"- **Solvers**: {len(self._solvers)}")

            # Units information
            ref_qty = self.get_reference_quantities()
            content.append(f"\n### Units Configuration")

            if ref_qty:
                content.append(f"✅ **Reference quantities set** ({len(ref_qty)} quantities):")
                for name, info in ref_qty.items():
                    content.append(f"- **{name}**: `{info['value']}`")

                # Show derived fundamental scalings
                scalings = self.derive_fundamental_scalings()
                if scalings:
                    content.append(f"\n**Derived Fundamental Scalings:**")
                    derivation_info = self.metadata.get('derived_scalings', {}).get('derivation_info', {})
                    for dim in ['[length]', '[time]', '[mass]', '[temperature]']:
                        if dim in scalings:
                            value = scalings[dim]
                            source = derivation_info.get(dim, 'direct')
                            content.append(f"- **{dim.strip('[]').title()}**: `{value}` _{source}_")

                    content.append("\n💡 *Use `model.show_optimal_units()` to see recommended units for your problem*")
            else:
                content.append("⚠️ **No reference quantities set**")
                content.append("\nTo set up dimensional analysis:")
                content.append("```python")
                content.append("model.set_reference_quantities(")
                content.append("    mantle_temperature=1500*uw.units.K,")
                content.append("    mantle_viscosity=1e21*uw.units.Pa*uw.units.s,")
                content.append("    plate_velocity=5*uw.units.cm/uw.units.year")
                content.append(")")
                content.append("```")

            # Materials information
            if self.materials:
                content.append(f"\n### Materials ({len(self.materials)})")
                for mat_name, properties in self.materials.items():
                    content.append(f"- **{mat_name}**: {len(properties)} properties")

            # Additional metadata
            if self.metadata:
                non_ref_metadata = {k: v for k, v in self.metadata.items() if k != 'reference_quantities'}
                if non_ref_metadata:
                    content.append(f"\n### Metadata")
                    content.append(f"- **Entries**: {len(non_ref_metadata)}")

            display(Markdown("\n".join(content)))

        except ImportError:
            # Fallback for non-Jupyter environments using uw.pprint
            import underworld3 as uw

            uw.pprint(f"Model: {self.name}")
            uw.pprint("=" * 40)
            uw.pprint(f"State: {self.state.value}")
            uw.pprint(f"Version: {self.version}")

            # Mesh info
            if self.mesh:
                uw.pprint(f"\nPrimary Mesh: {type(self.mesh).__name__}")
                if hasattr(self.mesh, 'dim'):
                    uw.pprint(f"  Dimension: {self.mesh.dim}")

            # Components
            uw.pprint(f"\nComponents:")
            uw.pprint(f"  Variables: {len(self._variables)}")
            uw.pprint(f"  Swarms: {len(self._swarms)}")
            uw.pprint(f"  Solvers: {len(self._solvers)}")

            # Units
            ref_qty = self.get_reference_quantities()
            uw.pprint(f"\nUnits Configuration:")
            if ref_qty:
                uw.pprint(f"  Reference quantities: {len(ref_qty)} set")
                for name, info in ref_qty.items():
                    uw.pprint(f"    {name}: {info['value']}")

                # Show derived fundamental scalings
                scalings = self.derive_fundamental_scalings()
                if scalings:
                    uw.pprint(f"\n  Derived Fundamental Scalings:")
                    derivation_info = self.metadata.get('derived_scalings', {}).get('derivation_info', {})
                    for dim in ['[length]', '[time]', '[mass]', '[temperature]']:
                        if dim in scalings:
                            value = scalings[dim]
                            source = derivation_info.get(dim, 'direct')
                            uw.pprint(f"    {dim.strip('[]').title()}: {value} ({source})")

                    uw.pprint(f"\n  Use model.show_optimal_units() for detailed recommendations")
            else:
                uw.pprint("  No reference quantities set")
                uw.pprint("  To set up: model.set_reference_quantities(...)")

            # Materials
            if self.materials:
                uw.pprint(f"\nMaterials: {len(self.materials)}")
                for mat_name, properties in self.materials.items():
                    uw.pprint(f"  {mat_name}: {len(properties)} properties")


# Global default model for automatic registration
_default_model = None

def get_default_model():
    """
    Get or create the default model for this UW3 session.
    
    The default model automatically registers all meshes, swarms, variables,
    and solvers created during the session, enabling serialization and 
    model orchestration without explicit user interaction.
    
    Returns
    -------
    Model
        The default model instance for this session
        
    Example
    -------
    >>> import underworld3 as uw
    >>> model = uw.get_default_model()
    >>> print(model)  # See all registered objects
    >>> config = model.to_dict()  # Serialize model
    """
    global _default_model
    if _default_model is None:
        _default_model = Model(name="default")
    return _default_model

def reset_default_model():
    """
    Reset the default model to a fresh instance.
    
    Useful for testing or starting a new simulation in an interactive session.
    All previously registered objects will be orphaned from the model registry.
    
    Returns
    -------
    Model
        New default model instance
        
    Example
    -------
    >>> import underworld3 as uw
    >>> uw.reset_default_model()  # Start fresh
    """
    global _default_model
    _default_model = Model(name="default")
    return _default_model

# Example specialized model configuration classes

class ThermalConvectionConfig(BaseModel):
    """
    Configuration model for thermal convection simulations.

    Demonstrates how to create specialized parameter configurations
    that work with the enhanced Model infrastructure.
    """

    model_config = ConfigDict(extra="allow")  # Allow additional user-defined parameters

    # Mesh parameters
    mesh_type: str = Field(default="UnstructuredSimplexBox", description="Mesh generation function name")
    cellsize: float = Field(default=0.1, gt=0, description="Characteristic cell size")
    qdegree: int = Field(default=2, ge=1, description="Quadrature degree")

    # Physical parameters
    rayleigh_number: float = Field(default=1e4, gt=0, description="Rayleigh number")
    viscosity: float = Field(default=1.0, gt=0, description="Reference viscosity")
    thermal_diffusivity: float = Field(default=1.0, gt=0, description="Thermal diffusivity")

    # Boundary conditions
    temperature_top: float = Field(default=0.0, description="Top boundary temperature")
    temperature_bottom: float = Field(default=1.0, description="Bottom boundary temperature")
    velocity_boundary: str = Field(default="slip", description="Velocity boundary condition type")

    # Solver parameters
    stokes_solver_type: str = Field(default="pcdksp", description="Stokes solver type")
    stokes_tolerance: float = Field(default=1e-6, gt=0, description="Stokes solver tolerance")
    advdiff_solver_type: str = Field(default="lu", description="Advection-diffusion solver type")

    # Time stepping
    dt: float = Field(default=0.01, gt=0, description="Time step size")
    max_time: float = Field(default=1.0, gt=0, description="Maximum simulation time")
    max_steps: int = Field(default=100, ge=1, description="Maximum time steps")

    def to_petsc_options(self) -> Dict[str, str]:
        """
        Convert configuration to PETSc options dictionary.

        Returns
        -------
        dict
            PETSc options suitable for setting via Model.set_petsc_option()
        """
        petsc_opts = {}

        # Stokes solver options
        if self.stokes_solver_type == "pcdksp":
            petsc_opts.update({
                'ksp_type': 'fgmres',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'schur',
                'ksp_rtol': str(self.stokes_tolerance)
            })
        elif self.stokes_solver_type == "lu":
            petsc_opts.update({
                'ksp_type': 'preonly',
                'pc_type': 'lu',
            })

        # Advection-diffusion options
        if self.advdiff_solver_type == "lu":
            petsc_opts.update({
                'advdiff_ksp_type': 'preonly',
                'advdiff_pc_type': 'lu'
            })

        return petsc_opts

    def to_materials_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert configuration to materials dictionary format.

        Returns
        -------
        dict
            Materials dictionary suitable for Model.materials
        """
        return {
            'fluid': {
                'viscosity': self.viscosity,
                'rayleigh_number': self.rayleigh_number,
                'thermal_diffusivity': self.thermal_diffusivity
            }
        }


# Convenience functions for backward compatibility
def create_model(name: Optional[str] = None) -> Model:
    """Create a new Model instance"""
    return Model(name)

def create_thermal_convection_model(config: ThermalConvectionConfig, name: str = "thermal_convection") -> Model:
    """
    Create a Model instance configured for thermal convection.

    Demonstrates integration between specialized configs and Model infrastructure.

    Parameters
    ----------
    config : ThermalConvectionConfig
        Configuration object with all simulation parameters
    name : str
        Model name

    Returns
    -------
    Model
        Configured model ready for thermal convection simulation
    """
    model = Model(name=name)

    # Set materials from config
    model.materials.update(config.to_materials_dict())

    # Set PETSc options from config
    petsc_options = config.to_petsc_options()
    for option, value in petsc_options.items():
        model.set_petsc_option(option, value)

    # Store config in metadata for reproducibility
    model.metadata['simulation_config'] = config.dict()
    model.metadata['config_type'] = 'ThermalConvectionConfig'

    return model