"""
Underworld3 Model Architecture

This module provides the Model class that serves as a central orchestrator for 
simulation components, parameter management, and object lifecycle management.

The Model object eliminates circular references by becoming the central authority
for mesh, swarms, variables, solvers, and other simulation components.
"""

import weakref
from typing import Optional, Dict, Any, Union
from enum import Enum
import underworld3 as uw
from .parameters import ParameterRegistry, ParameterType
from .materials import MaterialRegistry, MaterialProperty


class ModelState(Enum):
    """Model lifecycle states"""
    INITIALIZING = "initializing"
    CONFIGURED = "configured" 
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class Model:
    """
    Central orchestrator for Underworld3 simulations.
    
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
    
    Example:
        >>> model = uw.Model()
        >>> model.set_mesh(mesh)
        >>> swarm = model.create_swarm()
        >>> temperature = model.add_variable("temperature", mesh, uw.VarType.SCALAR)
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a new Model instance.
        
        Parameters:
        -----------
        name : str, optional
            Human-readable name for this model instance
        """
        self.name = name or f"Model_{id(self)}"
        self.state = ModelState.INITIALIZING
        
        # Core simulation components (using weak references where appropriate)
        self._mesh = None
        self._swarms = weakref.WeakValueDictionary()
        self._variables = weakref.WeakValueDictionary()  # All variables (mesh + swarm)
        self._solvers = weakref.WeakValueDictionary()
        
        # Model configuration
        self.parameters = ParameterRegistry()
        self.materials = MaterialRegistry()
        
        # Lifecycle tracking
        self._version = 0  # Incremented when model structure changes
        
        self.state = ModelState.CONFIGURED
    
    @property
    def mesh(self):
        """The primary mesh for this model"""
        return self._mesh
    
    def set_mesh(self, mesh):
        """
        Set or replace the primary mesh for this model.
        
        This method handles:
        - Notifying all swarms of mesh changes
        - Triggering variable updates
        - Invalidating solver caches
        
        Parameters:
        -----------
        mesh : uw.discretisation.Mesh
            The new mesh to use for this model
        """
        old_mesh = self._mesh
        self._mesh = mesh
        
        # Increment version to signal structural change
        self._version += 1
        
        # TODO: Implement mesh change notifications
        # - Notify swarms of mesh change
        # - Update variable dependencies  
        # - Invalidate solver caches
        
        if old_mesh is not None:
            print(f"Model '{self.name}': Replaced mesh (version {self._version})")
        else:
            print(f"Model '{self.name}': Set initial mesh")
    
    def create_swarm(self, **kwargs) -> 'uw.swarm.Swarm':
        """
        Create a new swarm associated with this model.
        
        The swarm will reference the model instead of directly referencing the mesh,
        breaking circular dependencies.
        
        Parameters:
        -----------
        **kwargs : dict
            Arguments passed to Swarm constructor
            
        Returns:
        --------
        uw.swarm.Swarm
            New swarm instance registered with this model
        """
        if self._mesh is None:
            raise RuntimeError("Model must have a mesh before creating swarms")
        
        # TODO: Modify Swarm constructor to accept model parameter
        # For now, create swarm with mesh directly but register with model
        swarm = uw.swarm.Swarm(mesh=self._mesh, **kwargs)
        
        # Register swarm with model
        swarm_name = f"swarm_{len(self._swarms)}"
        self._swarms[swarm_name] = swarm
        
        return swarm
    
    def add_variable(self, name: str, container, vtype, **kwargs):
        """
        Add a variable to the model registry.
        
        Variables are tracked centrally to enable cross-container operations
        and lifecycle management.
        
        Parameters:
        -----------
        name : str
            Variable name
        container : uw.discretisation.Mesh or uw.swarm.Swarm
            Container for the variable
        vtype : uw.VarType
            Variable type (SCALAR, VECTOR, TENSOR, etc.)
        **kwargs : dict
            Additional arguments for variable constructor
            
        Returns:
        --------
        Variable instance (MeshVariable or SwarmVariable)
        """
        if isinstance(container, uw.discretisation.Mesh):
            variable = uw.discretisation.MeshVariable(
                name, container, vtype, **kwargs
            )
        elif hasattr(container, 'add_variable'):  # Swarm-like
            variable = container.add_variable(name, vtype, **kwargs)
        else:
            raise TypeError(f"Unsupported container type: {type(container)}")
        
        # Register variable with model
        self._variables[name] = variable
        
        return variable
    
    def get_variable(self, name: str):
        """Get a variable by name from the model registry"""
        return self._variables.get(name)
    
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
    
    def define_parameter(self, name: str, ptype: ParameterType, **kwargs):
        """
        Define a new parameter with validation rules.
        
        Parameters:
        -----------
        name : str
            Parameter path (e.g., 'material.viscosity', 'solver.tolerance')
        ptype : ParameterType
            Parameter type for validation
        **kwargs : dict
            Additional arguments passed to ParameterRegistry.define_parameter()
        """
        self.parameters.define_parameter(name, ptype, **kwargs)
    
    def set_parameter(self, path: str, value: Any, validate: bool = True):
        """
        Set a model parameter with validation.
        
        Parameters:
        -----------
        path : str
            Parameter path (e.g., 'material.viscosity', 'solver.tolerance')
        value : Any
            Parameter value
        validate : bool
            Whether to validate the value against parameter definition
        """
        if not self.parameters.has_parameter(path):
            # Auto-define parameter if not already defined
            self.parameters.define_parameter(path, ParameterType.OBJECT, default=value)
        
        self.parameters.set_parameter(path, value, validate=validate)
        print(f"Model '{self.name}': Set parameter {path} = {value}")
    
    def get_parameter(self, path: str, default=None):
        """Get a parameter value by path"""
        return self.parameters.get_parameter(path, default)
    
    def create_material(self, name: str, description: str = "", reference: str = ""):
        """
        Create a new material definition for this model.
        
        Parameters:
        -----------
        name : str
            Material name (e.g., 'mantle', 'crust', 'plume')
        description : str
            Human-readable description
        reference : str
            Literature reference
            
        Returns:
        --------
        MaterialDefinition
            New material instance
        """
        return self.materials.create_material(name, description, reference)
    
    def get_material(self, name: str):
        """Get a material by name"""
        return self.materials.get_material(name)
    
    def assign_material_to_region(self, material_name: str, region_id: int):
        """Assign a material to a mesh region"""
        self.materials.assign_to_region(material_name, region_id)
        print(f"Model '{self.name}': Assigned material '{material_name}' to region {region_id}")
    
    def setup_standard_materials(self):
        """Create standard mantle and crust materials"""
        from .materials import create_standard_mantle_material, create_standard_crust_material
        
        mantle = create_standard_mantle_material(self.materials)
        crust = create_standard_crust_material(self.materials)
        
        print(f"Model '{self.name}': Created standard mantle and crust materials")
        return mantle, crust
    
    def setup_thermal_convection_parameters(self):
        """Configure standard parameters for thermal convection models"""
        from .parameters import define_thermal_convection_parameters
        define_thermal_convection_parameters(self.parameters)
        print(f"Model '{self.name}': Configured thermal convection parameters")
        
    def setup_stokes_flow_parameters(self):
        """Configure standard parameters for Stokes flow models"""
        from .parameters import define_stokes_flow_parameters
        define_stokes_flow_parameters(self.parameters)
        print(f"Model '{self.name}': Configured Stokes flow parameters")
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export complete model configuration including parameters and materials"""
        config = {
            'model_name': self.name,
            'model_version': self._version,
            'state': self.state.value,
            'parameters': self.parameters.export_config(),
            'materials': self.materials.export_config(),
            'variables': list(self._variables.keys()),
            'swarms': list(self._swarms.keys()),
            'solvers': list(self._solvers.keys())
        }
        return config
    
    def import_configuration(self, config: Dict[str, Any]):
        """Import model configuration from exported dict"""
        if 'parameters' in config:
            self.parameters.import_config(config['parameters'])
        if 'materials' in config:
            self.materials.import_config(config['materials'])
        print(f"Model '{self.name}': Imported configuration")
    
    def __repr__(self):
        mesh_info = f"mesh={type(self._mesh).__name__}" if self._mesh else "no mesh"
        var_count = len(self._variables)
        swarm_count = len(self._swarms)
        
        return (f"Model('{self.name}', {mesh_info}, "
                f"{var_count} variables, {swarm_count} swarms)")


# Convenience functions for backward compatibility
def create_model(name: Optional[str] = None) -> Model:
    """Create a new Model instance"""
    return Model(name)