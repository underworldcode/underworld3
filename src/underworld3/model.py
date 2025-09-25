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
import sympy


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
        self._variables = {}  # name -> variable mapping (changed to dict for simpler implementation)
        self._solvers = {}    # name -> solver mapping (changed to dict)
        
        # Simple configuration (no complex validation)
        self.materials = {}   # material_name -> {property: value/expression}
        self.metadata = {}    # User-defined metadata for serialization
        
        # Lifecycle tracking
        self._version = 0  # Incremented when model structure changes
        
        self.state = ModelState.CONFIGURED
    
    @property
    def mesh(self):
        """The primary mesh for this model"""
        return self._mesh
    
    def _register_mesh(self, mesh):
        """
        Internal method to register a mesh with this model.
        Called automatically from Mesh.__init__
        
        Parameters:
        -----------
        mesh : uw.discretisation.Mesh
            Mesh instance to register
        """
        self._mesh = mesh
        self._version += 1
    
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
        
        Parameters:
        -----------
        name : str
            Variable name
        variable : MeshVariable or SwarmVariable
            Variable instance to register
        """
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
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export model configuration to dictionary for serialization.
        
        Returns basic structure that can be serialized to JSON/YAML.
        Future versions will include full expression serialization.
        
        Returns:
        --------
        dict
            Model configuration dictionary
        """
        config = {
            'model_name': self.name,
            'model_version': self._version,
            'state': self.state.value,
            'mesh_type': type(self._mesh).__name__ if self._mesh else None,
            'variables': list(self._variables.keys()),
            'swarm_count': len(self._swarms),
            'solver_count': len(self._solvers),
            'materials': self.materials,  # Simple dict
            'metadata': self.metadata
        }
        
        return config
    
    def export_configuration(self) -> Dict[str, Any]:
        """Alias for to_dict() for backward compatibility"""
        return self.to_dict()
    
    def from_dict(self, config: Dict[str, Any]):
        """
        Import model configuration from dictionary.
        
        Note: Only imports materials and metadata for now.
        Mesh/variables/swarms must be recreated.
        """
        if 'materials' in config:
            self.materials = config['materials']
        if 'metadata' in config:
            self.metadata = config['metadata']
        print(f"Model '{self.name}': Imported configuration")
    
    def __repr__(self):
        mesh_info = f"mesh={type(self._mesh).__name__}" if self._mesh else "no mesh"
        var_count = len(self._variables)
        swarm_count = len(self._swarms)
        
        return (f"Model('{self.name}', {mesh_info}, "
                f"{var_count} variables, {swarm_count} swarms)")


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

# Convenience functions for backward compatibility
def create_model(name: Optional[str] = None) -> Model:
    """Create a new Model instance"""
    return Model(name)