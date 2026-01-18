"""
Parameter Management System for Underworld3 Models

This module provides structured parameter management with validation,
dependency tracking, type checking, and automatic propagation of changes
to dependent simulation components.
"""

import weakref
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ParameterType(Enum):
    """Supported parameter types with validation"""

    SCALAR = "scalar"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    STRING = "string"
    ARRAY = "array"
    ENUM = "enum"
    OBJECT = "object"


@dataclass
class ParameterDefinition:
    """
    Definition of a parameter including validation rules and metadata.

    Attributes:
    -----------
    name : str
        Parameter name (e.g., 'viscosity', 'Ra', 'solver.tolerance')
    ptype : ParameterType
        Parameter type for validation
    default : Any
        Default value
    bounds : tuple, optional
        (min, max) bounds for numerical parameters
    choices : list, optional
        Valid choices for enum parameters
    description : str
        Human-readable description
    units : str
        Physical units (for documentation)
    validator : callable, optional
        Custom validation function
    dependencies : list
        List of parameter paths that depend on this parameter
    """

    name: str
    ptype: ParameterType
    default: Any = None
    bounds: Optional[tuple] = None
    choices: Optional[list] = None
    description: str = ""
    units: str = ""
    validator: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)

    def validate(self, value: Any) -> bool:
        """
        Validate a parameter value against this definition.

        Returns:
        --------
        bool
            True if value is valid, raises ValueError if not
        """
        if value is None and self.default is not None:
            return True

        # Type validation
        if self.ptype == ParameterType.SCALAR:
            if not isinstance(value, (int, float, np.number)):
                raise ValueError(f"Parameter '{self.name}' must be a number, got {type(value)}")
            if self.bounds and not (self.bounds[0] <= value <= self.bounds[1]):
                raise ValueError(
                    f"Parameter '{self.name}' must be in range {self.bounds}, got {value}"
                )

        elif self.ptype == ParameterType.INTEGER:
            if not isinstance(value, (int, np.integer)):
                raise ValueError(f"Parameter '{self.name}' must be an integer, got {type(value)}")
            if self.bounds and not (self.bounds[0] <= value <= self.bounds[1]):
                raise ValueError(
                    f"Parameter '{self.name}' must be in range {self.bounds}, got {value}"
                )

        elif self.ptype == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError(f"Parameter '{self.name}' must be a boolean, got {type(value)}")

        elif self.ptype == ParameterType.STRING:
            if not isinstance(value, str):
                raise ValueError(f"Parameter '{self.name}' must be a string, got {type(value)}")

        elif self.ptype == ParameterType.ARRAY:
            if not isinstance(value, (list, tuple, np.ndarray)):
                raise ValueError(f"Parameter '{self.name}' must be array-like, got {type(value)}")

        elif self.ptype == ParameterType.ENUM:
            if self.choices and value not in self.choices:
                raise ValueError(
                    f"Parameter '{self.name}' must be one of {self.choices}, got {value}"
                )

        # Custom validation
        if self.validator and not self.validator(value):
            raise ValueError(f"Parameter '{self.name}' failed custom validation")

        return True


class ParameterRegistry:
    """
    Central registry for model parameters with validation and dependency tracking.

    Features:
    ---------
    - Hierarchical parameter organization (e.g., 'material.viscosity', 'solver.tolerance')
    - Type validation and bounds checking
    - Dependency tracking and automatic updates
    - Parameter history and versioning
    - Import/export for reproducible configurations

    Example:
    --------
    >>> registry = ParameterRegistry()
    >>> registry.define_parameter('Ra', ParameterType.SCALAR, default=1e5,
    ...                          bounds=(1e3, 1e8), description='Rayleigh number')
    >>> registry.set_parameter('Ra', 2e5)
    >>> registry.get_parameter('Ra')
    2e5
    """

    def __init__(self):
        self._definitions: Dict[str, ParameterDefinition] = {}
        self._values: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = {}  # Parameter change callbacks
        self._history: List[Dict] = []  # Parameter change history
        self._version = 0

    def define_parameter(
        self,
        name: str,
        ptype: ParameterType,
        default: Any = None,
        bounds: Optional[tuple] = None,
        choices: Optional[list] = None,
        description: str = "",
        units: str = "",
        validator: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """
        Define a new parameter with validation rules.

        Parameters:
        -----------
        name : str
            Parameter path (e.g., 'material.viscosity')
        ptype : ParameterType
            Parameter type for validation
        default : Any
            Default value
        bounds : tuple, optional
            (min, max) bounds for numerical parameters
        choices : list, optional
            Valid choices for enum parameters
        description : str
            Human-readable description
        units : str
            Physical units
        validator : callable, optional
            Custom validation function
        dependencies : list, optional
            Parameters that depend on this one
        """
        definition = ParameterDefinition(
            name=name,
            ptype=ptype,
            default=default,
            bounds=bounds,
            choices=choices,
            description=description,
            units=units,
            validator=validator,
            dependencies=dependencies or [],
        )

        self._definitions[name] = definition

        # Set default value if provided
        if default is not None:
            self._values[name] = default

    def set_parameter(self, name: str, value: Any, validate: bool = True) -> None:
        """
        Set a parameter value with validation and dependency updates.

        Parameters:
        -----------
        name : str
            Parameter path
        value : Any
            New parameter value
        validate : bool
            Whether to validate the value
        """
        # Get definition for validation
        if name not in self._definitions:
            raise KeyError(f"Parameter '{name}' not defined. Use define_parameter() first.")

        definition = self._definitions[name]

        # Validate value
        if validate:
            definition.validate(value)

        # Store old value for history
        old_value = self._values.get(name)

        # Set new value
        self._values[name] = value
        self._version += 1

        # Record change in history
        self._history.append(
            {
                "version": self._version,
                "parameter": name,
                "old_value": old_value,
                "new_value": value,
                "timestamp": None,  # Could add actual timestamp
            }
        )

        # Trigger callbacks for this parameter
        if name in self._callbacks:
            for callback in self._callbacks[name]:
                try:
                    callback(name, value, old_value)
                except Exception as e:
                    print(f"Warning: Callback failed for parameter '{name}': {e}")

        # Update dependent parameters if needed
        for dep_name in definition.dependencies:
            if dep_name in self._callbacks:
                for callback in self._callbacks[dep_name]:
                    try:
                        callback(dep_name, self._values.get(dep_name), None)
                    except Exception as e:
                        print(f"Warning: Dependency callback failed for '{dep_name}': {e}")

    def get_parameter(self, name: str, default=None) -> Any:
        """Get a parameter value by name"""
        return self._values.get(name, default)

    def has_parameter(self, name: str) -> bool:
        """Check if a parameter is defined"""
        return name in self._definitions

    def list_parameters(self) -> Dict[str, Any]:
        """List all parameter names and current values"""
        return dict(self._values)

    def get_definition(self, name: str) -> Optional[ParameterDefinition]:
        """Get parameter definition by name"""
        return self._definitions.get(name)

    def add_callback(self, name: str, callback: Callable) -> None:
        """
        Add a callback function to be called when parameter changes.

        Parameters:
        -----------
        name : str
            Parameter path
        callback : callable
            Function called as callback(param_name, new_value, old_value)
        """
        if name not in self._callbacks:
            self._callbacks[name] = []
        self._callbacks[name].append(callback)

    def export_config(self) -> Dict[str, Any]:
        """Export current parameter configuration"""
        return {
            "parameters": dict(self._values),
            "version": self._version,
            "definitions": {
                name: {
                    "type": defn.ptype.value,
                    "description": defn.description,
                    "units": defn.units,
                    "bounds": defn.bounds,
                    "choices": defn.choices,
                }
                for name, defn in self._definitions.items()
            },
        }

    def import_config(self, config: Dict[str, Any], validate: bool = True) -> None:
        """Import parameter configuration from exported dict"""
        if "parameters" in config:
            for name, value in config["parameters"].items():
                if name in self._definitions:
                    self.set_parameter(name, value, validate=validate)
                else:
                    print(f"Warning: Skipping undefined parameter '{name}'")

    def __repr__(self):
        return f"ParameterRegistry({len(self._definitions)} parameters, version {self._version})"


# Common parameter definitions for geodynamics models
def define_thermal_convection_parameters(registry: ParameterRegistry) -> None:
    """Define standard parameters for thermal convection models"""

    registry.define_parameter(
        "Ra",
        ParameterType.SCALAR,
        default=1e5,
        bounds=(1e3, 1e8),
        description="Rayleigh number",
        units="dimensionless",
    )

    registry.define_parameter(
        "material.viscosity",
        ParameterType.SCALAR,
        default=1.0,
        bounds=(1e-6, 1e6),
        description="Reference viscosity",
        units="Pa·s",
    )

    registry.define_parameter(
        "material.thermal_diffusivity",
        ParameterType.SCALAR,
        default=1e-6,
        bounds=(1e-9, 1e-3),
        description="Thermal diffusivity",
        units="m²/s",
    )

    registry.define_parameter(
        "solver.tolerance",
        ParameterType.SCALAR,
        default=1e-6,
        bounds=(1e-12, 1e-3),
        description="Solver convergence tolerance",
        units="dimensionless",
    )

    registry.define_parameter(
        "solver.max_iterations",
        ParameterType.INTEGER,
        default=1000,
        bounds=(10, 10000),
        description="Maximum solver iterations",
        units="count",
    )

    registry.define_parameter(
        "mesh.resolution",
        ParameterType.ARRAY,
        default=[32, 32],
        description="Mesh resolution in each dimension",
        units="elements",
    )


def define_stokes_flow_parameters(registry: ParameterRegistry) -> None:
    """Define standard parameters for Stokes flow models"""

    registry.define_parameter(
        "viscosity_contrast",
        ParameterType.SCALAR,
        default=1e3,
        bounds=(1, 1e8),
        description="Viscosity contrast between materials",
        units="dimensionless",
    )

    registry.define_parameter(
        "boundary_conditions.velocity.top",
        ParameterType.ENUM,
        choices=["free_slip", "no_slip", "prescribed"],
        default="free_slip",
        description="Top boundary velocity condition",
    )

    registry.define_parameter(
        "boundary_conditions.velocity.bottom",
        ParameterType.ENUM,
        choices=["free_slip", "no_slip", "prescribed"],
        default="no_slip",
        description="Bottom boundary velocity condition",
    )
