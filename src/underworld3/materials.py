"""
Material Management System for Underworld3 Models

This module provides structured material management with property definitions,
region assignments, and automatic propagation to constitutive models and solvers.
"""

import weakref
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class MaterialProperty(Enum):
    """Standard material properties for geodynamic models"""

    # Mechanical properties
    VISCOSITY = "viscosity"
    DENSITY = "density"
    YIELD_STRESS = "yield_stress"
    COHESION = "cohesion"
    FRICTION_ANGLE = "friction_angle"

    # Thermal properties
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    THERMAL_DIFFUSIVITY = "thermal_diffusivity"
    HEAT_CAPACITY = "heat_capacity"
    THERMAL_EXPANSION = "thermal_expansion"

    # Elastic properties
    YOUNGS_MODULUS = "youngs_modulus"
    POISSONS_RATIO = "poissons_ratio"
    SHEAR_MODULUS = "shear_modulus"

    # Flow properties
    PERMEABILITY = "permeability"
    POROSITY = "porosity"

    # Custom properties
    CUSTOM = "custom"


@dataclass
class MaterialDefinition:
    """
    Definition of a material with its properties and metadata.

    Attributes:
    -----------
    name : str
        Material name (e.g., 'mantle', 'crust', 'plume')
    properties : dict
        Dictionary of property_name -> value mappings
    description : str
        Human-readable description
    reference : str
        Literature reference or source
    temperature_dependent : dict
        Temperature-dependent property functions
    pressure_dependent : dict
        Pressure-dependent property functions
    constitutive_model : object
        Associated constitutive model instance
    """

    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    reference: str = ""
    temperature_dependent: Dict[str, Callable] = field(default_factory=dict)
    pressure_dependent: Dict[str, Callable] = field(default_factory=dict)
    constitutive_model: Optional[Any] = None

    def set_property(self, prop: Union[MaterialProperty, str], value: Any):
        """Set a material property value"""
        prop_name = prop.value if isinstance(prop, MaterialProperty) else prop
        self.properties[prop_name] = value

    def get_property(self, prop: Union[MaterialProperty, str], default=None):
        """Get a material property value"""
        prop_name = prop.value if isinstance(prop, MaterialProperty) else prop
        return self.properties.get(prop_name, default)

    def has_property(self, prop: Union[MaterialProperty, str]) -> bool:
        """Check if material has a specific property"""
        prop_name = prop.value if isinstance(prop, MaterialProperty) else prop
        return prop_name in self.properties

    def evaluate_property(
        self, prop: Union[MaterialProperty, str], temperature=None, pressure=None
    ):
        """
        Evaluate a material property, accounting for temperature/pressure dependence.

        Parameters:
        -----------
        prop : MaterialProperty or str
            Property to evaluate
        temperature : float or array, optional
            Temperature for evaluation
        pressure : float or array, optional
            Pressure for evaluation

        Returns:
        --------
        Property value (scalar or array)
        """
        prop_name = prop.value if isinstance(prop, MaterialProperty) else prop

        # Get base value
        base_value = self.get_property(prop_name)
        if base_value is None:
            raise ValueError(f"Material '{self.name}' does not have property '{prop_name}'")

        # Apply temperature dependence
        if temperature is not None and prop_name in self.temperature_dependent:
            temp_func = self.temperature_dependent[prop_name]
            base_value = temp_func(base_value, temperature)

        # Apply pressure dependence
        if pressure is not None and prop_name in self.pressure_dependent:
            pressure_func = self.pressure_dependent[prop_name]
            base_value = pressure_func(base_value, pressure)

        return base_value


class MaterialRegistry:
    """
    Central registry for material definitions and region assignments.

    Features:
    ---------
    - Material property database with validation
    - Region-based material assignments
    - Temperature/pressure dependent properties
    - Integration with constitutive models
    - Automatic property propagation to solvers

    Example:
    --------
    >>> registry = MaterialRegistry()
    >>> mantle = registry.create_material('mantle')
    >>> mantle.set_property('viscosity', 1e21)
    >>> mantle.set_property('density', 3300)
    >>> registry.assign_to_region('mantle', region_id=1)
    """

    def __init__(self):
        self._materials: Dict[str, MaterialDefinition] = {}
        self._region_assignments: Dict[int, str] = {}  # region_id -> material_name
        self._callbacks: List[Callable] = []  # Material change callbacks
        self._version = 0

    def create_material(
        self, name: str, description: str = "", reference: str = ""
    ) -> MaterialDefinition:
        """
        Create a new material definition.

        Parameters:
        -----------
        name : str
            Material name
        description : str
            Human-readable description
        reference : str
            Literature reference

        Returns:
        --------
        MaterialDefinition
            New material instance
        """
        if name in self._materials:
            raise ValueError(f"Material '{name}' already exists")

        material = MaterialDefinition(name=name, description=description, reference=reference)

        self._materials[name] = material
        self._version += 1

        return material

    def get_material(self, name: str) -> Optional[MaterialDefinition]:
        """Get a material by name"""
        return self._materials.get(name)

    def list_materials(self) -> List[str]:
        """List all material names"""
        return list(self._materials.keys())

    def delete_material(self, name: str):
        """Delete a material definition"""
        if name in self._materials:
            del self._materials[name]
            # Remove any region assignments
            self._region_assignments = {
                region_id: mat_name
                for region_id, mat_name in self._region_assignments.items()
                if mat_name != name
            }
            self._version += 1

    def assign_to_region(self, material_name: str, region_id: int):
        """
        Assign a material to a mesh region.

        Parameters:
        -----------
        material_name : str
            Name of material to assign
        region_id : int
            Mesh region identifier
        """
        if material_name not in self._materials:
            raise ValueError(f"Material '{material_name}' does not exist")

        self._region_assignments[region_id] = material_name
        self._notify_callbacks("region_assignment", region_id, material_name)

    def get_region_material(self, region_id: int) -> Optional[str]:
        """Get the material assigned to a region"""
        return self._region_assignments.get(region_id)

    def get_material_regions(self, material_name: str) -> List[int]:
        """Get all regions assigned to a material"""
        return [
            region_id
            for region_id, mat_name in self._region_assignments.items()
            if mat_name == material_name
        ]

    def evaluate_property_field(
        self,
        prop: Union[MaterialProperty, str],
        region_field: np.ndarray,
        temperature: Optional[np.ndarray] = None,
        pressure: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluate a material property over a field of region IDs.

        Parameters:
        -----------
        prop : MaterialProperty or str
            Property to evaluate
        region_field : array
            Array of region IDs
        temperature : array, optional
            Temperature field for evaluation
        pressure : array, optional
            Pressure field for evaluation

        Returns:
        --------
        array
            Property values corresponding to each region
        """
        prop_name = prop.value if isinstance(prop, MaterialProperty) else prop

        # Initialize output array
        result = np.zeros_like(region_field, dtype=float)

        # Evaluate property for each unique region
        unique_regions = np.unique(region_field)

        for region_id in unique_regions:
            material_name = self.get_region_material(region_id)
            if material_name is None:
                raise ValueError(f"No material assigned to region {region_id}")

            material = self.get_material(material_name)
            if material is None:
                raise ValueError(f"Material '{material_name}' not found")

            # Get mask for this region
            mask = region_field == region_id

            # Extract temperature/pressure for this region if provided
            region_temp = temperature[mask] if temperature is not None else None
            region_pressure = pressure[mask] if pressure is not None else None

            # Evaluate property for this region
            prop_value = material.evaluate_property(prop_name, region_temp, region_pressure)

            # Assign to result
            result[mask] = prop_value

        return result

    def add_callback(self, callback: Callable):
        """
        Add a callback function for material changes.

        Parameters:
        -----------
        callback : callable
            Function called as callback(event_type, *args)
        """
        self._callbacks.append(callback)

    def _notify_callbacks(self, event_type: str, *args):
        """Notify all callbacks of a material change"""
        for callback in self._callbacks:
            try:
                callback(event_type, *args)
            except Exception as e:
                print(f"Warning: Material callback failed: {e}")

    def export_config(self) -> Dict[str, Any]:
        """Export material configuration"""
        return {
            "materials": {
                name: {
                    "properties": mat.properties,
                    "description": mat.description,
                    "reference": mat.reference,
                }
                for name, mat in self._materials.items()
            },
            "region_assignments": dict(self._region_assignments),
            "version": self._version,
        }

    def import_config(self, config: Dict[str, Any]):
        """Import material configuration from exported dict"""
        if "materials" in config:
            for name, mat_config in config["materials"].items():
                material = self.create_material(
                    name, mat_config.get("description", ""), mat_config.get("reference", "")
                )
                material.properties = mat_config.get("properties", {})

        if "region_assignments" in config:
            self._region_assignments = config["region_assignments"]

    def __repr__(self):
        return f"MaterialRegistry({len(self._materials)} materials, {len(self._region_assignments)} assignments)"


# Common material definitions for geodynamics
def create_standard_mantle_material(registry: MaterialRegistry) -> MaterialDefinition:
    """Create a standard mantle material with typical properties"""
    material = registry.create_material(
        "mantle",
        description="Standard upper mantle material",
        reference="Turcotte & Schubert (2014)",
    )

    material.set_property(MaterialProperty.VISCOSITY, 1e21)  # Pa·s
    material.set_property(MaterialProperty.DENSITY, 3300)  # kg/m³
    material.set_property(MaterialProperty.THERMAL_CONDUCTIVITY, 3.0)  # W/m/K
    material.set_property(MaterialProperty.THERMAL_DIFFUSIVITY, 1e-6)  # m²/s
    material.set_property(MaterialProperty.THERMAL_EXPANSION, 3e-5)  # K⁻¹

    return material


def create_standard_crust_material(registry: MaterialRegistry) -> MaterialDefinition:
    """Create a standard crustal material with typical properties"""
    material = registry.create_material(
        "crust",
        description="Standard continental crust material",
        reference="Turcotte & Schubert (2014)",
    )

    material.set_property(MaterialProperty.VISCOSITY, 1e22)  # Pa·s
    material.set_property(MaterialProperty.DENSITY, 2700)  # kg/m³
    material.set_property(MaterialProperty.THERMAL_CONDUCTIVITY, 2.5)  # W/m/K
    material.set_property(MaterialProperty.THERMAL_DIFFUSIVITY, 1e-6)  # m²/s
    material.set_property(MaterialProperty.THERMAL_EXPANSION, 3e-5)  # K⁻¹

    return material


def create_high_viscosity_material(
    registry: MaterialRegistry, name: str = "high_visc", viscosity_contrast: float = 1000
) -> MaterialDefinition:
    """Create a high viscosity material for inclusion studies"""
    material = registry.create_material(
        name,
        description=f"High viscosity material (contrast {viscosity_contrast}x)",
        reference="User defined",
    )

    # Base properties similar to mantle
    material.set_property(MaterialProperty.VISCOSITY, 1e21 * viscosity_contrast)
    material.set_property(MaterialProperty.DENSITY, 3300)
    material.set_property(MaterialProperty.THERMAL_CONDUCTIVITY, 3.0)
    material.set_property(MaterialProperty.THERMAL_DIFFUSIVITY, 1e-6)

    return material
