r"""
Particle swarm management for Lagrangian tracking.

This module provides particle swarm (point cloud) data structures for
tracking material properties through deformation. Swarms enable Lagrangian
representations of material history, composition, and other quantities
that move with the flow.

Key Components
--------------
SwarmType : enum
    PETSc swarm type specification (BASIC or PIC).
SwarmVariable : class
    Variable storing values at particle locations with mesh-based proxy
    for use in symbolic expressions.
IndexSwarmVariable : class
    Integer-valued swarm variable for material indexing.

The swarm module integrates with PETSc's DMSwarm for parallel particle
management and provides automatic population, advection, and repopulation
capabilities.

See Also
--------
underworld3.discretisation : Mesh discretisation classes.
underworld3.systems.ddt : Time derivative schemes using swarms.
"""
from posixpath import pardir
import petsc4py.PETSc as PETSc

import numpy as np
import sympy
import h5py
import os
import warnings
import weakref
from typing import Optional, Tuple

import underworld3 as uw
from underworld3.utilities._api_tools import Stateful
from underworld3.utilities._api_tools import uw_object
from underworld3.utilities.mathematical_mixin import MathematicalMixin

import underworld3.timing as timing

comm = uw.mpi.comm

from enum import Enum


# We can grab this type from the PETSc module
class SwarmType(Enum):
    """
    PETSc swarm type specification.

    Determines how particles are managed by PETSc's DMSwarm infrastructure.

    Attributes
    ----------
    DMSWARM_BASIC : int
        Basic point cloud without mesh association.
    DMSWARM_PIC : int
        Particle-in-cell mode with automatic mesh cell tracking.
        Particles are migrated between MPI ranks as they move across
        cell boundaries.
    """

    DMSWARM_BASIC = 0
    DMSWARM_PIC = 1


# We can grab this type from the PETSc module
# SwarmPICLayout has been moved to pic_swarm.py


# Note - much of the setup is necessarily the same as the MeshVariable
# and the duplication should be removed.

from underworld3.utilities.dimensionality_mixin import DimensionalityMixin


class SwarmVariable(DimensionalityMixin, MathematicalMixin, Stateful, uw_object):
    r"""
    Variable supported by a particle swarm (point cloud).

    A SwarmVariable stores values at discrete particle locations and provides
    a mesh-based proxy representation for use in symbolic expressions. This
    enables Lagrangian tracking of material properties through deformation.

    Parameters
    ----------
    name : str
        Identifier for this variable (must be unique within the swarm).
    swarm : Swarm
        The supporting particle swarm.
    size : int or tuple, optional
        Shape specification: int for vectors, tuple for matrices.
        If None, inferred from ``vtype``.
    vtype : VarType, optional
        Variable type (SCALAR, VECTOR, TENSOR, SYM_TENSOR, MATRIX).
        If None, inferred from ``size``.
    dtype : type, default=float
        Data type for storage (float or int).
    proxy_degree : int, default=1
        Polynomial degree for the mesh proxy variable.
    proxy_continuous : bool, default=True
        Whether the proxy uses continuous (True) or discontinuous (False)
        interpolation.
    varsymbol : str, optional
        LaTeX symbol for display. Defaults to ``name``.
    rebuild_on_cycle : bool, default=True
        If True, rebuild the proxy when particles cycle through periodic
        boundaries. Recommended for continuous fields.
    units : str or pint.Unit, optional
        Physical units for this variable (e.g., 'kelvin', 'Pa').
        Requires reference quantities to be set on the model.

    Attributes
    ----------
    data : numpy.ndarray
        Direct access to variable values at particle locations.
    sym : sympy.Matrix
        Symbolic representation for use in expressions.

    See Also
    --------
    MeshVariable : Variable supported by mesh nodes.
    Swarm : Container for particle locations.

    Examples
    --------
    Create a temperature field on a swarm:

    >>> swarm = uw.swarm.Swarm(mesh)
    >>> T = swarm.add_variable("T", size=1, vtype=uw.VarType.SCALAR)
    >>> T.data[:] = 1600.0  # Set initial temperature

    Create a velocity field:

    >>> v = swarm.add_variable("v", size=mesh.dim, vtype=uw.VarType.VECTOR)

    Notes
    -----
    SwarmVariables are essential for tracking material properties that
    advect with the flow. The mesh proxy enables their use in finite
    element formulations while particle storage preserves Lagrangian
    history.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        name,
        swarm,
        size=None,  # only needed if MATRIX type
        vtype=None,
        dtype=float,
        proxy_degree=1,
        proxy_continuous=True,
        _register=True,
        _proxy=True,
        _nn_proxy=False,
        varsymbol=None,
        rebuild_on_cycle=True,
        units=None,
        units_backend=None,
    ):
        if name in swarm.vars.keys():
            raise ValueError("Variable with name {} already exists on swarm.".format(name))

        import re
        import sympy
        import math

        if varsymbol is None:
            varsymbol = name

        self.name = name
        self.clean_name = re.sub(r"[^a-zA-Z0-9_]", "", name)
        self.symbol = varsymbol

        self._swarm_ref = weakref.ref(swarm)
        self.shape = size

        mesh = swarm.mesh

        if vtype == None:
            if isinstance(size, int) and size == 1:
                vtype = uw.VarType.SCALAR
            elif isinstance(size, int) and size == mesh.dim:
                vtype = uw.VarType.VECTOR
            elif isinstance(size, tuple):
                if size[0] == mesh.dim and size[1] == mesh.dim:
                    vtype = uw.VarType.TENSOR
                else:
                    vtype = uw.VarType.MATRIX
            else:
                raise ValueError(
                    "Unable to infer variable type from `num_components`. Please explicitly set the `vtype` parameter."
                )

        self.vtype = vtype

        # Store unit metadata for variable
        # Convert string units to Pint Unit objects for consistency with MeshVariable
        if units is not None:
            if isinstance(units, str):
                # Parse string units to Pint Unit object
                # uw.units('K') returns a Quantity (1 kelvin), so we extract .units to get the Unit
                self._units = uw.units(units).units
            elif hasattr(units, "dimensionality"):
                # Already a pint.Unit object
                self._units = units
            else:
                # Fallback: store as-is (shouldn't happen)
                self._units = units

            # units_backend parameter is deprecated - Pint is the only supported backend
            if units_backend is not None and units_backend != "pint":
                raise ValueError(
                    f"Unknown units backend: {units_backend}. Only 'pint' is supported."
                )
        else:
            self._units = None

        # STRICT UNITS MODE CHECK
        # Enforce units-scales contract: variables with units require reference quantities
        if units is not None:
            model = uw.get_default_model()

            # Check if strict mode is enabled
            if uw.is_strict_units_active() and not model.has_units():
                raise ValueError(
                    f"Strict units mode: Cannot create swarm variable '{name}' with units='{units}' "
                    f"when model has no reference quantities.\n\n"
                    f"Options:\n"
                    f"  1. Set reference quantities FIRST:\n"
                    f"     model = uw.get_default_model()\n"
                    f"     model.set_reference_quantities(\n"
                    f"         domain_depth=uw.quantity(1000, 'km'),\n"
                    f"         plate_velocity=uw.quantity(5, 'cm/year')\n"
                    f"     )\n\n"
                    f"  2. Remove units parameter (use plain numbers):\n"
                    f"     swarm.add_variable('{name}', ...)  # No units\n\n"
                    f"  3. Disable strict mode (not recommended):\n"
                    f"     uw.use_strict_units(False)\n"
                )

            # If not strict mode and no reference quantities, warn as before
            if not model.has_units():
                warnings.warn(
                    f"\nSwarm variable '{name}' has units '{units}' but no reference quantities are set.\n"
                    f"Call model.set_reference_quantities() before creating variables with units.\n"
                    f"Variable will use scaling_coefficient=1.0, which may lead to poor numerical conditioning.\n"
                    f"Consider enabling strict mode: uw.use_strict_units(True)",
                    UserWarning
                )

        if not isinstance(vtype, uw.VarType):
            raise ValueError(
                "'vtype' must be an instance of 'Variable_Type', for example `underworld.VarType.SCALAR`."
            )

        if vtype == uw.VarType.SCALAR:
            self.num_components = 1
            self.shape = (1, 1)
            self.cpt_map = 0
        elif vtype == uw.VarType.VECTOR:
            self.num_components = mesh.dim
            self.shape = (1, mesh.dim)
            self.cpt_map = tuple(range(0, mesh.dim))
        elif vtype == uw.VarType.TENSOR:
            self.num_components = mesh.dim * mesh.dim
            self.shape = (mesh.dim, mesh.dim)
        elif vtype == uw.VarType.SYM_TENSOR:
            self.num_components = math.comb(mesh.dim + 1, 2)
            self.shape = (mesh.dim, mesh.dim)
        elif vtype == uw.VarType.MATRIX:
            self.num_components = self.shape[0] * self.shape[1]

        self._data_container = np.empty(self.shape, dtype=object)

        if (dtype == float) or (dtype == "float") or (dtype == np.float64):
            self.dtype = float
            petsc_type = PETSc.ScalarType
        elif (dtype == int) or (dtype == "int") or (dtype == np.int32) or (dtype == np.int64):
            self.dtype = int
            petsc_type = PETSc.IntType
        else:
            raise TypeError(
                f"Provided dtype={dtype} is not supported. Supported types are 'int' and 'float'."
            )

        if _register:
            # Check if swarm is already populated - PETSc doesn't allow registering
            # new fields after DMSwarmFinalizeFieldRegister() has been called
            if self.swarm.local_size > 0:
                raise RuntimeError(
                    f"Cannot add variable '{name}' to swarm: swarm is already populated "
                    f"with {self.swarm.local_size} particles. Variables must be created "
                    f"before calling swarm.populate() or any other operation that adds particles.\n"
                    f"\nCorrect usage:\n"
                    f"  swarm = uw.swarm.Swarm(mesh)\n"
                    f"  variable = swarm.add_variable('{name}', {size})  # Create variables first\n"
                    f"  swarm.populate(fill_param=3)  # Then populate with particles"
                )

            self.swarm.dm.registerField(self.clean_name, self.num_components, dtype=petsc_type)

        self._data = None
        self._cached_data = None
        # add to swarms dict

        self.swarm._vars[self.clean_name] = self
        self._is_accessed = False

        # Initialize proxy flags first before creating proxy variable
        self._updating_proxy = False  # Flag to prevent recursive proxy updates
        self._proxy_stale = True  # Flag to track if proxy needs updating (lazy evaluation)

        # proxy variable
        self._proxy = _proxy
        self._vtype = vtype
        self._proxy_degree = proxy_degree
        self._proxy_continuous = proxy_continuous
        self._nn_proxy = _nn_proxy
        self._create_proxy_variable()

        # recycle swarm
        self._rebuild_on_cycle = rebuild_on_cycle
        self._register = _register

        from collections import namedtuple

        SwarmVariable_ij = namedtuple("SwarmVariable_ij", ["data", "sym"])

        if self._proxy:
            for i in range(0, self.shape[0]):
                for j in range(0, self.shape[1]):
                    self._data_container[i, j] = SwarmVariable_ij(
                        data=f"SwarmVariable[...].data is only available within mesh.access() context",
                        sym=self.sym[i, j],
                    )

        # Initialize DimensionalityMixin
        DimensionalityMixin.__init__(self)

        super().__init__()

        # Array interface now unified using NDArray_With_Callback (no legacy/enhanced split)
        self._array_cache = None
        self._flat_data_cache = None

        # Register with default model for orchestration
        uw.get_default_model()._register_variable(self.name, self)

        return

    @property
    def units(self):
        """Return the units associated with this variable."""
        return self._units

    @units.setter
    def units(self, value):
        """Set the units for this variable."""
        # Convert string units to Pint Unit objects for consistency
        if value is not None and isinstance(value, str):
            self._units = uw.units(value).units
        else:
            self._units = value

    @property
    def has_units(self):
        """Check if this variable has units."""
        return self._units is not None

    def _create_variable_array(self, initial_data=None):
        """
        Factory function to create NDArray_With_Callback for variable data.
        Follows the same pattern as swarm.points implementation.

        Parameters
        ----------
        initial_data : numpy.ndarray, optional
            Initial data for the array. If None, fetches current data from PETSc.

        Returns
        -------
        NDArray_With_Callback
            Array object with callback for automatic PETSc synchronization
        """
        if initial_data is None:
            initial_data = self.unpack_uw_data_from_petsc(squeeze=False, sync=True)

        # Create NDArray_With_Callback (following swarm._points pattern)
        array_obj = uw.utilities.NDArray_With_Callback(
            initial_data,
            owner=self,
            disable_inplace_operators=False,  # Allow operations like existing arrays
        )

        # Single callback function (following swarm_update_callback pattern)
        def variable_update_callback(array, change_context):
            """Callback to sync variable changes back to PETSc (like swarm.points)"""
            # Only act on data-changing operations (following swarm.points pattern)
            data_changed = change_context.get("data_has_changed", True)
            if not data_changed:
                return

            # Skip updates during coordinate changes to prevent corruption
            if hasattr(self.swarm, "_migration_disabled") and self.swarm._migration_disabled:
                return

            # Persist changes to PETSc (like swarm callback updates coordinates)
            self.pack_uw_data_to_petsc(array, sync=True)

        # Register the callback (following swarm.points pattern)
        array_obj.add_callback(variable_update_callback)
        return array_obj

    def _create_canonical_data_array(self, initial_data=None):
        """
        Create the single canonical data array with PETSc synchronization.
        This is the ONLY method that creates arrays with PETSc callbacks.

        Returns data in shape (-1, num_components) using pack_raw/unpack_raw methods.

        Parameters
        ----------
        initial_data : numpy.ndarray, optional
            Initial data for the array. If None, fetches current data from PETSc.

        Returns
        -------
        NDArray_With_Callback
            Canonical array object with callback for automatic PETSc synchronization
        """
        if initial_data is None:
            # Use unpack_raw to get flat format (-1, num_components)
            initial_data = self.unpack_raw_data_from_petsc(squeeze=False, sync=True)

            # Handle case where unpack returns None (swarm not initialized)
            if initial_data is None:
                initial_data = np.zeros((0, self.num_components))

        # Create NDArray_With_Callback for flat data
        array_obj = uw.utilities.NDArray_With_Callback(
            initial_data,
            owner=self,
            disable_inplace_operators=False,  # Allow operations like existing arrays
        )

        # Single canonical callback for PETSc synchronization
        def canonical_data_callback(array, change_context):
            """ONLY callback that handles PETSc synchronization - prevents conflicts"""
            # Only act on data-changing operations
            data_changed = change_context.get("data_has_changed", True)
            if not data_changed:
                return

            # Skip updates during migration to prevent corruption
            if hasattr(self.swarm, "_migration_disabled") and self.swarm._migration_disabled:
                return

            # Check for None array to prevent copy errors
            if array is None:
                return

            # STEP 1: Ensure array has correct canonical shape before PETSc sync
            # The callback might receive wrong-shaped arrays from array view operations
            import numpy as np

            canonical_array = np.atleast_2d(array)
            if canonical_array.shape != (array.shape[0], self.num_components):
                # Reshape to canonical format: (-1, num_components)
                canonical_array = canonical_array.reshape(-1, self.num_components)

            # STEP 1: Sync to PETSc using established method with correct shape
            self.pack_raw_data_to_petsc(canonical_array, sync=True)

            # STEP 2: Handle variable-specific updates (like IndexSwarmVariable proxy marking)
            if hasattr(self, "_on_data_changed"):
                self._on_data_changed()

        # Register the single canonical callback
        array_obj.add_callback(canonical_data_callback)
        return array_obj

    def _create_array_view(self):
        """
        Create array view of canonical data using appropriate conversion strategy.

        Strategy depends on variable complexity:
        - Scalars/Vectors: Simple reshape operations
        - 2D+ Tensors: Complex pack/unpack operations

        Returns
        -------
        ArrayView
            Array-like object that delegates changes back to canonical data
        """
        if self._is_simple_variable():
            return self._create_simple_array_view()
        else:
            return self._create_tensor_array_view()

    def _is_simple_variable(self):
        """Check if this is a simple scalar/vector variable (not a complex tensor)"""
        return len(self.shape) <= 1 or (len(self.shape) == 2 and self.shape[1] == 1)

    def _create_simple_array_view(self):
        """Array view for scalars/vectors using simple reshape operations"""
        import numpy as np

        class SimpleSwarmArrayView:
            def __init__(self, parent_var):
                self.parent = parent_var

            def _get_array_data(self):
                # Simple reshape: (-1, num_components) -> (N, a, b)
                data = self.parent.data
                # For simple variables, reshape to (N, a, b) format
                reshaped = data.reshape(data.shape[0], *self.parent.shape)

                # Apply dimensionalization if needed
                import underworld3 as uw
                from .utilities.unit_aware_array import UnitAwareArray

                # Check if variable has units and model has reference quantities
                model = uw.get_default_model()
                has_units = hasattr(self.parent, "units") and self.parent.units is not None

                if has_units and model.has_units():
                    # Variable has units - wrap with UnitAwareArray
                    from .scaling import units as ureg
                    var_units = self.parent.units
                    if isinstance(var_units, str):
                        var_units = ureg(var_units)

                    # If ND scaling is active, data is non-dimensional and needs dimensionalization
                    if uw.is_nondimensional_scaling_active():
                        # Get dimensionality
                        pint_qty = 1.0 * var_units
                        dimensionality = dict(pint_qty.dimensionality)

                        # Dimensionalize: ND → dimensional using model reference scales
                        # This returns a UnitAwareArray with SI base units
                        dimensional_values = uw.dimensionalise(reshaped, target_dimensionality=dimensionality)

                        # Convert from SI base units to variable's units (e.g., m/s → cm/yr)
                        return dimensional_values.to(var_units)
                    else:
                        # ND scaling not active - data is already dimensional
                        return UnitAwareArray(reshaped, units=var_units)
                else:
                    # No units - return plain array
                    return reshaped

            def __getitem__(self, key):
                return self._get_array_data()[key]

            def __setitem__(self, key, value):
                import underworld3 as uw

                # PRINCIPLE (2025-11-27): When units are active and variable has units,
                # we REQUIRE unit-aware input to avoid ambiguity. Plain arrays are ambiguous:
                # are they dimensional or non-dimensional? We don't guess.
                #
                # - Use .array for unit-aware assignment (requires UnitAwareArray)
                # - Use .data for non-dimensional assignment (plain arrays OK)

                has_unit_info = hasattr(value, 'magnitude') or hasattr(value, 'value')
                model = uw.get_default_model()
                var_has_units = hasattr(self.parent, 'units') and self.parent.units is not None
                units_active = model.has_units() and uw.is_nondimensional_scaling_active()

                if not has_unit_info and var_has_units and units_active:
                    # Plain array assigned to unit-aware variable with scaling active
                    # This is ambiguous - reject with helpful error
                    var_units = self.parent.units
                    raise ValueError(
                        f"Cannot assign plain array to '{self.parent.name}.array' when units are active.\n"
                        f"\n"
                        f"The variable '{self.parent.name}' has units '{var_units}', but the assigned\n"
                        f"value has no unit information. This is ambiguous: should the values be\n"
                        f"interpreted as dimensional (in {var_units}) or non-dimensional?\n"
                        f"\n"
                        f"Solutions:\n"
                        f"  1. Wrap with units: UnitAwareArray(data, units='{var_units}')\n"
                        f"  2. Use uw.function.evaluate() which returns unit-aware arrays\n"
                        f"  3. For non-dimensional values, use: {self.parent.name}.data[...] = value\n"
                    )

                # Get current NON-DIMENSIONAL array data
                # Note: We use data directly here, not _get_array_data() which dimensionalizes
                raw_data = self.parent.data
                array_data = raw_data.reshape(raw_data.shape[0], *self.parent.shape)
                # Create a copy to modify (avoid modifying view directly)
                modified_data = array_data.copy()

                if has_unit_info:
                    # Value has units - need full conversion pipeline
                    if model.has_units() and var_has_units:
                        from .scaling import units as ureg

                        # Step 1: Convert to variable's units
                        target_units_str = self.parent.units if isinstance(self.parent.units, str) else str(self.parent.units)
                        converted = value.to(target_units_str)

                        # Extract numerical value
                        if hasattr(converted, 'value'):
                            dimensional_value = converted.value
                        elif hasattr(converted, 'magnitude'):
                            dimensional_value = converted.magnitude
                        else:
                            dimensional_value = float(converted)

                        # Step 2: Non-dimensionalize if scaling active
                        if uw.is_nondimensional_scaling_active():
                            target_units = ureg(target_units_str)
                            temp_qty = uw.quantity(dimensional_value, target_units)
                            nd_value = uw.non_dimensionalise(temp_qty)
                            if hasattr(nd_value, 'value'):
                                value = nd_value.value
                            elif hasattr(nd_value, 'magnitude'):
                                value = nd_value.magnitude
                            else:
                                value = nd_value
                        else:
                            value = dimensional_value
                    else:
                        # No units mode - just extract value/magnitude
                        if hasattr(value, 'value'):
                            value = value.value
                        elif hasattr(value, 'magnitude'):
                            value = value.magnitude
                        else:
                            value = float(value)

                # Update the specific elements
                modified_data[key] = value
                # Reshape back to canonical data format: ensure exact shape match
                reshaped_data = modified_data.reshape(-1, self.parent.num_components)
                self.parent.data[:] = reshaped_data

            # Forward common array methods
            def max(self):
                return self._get_array_data().max()

            def min(self):
                return self._get_array_data().min()

            def mean(self):
                """
                Compute mean value of swarm particles.

                ⚠️  WARNING: This computes a simple arithmetic mean of the particle values.
                Since swarm particles are typically non-uniformly distributed in space,
                this is an APPROXIMATION of the spatial mean. For accurate spatial
                statistics, consider using integration via swarm proxy variables or
                computing mesh integrals of the proxy field.

                Returns
                -------
                float or tuple
                    Mean value (float for scalar variables, tuple for multi-component)
                """
                return self._get_array_data().mean()

            def sum(self):
                return self._get_array_data().sum()

            def std(self):
                """
                Compute standard deviation of swarm particles.

                ⚠️  WARNING: This computes a simple numpy std of the particle values.
                Since swarm particles are typically non-uniformly distributed in space,
                this is an APPROXIMATION of the spatial standard deviation. For accurate
                spatial statistics, consider using integration via swarm proxy variables
                or computing mesh integrals of the proxy field.

                Returns
                -------
                float or tuple
                    Standard deviation (float for scalar variables, tuple for multi-component)
                """
                return self._get_array_data().std()

            @property
            def shape(self):
                return self._get_array_data().shape

            @property
            def dtype(self):
                return self._get_array_data().dtype

            def __array__(self):
                """Support for numpy functions like np.allclose(), np.isfinite(), etc."""
                return self._get_array_data()

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                """Support for numpy universal functions"""
                # Convert all SimpleSwarmArrayView inputs to arrays
                converted_inputs = []
                for input in inputs:
                    if hasattr(input, "_get_array_data"):  # Duck typing for array views
                        converted_inputs.append(input._get_array_data())
                    else:
                        converted_inputs.append(input)

                # Apply the ufunc to the converted inputs
                return ufunc(*converted_inputs, **kwargs)

            def delay_callback(self, description="array operation"):
                """Delegate to parent's canonical data delay_callback method"""
                return self.parent.data.delay_callback(description)

        return SimpleSwarmArrayView(self)

    def _create_tensor_array_view(self):
        """Array view for complex tensors using pack/unpack operations"""
        import numpy as np

        class TensorSwarmArrayView:
            def __init__(self, parent_var):
                self.parent = parent_var

            def _get_array_data(self):
                # Use complex pack/unpack for tensor layouts
                unpacked = self.parent.unpack_uw_data_from_petsc(squeeze=False)

                # Apply dimensionalization if needed
                import underworld3 as uw
                from .utilities.unit_aware_array import UnitAwareArray

                # Check if variable has units and model has reference quantities
                model = uw.get_default_model()
                has_units = hasattr(self.parent, "units") and self.parent.units is not None

                if has_units and model.has_units():
                    # Variable has units - wrap with UnitAwareArray
                    from .scaling import units as ureg
                    var_units = self.parent.units
                    if isinstance(var_units, str):
                        var_units = ureg(var_units)

                    # If ND scaling is active, data is non-dimensional and needs dimensionalization
                    if uw.is_nondimensional_scaling_active():
                        # Get dimensionality
                        pint_qty = 1.0 * var_units
                        dimensionality = dict(pint_qty.dimensionality)

                        # Dimensionalize: ND → dimensional using model reference scales
                        # This returns a UnitAwareArray with SI base units
                        dimensional_values = uw.dimensionalise(unpacked, target_dimensionality=dimensionality)

                        # Convert from SI base units to variable's units (e.g., m/s → cm/yr)
                        return dimensional_values.to(var_units)
                    else:
                        # ND scaling not active - data is already dimensional
                        return UnitAwareArray(unpacked, units=var_units)
                else:
                    # No units - return plain array
                    return unpacked

            def __getitem__(self, key):
                return self._get_array_data()[key]

            def __setitem__(self, key, value):
                import underworld3 as uw

                # PRINCIPLE (2025-11-27): When units are active and variable has units,
                # we REQUIRE unit-aware input to avoid ambiguity. Plain arrays are ambiguous:
                # are they dimensional or non-dimensional? We don't guess.
                #
                # - Use .array for unit-aware assignment (requires UnitAwareArray)
                # - Use .data for non-dimensional assignment (plain arrays OK)

                has_unit_info = hasattr(value, 'magnitude') or hasattr(value, 'value')
                model = uw.get_default_model()
                var_has_units = hasattr(self.parent, 'units') and self.parent.units is not None
                units_active = model.has_units() and uw.is_nondimensional_scaling_active()

                if not has_unit_info and var_has_units and units_active:
                    # Plain array assigned to unit-aware variable with scaling active
                    # This is ambiguous - reject with helpful error
                    var_units = self.parent.units
                    raise ValueError(
                        f"Cannot assign plain array to '{self.parent.name}.array' when units are active.\n"
                        f"\n"
                        f"The variable '{self.parent.name}' has units '{var_units}', but the assigned\n"
                        f"value has no unit information. This is ambiguous: should the values be\n"
                        f"interpreted as dimensional (in {var_units}) or non-dimensional?\n"
                        f"\n"
                        f"Solutions:\n"
                        f"  1. Wrap with units: UnitAwareArray(data, units='{var_units}')\n"
                        f"  2. Use uw.function.evaluate() which returns unit-aware arrays\n"
                        f"  3. For non-dimensional values, use: {self.parent.name}.data[...] = value\n"
                    )

                # Get current NON-DIMENSIONAL array data from PETSc
                # Note: We use unpack directly here, not _get_array_data() which dimensionalizes
                array_data = self.parent.unpack_uw_data_from_petsc(squeeze=False)
                # Create a copy to modify (avoid modifying view directly)
                modified_data = array_data.copy()

                if has_unit_info:
                    # Value has units - need full conversion pipeline
                    if model.has_units() and var_has_units:
                        from .scaling import units as ureg

                        # Step 1: Convert to variable's units
                        target_units_str = self.parent.units if isinstance(self.parent.units, str) else str(self.parent.units)
                        converted = value.to(target_units_str)

                        # Extract numerical value
                        if hasattr(converted, 'value'):
                            dimensional_value = converted.value
                        elif hasattr(converted, 'magnitude'):
                            dimensional_value = converted.magnitude
                        else:
                            dimensional_value = float(converted)

                        # Step 2: Non-dimensionalize if scaling active
                        if uw.is_nondimensional_scaling_active():
                            target_units = ureg(target_units_str)
                            temp_qty = uw.quantity(dimensional_value, target_units)
                            nd_value = uw.non_dimensionalise(temp_qty)
                            if hasattr(nd_value, 'value'):
                                value = nd_value.value
                            elif hasattr(nd_value, 'magnitude'):
                                value = nd_value.magnitude
                            else:
                                value = nd_value
                        else:
                            value = dimensional_value
                    else:
                        # No units mode - just extract value/magnitude
                        if hasattr(value, 'value'):
                            value = value.value
                        elif hasattr(value, 'magnitude'):
                            value = value.magnitude
                        else:
                            value = float(value)

                # Update the specific elements
                modified_data[key] = value
                # Pack back to canonical data format
                packed_data = self.parent._pack_array_to_data_format(modified_data)
                self.parent.data[:] = packed_data

            # Forward common array methods
            def max(self):
                return self._get_array_data().max()

            def min(self):
                return self._get_array_data().min()

            def mean(self):
                """
                Compute mean value of swarm particles.

                ⚠️  WARNING: This computes a simple arithmetic mean of the particle values.
                Since swarm particles are typically non-uniformly distributed in space,
                this is an APPROXIMATION of the spatial mean. For accurate spatial
                statistics, consider using integration via swarm proxy variables or
                computing mesh integrals of the proxy field.

                Returns
                -------
                float or tuple
                    Mean value (float for scalar variables, tuple for multi-component)
                """
                return self._get_array_data().mean()

            def sum(self):
                return self._get_array_data().sum()

            def std(self):
                """
                Compute standard deviation of swarm particles.

                ⚠️  WARNING: This computes a simple numpy std of the particle values.
                Since swarm particles are typically non-uniformly distributed in space,
                this is an APPROXIMATION of the spatial standard deviation. For accurate
                spatial statistics, consider using integration via swarm proxy variables
                or computing mesh integrals of the proxy field.

                Returns
                -------
                float or tuple
                    Standard deviation (float for scalar variables, tuple for multi-component)
                """
                return self._get_array_data().std()

            @property
            def shape(self):
                return self._get_array_data().shape

            @property
            def dtype(self):
                return self._get_array_data().dtype

            def __array__(self):
                """Support for numpy functions like np.allclose(), np.isfinite(), etc."""
                return self._get_array_data()

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                """Support for numpy universal functions"""
                # Convert all TensorSwarmArrayView inputs to arrays
                converted_inputs = []
                for input in inputs:
                    if hasattr(input, "_get_array_data"):  # Duck typing for array views
                        converted_inputs.append(input._get_array_data())
                    else:
                        converted_inputs.append(input)

                # Apply the ufunc to the converted inputs
                return ufunc(*converted_inputs, **kwargs)

            def delay_callback(self, description="array operation"):
                """Delegate to parent's canonical data delay_callback method"""
                return self.parent.data.delay_callback(description)

        return TensorSwarmArrayView(self)

    def _pack_array_to_data_format(self, array_data):
        """Convert array format (N,a,b) back to canonical data format (N,components)"""
        # Use existing pack logic but return numpy array instead of writing to PETSc
        # This is a pure conversion method - no PETSc access
        return array_data.reshape(array_data.shape[0], -1)

    # Legacy methods preserved for backward compatibility (now do nothing)
    def use_legacy_array(self):
        """Deprecated: Array interface is now unified using NDArray_With_Callback"""
        pass

    def use_enhanced_array(self):
        """Deprecated: Array interface is now unified using NDArray_With_Callback"""
        pass

    def sync_disabled(self, description="batch operation"):
        """
        Context manager to disable automatic synchronization for batch operations.
        Now uses NDArray_With_Callback's delay_callback mechanism.

        Parameters
        ----------
        description : str
            Description of the batch operation for debugging
        """
        # Use NDArray_With_Callback's built-in delay mechanism
        return self.array.delay_callback(description)

    ## Should be a single master copy (mesh variable / swarm variable)
    def _data_layout(self, i, j=None):
        # mapping

        if self.vtype == uw.VarType.SCALAR:
            return 0
        if self.vtype == uw.VarType.VECTOR:
            if i < 0 or j < 0:
                return self.swarm.dim
            else:
                if j is None:
                    return i
                elif i == 0:
                    return j
                else:
                    raise IndexError(
                        f"Vectors have shape {self.swarm.dim} or {(1, self.swarm.dim)} "
                    )
        if self.vtype == uw.VarType.TENSOR:
            if self.swarm.dim == 2:
                if i < 0 or j < 0:
                    return 4
                else:
                    return ((0, 1), (2, 3))[i][j]
            else:
                if i < 0 or j < 0:
                    return 9
                else:
                    return ((0, 1, 2), (3, 4, 5), (6, 7, 8))[i][j]

        if self.vtype == uw.VarType.SYM_TENSOR:
            if self.swarm.dim == 2:
                if i < 0 or j < 0:
                    return 3
                else:
                    return ((0, 2), (2, 1))[i][j]
            else:
                if i < 0 or j < 0:
                    return 6
                else:
                    return ((0, 3, 4), (3, 1, 5), (4, 5, 2))[i][j]

        if self.vtype == uw.VarType.MATRIX:
            if i < 0 or j < 0:
                return self.shape[0] * self.shape[1]
            else:
                return i + j * self.shape[0]

    def _create_proxy_variable(self):
        # release if defined
        old_meshVar = getattr(self, "_meshVar", None)
        self._meshVar = None

        if self._proxy:
            self._meshVar = uw.discretisation.MeshVariable(
                "proxy_" + self.clean_name,
                self.swarm.mesh,
                self.shape,
                self._vtype,
                degree=self._proxy_degree,
                continuous=self._proxy_continuous,
                varsymbol=r"\left<" + self.symbol + r"\right>",
            )

    def _update(self):
        """
        Mark proxy mesh variable as stale for lazy evaluation.
        The actual update happens when the proxy is accessed.
        """

        # if not proxied, nothing to do. return.
        if not self._meshVar:
            return

        # Mark proxy as stale for lazy evaluation (avoids immediate PETSc access conflicts)
        self._proxy_stale = True

        return

    def _update_proxy_if_stale(self):
        """
        Actually update the proxy mesh variable if it's marked as stale.
        This implements lazy evaluation to avoid PETSc access conflicts.
        """

        # if not proxied, nothing to do. return.
        if not self._meshVar:
            return

        # Only update if stale and not already updating
        if not self._proxy_stale or self._updating_proxy:
            return

        try:
            self._updating_proxy = True
            self._rbf_to_meshVar(self._meshVar)
            self._proxy_stale = False  # Mark as fresh
        finally:
            self._updating_proxy = False

        return

    # Maybe rbf_interpolate for this one and meshVar is a special case
    def _rbf_to_meshVar(self, meshVar, nnn=None, verbose=False):
        """
        Here is how it works: for each particle, create a distance-weighted average on the node data

        Todo: caching the k-d trees etc for the proxy-mesh-variable nodal points
        Todo: some form of global fall-back for when there are no particles on a processor
        """

        # Mapping to the coordinates of the variable from the
        # particle coords

        if nnn is None:
            nnn = self.swarm.mesh.dim + 1

        if meshVar.mesh != self.swarm.mesh:
            # If this is our own proxy variable and mesh has changed, recreate it
            if hasattr(self, "_meshVar") and meshVar is self._meshVar:
                self._create_proxy_variable()
                # Use the newly created proxy variable
                meshVar = self._meshVar
            else:
                raise RuntimeError("Cannot map a swarm to a different mesh")

        new_coords = meshVar.coords

        Values = self.rbf_interpolate(new_coords, verbose=verbose, nnn=nnn)

        meshVar.data[...] = Values[...]

        return

    def _rbf_reduce_to_meshVar(self, meshVar, verbose=False):
        """
        This method updates a mesh variable for the current
        swarm & particle variable state by reducing the swarm to
        the nearest point for each particle

        Here is how it works:

            1) for each particle, create a distance-weighted average on the node data
            2) check to see which nodes have zero weight / zero contribution and replace with nearest particle value

        Todo: caching the k-d trees etc for the proxy-mesh-variable nodal points
        Todo: some form of global fall-back for when there are no particles on a processor

        """

        # if not proxied, nothing to do. return.
        if not self._meshVar:
            return

        # 1 - Average particles to nodes with distance weighted average

        # Use non-dimensional coordinates for internal KDTree (matches swarm.data coordinate system)
        kd = uw.kdtree.KDTree(meshVar.coords_nd)

        with self.swarm.access():
            d, n = kd.query(self.swarm.data, k=1, sqr_dists=False)  # need actual distances

            node_values = np.zeros((meshVar.coords.shape[0], self.num_components))
            w = np.zeros(meshVar.coords.shape[0])

            if not self._nn_proxy:
                for i in range(self.local_size):
                    # if b[i]:
                    node_values[n[i], :] += self.data[i, :] / (1.0e-24 + d[i])
                    w[n[i]] += 1.0 / (1.0e-24 + d[i])

                node_values[np.where(w > 0.0)[0], :] /= w[np.where(w > 0.0)[0]].reshape(-1, 1)

        # 2 - set NN vals on mesh var where w == 0.0

        p_nnmap = self.swarm._get_map(self)

        meshVar.data[...] = node_values[...]
        meshVar.data[np.where(w == 0.0), :] = self.data[p_nnmap[np.where(w == 0.0)], :]

        return

    # # Need to be able to unpack as well
    # def pack_raw_data_to_petsc(self, data_array):
    #     """Convert an array in the correct shape for the underlying variable into something that can be loaded into
    #     the flat storage structure used by PETSc in a numpy assigment (with index broadcasting etc)
    #     """

    #     shape = self.shape
    #     storage_size = self._data_layout(-1)
    #     data_array_3d = data_array.reshape(-1, *self.shape)

    #     with self.swarm.access(self):
    #         for i in range(shape[0]):
    #             for j in range(shape[1]):
    #                 ij = self._data_layout(i, j)
    #                 self._data[:, ij] = data_array_3d[:, i, j]

    #     return

    def pack_uw_data_to_petsc(self, data_array, sync=True):
        """
        Enhanced pack method that directly accesses PETSc field without access() context.
        Designed for the new swarmVariable.array interface.

        Parameters
        ----------
        data_array : numpy.ndarray
            Array data to pack into PETSc field
        sync : bool
            Whether to sync parallel operations (default True)
        """
        shape = self.shape
        data_array_3d = data_array.reshape(-1, *self.shape)

        # Direct PETSc field access without context manager
        petsc_data = self.swarm.dm.getField(self.clean_name).reshape((-1, self.num_components))

        try:
            # Pack data using same layout as original method
            for i in range(shape[0]):
                for j in range(shape[1]):
                    ij = self._data_layout(i, j)
                    petsc_data[:, ij] = data_array_3d[:, i, j]

            # Increment variable state to track changes
            self._increment()

            # Update the proxy mesh variable if one exists (for integral calculations)
            self._update()

            # Sync parallel operations if requested
            if sync:
                # TODO: Add parallel sync logic here if needed
                pass

        finally:
            # Always restore the field
            self.swarm.dm.restoreField(self.clean_name)

    # def unpack_raw_data_to_petsc(self, squeeze=True):
    #     """Return an array in the correct shape for the underlying variable from
    #     the flat storage structure used by PETSc. By default, use numpy squeeze to remove additional
    #     dimensions (keep those dimensions to leave all data as 3D array - scalars being shape (1,1), vectors
    #     being (1,dim) and so on)
    #     """

    #     shape = self.shape

    #     with self.swarm.access():
    #         points = self._data.shape[0]
    #         data_array_3d = np.empty(shape=(points, *shape), dtype=self._data.dtype)

    #         for i in range(shape[0]):
    #             for j in range(shape[1]):
    #                 ij = self._data_layout(i, j)
    #                 data_array_3d[:, i, j] = self._data[:, ij]

    #     if squeeze:
    #         return data_array_3d.squeeze()
    #     else:
    #         return data_array_3d

    def unpack_uw_data_from_petsc(self, squeeze=True, sync=True):
        """
        Enhanced unpack method that directly accesses PETSc field without access() context.
        Designed for the new swarmVariable.array interface.

        Parameters
        ----------
        squeeze : bool
            Whether to squeeze singleton dimensions (default True)
        sync : bool
            Whether to sync parallel operations (default True)
        """
        shape = self.shape

        # Direct PETSc field access without context manager
        petsc_data = self.swarm.dm.getField(self.clean_name).reshape((-1, self.num_components))

        try:
            # Sync parallel operations if requested
            if sync:
                # TODO: Add parallel sync logic here if needed
                pass

            # Unpack data using same layout as original method
            points = petsc_data.shape[0]
            data_array_3d = np.empty(shape=(points, *shape), dtype=petsc_data.dtype)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    ij = self._data_layout(i, j)
                    data_array_3d[:, i, j] = petsc_data[:, ij]

        finally:
            # Always restore the field
            self.swarm.dm.restoreField(self.clean_name)

        if squeeze:
            return data_array_3d.squeeze()
        else:
            return data_array_3d

    def pack_raw_data_to_petsc(self, data_array, sync=True):
        """
        Pack data array to PETSc using traditional data shape (-1, num_components).
        Direct PETSc access without access() context for backward compatibility.

        Parameters
        ----------
        data_array : numpy.ndarray
            Array data in traditional flat format (-1, num_components)
        sync : bool
            Whether to sync parallel operations (default True)
        """
        import numpy as np

        # Convert to expected shape: (-1, num_components)
        data_array = np.atleast_2d(data_array)
        if data_array.shape[1] != self.num_components:
            raise ValueError(
                f"Data array must have shape (-1, {self.num_components}), got {data_array.shape}"
            )

        # Direct PETSc field access without context manager
        petsc_data = self.swarm.dm.getField(self.clean_name).reshape((-1, self.num_components))

        try:
            # Direct assignment in traditional flat format
            petsc_data[:] = data_array

            # Increment variable state to track changes
            self._increment()

            # Update the proxy mesh variable if one exists (for integral calculations)
            self._update()

            # Sync parallel operations if requested
            if sync:
                # TODO: Add parallel sync logic here if needed
                pass

        finally:
            # Always restore the field
            self.swarm.dm.restoreField(self.clean_name)

        return

    def unpack_raw_data_from_petsc(self, squeeze=True, sync=True):
        """
        Unpack data from PETSc in traditional data shape (-1, num_components).
        Direct PETSc access without access() context for backward compatibility.

        Parameters
        ----------
        squeeze : bool
            Whether to remove singleton dimensions (default True)
        sync : bool
            Whether to sync parallel operations (default True)

        Returns
        -------
        numpy.ndarray
            Array data in traditional flat format (-1, num_components)
        """
        import numpy as np

        # Check if swarm has any particles before accessing field
        swarm_size = self.swarm.local_size
        if swarm_size <= 0:
            # Swarm not populated yet, return empty array
            return np.zeros((0, self.num_components))

        # Direct PETSc field access without context manager
        field_data = self.swarm.dm.getField(self.clean_name)
        if field_data is None:
            # Field not properly initialized, restore and return empty array
            self.swarm.dm.restoreField(self.clean_name)
            return np.zeros((0, self.num_components))

        petsc_data = field_data.reshape((-1, self.num_components))

        try:
            # Return data in traditional flat format
            result = petsc_data.copy()

            # Sync parallel operations if requested
            if sync:
                # TODO: Add parallel sync logic here if needed
                pass

        finally:
            # Always restore the field
            self.swarm.dm.restoreField(self.clean_name)

        if squeeze:
            return result.squeeze()
        else:
            return result

    def _object_viewer(self):
        """This will substitute specific information about this object"""
        from IPython.display import Latex, Markdown, display
        from textwrap import dedent

        # feedback on this instance
        #
        display(
            Markdown(
                f"""**SwarmVariable:**
  > symbol:  ${self.symbol}$\n
  > shape:   ${self.shape}$\n
  > proxy:   ${self._proxy}$\n
  > proxy_degree:  ${self._proxy_degree}$\n
  > proxy_continuous:  `{self._proxy_continuous}`\n
  > type:    `{self.vtype.name}`"""
            ),
        )

        display(self.data),
        return

    def rbf_interpolate(self, new_coords, verbose=False, nnn=None):
        """
        Radial basis function interpolation of particle data to arbitrary points.

        Uses inverse-distance weighting to interpolate particle values
        to new coordinate locations.

        Parameters
        ----------
        new_coords : numpy.ndarray
            Target coordinates of shape (N, dim) to interpolate to.
        verbose : bool, default=False
            Print diagnostic information during interpolation.
        nnn : int, optional
            Number of nearest neighbors to use. Defaults to ``mesh.dim + 1``.

        Returns
        -------
        numpy.ndarray
            Interpolated values at the target coordinates.
        """
        # An inverse-distance mapping is quite robust here ... as long
        # as we take care of the case where some nodes coincide (likely if used with mesh2mesh)
        # We try to eliminate contributions from recently remeshed particles

        import numpy as np

        # Get data directly from PETSc to avoid circular callback dependencies
        raw_data = self.unpack_raw_data_from_petsc(squeeze=False, sync=False)
        data_size = raw_data.shape

        # What to do if there are no particles
        if data_size[0] <= 1:
            return np.zeros((new_coords.shape[0], data_size[1]))

        if nnn is None:
            nnn = self.swarm.mesh.dim + 1

        if nnn > data_size[0]:
            nnn = data_size[0]

        # Use direct PETSc access to avoid callback circular dependency
        if self.swarm.recycle_rate > 1:
            not_remeshed = self.swarm._remeshed.data[:, 0] != 0
            D = raw_data[not_remeshed].copy()

            kdt = uw.kdtree.KDTree(self.swarm._particle_coordinates.data[not_remeshed, :])
        else:
            D = raw_data.copy()
            kdt = uw.kdtree.KDTree(self.swarm._particle_coordinates.data[:, :])

            # kdt.build_index()

            values = kdt.rbf_interpolator_local(new_coords, D, nnn, 2, verbose)

            del kdt

        return values

    @property
    def swarm(self):
        """
        The swarm this variable belongs to (accessed via weak reference).
        Raises RuntimeError if the swarm has been garbage collected.
        """
        if self._swarm_ref is None:
            raise RuntimeError("SwarmVariable has no swarm reference (internal error)")

        swarm = self._swarm_ref()
        if swarm is None:
            raise RuntimeError(
                f"Swarm for variable '{self.clean_name}' has been garbage collected. "
                "Variables cannot outlive their parent swarm."
            )
        return swarm

    @property
    def old_data(self):
        """TESTING: Original data property implementation."""
        if self._data is None:
            raise RuntimeError("Data must be accessed via the swarm `access()` context manager.")
        return self._data

    @property
    def data(self):
        """
        Canonical data storage with PETSc synchronization.
        Shape: (-1, num_components) - flat format for backward compatibility.

        This is the ONLY property that handles PETSc synchronization to avoid conflicts.
        The .array property uses this as its underlying storage with format conversion.

        Returns
        -------
        NDArray_With_Callback
            Array with shape (-1, num_components) with automatic PETSc synchronization
        """
        # Cache and reuse canonical data object to avoid field access conflicts
        # Use direct __dict__ check to avoid MathematicalMixin recursion
        if "_canonical_data" not in self.__dict__ or self._canonical_data is None:
            # Create the single canonical data array with PETSc sync
            self._canonical_data = self._create_canonical_data_array()

        return self._canonical_data

    @property
    def array(self):
        """
        Array view of canonical data with automatic format conversion.
        Shape: (N, a, b) for tensor shape (a, b).

        This property is ALWAYS a view of the canonical .data property.
        No direct PETSc access - all changes delegate back to canonical storage.
        """
        return self._create_array_view()

    @array.setter
    def array(self, array_value):
        """
        Set variable data through canonical data property with format conversion.
        """
        if self._is_simple_variable():
            # Simple case: reshape array format (N,a,b) to canonical format (N,components)
            canonical_data = array_value.reshape(array_value.shape[0], -1)
        else:
            # Complex case: use pack operations for tensor layout conversion
            canonical_data = self._pack_array_to_data_format(array_value)

        # Assign to canonical data property (triggers PETSc sync)
        self.data[:] = canonical_data

    @property
    def sym(self):
        r"""Symbolic representation for use in equations.

        Returns the symbolic expression from the proxy mesh variable,
        which can be used in SymPy expressions for constitutive models,
        boundary conditions, etc.

        Returns
        -------
        sympy.Matrix
            Symbolic matrix expression.

        Notes
        -----
        The proxy is automatically updated if particle data has changed.
        """
        # Ensure proxy is up to date before returning symbolic representation
        self._update_proxy_if_stale()
        return self._meshVar.sym

    @property
    def sym_1d(self):
        r"""Flattened symbolic representation.

        Returns the symbolic expression as a 1D (column) vector form,
        useful for Voigt notation in tensor calculations.

        Returns
        -------
        sympy.Matrix
            Flattened symbolic expression.
        """
        # Ensure proxy is up to date before returning symbolic representation
        self._update_proxy_if_stale()
        return self._meshVar.sym_1d

    # Global statistics methods (MPI-aware) for particle data
    # Note: Only methods that make sense for non-uniformly distributed particles
    # are provided. Mean/RMS/variance are NOT provided because particles cluster
    # unevenly in the domain, making these statistics misleading.

    @uw.collective_operation
    def global_max(self, axis=None, out=None, keepdims=False):
        """
        Maximum value across all MPI ranks.

        Finds the maximum value of the particle property across all processors.
        Useful for finding extreme values in particle swarm data.

        Parameters
        ----------
        axis : None, int, or tuple of ints, optional
            Axis or axes along which to operate. By default, flattened input is used.
        out : None, optional
            Alternative output array (not supported, kept for API compatibility).
        keepdims : bool, optional
            If True, reduced axes are left as dimensions with size one.

        Returns
        -------
        UWQuantity or scalar
            Maximum value with units preserved (if variable has units).

        Examples
        --------
        >>> max_temp = temperature_swarm.global_max()
        >>> print(f"Maximum temperature: {max_temp}")

        Notes
        -----
        This is a collective operation - all ranks must call it.
        The result is identical on all ranks.
        """
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        # Wrap data in UnitAwareArray to use its global_max implementation
        temp_array = UnitAwareArray(self.data, units=self._units)
        return temp_array.global_max(axis=axis, out=out, keepdims=keepdims)

    @uw.collective_operation
    def global_min(self, axis=None, out=None, keepdims=False):
        """
        Minimum value across all MPI ranks.

        Finds the minimum value of the particle property across all processors.
        Useful for finding extreme values in particle swarm data.

        Parameters
        ----------
        axis : None, int, or tuple of ints, optional
            Axis or axes along which to operate. By default, flattened input is used.
        out : None, optional
            Alternative output array (not supported, kept for API compatibility).
        keepdims : bool, optional
            If True, reduced axes are left as dimensions with size one.

        Returns
        -------
        UWQuantity or scalar
            Minimum value with units preserved (if variable has units).

        Examples
        --------
        >>> min_pressure = pressure_swarm.global_min()
        >>> print(f"Minimum pressure: {min_pressure}")

        Notes
        -----
        This is a collective operation - all ranks must call it.
        The result is identical on all ranks.
        """
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        temp_array = UnitAwareArray(self.data, units=self._units)
        return temp_array.global_min(axis=axis, out=out, keepdims=keepdims)

    @uw.collective_operation
    def global_sum(self, axis=None, out=None, keepdims=False):
        """
        Sum of values across all MPI ranks.

        Computes the sum of particle property values across all processors.

        Parameters
        ----------
        axis : None, int, or tuple of ints, optional
            Axis or axes along which to operate. By default, flattened input is used.
        out : None, optional
            Alternative output array (not supported, kept for API compatibility).
        keepdims : bool, optional
            If True, reduced axes are left as dimensions with size one.

        Returns
        -------
        UWQuantity or scalar
            Sum with units preserved (if variable has units).

        Notes
        -----
        This is a collective operation - all ranks must call it.
        The result is identical on all ranks.

        Warning: This sum is NOT a physical domain-integrated quantity because
        particles are non-uniformly distributed. For domain integration, use
        the proxy mesh variable with uw.maths.Integral().
        """
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        temp_array = UnitAwareArray(self.data, units=self._units)
        return temp_array.global_sum(axis=axis, out=out, keepdims=keepdims)

    @uw.collective_operation
    def global_norm(self, ord=None):
        """
        L2 norm (Frobenius norm) across all MPI ranks.

        Computes the L2 norm of particle property values: sqrt(sum(x**2))
        across all processors.

        Parameters
        ----------
        ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
            Order of the norm (default: None = 2-norm)

        Returns
        -------
        UWQuantity or scalar
            L2 norm with units preserved (if variable has units).

        Notes
        -----
        This is a collective operation - all ranks must call it.
        The result is identical on all ranks.

        For vectors, computes the Frobenius norm treating the array as flattened.

        Warning: This norm is NOT a physical domain-integrated quantity because
        particles are non-uniformly distributed.
        """
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        temp_array = UnitAwareArray(self.data, units=self._units)
        return temp_array.global_norm(ord=ord)

    @uw.collective_operation
    def global_size(self):
        """
        Total particle count across all MPI ranks.

        Returns the total number of particles across all processors.
        Useful for population monitoring and load balancing diagnostics.

        Returns
        -------
        int
            Total number of particles across all ranks.

        Examples
        --------
        >>> total_particles = swarm_var.global_size()
        >>> local_particles = swarm_var.data.shape[0]
        >>> print(f"Rank has {local_particles} of {total_particles} particles")

        Notes
        -----
        This is a collective operation - all ranks must call it.
        The result is identical on all ranks.
        """
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        temp_array = UnitAwareArray(self.data, units=self._units)
        return temp_array.global_size()

    @timing.routine_timer_decorator
    def save(
        self,
        filename: int,
        compression: Optional[bool] = False,
        compressionType: Optional[str] = "gzip",
        force_sequential=False,
    ):
        """

        Save the swarm variable to a h5 file.

        Parameters
        ----------
        filename :
            The filename of the swarm variable to save to disk.
        compression :
            Add compression to the h5 files (saves space but increases write times with increasing no. of processors)
        compressionType :
            Type of compression to use, 'gzip' and 'lzf' supported. 'gzip' is default. Compression also needs to be set to 'True'.

        force_sequential : activate the serial version of hdf5

        """
        if h5py.h5.get_config().mpi == False and comm.size > 1 and comm.rank == 0:
            warnings.warn(
                "Collective IO not possible as h5py not available in parallel mode. Switching to sequential. This will be slow for models running on multiple processors",
                stacklevel=2,
            )
        if compression == True and comm.rank == 0:
            warnings.warn("Compression may slow down write times", stacklevel=2)
        if filename.endswith(".h5") == False:
            raise RuntimeError("The filename must end with .h5")

        if h5py.h5.get_config().mpi == True and not force_sequential:
            with h5py.File(f"{filename[:-3]}.h5", "w", driver="mpio", comm=comm) as h5f:
                if compression == True:
                    h5f.create_dataset("data", data=self.data[:], compression=compressionType)
                else:
                    h5f.create_dataset("data", data=self.data[:])
        else:
            if comm.rank == 0:
                with h5py.File(f"{filename[:-3]}.h5", "w") as h5f:
                    if compression == True:
                        h5f.create_dataset(
                            "data",
                            data=self.data[:],
                            chunks=True,
                            maxshape=(None, self.data.shape[1]),
                            compression=compressionType,
                        )
                    else:
                        h5f.create_dataset(
                            "data",
                            data=self.data[:],
                            chunks=True,
                            maxshape=(None, self.data.shape[1]),
                        )
                comm.barrier()
                for proc in range(1, comm.size):
                    if comm.rank == proc:
                        if self.local_size > 0:
                            with h5py.File(f"{filename[:-3]}.h5", "a") as h5f:
                                incoming_size = h5f["data"].shape[0]
                                h5f["data"].resize((h5f["data"].shape[0] + self.local_size), axis=0)
                                h5f["data"][incoming_size:] = self.data[:, ...]
                    comm.barrier()
                comm.barrier()

        ## Add swarm variable unit metadata to the file
        import json

        # Use preferred selective_ranks pattern for unit metadata
        with uw.selective_ranks(0) as should_execute:
            if should_execute:
                with h5py.File(f"{filename[:-3]}.h5", "a") as h5f:
                    # Add swarm variable unit metadata
                    swarm_metadata = {
                        "coordinate_units": (
                            str(self.swarm.coordinate_units)
                            if hasattr(self.swarm, "coordinate_units")
                            else None
                        ),
                        "variable_units": (
                            str(self.units) if hasattr(self, "units") and self.units else None
                        ),
                        "variable_dimensionality": (
                            str(self.dimensionality) if hasattr(self, "dimensionality") else None
                        ),
                        "units_backend": "pint" if self.has_units else None,
                        "proxy_degree": self._proxy_degree,
                        "num_components": self.num_components,
                        "variable_name": self.name,
                    }

                    # Store in dataset attributes
                    if "data" in h5f:
                        h5f["data"].attrs["units_metadata"] = json.dumps(swarm_metadata)

        return

    @timing.routine_timer_decorator
    def write_proxy(self, filename: str):
        # if not proxied, nothing to do. return.
        if not self._meshVar:
            uw.pprint("No proxy mesh variable that can be saved")
            return

        self._meshVar.write(filename)

        return

    @timing.routine_timer_decorator
    def read_timestep(
        self,
        data_filename: str,
        swarmID: str,
        data_name: str,
        index: int,
        outputPath="",
    ):
        # mesh.write_timestep( "test", meshUpdates=False, meshVars=[X], outputPath="", index=0)
        # swarm.write_timestep("test", "swarm", swarmVars=[var], outputPath="", index=0)

        output_base_name = os.path.join(outputPath, data_filename)
        swarmFilename = output_base_name + f".{swarmID}.{index:05}.h5"
        filename = output_base_name + f".{swarmID}.{data_name}.{index:05}.h5"

        # check if swarmFilename exists
        if os.path.isfile(os.path.abspath(swarmFilename)):  # easier to debug abs path
            print(f"Reading swarm information from {swarmFilename}", flush=True)
            pass
        else:
            raise RuntimeError(f"{os.path.abspath(swarmFilename)} does not exist")

        if os.path.isfile(os.path.abspath(filename)):
            print(f"Reading variable information from {filename}", flush=True)

            pass
        else:
            raise RuntimeError(f"{os.path.abspath(filename)} does not exist")

        ### open up file with coords on all procs and open up data on all procs. May be problematic for large problems.
        with (
            h5py.File(f"{filename}", "r") as h5f_data,
            h5py.File(f"{swarmFilename}", "r") as h5f_swarm,
        ):

            # with self.swarm.access(self):
            var_dtype = self.dtype
            file_dtype = h5f_data["data"][:].dtype
            file_length = h5f_data["data"][:].shape[0]

            if var_dtype != file_dtype:
                if comm.rank == 0:
                    warnings.warn(
                        f"{os.path.basename(filename)} dtype ({file_dtype}) does not match {self.name} swarm variable dtype ({var_dtype}) which may result in a loss of data.",
                        stacklevel=2,
                    )

            # First work out which are local points and ignore the rest
            # This might help speed up the load by dropping lots of particles

            all_coords = h5f_swarm["coordinates"][()]
            all_data = h5f_data["data"][()]

            local_coords = all_coords  # [local]
            local_data = all_data  # [local]

            kdt = uw.kdtree.KDTree(local_coords)

            self.array[:, 0, :] = kdt.rbf_interpolator_local(
                self.swarm._particle_coordinates.data, local_data, nnn=1
            )

        return


class IndexSwarmVariable(SwarmVariable):
    """
    Integer-valued swarm variable for material tracking.

    IndexSwarmVariable stores integer indices at particle locations, typically
    used for tracking distinct material types. It automatically generates
    symbolic mask expressions for each material index, enabling material-
    dependent properties in constitutive models.

    Parameters
    ----------
    name : str
        Variable name for identification and I/O.
    swarm : Swarm
        Parent swarm object.
    indices : int
        Number of distinct material indices (default 1).
    proxy_degree : int
        Polynomial degree for mesh projection (default 1).
    proxy_continuous : bool
        Whether mesh proxy is continuous (default True).

    Attributes
    ----------
    sym : list of sympy.Expr
        Symbolic mask expressions for each material index.

    Examples
    --------
    >>> material = IndexSwarmVariable("M", swarm, indices=3)
    >>> material.data[:] = 0  # Set all particles to material 0
    >>> # Use sym[i] as multiplier for material i properties
    >>> viscosity = material.sym[0] * 1e20 + material.sym[1] * 1e21

    See Also
    --------
    SwarmVariable : Base class for particle-supported variables.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        name,
        swarm,
        indices=1,
        proxy_degree=1,
        proxy_continuous=True,
        update_type=0,
        npoints=5,
        radius=0.5,
        npoints_bc=2,
        ind_bc=None,
        varsymbol=None,
    ):
        self.indices = indices
        self.nnn = npoints
        self.radius_s = radius  # **2 # changed to radius
        self.update_type = update_type
        if self.update_type == 1:
            self.nnn_bc = npoints_bc
            self.ind_bc = ind_bc

        # These are the things we require of the generic swarm variable type
        super().__init__(
            name,
            swarm,
            size=1,
            vtype=None,
            dtype=int,
            proxy_degree=proxy_degree,
            proxy_continuous=proxy_continuous,
            _proxy=False,
            varsymbol=varsymbol,
        )
        """
        vtype = (None,)
        dtype = (float,)
        proxy_degree = (1,)
        proxy_continuous = (True,)
        _register = (True,)
        _proxy = (True,)
        _nn_proxy = (False,)
        varsymbol = (None,)
        rebuild_on_cycle = (True,)
        """
        # The indices variable defines how many "level set" maps we create as components in the proxy variable

        import sympy

        self._MaskArray = sympy.Matrix.zeros(1, self.indices)
        self._meshLevelSetVars = [None] * self.indices

        for i in range(indices):
            self._meshLevelSetVars[i] = uw.discretisation.MeshVariable(
                name + R"^{[" + str(i) + R"]}",
                self.swarm.mesh,
                num_components=1,
                degree=proxy_degree,
                continuous=proxy_continuous,
            )
            self._MaskArray[0, i] = self._meshLevelSetVars[i].sym[0, 0]

        # Initialize lazy evaluation state
        self._proxy_stale = True  # Proxy variables need initial update

        return

    def _update(self):
        """
        Backward compatibility wrapper for _update_proxy_variables.

        Maintains existing API while implementing lazy evaluation internally.
        """
        self._update_proxy_variables()

    def _on_data_changed(self):
        """
        Hook called by unified data callback when canonical data changes.

        For IndexSwarmVariable, this marks proxy variables as stale for lazy evaluation.
        This replaces the complex custom array override with a simple hook.
        """
        self._proxy_stale = True

    # This is the sympy vector interface - it's meaningless if these are not spatial arrays
    @property
    def sym(self):
        """
        Lazy evaluation of symbolic mask array.

        Only updates proxy variables when they're actually needed (when sym is accessed)
        and only if the proxy variables are marked as stale due to data changes.
        This avoids expensive RBF interpolation during data assignment operations.
        """
        if self._proxy_stale:
            self._update_proxy_variables()
            self._proxy_stale = False
        return self._MaskArray

    @property
    def sym_1d(self):
        """
        One-dimensional symbolic mask array (alias for :attr:`sym`).

        Returns the same symbolic mask array as :attr:`sym`, provided for API
        compatibility with other variable types that distinguish between
        multi-dimensional and flattened representations.

        Returns
        -------
        sympy.Matrix
            Symbolic mask array of shape (indices, 1).

        See Also
        --------
        sym : Primary symbolic mask array access.
        """
        return self._MaskArray

    # We can  also add a __getitem__ call to access each mask

    def __getitem__(self, index):
        return self.sym[index]

    def createMask(self, funcsList):
        """
        Create a material-weighted symbolic expression from per-material values.

        This method creates a SymPy expression that combines multiple material
        properties using the index variable's symbolic masks. The result can be
        used directly in Underworld's solver equations.

        Parameters
        ----------
        funcsList : list or tuple
            List of values or symbolic expressions, one per material index.
            Length must equal :attr:`indices`.

        Returns
        -------
        sympy.Basic
            Symbolic expression: ``sum(funcsList[i] * mask[i] for i in indices)``.

        Raises
        ------
        RuntimeError
            If ``funcsList`` is not a list/tuple or has wrong length.

        Examples
        --------
        >>> # Define viscosity per material
        >>> viscosity = material.createMask([1e21, 1e20, 1e22])  # 3 materials
        >>> # Use in solver
        >>> stokes.constitutive_model.viscosity = viscosity

        See Also
        --------
        visMask : Create visualization mask showing material indices.
        """

        if not isinstance(funcsList, (tuple, list)):
            raise RuntimeError("Error input for createMask() - wrong type of input")

        if len(funcsList) != self.indices:
            raise RuntimeError("Error input for createMask() - wrong length of input")

        symo = sympy.simplify(0)
        for i in range(self.indices):
            symo += funcsList[i] * self._MaskArray[i]

        return symo

    def viewMask(self, expr):
        """
        Decompose a masked expression into per-material components.

        .. note::
            This method is not yet implemented. Currently returns None.

        Takes a symbolic expression created by :meth:`createMask` and extracts
        the individual material-specific components.

        Parameters
        ----------
        expr : sympy.Basic
            A masked symbolic expression created by :meth:`createMask`.

        Returns
        -------
        list or None
            List of symbolic expressions, one per material index.
            Currently returns None (not implemented).

        See Also
        --------
        createMask : Create a masked expression from per-material values.
        """
        # TODO: Implement decomposition of masked expressions
        # output = []
        # for i in range(self.indices):
        #     tmp = {}
        #     for j in range(self.indices):
        #         if i == j: pass
        #         tmp
        # return output
        pass

    def visMask(self):
        """
        Create a visualization mask showing material indices.

        Returns a symbolic expression where each material region shows its
        index value (0, 1, 2, ...). Useful for visualization and debugging
        of material distributions.

        Returns
        -------
        sympy.Basic
            Symbolic expression evaluating to material index at each point.

        Examples
        --------
        >>> vis_field = material.visMask()
        >>> values = uw.function.evaluate(vis_field, swarm.data)
        >>> # values[i] gives material index at particle i

        See Also
        --------
        createMask : Create arbitrary material-weighted expressions.
        """
        return self.createMask(list(range(self.indices)))

    def view(self):
        """
        Show information on IndexSwarmVariable
        """
        uw.pprint(f"IndexSwarmVariable {self}")
        uw.pprint(f"Numer of indices {self.indices}")

    def _update_proxy_variables(self):
        """
        This method updates the proxy mesh (vector) variable for the index variable on the current swarm locations

        Here is how it works:

            1) for each particle, create a distance-weighted average on the node data
            2) for each index in the set, we create a mask mesh variable by mapping 1.0 wherever the
               index matches and 0.0 where it does not.

        NOTE: If no material is identified with a given nodal value, the default is to impose
        a near-neighbour hunt for a valid material and set that one

        ## ToDo: This should be revisited to match the updated master copy of _update

        update_type 0: assign the particles to the nearest mesh_levelset nodes, and calculate the value on nodes from them.
        update_type 1: calculate the material property value on mesh_levelset nodes from the nearest N particles directly.

        """
        if self.update_type == 0:
            # Use non-dimensional coordinates for internal level set KDTree
            kd = uw.kdtree.KDTree(self._meshLevelSetVars[0].coords_nd)

            n_distance, n_indices = kd.query(
                self.swarm._particle_coordinates.data, k=self.nnn, sqr_dists=False
            )
            kd_swarm = uw.kdtree.KDTree(self.swarm._particle_coordinates.data)
            # n, d, b = kd_swarm.find_closest_point(self._meshLevelSetVars[0].coords)
            d, n = kd_swarm.query(self._meshLevelSetVars[0].coords, k=1, sqr_dists=False)

            for ii in range(self.indices):
                meshVar = self._meshLevelSetVars[ii]

                with self.swarm.mesh.access(meshVar), self.swarm.access():
                    node_values = np.zeros((meshVar.data.shape[0],))
                    w = np.zeros((meshVar.data.shape[0],))

                    for i in range(self.swarm.local_size):
                        tem = np.isclose(n_distance[i, :], n_distance[i, 0])
                        dist = n_distance[i, tem]
                        indices = n_indices[i, tem]
                        tem = dist < self.radius_s
                        dist = dist[tem]
                        indices = indices[tem]
                        for j, ind in enumerate(indices):
                            node_values[ind] += (
                                np.isclose(self.data[i], ii) / (1.0e-16 + dist[j])
                            )[0]
                            w[ind] += 1.0 / (1.0e-16 + dist[j])

                    node_values[np.where(w > 0.0)[0]] /= w[np.where(w > 0.0)[0]]
                    meshVar.data[:, 0] = node_values[...]

                    # if there is no material found,
                    # impose a near-neighbour hunt for a valid material and set that one
                    ind_w0 = np.where(w == 0.0)[0]
                    if len(ind_w0) > 0:
                        ind_ = np.where(self.data[n[ind_w0]] == ii)[0]
                        if len(ind_) > 0:
                            meshVar.data[ind_w0[ind_]] = 1.0
        elif self.update_type == 1:
            kd = uw.kdtree.KDTree(self.swarm._particle_coordinates.data)
            n_distance, n_indices = kd.query(
                self._meshLevelSetVars[0].coords, k=self.nnn, sqr_dists=False
            )

            for ii in range(self.indices):
                meshVar = self._meshLevelSetVars[ii]
                node_values = np.zeros((meshVar.data.shape[0],))
                w = np.zeros((meshVar.data.shape[0],))
                for i in range(meshVar.data.shape[0]):
                    if i not in self.ind_bc:
                        ind = np.where(n_distance[i, :] < self.radius_s)
                        a = 1.0 / (n_distance[i, ind] + 1.0e-16)
                        w[i] = np.sum(a)
                        b = np.isclose(self.data[n_indices[i, ind]], ii)
                        node_values[i] = np.sum(np.dot(a, b))
                        if ind[0].size == 0:
                            w[i] = 0
                    else:
                        ind = np.where(n_distance[i, : self.nnn_bc] < self.radius_s)
                        a = 1.0 / (n_distance[i, : self.nnn_bc][ind] + 1.0e-16)
                        w[i] = np.sum(a)
                        b = np.isclose(self.data[n_indices[i, : self.nnn_bc][ind]], ii)
                        node_values[i] = np.sum(np.dot(a, b))
                        if ind[0].size == 0:
                            w[i] = 0

                node_values[np.where(w > 0.0)[0]] /= w[np.where(w > 0.0)[0]]
                meshVar.data[:, 0] = node_values[...]

                # if there is no material found,
                # impose a near-neighbour hunt for a valid material and set that one
                ind_w0 = np.where(w == 0.0)[0]
                if len(ind_w0) > 0:
                    ind_ = np.where(self.data[n_indices[ind_w0]] == ii)[0]
                    if len(ind_) > 0:
                        meshVar.data[ind_w0[ind_]] = 1.0
        return


## Import PIC-related classes from separate module to maintain compatibility
# from .pic_swarm import PICSwarm, NodalPointPICSwarm, SwarmPICLayout

## This should be the basic swarm, and we can then create a sub-class that will
## be a PIC swarm

# PICSwarm and NodalPointPICSwarm classes have been moved to pic_swarm.py


## New - Basic Swarm (no PIC skillz)
## What is missing:
##  - no celldm
##  - PIC layouts of particles are not directly available / must be done by hand
##  - No automatic migration - must compute ranks for the particle swarms
##  - No automatic definition of coordinate fields (need to add by hand)


class Swarm(Stateful, uw_object):
    """
    A basic particle swarm implementation for Lagrangian particle tracking and data storage.

    The UW `Swarm` class provides a simplified particle management system that uses
    PETSc's DMSWARM_BASIC type. Unlike the standard `Swarm` class, this implementation
    does not rely on PETSc to determine ranks for particle migration but instead uses
    our own kdtree neighbour-domain computations.

    This class is preferred for most operations except where particle / cell relationships
    are always required.

    Parameters
    ----------
    mesh : uw.discretisation.Mesh
        The mesh object that defines the computational domain for particle operations.
        Particles will be associated with this mesh for spatial queries and operations.
    recycle_rate : int, optional
        Rate at which particles are recycled for streak management. If > 1, enables
        streak particle functionality where particles are duplicated and tracked
        across multiple cycles. Default is 0 (no recycling).
    verbose : bool, optional
        Enable verbose output for debugging and monitoring particle operations.
        Default is False.

    Attributes
    ----------
    mesh : uw.discretisation.Mesh
        Reference to the associated mesh object.
    dim : int
        Spatial dimension of the mesh (2D or 3D).
    cdim : int
        Coordinate dimension of the mesh.
    data : numpy.ndarray
        Direct access to particle coordinate data.
    particle_coordinates : SwarmVariable
        SwarmVariable containing particle coordinate information.
    recycle_rate : int
        Current recycle rate for streak management.
    cycle : int
        Current cycle number for streak particles.

    Methods
    -------
    populate(fill_param=1)
        Populate the swarm with particles throughout the domain.
    migrate(remove_sent_points=True, delete_lost_points=True, max_its=10)
        Manually migrate particles across MPI processes after coordinate updates.
    add_particles_with_coordinates(coords)
        Add new particles at specified coordinate locations.
    add_particles_with_global_coordinates(coords)
        Add particles using global coordinate system.
    add_variable(name, size, dtype=float)
        Add a new variable to track additional particle properties.
    save(filename, meshUnits=1.0, swarmUnits=1.0, units="dimensionless")
        Save swarm data to file.
    read_timestep(filename, step_name, outputPath="./output/")
        Read swarm data from a specific timestep file.
    advection(V_fn, delta_t, evalf=False, corrector=True, restore_points_func=None)
        Advect particles using a velocity field.
    estimate_dt(V_fn, dt_min=1.0e-15, dt_max=1.0)
        Estimate appropriate timestep for particle advection.

    Examples
    --------
    Create a basic swarm and populate with particles:

    >>> import underworld3 as uw
    >>> mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(1,1))
    >>> swarm = uw.swarm.Swarm(mesh=mesh)
    >>> swarm.populate(fill_param=2)

    Create a streak swarm with recycling:

    >>> streak_swarm = uw.swarm.Swarm(mesh=mesh, recycle_rate=5)
    >>> streak_swarm.populate(fill_param=1)

    Add custom particle data:

    >>> temperature = swarm.add_variable("temperature", 1)
    >>> velocity = swarm.add_variable("velocity", mesh.dim)

    Manual particle migration after coordinate updates:

    Note: particle migration is still called automatically when we
    `access` and update the particle_coordinates variables

    Note: `swarm.populate` uses a the mesh point locations for discontinuous interpolants to
    determine the particle locations.

    """

    instances = 0

    @timing.routine_timer_decorator
    def __init__(self, mesh, recycle_rate=0, verbose=False, clip_to_mesh=True):
        Swarm.instances += 1

        self.verbose = verbose
        self._clip_to_mesh = clip_to_mesh

        # Store reference to model instead of direct mesh reference
        # This enables dynamic mesh handover while maintaining access to mesh services
        import underworld3 as uw

        model = uw.get_default_model()

        # Register mesh with model if not already present
        model._register_mesh(mesh)

        # Store reference to this swarm's specific mesh for proxy operations
        self._mesh_id = id(mesh)

        self._model_ref = weakref.ref(model)
        self.dim = mesh.dim
        self.cdim = mesh.cdim

        # Mesh version tracking for coordinate change detection
        self._mesh_version = mesh._mesh_version

        # Register this swarm with the mesh for coordinate change notifications
        mesh.register_swarm(self)

        self.dm = PETSc.DMSwarm().create()
        self.dm.setDimension(self.dim)
        self.dm.setType(SwarmType.DMSWARM_BASIC.value)
        self._data = None

        # Add data structure to hold point location information in
        # an array with a callback that resets the relevant parts of the
        # swarm variable stack when the data structure is modified.

        self._coords = None

        ####

        # Is the swarm a streak-swarm ?
        self.recycle_rate = recycle_rate
        self.cycle = 0

        # dictionary for variables
        # Using WeakValueDictionary to prevent circular references
        self._vars = weakref.WeakValueDictionary()

        # add variable to handle particle coords - match name from PIC_Swarm for consistency
        self._coord_var = SwarmVariable(
            "DMSwarmPIC_coor",
            self,
            self.cdim,
            dtype=float,
            _register=True,
            _proxy=False,
            rebuild_on_cycle=False,
        )

        # add variable to handle particle ranks - this exists on the PETSc machinery already
        self._rank_var = SwarmVariable(
            "DMSwarm_rank",
            self,
            1,
            dtype=int,
            _register=False,
            _proxy=False,
            rebuild_on_cycle=False,
        )

        # add variable to hold swarm coordinates during position updates
        self._X0 = uw.swarm.SwarmVariable(
            "DMSwarm_X0",
            self,
            self.cdim,
            dtype=float,
            _register=True,
            _proxy=False,
            rebuild_on_cycle=False,
        )

        # This is for swarm streak management:
        # add variable to hold swarm origins

        if self.recycle_rate > 1:

            self._remeshed = uw.swarm.SwarmVariable(
                "DMSwarm_remeshed",
                self,
                1,
                dtype=int,
                _register=True,
                _proxy=False,
                rebuild_on_cycle=False,
            )

        self._X0_uninitialised = True
        self._index = None
        self._nnmapdict = {}
        self._migration_disabled = False

        super().__init__()

        # Register with default model for orchestration
        uw.get_default_model()._register_swarm(self)

    def __del__(self):
        """Cleanup swarm by unregistering from mesh to prevent memory leaks"""
        try:
            if hasattr(self, "mesh") and self.mesh is not None:
                self.mesh.unregister_swarm(self)
        except (AttributeError, ReferenceError, RuntimeError):
            # Mesh/Model may have already been garbage collected, which is fine
            pass

    @property
    def mesh(self):
        """The mesh this swarm operates on"""
        model = self._model_ref()
        if model is None:
            raise RuntimeError("Model has been garbage collected")
        return model.get_mesh(self._mesh_id)

    @mesh.setter
    def mesh(self, new_mesh):
        """
        Assign swarm to a new mesh with dimensional validation and proxy updates.

        Parameters
        ----------
        new_mesh : uw.discretisation.Mesh
            New mesh to assign this swarm to

        Raises
        ------
        ValueError
            If new mesh has incompatible dimensions
        """
        model = self._model_ref()
        if model is None:
            raise RuntimeError("Model has been garbage collected")

        # Register new mesh with model
        model._register_mesh(new_mesh)

        if id(new_mesh) == self._mesh_id:
            # Check if swarm is already compatible with target mesh
            if self.dim == new_mesh.dim and self.cdim == new_mesh.cdim:
                # Dimensions match, check if proxy variables need updating
                proxy_vars_updated = True
                for var in self._vars.values():
                    if (
                        hasattr(var, "_proxy")
                        and var._proxy
                        and hasattr(var, "_meshVar")
                        and var._meshVar
                    ):
                        if var._meshVar.mesh is not new_mesh:
                            proxy_vars_updated = False
                            break
                if proxy_vars_updated:
                    return  # No change needed

        # Use swarm's current dimensions for validation (not model.mesh which may have been auto-updated)
        current_dim = self.dim
        current_cdim = self.cdim

        # Critical dimensional validation
        if new_mesh.dim != current_dim:
            raise ValueError(
                f"Cannot assign swarm to mesh with different coordinate dimension. "
                f"Current swarm dim={current_dim}, new mesh dim={new_mesh.dim}. "
                f"Swarm particles and variables are sized for {current_dim}D space."
            )

        if new_mesh.cdim != current_cdim:
            raise ValueError(
                f"Cannot assign swarm to mesh with different embedding dimension. "
                f"Current swarm cdim={current_cdim}, new mesh cdim={new_mesh.cdim}."
            )

        # Update model's mesh and handle all swarm transitions
        model._update_mesh_for_swarm(self, new_mesh)

        # Update swarm's mesh reference
        self._mesh_id = id(new_mesh)

        # Update swarm's cached dimensions
        self.dim = new_mesh.dim
        self.cdim = new_mesh.cdim

        # Recreate all proxy variables for new mesh
        for var in self._vars.values():
            var._create_proxy_variable()  # Safe for all variables (proxied or not)

        # Update mesh version tracking
        self._mesh_version = new_mesh._mesh_version

    @property
    def local_size(self):
        r"""Number of particles on this MPI rank.

        Returns
        -------
        int
            Local particle count.

        See Also
        --------
        dm.getLocalSize : Underlying PETSc method.
        """
        return self.dm.getLocalSize()

    # We could probably use a global_size property too

    @property
    def data(self):
        r"""Particle coordinates (alias for :attr:`points`).

        .. deprecated:: 0.99.0
            Use direct DM field access for particle coordinates.

        Returns
        -------
        numpy.ndarray
            Particle coordinate array of shape ``(n_particles, dim)``.
        """
        return self.points

    @property
    def points(self):
        """
        Swarm particle coordinates in physical units.

        .. deprecated:: 0.99.0
            Use swarm variables or direct DM access instead.
            ``swarm.points`` is being deprecated.

        When the mesh has coordinate scaling applied (via model units),
        this property automatically converts from internal model coordinates
        to physical coordinates for user access.

        When the mesh has coordinate units specified, returns a unit-aware array.

        Returns:
            numpy.ndarray or UnitAwareArray: Particle coordinates (with units if mesh.units is set)
        """
        import warnings

        warnings.warn("swarm.points is deprecated", DeprecationWarning, stacklevel=2)

        # Check for mesh coordinate changes and trigger migration if needed
        if hasattr(self, "_mesh_version") and self._mesh_version != self.mesh._mesh_version:
            # Mesh coordinates have changed, force migration to update swarm
            self._force_migration_after_mesh_change()
            # Update our mesh version to match
            self._mesh_version = self.mesh._mesh_version

        # Get current coordinate data from PETSc (these are in model coordinates)
        model_coords = (self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.dim))).copy()
        self.dm.restoreField("DMSwarmPIC_coor")

        # Apply scaling to convert model coordinates to physical coordinates
        if hasattr(self.mesh.CoordinateSystem, "_scaled") and self.mesh.CoordinateSystem._scaled:
            scale_factor = self.mesh.CoordinateSystem._length_scale
            coords = model_coords * scale_factor
        else:
            coords = model_coords

        # Cache and reuse NDArray_With_Callback object for consistent object identity
        if not hasattr(self, "_coords") or self._coords is None:
            # First access: create new NDArray_With_Callback object
            self._coords = uw.utilities.NDArray_With_Callback(
                coords,
                owner=self,
                disable_inplace_operators=True,
            )

            # Define the callback function (only once)
            def swarm_update_callback(array, change_context):
                # print(
                #     f"Swarm update callback - {self.dm.getLocalSize()}",
                #     flush=True,
                # )

                # Check if this operation may have changed data
                # Skip expensive operations for read-only sync operations
                data_changed = change_context.get("data_has_changed", True)

                if not data_changed:
                    # print(
                    #     "Swarm callback: Skipping migration - read-only sync operation"
                    # )
                    return

                # Check if sizes match before attempting to copy back
                petsc_size = self.dm.getLocalSize()
                points_size = array.shape[0]

                if petsc_size == points_size:
                    # Update PETSc state
                    # We could do this directly which would be more efficient and bypass the access manager (appropriately, here)
                    self._coord_var.array[:, 0, :] = array[...]

                    # Migrate by default (unless user has disabled it)
                    if not self._migration_disabled:
                        self.migrate()
                        for var in self._vars.values():
                            var._update()

                else:
                    # This means a migration call has been made before we have
                    # had a chance to update the swarm consistently. This is an error
                    # condition. We raise an exception to prevent further errors.

                    print(
                        f"Size mismatch: PETSc={petsc_size}, Points={points_size}\n",
                        f"The swarm migration state has become corrupted",
                    )
                    raise RuntimeError

                return

            # Add callback to the cached object
            self._coords.add_callback(swarm_update_callback)
        else:
            # Subsequent accesses: efficiently sync new coordinate data
            # This preserves callbacks and delay contexts, updating object reference if size
            # changed as a result of migration operations

            self._coords = self._coords.sync_data(coords)

        # Wrap with unit-aware array if mesh has units
        if hasattr(self.mesh, "units") and self.mesh.units is not None:
            from underworld3.utilities.unit_aware_array import UnitAwareArray

            return UnitAwareArray(self._coords, units=self.mesh.units)

        return self._coords

    @points.setter
    def points(self, value):
        """
        Set swarm particle coordinates from physical units.

        .. deprecated:: 0.99.0
            Use swarm variables or direct DM access instead.

        When the mesh has coordinate scaling applied (via model units),
        this property automatically converts from physical coordinates
        to internal model coordinates for PETSc storage.

        Args:
            value (numpy.ndarray): Particle coordinates in physical units
        """
        import warnings

        warnings.warn("swarm.points is deprecated", DeprecationWarning, stacklevel=2)

        if value.shape[0] != self.local_size:
            raise TypeError(
                f"Points must be a numpy array with the same size as the swarm",
                f"  - partial allocation to the swarm may trigger migration or point removal",
                f"  - either change all the swarm points at once or use the `with migration_control()` manager",
            )

        # Apply inverse scaling to convert physical coordinates to model coordinates
        if hasattr(self.mesh.CoordinateSystem, "_scaled") and self.mesh.CoordinateSystem._scaled:
            scale_factor = self.mesh.CoordinateSystem._length_scale
            model_coords = value / scale_factor
        else:
            model_coords = value

        # Update the cached NDArray (triggers callback) - use physical coordinates for cache
        self._coords[...] = value[...]

        # Update PETSc DM field directly with model coordinates for immediate consistency
        coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.dim))
        coords[...] = model_coords[...]
        self.dm.restoreField("DMSwarmPIC_coor")

    # @points.setter
    # def points(self, value):

    #     if isinstance(value, np.ndarray):
    #         if value.shape[0] != self.local_size:
    #             message = (
    #                 "Points must be a numpy array with the same size as the swarm."
    #                 + "Partial allocation to the swarm may trigger particle migration"
    #                 + "either change all the swarm points at once or use the `with migration_disabled()` manager",
    #             )
    #             raise TypeError(message)

    #     self._coords[...] = value[...]

    @property
    def _particle_coordinates(self):
        return self._coord_var

    @property
    def coords(self):
        """
        Swarm particle coordinates in physical units.

        This is the primary public interface for accessing particle coordinates.
        Coordinates are automatically converted from internal model units to
        physical units based on the model's reference quantities.

        Returns
        -------
        UWQuantity or numpy.ndarray
            Particle coordinates in physical units with shape (n_particles, dim).
            If model has reference quantities, returns UWQuantity with appropriate
            length units. Otherwise returns plain array.

        Notes
        -----
        - Coordinates are converted from model units to physical units automatically
        - For internal use with model units, access `swarm._particle_coordinates.data`
        - Setting coordinates accepts either physical units or plain numbers

        Examples
        --------
        >>> coords_physical = swarm.coords  # Get physical coordinates
        >>> swarm.coords = new_coords_with_units  # Set from physical units

        See Also
        --------
        swarm.units : Get the unit specification for coordinates
        """
        # Get internal model-unit coordinates
        model_coords = self._particle_coordinates.data

        # Convert to physical units
        import underworld3 as uw

        model = uw.get_default_model()

        # Use from_model_magnitude to convert back to physical
        return model.from_model_magnitude(model_coords, "[length]")

    @coords.setter
    def coords(self, value):
        """
        Set swarm particle coordinates from physical units.

        Accepts coordinates with units or plain numbers. If units are provided,
        they are converted to model units automatically. If plain numbers are
        provided, they are assumed to be in the correct unit system.

        Parameters
        ----------
        value : array-like or UWQuantity
            New coordinates. Can be:
            - Array with units (e.g., values * uw.units.km)
            - Plain array (assumed to be in model units or physical units depending on context)
        """
        import underworld3 as uw

        # Convert physical → non-dimensional units
        model_coords = uw.scaling.non_dimensionalise(value)

        # Set internal coordinates
        self._particle_coordinates.data[...] = model_coords

    @property
    def units(self):
        """
        Unit specification for swarm coordinates.

        Returns the physical unit string for coordinates based on the model's
        reference quantities. This indicates what units the coordinates are in
        when accessed via the `coords` property.

        Returns
        -------
        str or None
            Unit string for coordinates (e.g., 'kilometer', 'meter'), or None
            if no reference quantities are set

        Examples
        --------
        >>> print(swarm.units)  # 'kilometer' if length_scale was set in km
        >>> coords = swarm.coords  # Coordinates in kilometers
        """
        # Coordinates have length dimensions
        import underworld3 as uw

        model = uw.get_default_model()

        # Check if model has reference quantities
        if not hasattr(model, "_pint_registry"):
            return None

        # Get length scale from model
        try:
            scales = model.get_fundamental_scales()
            if "length" in scales:
                length_scale = scales["length"]
                if hasattr(length_scale, "_pint_qty"):
                    return str(length_scale._pint_qty.units)
                elif hasattr(length_scale, "units"):
                    return str(length_scale.units)
        except:
            pass

        return None

    @property
    def clip_to_mesh(self):
        """
        Whether particles are clipped to remain within mesh boundaries.

        When True (default), particles that move outside the mesh domain
        during advection or coordinate updates are removed or repositioned
        to stay within bounds. When False, particles can exist outside the
        mesh domain.

        Returns
        -------
        bool
            Current clipping state.

        See Also
        --------
        dont_clip_to_mesh : Context manager to temporarily disable clipping.
        """
        return self._clip_to_mesh

    @clip_to_mesh.setter
    def clip_to_mesh(self, value):
        """Set whether particles should be clipped to mesh boundaries."""
        self._clip_to_mesh = bool(value)

    def dont_clip_to_mesh(self):
        """
        Context manager that temporarily disables mesh clipping for the swarm.
        `swarm.migrate` is called automatically when exiting the context.

        Usage:
            with swarm.dont_clip_to_mesh():
                # swarm operations that should not be clipped to mesh
                swarm.data = new_positions

        """

        class _ClipToggleContext:
            def __init__(self, swarm):
                self.swarm = swarm
                self.original_value = None

            def __enter__(self):
                self.original_value = self.swarm._clip_to_mesh
                self.swarm._clip_to_mesh = False
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.swarm._clip_to_mesh = self.original_value
                self.swarm.migrate()

        return _ClipToggleContext(self)

    def migration_disabled(self):
        """
        Legacy context manager that completely disables migration.
        Use migration_control(disable=True) for new code.

        Context manager that temporarily disables particle migration for the swarm.
        Migration is NOT called when exiting the context.

        Usage:
            with swarm.migration_disabled():
                # swarm operations that should not trigger migration
                swarm.data = new_positions
                # ... other operations ...
                # migrate() will be skipped during these operations

        """
        return self.migration_control(disable=True)

    def migration_control(self, disable=False):
        """
        Context manager to control particle migration behavior.

        Parameters
        ----------
        disable : bool
            If False (default), migration is deferred until context exit.
            If True, migration is completely disabled.

        Usage:
            # Defer migration until end (default)
            with swarm.migration_control():
                swarm.points[mask1] += delta1
                swarm.points[mask2] *= scale
                # Migration happens HERE on exit

            # Completely disable migration
            with swarm.migration_control(disable=True):
                # Operations where migration should never happen
                # No migration on exit
        """

        class _MigrationControlContext:
            def __init__(self, swarm, disable):
                self.swarm = swarm
                self.disable = disable
                self.original_value = None
                self.initial_size = None

            def __enter__(self):
                self.original_value = self.swarm._migration_disabled
                self.swarm._migration_disabled = True
                if not self.disable:
                    self.initial_size = self.swarm.local_size
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.swarm._migration_disabled = self.original_value

                # Perform deferred migration if not disabled and not still blocked
                if not self.disable and not self.swarm._migration_disabled:
                    # Check if particle positions might have changed
                    if self.swarm.local_size == self.initial_size:
                        self.swarm.migrate()
                        for var in self.swarm._vars.values():
                            var._update()

        return _MigrationControlContext(self, disable)

    @timing.routine_timer_decorator
    @uw.collective_operation
    def populate(
        self,
        fill_param: Optional[int] = 1,
    ):
        """
        Populate the swarm with particles throughout the domain.

        Parameters
        ----------
        fill_param:
            Parameter determining the particle count per cell (per dimension)
            for the given layout, using the mesh degree.

        Raises
        ------
        RuntimeError
            If the swarm has already been initialized with particles.
        """

        if self.local_size > 0:
            raise RuntimeError(
                f"Cannot populate swarm that already has {self.local_size} particles. "
                "populate() is only for swarm initialization."
            )

        self.fill_param = fill_param

        newp_coords0 = self.mesh._get_coords_for_basis(fill_param, continuous=False)
        newp_cells0 = self.mesh.get_closest_local_cells(newp_coords0)

        valid = newp_cells0 != -1
        newp_coords = newp_coords0[valid]
        newp_cells = newp_cells0[valid]

        self.dm.finalizeFieldRegister()

        # PETSc < 3.24 has an off-by-one bug in addNPoints when swarm size is initially zero
        # It allocates N-1 instead of N, so we add +1 to compensate
        # PETSc 3.24+ fixed this bug, so we use the exact count
        from petsc4py import PETSc
        if PETSc.Sys.getVersion() >= (3, 24, 0):
            self.dm.addNPoints(newp_coords.shape[0])
        else:
            self.dm.addNPoints(newp_coords.shape[0] + 1)

        coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.dim))
        ranks = self.dm.getField("DMSwarm_rank")
        coords[...] = newp_coords[...]
        ranks[...] = uw.mpi.rank
        self.dm.restoreField("DMSwarmPIC_coor")
        self.dm.restoreField("DMSwarm_rank")

        if self.recycle_rate > 1:
            with self.access():
                # This is a mesh-local quantity, so let's just
                # store it on the mesh in an ad_hoc fashion for now

                self.mesh.particle_X_orig = self._particle_coordinates.data.copy()

            with self.access():
                swarm_orig_size = self.local_size
                all_local_coords = np.vstack(
                    (self._particle_coordinates.data,) * (self.recycle_rate)
                )

                swarm_new_size = all_local_coords.shape[0]

            self.dm.addNPoints(swarm_new_size - swarm_orig_size)

            coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.dim))

            # Compute perturbation - extract magnitude if coordinates have units
            # numpy.array(..., dtype=float64) forces conversion to plain array
            coord_data = np.array(all_local_coords, dtype=np.float64)
            search_lengths = np.array(self.mesh._search_lengths[all_local_cells], dtype=np.float64)

            perturbation = (
                (0.33 / (1 + fill_param))
                * (np.random.random(size=coord_data.shape) - 0.5)
                * 0.00001
                * search_lengths  # typical cell size
            )

            # Add perturbation (coords array stores dimensionless values)
            coords[...] = coord_data + perturbation

            self.dm.restoreField("DMSwarmPIC_coor")

            ## Now set the cycle values

            with self.access(self._remeshed):
                for i in range(0, self.recycle_rate):
                    offset = swarm_orig_size * i
                    self._remeshed.data[offset::, 0] = i

        return

    @timing.routine_timer_decorator
    @uw.collective_operation
    def migrate(
        self,
        remove_sent_points=True,
        delete_lost_points=None,
        max_its=10,
    ):
        """
        Migrate swarm across processes after coordinates have been updated.

        The algorithm uses a global kD-tree for the centroids of the domains to decide the particle mpi.rank (send to the closest)
        If the particles are mis-assigned to a particular mpi.rank, the next choice is the second-closest and so on.

        A few particles are still not found after this distribution process which probably means they are just outside the mesh.
        If some points remain lost, they will be deleted if `delete_lost_points` is set.

        Implementation note:
            We retained (above) the name `DMSwarmPIC_coor` for the particle field to allow this routine to be inherited by a PIC swarm
            which has this field pre-defined. (We'd need to add a cellid field as well, and re-compute it upon landing)

        Note: This is a COLLECTIVE operation - all MPI ranks must call it.
        """

        if self._migration_disabled:
            return

        from time import time

        if delete_lost_points is None:
            delete_lost_points = self.clip_to_mesh

        centroids = self.mesh._get_domain_centroids()
        mesh_domain_kdtree = uw.kdtree.KDTree(centroids)

        # This will only worry about particles that are not already claimed !
        #

        swarm_coord_array = (self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.dim))).copy()
        self.dm.restoreField("DMSwarmPIC_coor")

        in_or_not = self.mesh.points_in_domain(
            swarm_coord_array,
        )

        num_points_in_domain = np.count_nonzero(in_or_not == True)
        num_points_not_in_domain = np.count_nonzero(in_or_not == False)
        not_my_points = np.where(in_or_not == False)[0]

        uw.mpi.barrier()

        global_unclaimed_points = int(
            uw.utilities.gather_data(num_points_not_in_domain, bcast=True, dtype=int).sum()
        )

        global_claimed_points = int(
            uw.utilities.gather_data(num_points_in_domain, bcast=True, dtype=int).sum()
        )

        # Unlikely, but we should check this
        uw.mpi.barrier()
        if global_unclaimed_points == 0:
            return

        # Migrate particles between processes (if there are more than one of them)

        if uw.mpi.size > 1:
            for it in range(0, min(max_its, uw.mpi.size)):

                # Send unclaimed points to next processor in line

                swarm_rank_array = self.dm.getField("DMSwarm_rank")
                swarm_coord_array = self.dm.getField("DMSwarmPIC_coor").reshape(-1, self.dim)

                if not_my_points.shape[0] > 0:
                    dist, rank = mesh_domain_kdtree.query(
                        swarm_coord_array[not_my_points], k=it + 1, sqr_dists=False
                    )
                    swarm_rank_array[not_my_points, 0] = rank.reshape(-1, it + 1)[:, it]

                self.dm.restoreField("DMSwarm_rank")
                self.dm.restoreField("DMSwarmPIC_coor")

                # Now we send the points (basic migration)
                self.dm.migrate(remove_sent_points=True)
                uw.mpi.barrier()

                swarm_coord_array = self.dm.getField("DMSwarmPIC_coor").reshape(-1, self.dim)
                in_or_not = self.mesh.points_in_domain(swarm_coord_array)
                self.dm.restoreField("DMSwarmPIC_coor")

                num_points_in_domain = np.count_nonzero(in_or_not == True)
                num_points_not_in_domain = np.count_nonzero(in_or_not == False)
                not_my_points = np.where(in_or_not == False)[0]

                unclaimed_points_last_iteration = global_unclaimed_points
                claimed_points_last_iteration = global_claimed_points

                uw.mpi.barrier()

                global_unclaimed_points = int(
                    uw.utilities.gather_data(
                        num_points_not_in_domain,
                        bcast=True,
                        dtype=int,
                    ).sum()
                )

                global_claimed_points = int(
                    uw.utilities.gather_data(num_points_in_domain, bcast=True, dtype=int).sum()
                )

                if global_unclaimed_points == 0:
                    break

                if (
                    global_unclaimed_points == unclaimed_points_last_iteration
                    and global_claimed_points == claimed_points_last_iteration
                ):
                    break

        # Missing points for deletion if required
        if delete_lost_points:
            uw.mpi.barrier()
            if len(not_my_points) > 0:
                indices = np.sort(not_my_points)[::-1]
                for index in indices:
                    self.dm.removePointAtIndex(index)

                # CRITICAL FIX: Invalidate cached data after removing particles
                # The _particle_coordinates variable caches data - must refresh after DM changes
                self._particle_coordinates._canonical_data = None

                # Also invalidate caches for all swarm variables
                for var in self._vars.values():
                    if hasattr(var, "_canonical_data"):
                        var._canonical_data = None

        return

    def _force_migration_after_mesh_change(self):
        """
        Force migration of swarm particles after mesh coordinate changes.

        This method bypasses the normal migration_disabled check since mesh
        coordinate changes require swarm particles to be re-distributed
        regardless of migration disabled state.
        """
        # Temporarily override migration disabled state
        original_migration_disabled = self._migration_disabled
        self._migration_disabled = False

        try:
            # Disable variable array callbacks during migration to prevent corruption
            # Collect all variable arrays and disable their callbacks
            disabled_arrays = []
            for var in self._vars.values():
                if hasattr(var, "_array_cache") and var._array_cache is not None:
                    var._array_cache.disable_callbacks()
                    disabled_arrays.append(var._array_cache)

            try:
                # Perform standard migration
                self.migrate(remove_sent_points=True, delete_lost_points=True)
            finally:
                # Re-enable variable array callbacks
                for array_cache in disabled_arrays:
                    array_cache.enable_callbacks()

        finally:
            # Restore original migration disabled state
            self._migration_disabled = original_migration_disabled

    @timing.routine_timer_decorator
    def add_particles_with_coordinates(self, coordinatesArray) -> int:
        """
        Add particles to the swarm using particle coordinates provided
        using a numpy array.

        Note that particles with coordinates NOT local to the current processor will
        be rejected / ignored.

        Either include an array with all coordinates to all processors
        or an array with the local coordinates.

        Parameters
        ----------
        coordinatesArray : numpy.ndarray
            The numpy array containing the coordinate of the new particles. Array is
            expected to take shape n*dim, where n is the number of new particles, and
            dim is the dimensionality of the swarm's supporting mesh.

        Returns
        --------
        npoints: int
            The number of points added to the local section of the swarm.
        """

        if not isinstance(coordinatesArray, np.ndarray):
            raise TypeError("'coordinateArray' must be provided as a numpy array")
        if not len(coordinatesArray.shape) == 2:
            raise ValueError("The 'coordinateArray' is expected to be two dimensional.")
        if not coordinatesArray.shape[1] == self.mesh.dim:
            #### petsc appears to ignore columns that are greater than the mesh dim, but still worth including
            raise ValueError(
                """The 'coordinateArray' must have shape n*dim, where 'n' is the
                              number of particles to add, and 'dim' is the dimensionality of
                              the supporting mesh ({}).""".format(
                    self.mesh.dim
                )
            )

        valid = self.mesh.points_in_domain(coordinatesArray, strict_validation=True)
        valid_coordinates = coordinatesArray[valid]
        npoints = len(valid_coordinates)
        swarm_size = self.dm.getLocalSize()

        # -1 means no particles have been added yet (PETSc interface change)
        if swarm_size == -1:
            swarm_size = 0
            # PETSc < 3.24 has an off-by-one bug in addNPoints when swarm size is initially zero
            # It allocates N-1 instead of N, so we add +1 to compensate
            # PETSc 3.24+ fixed this bug, so we use the exact count
            from petsc4py import PETSc
            if PETSc.Sys.getVersion() < (3, 24, 0):
                npoints = npoints + 1

        self.dm.finalizeFieldRegister()
        self.dm.addNPoints(npoints=npoints)

        if npoints > 0:
            coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.dim))
            ranks = self.dm.getField("DMSwarm_rank")
            coords[swarm_size::, :] = valid_coordinates[:, :]
            ranks[swarm_size::] = uw.mpi.rank
            self.dm.restoreField("DMSwarm_rank")
            self.dm.restoreField("DMSwarmPIC_coor")

        # Here we update the swarm cycle values as required

        if self.recycle_rate > 1:
            with self.access(self._remeshed):
                # self._Xorig.data[...] = coordinatesArray
                self._remeshed.data[...] = 0

        self.dm.migrate(remove_sent_points=True)
        return npoints

    @timing.routine_timer_decorator
    def add_particles_with_global_coordinates(
        self,
        globalCoordinatesArray,
        migrate=True,
        delete_lost_points=True,
    ) -> int:
        """
        Add particles to the swarm using particle coordinates provided
        using a numpy array.

        global coordinates: particles will be appropriately migrated

        Parameters
        ----------
        globalCoordinatesArray : numpy.ndarray
            The numpy array containing the coordinate of the new particles. Array is
            expected to take shape n*dim, where n is the number of new particles, and
            dim is the dimensionality of the swarm's supporting mesh.

        Returns
        --------
        npoints: int
            The number of points added to the local section of the swarm.
        """

        if not isinstance(globalCoordinatesArray, np.ndarray):
            raise TypeError("'coordinateArray' must be provided as a numpy array")
        if not len(globalCoordinatesArray.shape) == 2:
            raise ValueError("The 'coordinateArray' is expected to be two dimensional.")
        if not globalCoordinatesArray.shape[1] == self.mesh.dim:
            #### petsc appears to ignore columns that are greater than the mesh dim, but still worth including
            raise ValueError(
                """The 'coordinateArray' must have shape n*dim, where 'n' is the
                                number of particles to add, and 'dim' is the dimensionality of
                                the supporting mesh ({}).""".format(
                    self.mesh.dim
                )
            )

        npoints = len(globalCoordinatesArray)
        swarm_size = self.dm.getLocalSize()

        # -1 means no particles have been added yet
        if swarm_size == -1:
            swarm_size = 0
            # PETSc < 3.24 has an off-by-one bug in addNPoints when swarm size is initially zero
            # It allocates N-1 instead of N, so we add +1 to compensate
            # PETSc 3.24+ fixed this bug, so we use the exact count
            from petsc4py import PETSc
            if PETSc.Sys.getVersion() < (3, 24, 0):
                npoints = npoints + 1

        self.dm.finalizeFieldRegister()
        self.dm.addNPoints(npoints=npoints)

        # Add new points with provided coords
        # Record the current rank (migration needs to know where we start from !)

        coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.dim))
        ranks = self.dm.getField("DMSwarm_rank")
        coords[swarm_size::, :] = globalCoordinatesArray[:, :]
        ranks[swarm_size::] = uw.mpi.rank
        self.dm.restoreField("DMSwarm_rank")
        self.dm.restoreField("DMSwarmPIC_coor")

        # Here we update the swarm cycle values as required

        if self.recycle_rate > 1:
            with self.access(self._remeshed):
                # self._Xorig.data[...] = globalCoordinatesArray
                self._remeshed.data[...] = 0

        if migrate:
            self.migrate(remove_sent_points=True, delete_lost_points=delete_lost_points)

        return npoints

    @timing.routine_timer_decorator
    def save(
        self,
        filename: int,
        compression: Optional[bool] = False,
        compressionType: Optional[str] = "gzip",
        force_sequential=False,
    ):
        """

        Save the swarm coordinates to a h5 file.

        Parameters
        ----------
        filename :
            The filename of the swarm checkpoint file to save to disk.
        compression :
            Add compression to the h5 files (saves space but increases write times with increasing no. of processors)
        compressionType :
            Type of compression to use, 'gzip' and 'lzf' supported. 'gzip' is default. Compression also needs to be set to 'True'.



        """
        if h5py.h5.get_config().mpi == False and comm.size > 1 and comm.rank == 0:
            warnings.warn(
                "Collective IO not possible as h5py not available in parallel mode. Switching to sequential. This will be slow for models running on multiple processors",
                stacklevel=2,
            )
        if filename.endswith(".h5") == False:
            raise RuntimeError("The filename must end with .h5")
        if compression == True and comm.rank == 0:
            warnings.warn("Compression may slow down write times", stacklevel=2)

        if h5py.h5.get_config().mpi == True and not force_sequential:
            # It seems to be a bad idea to mix mpi barriers with the access
            # context manager so the copy-free version of this seems to hang
            # when there are many active cores. This is probably why the parallel
            # h5py write hangs

            points_data_copy = self._particle_coordinates.data[:].copy()

            with h5py.File(f"{filename[:-3]}.h5", "w", driver="mpio", comm=comm) as h5f:
                if compression == True:
                    h5f.create_dataset(
                        "coordinates",
                        data=points_data_copy,
                        compression=compressionType,
                    )
                else:
                    h5f.create_dataset("coordinates", data=points_data_copy)

            del points_data_copy

        else:
            # It seems to be a bad idea to mix mpi barriers with the access
            # context manager so the copy-free version of this seems to hang
            # when there are many active cores

            points_data_copy = self.points[:].copy()

            if comm.rank == 0:
                with h5py.File(f"{filename[:-3]}.h5", "w") as h5f:
                    if compression == True:
                        h5f.create_dataset(
                            "coordinates",
                            data=points_data_copy,
                            chunks=True,
                            maxshape=(None, points_data_copy.shape[1]),
                            compression=compressionType,
                        )
                    else:
                        h5f.create_dataset(
                            "coordinates",
                            data=points_data_copy,
                            chunks=True,
                            maxshape=(None, points_data_copy.shape[1]),
                        )

            comm.barrier()
            for i in range(1, comm.size):
                if comm.rank == i:
                    with h5py.File(f"{filename[:-3]}.h5", "a") as h5f:
                        h5f["coordinates"].resize(
                            (h5f["coordinates"].shape[0] + points_data_copy.shape[0]),
                            axis=0,
                        )
                        # passive swarm, zero local particles is not unusual
                        if data_copy.shape[0] > 0:
                            h5f["coordinates"][-points_data_copy.shape[0] :] = points_data_copy[:]
                comm.barrier()
            comm.barrier()

            del points_data_copy

        ## Add swarm coordinate unit metadata to the file
        import json

        # Use preferred selective_ranks pattern for coordinate metadata
        with uw.selective_ranks(0) as should_execute:
            if should_execute:
                with h5py.File(f"{filename[:-3]}.h5", "a") as h5f:
                    # Add swarm coordinate unit metadata
                    swarm_coord_metadata = {
                        "coordinate_units": (
                            str(self.coordinate_units)
                            if hasattr(self, "coordinate_units")
                            else None
                        ),
                        "coordinate_dimensionality": (
                            str(self.coordinate_dimensionality)
                            if hasattr(self, "coordinate_dimensionality")
                            else None
                        ),
                        "swarm_type": type(self).__name__,
                        "mesh_type": type(self.mesh).__name__ if hasattr(self, "mesh") else None,
                        "dimension": self.dim,
                    }

                    # Store in coordinates dataset attributes
                    if "coordinates" in h5f:
                        h5f["coordinates"].attrs["swarm_metadata"] = json.dumps(
                            swarm_coord_metadata
                        )

        return

    @timing.routine_timer_decorator
    def read_timestep(
        self,
        base_filename: str,
        swarm_id: str,
        index: int,
        outputPath: Optional[str] = "",
        migrate=True,
    ):
        output_base_name = os.path.join(outputPath, base_filename)
        swarm_file = output_base_name + f".{swarm_id}.{index:05}.h5"

        ### open up file with coords on all procs
        with h5py.File(f"{swarm_file}", "r") as h5f:
            coordinates = h5f["coordinates"][:]

        # We make it possible not to migrate the swarm because this
        # will also delete points outside the mesh. We may not want to do
        # that (either for debugging / visualisation, or when adapting the mesh)

        self.add_particles_with_global_coordinates(coordinates, migrate=migrate)

        return

    @timing.routine_timer_decorator
    def add_variable(
        self,
        name,
        size=1,
        dtype=float,
        proxy_degree=2,
        _nn_proxy=False,
        units=None,
    ):
        """
        Add a variable to the swarm.

        Variables must be created before the swarm is populated with particles.
        Once swarm.populate() or similar methods are called, PETSc finalizes
        field registration and no new variables can be added.

        Parameters
        ----------
        name : str
            Variable name
        size : int, default 1
            Number of components (1 for scalar, 2-3 for vector, etc.)
        dtype : type, default float
            Data type (float or int)
        proxy_degree : int, default 2
            Degree for mesh proxy variable interpolation
        _nn_proxy : bool, default False
            Internal parameter for nearest-neighbor proxy
        units : str, optional
            Physical units for this variable (e.g., "kg/m^3", "m/s")

        Returns
        -------
        SwarmVariable
            The created swarm variable

        Raises
        ------
        RuntimeError
            If swarm is already populated with particles

        Examples
        --------
        Correct usage:
        >>> swarm = uw.swarm.Swarm(mesh)
        >>> material = swarm.add_variable("material", 1, dtype=int)
        >>> temperature = swarm.add_variable("temperature", 1)
        >>> swarm.populate(fill_param=3)  # Populate after creating variables

        Incorrect usage (will raise error):
        >>> swarm = uw.swarm.Swarm(mesh)
        >>> swarm.populate(fill_param=3)
        >>> material = swarm.add_variable("material", 1)  # ERROR!
        """
        # Check early to provide a clear error message
        if self.local_size > 0:
            raise RuntimeError(
                f"Cannot add variable '{name}' to swarm: swarm is already populated "
                f"with {self.local_size} particles. Variables must be created "
                f"before calling swarm.populate() or any other operation that adds particles.\n"
                f"\nCorrect usage:\n"
                f"  swarm = uw.swarm.Swarm(mesh)\n"
                f"  variable = swarm.add_variable('{name}', {size})  # Create variables first\n"
                f"  swarm.populate(fill_param=3)  # Then populate with particles"
            )

        return SwarmVariable(
            name,
            self,
            size,
            dtype=dtype,
            proxy_degree=proxy_degree,
            _nn_proxy=_nn_proxy,
            units=units,
        )

    @timing.routine_timer_decorator
    def petsc_save_checkpoint(
        self,
        swarmName: str,
        index: int,
        outputPath: Optional[str] = "",
    ):
        """

        Use PETSc to save the swarm and attached data to a .pbin and xdmf file.

        Parameters
        ----------
        swarmName :
            Name of the swarm to save.
        index :
            An index which might correspond to the timestep or output number (for example).
        outputPath :
            Path to save the data. If left empty it will save the data in the current working directory.
        """

        x_swarm_fname = f"{outputPath}{swarmName}_{index:05d}.xmf"
        self.dm.viewXDMF(x_swarm_fname)

    @timing.routine_timer_decorator
    def write_timestep(
        self,
        filename: str,
        swarmname: str,
        index: int,
        swarmVars: Optional[list] = None,
        outputPath: Optional[str] = "",
        time: Optional[int] = None,
        compression: Optional[bool] = False,
        compressionType: Optional[str] = "gzip",
        force_sequential: Optional[bool] = False,
    ):
        """

        Save data to h5 and a corresponding xdmf for visualisation using h5py.

        Parameters
        ----------
        swarmName :
            Name of the swarm to save.
        swarmVars :
            List of swarm objects to save.
        index :
            An index which might correspond to the timestep or output number (for example).
        outputPath :
            Path to save the data. If left empty it will save the data in the current working directory.
        time :
            Attach the time to the generated xdmf.
        compression :
            Whether to compress the h5 files [bool].
        compressionType :
            The type of compression to use. 'gzip' and 'lzf' are the supported types, with 'gzip' as the default.
        """

        # This will eliminate the issue of whether or not to put path separators in the
        # outputPath. Also does the right thing if outputPath is ""

        output_base_name = os.path.join(outputPath, filename) + "." + swarmname

        # check the directory where we will write checkpoint
        dir_path = os.path.dirname(output_base_name)  # get directory

        # check if path exists
        if os.path.exists(os.path.abspath(dir_path)):  # easier to debug abs
            pass
        else:
            raise RuntimeError(f"{os.path.abspath(dir_path)} does not exist")

        # check if we have write access
        if os.access(os.path.abspath(dir_path), os.W_OK):
            pass
        else:
            raise RuntimeError(f"No write access to {os.path.abspath(dir_path)}")

        # could also try to coerce this to be a list and raise if it fails (tuple, singleton ... )
        # also ... why the typechecking if this can still happen

        if swarmVars is not None and not isinstance(swarmVars, list):
            raise RuntimeError("`swarmVars` does not appear to be a list.")

        else:
            ### save the swarm particle location
            self.save(
                filename=f"{output_base_name}.{index:05d}.h5",
                compression=compression,
                compressionType=compressionType,
                force_sequential=force_sequential,
            )

        #### Generate a h5 file for each field
        if swarmVars != None:
            for field in swarmVars:
                field.save(
                    filename=f"{output_base_name}.{field.name}.{index:05d}.h5",
                    compression=compression,
                    compressionType=compressionType,
                    force_sequential=force_sequential,
                )

        if uw.mpi.rank == 0:
            ### only need to combine the h5 files to a single xdmf on one proc
            with open(f"{output_base_name}.{index:05d}.xdmf", "w") as xdmf:
                # Write the XDMF header
                xdmf.write('<?xml version="1.0" ?>\n')
                xdmf.write('<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">\n')
                xdmf.write("<Domain>\n")
                xdmf.write(f'<Grid Name="{output_base_name}.{index:05d}" GridType="Uniform">\n')

                if time != None:
                    xdmf.write(f'	<Time Value="{time}" />\n')

                # Write the grid element for the HDF5 dataset
                with h5py.File(f"{output_base_name}.{index:05}.h5", "r") as h5f:
                    xdmf.write(
                        f'	<Topology Type="POLYVERTEX" NodesPerElement="{h5f["coordinates"].shape[0]}"> </Topology>\n'
                    )
                    if h5f["coordinates"].shape[1] == 2:
                        xdmf.write('		<Geometry Type="XY">\n')
                    elif h5f["coordinates"].shape[1] == 3:
                        xdmf.write('		<Geometry Type="XYZ">\n')
                    xdmf.write(
                        f'			<DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{h5f["coordinates"].shape[0]} {h5f["coordinates"].shape[1]}">{os.path.basename(h5f.filename)}:/coordinates</DataItem>\n'
                    )
                    xdmf.write("		</Geometry>\n")

                # Write the attribute element for the field
                if swarmVars != None:
                    for field in swarmVars:
                        with h5py.File(
                            f"{output_base_name}.{field.name}.{index:05d}.h5", "r"
                        ) as h5f:
                            if h5f["data"].dtype == np.int32:
                                xdmf.write(
                                    f'	<Attribute Type="Scalar" Center="Node" Name="{field.name}">\n'
                                )
                                xdmf.write(
                                    f'			<DataItem Format="HDF" NumberType="Int" Precision="4" Dimensions="{h5f["data"].shape[0]} {h5f["data"].shape[1]}">{os.path.basename(h5f.filename)}:/data</DataItem>\n'
                                )
                            elif h5f["data"].shape[1] == 1:
                                xdmf.write(
                                    f'	<Attribute Type="Scalar" Center="Node" Name="{field.name}">\n'
                                )
                                xdmf.write(
                                    f'			<DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{h5f["data"].shape[0]} {h5f["data"].shape[1]}">{os.path.basename(h5f.filename)}:/data</DataItem>\n'
                                )
                            elif h5f["data"].shape[1] == 2 or h5f["data"].shape[1] == 3:
                                xdmf.write(
                                    f'	<Attribute Type="Vector" Center="Node" Name="{field.name}">\n'
                                )
                                xdmf.write(
                                    f'			<DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{h5f["data"].shape[0]} {h5f["data"].shape[1]}">{os.path.basename(h5f.filename)}:/data</DataItem>\n'
                                )
                            else:
                                xdmf.write(
                                    f'	<Attribute Type="Tensor" Center="Node" Name="{field.name}">\n'
                                )
                                xdmf.write(
                                    f'			<DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{h5f["data"].shape[0]} {h5f["data"].shape[1]}">{os.path.basename(h5f.filename)}:/data</DataItem>\n'
                                )

                            xdmf.write("	</Attribute>\n")
                else:
                    pass

                # Write the XDMF footer
                xdmf.write("</Grid>\n")
                xdmf.write("</Domain>\n")
                xdmf.write("</Xdmf>\n")

    @property
    def vars(self):
        r"""List of SwarmVariables attached to this swarm.

        Returns
        -------
        list
            List of :class:`SwarmVariable` objects defined on this swarm.
        """
        return self._vars

    def _legacy_access(self, *writeable_vars: SwarmVariable):
        """
        This context manager makes the underlying swarm variables data available to
        the user. The data should be accessed via the variables `data` handle.

        As default, all data is read-only. To enable writeable data, the user should
        specify which variable they wish to modify.

        At the conclusion of the users context managed block, numerous further operations
        will be automatically executed. This includes swarm parallel migration routines
        where the swarm's `particle_coordinates` variable has been modified. The swarm
        variable proxy mesh variables will also be updated for modifed swarm variables.

        Parameters
        ----------
        writeable_vars
            The variables for which data write access is required.

        Example
        -------

        >>> import underworld3 as uw
        >>> someMesh = uw.discretisation.FeMesh_Cartesian()
        >>> with someMesh._deform_mesh():
        ...     someMesh.data[0] = [0.1,0.1]
        >>> someMesh.data[0]
        array([ 0.1,  0.1])
        """
        import time

        uw.timing._incrementDepth()
        stime = time.time()

        deaccess_list = []
        for var in self._vars.values():
            # if already accessed within higher level context manager, continue.
            if var._is_accessed == True:
                continue
            # set flag so variable status can be known elsewhere
            var._is_accessed = True
            # add to de-access list to rewind this later
            deaccess_list.append(var)
            # grab numpy object, setting read only if necessary
            var._data = self.dm.getField(var.clean_name).reshape((-1, var.num_components))
            assert var._data is not None
            if var not in writeable_vars:
                var._old_data_flag = var._data.flags.writeable
                var._data.flags.writeable = False
            else:
                # increment variable state
                var._increment()

            # make *view* for each var component
            if var._proxy:
                for i in range(0, var.shape[0]):
                    for j in range(0, var.shape[1]):
                        var._data_container[i, j] = var._data_container[i, j]._replace(
                            data=var._data[:, var._data_layout(i, j)],
                        )

        # if particles moving, update swarm state
        if self._particle_coordinates in writeable_vars:
            self._increment()

        # Create a class which specifies the required context
        # manager hooks (`__enter__`, `__exit__`).
        class exit_manager:
            def __init__(self, swarm):
                self.em_swarm = swarm

            def __enter__(self):

                pass

            def __exit__(self, *args):

                for var in self.em_swarm.vars.values():
                    # only de-access variables we have set access for.
                    if var not in deaccess_list:
                        continue
                    # set this back, although possibly not required.
                    if var not in writeable_vars:
                        var._data.flags.writeable = var._old_data_flag
                    var._data = None
                    self.em_swarm.dm.restoreField(var.clean_name)
                    var._is_accessed = False
                # do particle migration if coords changes

                if self.em_swarm._particle_coordinates in writeable_vars:
                    # let's use the mesh index to update the particles owning cells.
                    # note that the `petsc4py` interface is more convenient here as the
                    # `SwarmVariable.data` interface is controlled by the context manager
                    # that we are currently within, and it is therefore too easy to
                    # get things wrong that way.
                    #
                    #

                    # if uw.mpi.size > 1:
                    #     coords = self.em_swarm.dm.getField("DMSwarmPIC_coor").reshape(
                    #         (-1, self.em_swarm.dim)
                    #     )

                    #     self.em_swarm.dm.restoreField("DMSwarmPIC_coor")

                    #     ## We'll need to identify the new processes here and update the particle rank value accordingly
                    #

                    # Even if only on one process, migrate needs to be called to remove particles that are
                    # not in the domain.

                    self.em_swarm.migrate(
                        remove_sent_points=True,
                        delete_lost_points=self.em_swarm._clip_to_mesh,
                    )

                    # void these things too
                    self.em_swarm._index = None
                    self.em_swarm._nnmapdict = {}

                # do var updates
                for var in self.em_swarm.vars.values():
                    # if swarm migrated, update all.
                    # if var updated, update var.
                    if (self.em_swarm._particle_coordinates in writeable_vars) or (
                        var in writeable_vars
                    ):
                        var._update()

                    if var._proxy:
                        for i in range(0, var.shape[0]):
                            for j in range(0, var.shape[1]):
                                # var._data_ij[i, j] = None
                                var._data_container[i, j] = var._data_container[i, j]._replace(
                                    data=f"SwarmVariable[...].data is only available within mesh.access() context",
                                )

                uw.timing._decrementDepth()
                uw.timing.log_result(time.time() - stime, "Swarm.access", 1)

        return exit_manager(self)

    def access(self, *writeable_vars: SwarmVariable):
        """
        Dummy access manager that provides deferred sync for backward compatibility.
        Uses NDArray_With_Callback.delay_callbacks_global() internally.

        This is a compatibility wrapper that allows existing code using the access()
        context manager to work with the new direct-access variable interfaces.
        All variable modifications are deferred and synchronized at context exit.

        Parameters
        ----------
        writeable_vars
            Variables that will be modified (ignored - all variables are writable
            with the new interface, this parameter is kept for API compatibility)

        Returns
        -------
        Context manager that defers variable synchronization until exit

        Notes
        -----
        This method is deprecated. New code should access variable.data or
        variable.array directly without requiring an access context.
        """
        import underworld3.utilities

        class DummyAccessContext:
            def __init__(self, swarm, writeable_vars):
                self.swarm = swarm
                self.writeable_vars = writeable_vars
                self.delay_context = None

            def __enter__(self):
                # Use NDArray_With_Callback global delay context for deferred sync
                self.delay_context = (
                    underworld3.utilities.NDArray_With_Callback.delay_callbacks_global(
                        "swarm.access compatibility"
                    )
                )
                return self.delay_context.__enter__()

            def __exit__(self, exc_type, exc_val, exc_tb):
                # This triggers all accumulated callbacks from all variables
                if self.delay_context:
                    return self.delay_context.__exit__(exc_type, exc_val, exc_tb)
                return False

        return DummyAccessContext(self, writeable_vars)

    ## Better to have one master copy - this one is cut'n'pasted from
    ## the MeshVariable class

    def _data_layout(self, i, j=None):
        # mapping

        if self.vtype == uw.VarType.SCALAR:
            return 0
        if self.vtype == uw.VarType.VECTOR:
            if j is None:
                return i
            elif i == 0:
                return j
            else:
                raise IndexError(f"Vectors have shape {self.mesh.dim} or {(1, self.mesh.dim)} ")
        if self.vtype == uw.VarType.TENSOR:
            if self.mesh.dim == 2:
                return ((0, 1), (2, 3))[i][j]
            else:
                return ((0, 1, 2), (3, 4, 5), (6, 7, 8))[i][j]

        if self.vtype == uw.VarType.SYM_TENSOR:
            if self.mesh.dim == 2:
                return ((0, 2), (2, 1))[i][j]
            else:
                return ((0, 3, 4), (3, 1, 5), (4, 5, 2))[i][j]

        if self.vtype == uw.VarType.MATRIX:
            return i + j * self.shape[0]

    ## Check this - the interface to kdtree has changed, are we picking the correct field ?
    @timing.routine_timer_decorator
    def _get_map(self, var):
        # generate tree if not avaiable
        if not self._index:
            self._index = uw.kdtree.KDTree(self.data)

        # get or generate map
        meshvar_coords = var._meshVar.coords
        # we can't use numpy arrays directly as keys in python dicts, so
        # we'll use `xxhash` to generate a hash of array.
        # this shouldn't be an issue performance wise but we should test to be
        # sufficiently confident of this.
        import xxhash

        h = xxhash.xxh64()
        h.update(meshvar_coords)
        digest = h.intdigest()
        if digest not in self._nnmapdict:
            self._nnmapdict[digest] = self._index.query(meshvar_coords, k=1, sqr_dists=False)[1]
        return self._nnmapdict[digest]

    @timing.routine_timer_decorator
    def advection(
        self,
        V_fn,
        delta_t,
        order=2,
        corrector=False,
        restore_points_to_domain_func=None,
        evalf=False,
        step_limit=False,
    ):
        # Convert delta_t to model units if it has units
        # This ensures consistent arithmetic: velocity is in model units, so time must be too
        import underworld3 as uw

        delta_t_model = uw.scaling.non_dimensionalise(delta_t)

        dt_limit = self.estimate_dt(V_fn)

        if step_limit and dt_limit is not None:
            substeps = int(max(1, round(abs(delta_t_model) / dt_limit)))
        else:
            substeps = 1

        if uw.mpi.rank == 0 and self.verbose:
            print(f"Substepping {substeps} / {abs(delta_t) / dt_limit}, {delta_t} ")

        # X0 holds the particle location at the start of advection
        # This is needed because the particles may be migrated off-proc
        # during timestepping. Probably not needed - use global evaluation instead

        X0 = self._X0

        V_fn_matrix = self.mesh.vector.to_matrix(V_fn)

        # Use current velocity to estimate where the particles would have
        # landed in an implicit step. WE CANT DO THIS WITH SUB-STEPPING unless
        # We have a lot more information about the previous launch point / timestep
        # Also: how does this interact with the particle restoration function ?

        # if corrector == True and not self._X0_uninitialised:
        #     with self.access(self._particle_coordinates):
        #         v_at_Vpts = np.zeros_like(self.data)

        #         if evalf:
        #             for d in range(self.dim):
        #                 v_at_Vpts[:, d] = uw.function.evalf(
        #                     V_fn_matrix[d], self.data
        #                 ).reshape(-1)
        #         else:
        #             for d in range(self.dim):
        #                 v_at_Vpts[:, d] = uw.function.evaluate(
        #                     V_fn_matrix[d], self.data
        #                 ).reshape(-1)

        #         corrected_position = X0.data.copy() + delta_t * v_at_Vpts
        #         if restore_points_to_domain_func is not None:
        #             corrected_position = restore_points_to_domain_func(
        #                 corrected_position
        #             )

        #         updated_current_coords = 0.5 * (corrected_position + self.data.copy())

        #         # validate_coords to ensure they live within the domain (or there will be trouble)

        #         if restore_points_to_domain_func is not None:
        #             updated_current_coords = restore_points_to_domain_func(
        #                 updated_current_coords
        #             )

        #         self.data[...] = updated_current_coords[...]

        #         del updated_current_coords
        #         del v_at_Vpts

        # Wrap this whole thing in sub-stepping loop
        for step in range(0, substeps):

            X0.array[:, 0, :] = self._particle_coordinates.data[...]

            # Mid point algorithm (2nd order)

            if order == 2:
                print(f"Advection (2nd): {self.local_size} - swarm points", flush=True)

                # Use internal model-unit coordinates directly (no conversion needed)
                v_at_Vpts = np.zeros_like(self._particle_coordinates.data[...])

                # First evaluate the velocity at the particle locations
                # (this is a local operation)

                v_at_Vpts[...] = uw.function.evaluate(V_fn_matrix, self._particle_coordinates.data)[
                    :, 0, :
                ]

                mid_pt_coords = (
                    self._particle_coordinates.data[...]
                    + 0.5 * delta_t_model * v_at_Vpts / substeps
                )

                # This will re-position particles in periodic domains (etc)
                if self.mesh.return_coords_to_bounds is not None:
                    mid_pt_coords = self.mesh.return_coords_to_bounds(mid_pt_coords)

                # Now do a **Global** evaluation
                # (since the mid-points might have moved off-proc)
                #

                v_at_Vpts[...] = uw.function.global_evaluate(V_fn_matrix, mid_pt_coords)[:, 0, :]

                new_coords = X0.array[:, 0, :] + delta_t_model * v_at_Vpts / substeps

                if self.mesh.return_coords_to_bounds is not None:
                    new_coords = self.mesh.return_coords_to_bounds(new_coords)

                # Set the new particle positions (and automatically migrate)
                self._particle_coordinates.data[...] = new_coords[...]

                del new_coords
                del v_at_Vpts

            # forward Euler (1st order)
            else:
                coords = self._particle_coordinates.data
                print(
                    f"1. Advection (1st): {coords.shape} v {self.local_size} - swarm point shape",
                    flush=True,
                )

                v_at_Vpts = np.zeros_like(coords)
                v_at_Vpts[...] = uw.function.global_evaluate(V_fn_matrix, coords[...])[:, 0, :]

                print(
                    f"2. Advection (1st): {coords.shape} v {self.local_size} - swarm point shape",
                    flush=True,
                )

                new_coords = coords[...] + delta_t_model * v_at_Vpts / substeps

                print(
                    f"3. Advection (1st): {coords.shape} v {self.local_size} - swarm point shape",
                    flush=True,
                )

                if self.mesh.return_coords_to_bounds is not None:
                    new_coords = self.mesh.return_coords_to_bounds(new_coords)

                self._particle_coordinates.data[...] = new_coords[...]

        ## End of substepping loop

        ## Cycling of the swarm is a cheap and cheerful version of population control for particles. It turns the
        ## swarm into a streak-swarm where particles are Lagrangian for a number of steps and then reset to their
        ## original location.

        if self.recycle_rate > 1:
            # Restore particles which have cycle == cycle rate (use >= just in case)

            # Remove remesh points and recreate a new set at the mesh-local
            # locations that we already have stored.

            with self.access(self._particle_coordinates, self._remeshed):
                remeshed = self._remeshed.data[:, 0] == 0
                # This is one way to do it ... we can do this better though
                self.data[remeshed, 0] = 1.0e100

            swarm_size = self.dm.getLocalSize()

            num_remeshed_points = self.mesh.particle_X_orig.shape[0]

            self.dm.addNPoints(num_remeshed_points)

            ## cellid = self.dm.getField("DMSwarm_cellid")
            coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.dim))
            rmsh = self.dm.getField("DMSwarm_remeshed")

            # print(f"cellid -> {cellid.shape}")
            # print(f"particle coords -> {coords.shape}")
            # print(f"remeshed points  -> {num_remeshed_points}")

            # Compute perturbation - extract magnitude if coordinates have units
            # numpy.array(..., dtype=float64) forces conversion to plain array
            coord_data = np.array(self.mesh.particle_X_orig[:, :], dtype=np.float64)
            radii_data = np.array(self.mesh._radii[cellid[swarm_size::]], dtype=np.float64)

            perturbation = 0.00001 * (
                (0.33 / (1 + self.fill_param))
                * (np.random.random(size=(num_remeshed_points, self.dim)) - 0.5)
                * radii_data.reshape(-1, 1)
            )

            # Add perturbation (coords array stores dimensionless values)
            coords[swarm_size::] = coord_data + perturbation
            ## cellid[swarm_size::] = self.mesh.particle_CellID_orig[:, 0]
            rmsh[swarm_size::] = 0

            # self.dm.restoreField("DMSwarm_cellid")
            self.dm.restoreField("DMSwarmPIC_coor")
            self.dm.restoreField("DMSwarm_remeshed")

            # when we let this go, the particles may be re-distributed to
            # other processors, and we will need to rebuild the remeshed
            # array before trying to compute / assign values to variables

            for swarmVar in self.vars.values():
                if swarmVar._rebuild_on_cycle:
                    with self.access(swarmVar):
                        if swarmVar.dtype is int:
                            nnn = 1
                        else:
                            nnn = self.mesh.dim + 1  # 3 for triangles, 4 for tets ...

                        interpolated_values = (
                            swarmVar.rbf_interpolate(self.mesh.particle_X_orig, nnn=nnn)
                            #     swarmVar._meshVar.fn, self.mesh.particle_X_orig
                            # )
                        ).astype(swarmVar.dtype)

                        swarmVar.data[swarm_size::] = interpolated_values

            ##
            ## Determine RANK
            ##

            # Migrate will already have been called by the access manager.
            # Maybe we should hash the local particle coords to make this
            # a little more user-friendly

            # self.dm.migrate(remove_sent_points=True)

            with self.access(self._remeshed):
                self._remeshed.data[...] = np.mod(self._remeshed.data[...] - 1, self.recycle_rate)

            self.cycle += 1

            ## End of cycle_swarm loop
            #
            #

        # Remove points no longer in the domain
        self.migrate(
            delete_lost_points=True,
            max_its=1,
        )

        return

    @timing.routine_timer_decorator
    def estimate_dt(self, V_fn):
        """
        Calculates an appropriate advective timestep for the given
        mesh and velocity configuration.
        """
        # we'll want to do this on an element by element basis
        # for more general mesh

        # first let's extract a max global velocity magnitude
        import math
        import numpy as np

        vel = uw.function.evaluate(V_fn, self._particle_coordinates.data, evalf=True)

        # If vel is unit-aware (UnitAwareArray), nondimensionalise it to get
        # consistent nondimensional values that match mesh._radii
        # Note: .magnitude returns physical units, which would be wrong here
        if hasattr(vel, "units") and vel.units is not None:
            vel = uw.non_dimensionalise(vel)
        elif hasattr(vel, "magnitude"):
            # Plain UWQuantity without units context - use magnitude
            vel = vel.magnitude

        # Ensure vel is a plain numpy array
        vel = np.asarray(vel)

        try:
            magvel_squared = vel[:, 0] ** 2 + vel[:, 1] ** 2
            if self.mesh.dim == 3:
                magvel_squared += vel[:, 2] ** 2

            max_magvel = math.sqrt(magvel_squared.max())

        except (ValueError, IndexError):
            max_magvel = 0.0

        from mpi4py import MPI

        max_magvel_glob = comm.allreduce(max_magvel, op=MPI.MAX)

        min_dx = self.mesh.get_min_radius()

        # The assumption should be that we cross one or two elements (2-4 radii), not more,
        # in a single step (order 2, means one element per half-step or something
        # that we can broadly interpret that way)

        if max_magvel_glob != 0.0:
            return min_dx / max_magvel_glob
        else:
            return None


class NodalPointSwarm(Swarm):
    r"""BASIC_Swarm with particles located at the coordinate points of a meshVariable

    The swarmVariable `X0` is defined so that the particles can "snap back" to their original locations
    after they have been moved.

    The purpose of this Swarm is to manage sample points for advection schemes based on upstream sampling
    (method of characteristics etc)"""

    def __init__(
        self,
        trackedVariable: uw.discretisation.MeshVariable,
        verbose=False,
    ):
        self.trackedVariable = trackedVariable
        self.swarmVariable = None

        mesh = trackedVariable.mesh

        # Set up a standard swarm
        super().__init__(mesh, verbose, clip_to_mesh=False)

        nswarm = self

        meshVar_name = trackedVariable.clean_name
        meshVar_symbol = trackedVariable.symbol

        ks = str(self.instance_number)
        name = f"{meshVar_name}_star"
        symbol = rf"{{ {meshVar_symbol} }}^{{ <*> }}"

        self.swarmVariable = uw.swarm.SwarmVariable(
            name,
            nswarm,
            vtype=trackedVariable.vtype,
            _proxy=False,
            varsymbol=symbol,
        )

        # The launch point location
        name = f"ns_X0_{ks}"
        symbol = r"X0^{*^{{[" + ks + "]}}}"
        nX0 = uw.swarm.SwarmVariable(name, nswarm, nswarm.dim, _proxy=False)

        # The launch point index
        name = f"ns_I_{ks}"
        symbol = r"I^{*^{{[" + ks + "]}}}"
        nI0 = uw.swarm.SwarmVariable(name, nswarm, 1, dtype=int, _proxy=False)

        # The launch point processor rank
        name = f"ns_R0_{ks}"
        symbol = r"R0^{*^{{[" + ks + "]}}}"
        nR0 = uw.swarm.SwarmVariable(name, nswarm, 1, dtype=int, _proxy=False)

        nswarm.dm.finalizeFieldRegister()
        # PETSc < 3.24 has an off-by-one bug in addNPoints when swarm size is initially zero
        # It allocates N-1 instead of N, so we add +1 to compensate
        # PETSc 3.24+ fixed this bug, so we use the exact count
        from petsc4py import PETSc
        npts_to_add = trackedVariable.coords.shape[0]
        if PETSc.Sys.getVersion() < (3, 24, 0):
            npts_to_add = npts_to_add + 1
        nswarm.dm.addNPoints(npts_to_add)

        coords = nswarm.dm.getField("DMSwarmPIC_coor").reshape((-1, nswarm.dim))
        ranks = nswarm.dm.getField("DMSwarm_rank").reshape((-1, 1))
        coords[...] = trackedVariable.coords[...]
        ranks[...] = uw.mpi.rank

        cellid = self.mesh.get_closest_cells(
            coords,
        )

        # Move slightly within the chosen cell to avoid edge effects
        centroid_coords = self.mesh._centroids[cellid]

        shift = 0.001
        coords[:, :] = (1.0 - shift) * coords[:, :] + shift * centroid_coords[:, :]

        nswarm.dm.restoreField("DMSwarmPIC_coor")
        nswarm.dm.restoreField("DMSwarm_rank")

        nswarm.dm.migrate(remove_sent_points=True)

        with nswarm.access(nX0, nI0):
            nX0.data[:, :] = coords
            nI0.data[:, 0] = range(0, coords.shape[0])

        self._nswarm = nswarm
        self._nX0 = nX0
        self._nI0 = nI0
        self._nR0 = nR0

        return

    @timing.routine_timer_decorator
    def advection(
        self,
        V_fn,
        delta_t,
        order=2,
        corrector=False,
        restore_points_to_domain_func=None,
        evalf=False,
        step_limit=True,
    ):

        with self.access(self._X0):
            self._X0.data[...] = self._nX0.data[...]

        with self.access(self._nR0):
            self._nR0.data[...] = uw.mpi.rank

        super().advection(
            V_fn,
            delta_t,
            order,
            corrector,
            restore_points_to_domain_func,
            evalf,
            step_limit,
        )

        return


## New - Basic Swarm (no PIC skillz)
## What is missing:
##  - no celldm
##  - PIC layouts of particles are not directly available / must be done by hand
##  - No automatic migration - must compute ranks for the particle swarms
##  - No automatic definition of coordinate fields (need to add by hand)
