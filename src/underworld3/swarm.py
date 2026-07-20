r"""
Particle swarm management for Lagrangian tracking.

This module provides particle swarm (point cloud) data structures for
tracking material properties through deformation. Swarms enable Lagrangian
representations of material history, composition, and other quantities
that move with the flow.

**SwarmType** -- PETSc swarm type specification (BASIC or PIC).

**SwarmVariable** -- Variable storing values at particle locations with
mesh-based proxy for use in symbolic expressions.

**IndexSwarmVariable** -- Integer-valued swarm variable for material indexing.

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
        No effect. Retained for backward compatibility with the removed
        particle-recycling (streak swarm) feature.
    units : str or pint.Unit, optional
        Physical units for this variable (e.g., 'kelvin', 'Pa').
        Requires reference quantities to be set on the model.

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
            # Note: on a cd-1 mesh (dim < cdim, e.g. SphericalManifold),
            # vector fields are stored with ``cdim`` components in the
            # embedded coordinate space (tangent-constrained 3-vectors
            # on a 2-manifold) and the SwarmVariable coord cache is
            # built with size == mesh.cdim. We accept either dim or
            # cdim as a vector match; on volume meshes dim == cdim so
            # the two branches coincide.
            if isinstance(size, int) and size == 1:
                vtype = uw.VarType.SCALAR
            elif isinstance(size, int) and (size == mesh.dim or size == mesh.cdim):
                vtype = uw.VarType.VECTOR
            elif isinstance(size, tuple):
                if (
                    (size[0] == mesh.dim and size[1] == mesh.dim)
                    or (size[0] == mesh.cdim and size[1] == mesh.cdim)
                ):
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
            # On manifold meshes (dim != cdim) coordinate-like vector
            # fields carry cdim components in the embedding space. Honor
            # the explicitly supplied size when it matches cdim; default
            # to mesh.dim otherwise. On volume meshes the two coincide.
            if isinstance(size, int) and size == mesh.cdim:
                self.num_components = mesh.cdim
                self.shape = (1, mesh.cdim)
                self.cpt_map = tuple(range(0, mesh.cdim))
            else:
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

        # Inert: kept for backward compatibility with the removed
        # particle-recycling (streak swarm) feature.
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
            initial_data = self.unpack_uw_data_from_petsc(squeeze=False)

        # Create NDArray_With_Callback (following swarm._points pattern)
        array_obj = uw.utilities.NDArray_With_Callback(
            initial_data,
            owner=self,
            disable_inplace_operators=False,  # Allow operations like existing arrays
        )

        # Single callback function (following swarm_update_callback pattern)
        def variable_update_callback(array, change_context):
            """Callback to sync variable changes back to PETSc (like swarm.points)"""
            var = array.owner
            if var is None:
                # This guard handles cases where the array is accessed during
                # object teardown (e.g. at application exit or mesh rebuilds),
                # where the owning Python variable has already been garbage
                # collected but the NDArray proxy still exists.
                return

            # Only act on data-changing operations (following swarm.points pattern)
            data_changed = change_context.get("data_has_changed", True)
            if not data_changed:
                return

            # While migration is suppressed, DEFER the PETSc pack rather than
            # discarding the write: the DMSwarm layout may be mid-change, so
            # packing now could corrupt it, but the user's values must survive.
            # They are flushed by Swarm._flush_pending_petsc_sync() when the
            # migration-control context exits (SWARM-04).
            if getattr(var.swarm, "_migration_disabled", False):
                var.swarm._pending_petsc_sync.add(var.clean_name)
                return

            # Persist changes to PETSc (like swarm callback updates coordinates)
            var.pack_uw_data_to_petsc(array)

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
            initial_data = self.unpack_raw_data_from_petsc(squeeze=False)

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

            # While migration is suppressed, DEFER the PETSc pack rather than
            # discarding the write: the DMSwarm layout may be mid-change, so
            # packing now could corrupt it, but the user's values must survive
            # in the canonical array. They are flushed to PETSc by
            # Swarm._flush_pending_petsc_sync() when the migration-control
            # context exits (SWARM-04: previously these writes were silently
            # lost — nothing re-packed, and migrate()'s trailing invalidation
            # destroyed the only copy).
            if getattr(self.swarm, "_migration_disabled", False):
                self.swarm._pending_petsc_sync.add(self.clean_name)
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
            self.pack_raw_data_to_petsc(canonical_array)

            # Coordinate writes may strand particles on the wrong rank. Mark
            # the swarm for DEFERRED migration — migrate() itself is
            # collective and must not run from a per-write callback (ranks
            # write unevenly → deadlock). The migration happens at the next
            # collective point: migration-control context exit or solve entry
            # (SWARM-03; the class docstring's automatic-migration promise).
            if getattr(self.swarm, "_coord_var", None) is self:
                self.swarm._needs_migration = True

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

            # Reduction methods follow the MeshVariable array-view contract:
            # scalar (float) for single-component variables, per-component
            # tuple for multi-component variables (LE-07 / BF-11).
            #
            # NOTE: these are simple arithmetic reductions over the particle
            # values. Swarm particles are generally non-uniformly distributed
            # in space, so mean()/std() only APPROXIMATE the spatial
            # statistics — use mesh integrals of the proxy field for
            # spatially-accurate statistics.

            def _per_component_reduction(self, reduction):
                data = self._get_array_data()
                if self.parent.num_components == 1:
                    return float(reduction(data))
                flat = np.asarray(data).reshape(data.shape[0], -1)
                return tuple(
                    float(reduction(flat[:, i])) for i in range(self.parent.num_components)
                )

            def max(self):
                """Maximum (float for scalars, per-component tuple otherwise)."""
                return self._per_component_reduction(np.max)

            def min(self):
                """Minimum (float for scalars, per-component tuple otherwise)."""
                return self._per_component_reduction(np.min)

            def mean(self):
                """Arithmetic particle mean (float for scalars, per-component tuple otherwise)."""
                return self._per_component_reduction(np.mean)

            def sum(self):
                """Sum (float for scalars, per-component tuple otherwise)."""
                return self._per_component_reduction(np.sum)

            def std(self):
                """Arithmetic particle standard deviation (float for scalars, per-component tuple otherwise)."""
                return self._per_component_reduction(np.std)

            @property
            def shape(self):
                return self._get_array_data().shape

            @property
            def dtype(self):
                return self._get_array_data().dtype

            def __array__(self, dtype=None, copy=None):
                """Support for numpy functions like np.allclose(), np.isfinite(), etc.

                numpy 2.0 calls __array__ with dtype/copy keywords; honour them.
                """
                arr = self._get_array_data()
                if dtype is not None:
                    arr = arr.astype(dtype, copy=bool(copy))
                elif copy:
                    arr = arr.copy()
                return arr

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

            # Reduction methods follow the MeshVariable array-view contract:
            # scalar (float) for single-component variables, per-component
            # tuple for multi-component variables (LE-07 / BF-11). Components
            # are ordered as in the flat canonical layout.
            #
            # NOTE: these are simple arithmetic reductions over the particle
            # values. Swarm particles are generally non-uniformly distributed
            # in space, so mean()/std() only APPROXIMATE the spatial
            # statistics — use mesh integrals of the proxy field for
            # spatially-accurate statistics.

            def _per_component_reduction(self, reduction):
                data = self._get_array_data()
                if self.parent.num_components == 1:
                    return float(reduction(data))
                flat = np.asarray(data).reshape(data.shape[0], -1)
                return tuple(
                    float(reduction(flat[:, i])) for i in range(self.parent.num_components)
                )

            def max(self):
                """Maximum (float for scalars, per-component tuple otherwise)."""
                return self._per_component_reduction(np.max)

            def min(self):
                """Minimum (float for scalars, per-component tuple otherwise)."""
                return self._per_component_reduction(np.min)

            def mean(self):
                """Arithmetic particle mean (float for scalars, per-component tuple otherwise)."""
                return self._per_component_reduction(np.mean)

            def sum(self):
                """Sum (float for scalars, per-component tuple otherwise)."""
                return self._per_component_reduction(np.sum)

            def std(self):
                """Arithmetic particle standard deviation (float for scalars, per-component tuple otherwise)."""
                return self._per_component_reduction(np.std)

            @property
            def shape(self):
                return self._get_array_data().shape

            @property
            def dtype(self):
                return self._get_array_data().dtype

            def __array__(self, dtype=None, copy=None):
                """Support for numpy functions like np.allclose(), np.isfinite(), etc.

                numpy 2.0 calls __array__ with dtype/copy keywords; honour them.
                """
                arr = self._get_array_data()
                if dtype is not None:
                    arr = arr.astype(dtype, copy=bool(copy))
                elif copy:
                    arr = arr.copy()
                return arr

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
        # Empty-partition guard: an N=0 array has total size 0, so numpy cannot
        # infer the -1 component dimension ("cannot reshape array of size 0 into
        # shape (0,newaxis)"). This bites a rank that owns no local particles
        # during a parallel read_timestep. Compute the component count from the
        # trailing dims explicitly.
        if array_data.size == 0:
            ncomp = int(np.prod(array_data.shape[1:])) if array_data.ndim > 1 else 1
            return array_data.reshape(array_data.shape[0], ncomp)
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
            # REINIT policy: a swarm proxy is re-projected from the
            # *particles* on next access, so a mesh adapt should NOT
            # interpolate the old proxy values onto the new node
            # layout — that would freeze stale per-particle data on the
            # new mesh and miss particle migration. The helper marks the
            # var stale via _mark_reinit_stale; we wire that callback
            # below to set ``self._proxy_stale = True`` so the next
            # access re-projects.
            self._meshVar = uw.discretisation.MeshVariable(
                "proxy_" + self.clean_name,
                self.swarm.mesh,
                self.shape,
                self._vtype,
                degree=self._proxy_degree,
                continuous=self._proxy_continuous,
                varsymbol=r"\left<" + self.symbol + r"\right>",
                remesh_policy="reinit",
            )
            # The remesh helper calls this on REINIT vars after an
            # adapt. Bound here so the closure captures ``self`` (the
            # SwarmVariable) rather than the proxy MeshVariable.
            self._meshVar._remesh_reinit_callback = (
                lambda _self=self: setattr(_self, "_proxy_stale", True))

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

    # NB: laziness is safe because freshness is enforced at CONSUMPTION:
    # the `.sym` accessors call this, and `Swarm._sync_before_assembly()`
    # (invoked from Mesh.update_lvec at solve entry) eagerly refreshes any
    # stale proxy before a solver reads the proxy DM directly
    # (issue #215 Bug 3 / issue #289).
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

        # Lifetime contract: variables hold their parent swarm by WEAK
        # reference (a strong back-reference would cycle with the swarm's
        # own strong _coord_var/_X0 members and defer DMSwarm destruction
        # from refcount-immediate to gc time — the transient-evaluation-
        # swarm leak guarded by tests/test_0006_memory_leak.py). A variable
        # that outlives its swarm is therefore a symbolic FOSSIL: its proxy
        # keeps the last projection and can never be refreshed — say so
        # loudly instead of raising from deep inside a .sym access.
        # Particle-data paths (.data, rbf_interpolate, ...) still raise
        # via the .swarm property guard.
        if self._swarm_ref is None or self._swarm_ref() is None:
            import warnings

            warnings.warn(
                f"SwarmVariable '{self.clean_name}': the parent swarm no "
                "longer exists; the proxy mesh variable retains its last "
                "projection and cannot be refreshed.",
                stacklevel=2,
            )
            self._proxy_stale = False  # nothing can ever refresh it again
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

        # Starved-rank guard (SWARM-07): with <= 1 local particles there is
        # nothing meaningful to interpolate — rbf_interpolate would return
        # silent zeros. Keep this rank's current proxy nodal values instead,
        # and say so. NB: MeshVariable reads/writes perform collective ghost
        # synchronisation, so EVERY rank must execute the same read-then-write
        # sequence; only the values differ on starved ranks.
        current_values = np.array(meshVar.data[...], copy=True)

        if self.swarm.local_size <= 1:
            # Warn only once the swarm has ever held particles: proxied
            # variables are created (and their .sym touched) before
            # populate(), and that expected pre-population state should not
            # generate noise.
            if self.swarm._population_generation > 0:
                import warnings

                warnings.warn(
                    f"Swarm proxy update: rank {uw.mpi.rank} holds "
                    f"{max(self.swarm.local_size, 0)} particles; proxy variable "
                    f"'{getattr(meshVar, 'clean_name', meshVar.name)}' left "
                    "unchanged on this rank.",
                    stacklevel=2,
                )
            Values = current_values
        else:
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

        # Use cached KDTree for interpolation (avoids redundant index construction)
        kd = meshVar._get_kdtree()

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

    @staticmethod
    def _warn_deprecated_sync(sync):
        """One-cycle keyword shim for the removed no-op ``sync=`` argument on
        the four pack/unpack methods (it never had an effect)."""
        if sync is not None:
            import warnings
            warnings.warn(
                "the 'sync' argument never had an effect and is deprecated; "
                "remove it",
                DeprecationWarning, stacklevel=3)

    def pack_uw_data_to_petsc(self, data_array, sync=None):
        """
        Enhanced pack method that directly accesses PETSc field without access() context.
        Designed for the new swarmVariable.array interface.

        Parameters
        ----------
        data_array : numpy.ndarray
            Array data to pack into PETSc field
        sync : deprecated
            Never had an effect; deprecated (one DeprecationWarning if passed).
        """
        self._warn_deprecated_sync(sync)
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

    def unpack_uw_data_from_petsc(self, squeeze=True, sync=None):
        """
        Enhanced unpack method that directly accesses PETSc field without access() context.
        Designed for the new swarmVariable.array interface.

        Parameters
        ----------
        squeeze : bool
            Whether to squeeze singleton dimensions (default True)
        sync : deprecated
            Never had an effect; deprecated (one DeprecationWarning if passed).
        """
        self._warn_deprecated_sync(sync)
        shape = self.shape

        # Direct PETSc field access without context manager
        petsc_data = self.swarm.dm.getField(self.clean_name).reshape((-1, self.num_components))

        try:
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

    def pack_raw_data_to_petsc(self, data_array, sync=None):
        """
        Pack data array to PETSc using traditional data shape (-1, num_components).
        Direct PETSc access without access() context for backward compatibility.

        Parameters
        ----------
        data_array : numpy.ndarray
            Array data in traditional flat format (-1, num_components)
        sync : deprecated
            Never had an effect; deprecated (one DeprecationWarning if passed).
        """
        self._warn_deprecated_sync(sync)
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

        finally:
            # Always restore the field
            self.swarm.dm.restoreField(self.clean_name)

        return

    def unpack_raw_data_from_petsc(self, squeeze=True, sync=None):
        """
        Unpack data from PETSc in traditional data shape (-1, num_components).
        Direct PETSc access without access() context for backward compatibility.

        Parameters
        ----------
        squeeze : bool
            Whether to remove singleton dimensions (default True)
        sync : deprecated
            Never had an effect; deprecated (one DeprecationWarning if passed).

        Returns
        -------
        numpy.ndarray
            Array data in traditional flat format (-1, num_components)
        """
        self._warn_deprecated_sync(sync)
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
        raw_data = self.unpack_raw_data_from_petsc(squeeze=False)
        data_size = raw_data.shape

        # What to do if there are no particles: never SILENTLY return zeros
        # (SWARM-07) — a starved rank writing these into a proxy corrupts it.
        # (Silent only for a swarm that has never been populated: proxied
        # variables legitimately touch this path at creation time.)
        if data_size[0] <= 1:
            if self.swarm._population_generation > 0:
                import warnings

                warnings.warn(
                    f"rbf_interpolate: rank {uw.mpi.rank} holds only "
                    f"{data_size[0]} particles of swarm variable "
                    f"'{self.clean_name}' — returning zeros for this rank's "
                    "query points.",
                    stacklevel=2,
                )
            return np.zeros((new_coords.shape[0], data_size[1]))

        if nnn is None:
            nnn = self.swarm.mesh.dim + 1

        if nnn > data_size[0]:
            nnn = data_size[0]

        # Use direct PETSc access to avoid callback circular dependency
        D = raw_data.copy()
        kdt = self.swarm._get_kdtree()
        values = kdt.rbf_interpolator_local(new_coords, D, nnn, 2, verbose)

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
        Canonical data storage in flat format for internal operations.

        Returns particle data in shape ``(-1, num_components)`` regardless of
        variable type. Values are always **non-dimensional** (no unit conversion).

        This property handles PETSc synchronization. The ``.array`` property is
        a view of this with shape conversion.

        When to Use
        -----------
        - **Variable-to-variable transfers**: Copying data between swarm variables
          avoids redundant unit conversions
        - **Low-level operations**: Direct access to particle data
        - **Backward compatibility**: Existing code using flat format

        For general user access, prefer ``.array`` which provides a structured shape.

        Returns
        -------
        NDArray_With_Callback
            Array with shape ``(-1, num_components)`` with automatic PETSc sync.

        Examples
        --------
        >>> # Efficient variable-to-variable copy
        >>> new_material.data[...] = old_material.data[...]

        >>> # Check data shape
        >>> scalar_var.data.shape  # (n_particles, 1)
        >>> vector_var.data.shape  # (n_particles, dim)

        See Also
        --------
        array : Structured format ``(N, a, b)`` for general access.
        sym : Symbolic representation for equations.
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
        Primary interface for reading and writing particle data.

        Returns a structured array view with shape ``(N, a, b)`` where ``N`` is
        the number of particles and ``(a, b)`` depends on the variable type:

        - Scalar: ``(N, 1, 1)``
        - Vector: ``(N, 1, dim)``
        - Tensor: ``(N, dim, dim)``

        Returns
        -------
        NDArray
            Array view that delegates changes back to canonical storage.

        Examples
        --------
        >>> # Scalar field initialization
        >>> material_property.array[:, 0, 0] = 1e21  # Set viscosity

        >>> # Vector field
        >>> velocity.array[:, 0, 0] = vx_values  # x-component
        >>> velocity.array[:, 0, 1] = vy_values  # y-component

        >>> # Reading values
        >>> max_visc = material_property.array[:, 0, 0].max()

        Notes
        -----
        This property is a view of the canonical ``.data`` property with
        automatic shape conversion. All modifications trigger proxy variable
        updates for mesh interpolation.

        See Also
        --------
        data : Flat format ``(-1, components)`` for variable-to-variable transfers.
        sym : Symbolic representation for use in equations.
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

        local_data = self.data[:]
        local_n = local_data.shape[0]
        n_components = self.num_components

        if h5py.h5.get_config().mpi == True and not force_sequential:
            # BUGFIX(#151): the previous parallel path called
            #   h5f.create_dataset("data", data=self.data[:])
            # collectively, but each rank passed its own local-sized array.
            # In parallel HDF5 every rank must specify the *same* dataset
            # shape on a collective create_dataset; passing different shapes
            # leaves HDF5's internal metadata inconsistent so the collective
            # close never synchronises, producing a silent hang.
            #
            # Fix: allgather the per-rank sizes, create the dataset at the
            # global shape, then each rank writes its own slice.
            sizes = comm.allgather(local_n)
            total_n = sum(sizes)
            offset = sum(sizes[: comm.rank])

            with h5py.File(f"{filename[:-3]}.h5", "w", driver="mpio", comm=comm) as h5f:
                if compression == True:
                    dset = h5f.create_dataset(
                        "data",
                        shape=(total_n, n_components),
                        dtype=local_data.dtype,
                        chunks=True,
                        compression=compressionType,
                    )
                else:
                    dset = h5f.create_dataset(
                        "data",
                        shape=(total_n, n_components),
                        dtype=local_data.dtype,
                    )
                if local_n > 0:
                    dset[offset : offset + local_n] = local_data
        else:
            # Sequential fallback: rank 0 creates the file and writes its slab,
            # then each higher rank appends in turn. Indentation here matters —
            # the barrier/loop must be outside the rank-0 branch so all ranks
            # synchronise (the previous version nested them inside, leaving
            # higher ranks' data unwritten and rank 0 deadlocked at the
            # barrier with no peers).
            if comm.rank == 0:
                with h5py.File(f"{filename[:-3]}.h5", "w") as h5f:
                    if compression == True:
                        h5f.create_dataset(
                            "data",
                            data=local_data,
                            chunks=True,
                            maxshape=(None, n_components),
                            compression=compressionType,
                        )
                    else:
                        h5f.create_dataset(
                            "data",
                            data=local_data,
                            chunks=True,
                            maxshape=(None, n_components),
                        )

            comm.barrier()
            for proc in range(1, comm.size):
                if comm.rank == proc and local_n > 0:
                    with h5py.File(f"{filename[:-3]}.h5", "a") as h5f:
                        incoming_size = h5f["data"].shape[0]
                        h5f["data"].resize((incoming_size + local_n), axis=0)
                        h5f["data"][incoming_size:] = local_data
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
        """Write this variable's proxy mesh variable to an HDF5 file.

        The proxy is the RBF-interpolated mesh-variable image of the
        particle data (built when the variable was created with
        ``proxy_degree > 0``); this writes that mesh field via
        ``MeshVariable.write``, which is often the most convenient
        checkpoint/visualisation form of particle data.

        Parameters
        ----------
        filename : str
            Output file path, passed directly to
            ``MeshVariable.write`` (conventionally ``*.h5``).

        Notes
        -----
        If the variable has no proxy (``proxy_degree=0`` /
        ``_proxy=False``), nothing is written; a message is printed and
        the call returns.
        """
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
        """Restore this variable's values from a saved timestep.

        Reads the checkpoint files written by ``Swarm.write_timestep``
        and maps the saved (coordinate, value) pairs onto the *current*
        particles by nearest-neighbour interpolation — the live swarm
        need not have the same particle count or positions as the saved
        one.

        Filename convention (matching ``Swarm.write_timestep(
        data_filename, swarmID, swarmVars=[...], outputPath=...,
        index=...)``):

        * swarm coordinates:
          ``{outputPath}/{data_filename}.{swarmID}.{index:05}.h5``
        * this variable's data:
          ``{outputPath}/{data_filename}.{swarmID}.{data_name}.{index:05}.h5``

        Both files must exist (``RuntimeError`` otherwise).

        Parameters
        ----------
        data_filename : str
            Base name used when the checkpoint was written. (Note: the
            corresponding parameter on ``Swarm.read_timestep`` is
            spelled ``base_filename`` — the two signatures predate a
            common convention.)
        swarmID : str
            Swarm identifier used when the checkpoint was written
            (``swarm_id`` on ``Swarm.read_timestep``).
        data_name : str
            The saved variable's ``name`` (``Swarm.write_timestep``
            embeds each variable's ``.name`` in its data filename).
        index : int
            Timestep index (zero-padded to five digits in the
            filename).
        outputPath : str, optional
            Directory holding the checkpoint files (default: current
            directory).

        Notes
        -----
        MPI-collective. Rank 0 reads the saved data in one shot; the
        saved points are then routed to their owning ranks with the
        same migration rule the live swarm uses, so the rank-local
        nearest-neighbour lookup always sees the correct neighbours.
        """
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

        # Memory-bounded parallel read. Rank 0 reads the saved file in one
        # shot (this is not chunked — the file is the smallest copy of the
        # data and we hold it for the duration of one ``add_particles`` call;
        # streaming hyperslabs is a follow-up if rank-0 RAM becomes the
        # binding constraint). Saved (coord, value) pairs are then pushed
        # into a transient routing swarm and migrated *with the same rule
        # the live swarm uses* — ``points_in_domain`` plus the centroid
        # fallback inside ``Swarm.migrate``. That co-locates a saved point
        # and the corresponding live particle on the same rank, so the
        # rank-local KDTree always sees the right neighbour.

        n_components = self.num_components
        dim = self.swarm.mesh.dim

        if uw.mpi.rank == 0:
            with (
                h5py.File(f"{filename}", "r") as h5f_data,
                h5py.File(f"{swarmFilename}", "r") as h5f_swarm,
            ):
                file_dtype = h5f_data["data"].dtype
                if self.dtype != file_dtype:
                    warnings.warn(
                        f"{os.path.basename(filename)} dtype ({file_dtype}) "
                        f"does not match {self.name} swarm variable dtype "
                        f"({self.dtype}) which may result in a loss of data.",
                        stacklevel=2,
                    )
                X_chunk = h5f_swarm["coordinates"][()].reshape(-1, dim)
                D_chunk = h5f_data["data"][()].reshape(-1, n_components)
        else:
            X_chunk = np.empty((0, dim), dtype=np.float64)
            D_chunk = np.empty((0, n_components), dtype=np.float64)

        tmp_swarm = uw.swarm.Swarm(self.swarm.mesh)
        saved = SwarmVariable(
            "_read_timestep_saved",
            tmp_swarm,
            vtype=uw.VarType.MATRIX,
            size=(1, n_components),
            dtype=float,
            _proxy=False,
            varsymbol=r"\cal{S}",
        )

        size_before = max(tmp_swarm.dm.getLocalSize(), 0)
        tmp_swarm.add_particles_with_global_coordinates(X_chunk, migrate=False)
        tmp_swarm._invalidate_canonical_data()
        saved.array[size_before:, 0, :] = D_chunk[:, :]

        # Use the same migration rule as the live swarm so saved points and
        # live particles at the same coordinate land on the same rank.
        # ``delete_lost_points=False`` keeps points that ``points_in_domain``
        # rejects on every rank; they fall through to the centroid fallback
        # inside ``Swarm.migrate`` and end up somewhere deterministic.
        tmp_swarm.migrate(remove_sent_points=True, delete_lost_points=False)

        landed_X = tmp_swarm._particle_coordinates.array[...].reshape(-1, dim)
        landed_D = saved.array[:, 0, :]

        if landed_X.shape[0] == 0:
            warnings.warn(
                f"[rank {uw.mpi.rank}] read_timestep: no saved swarm points "
                f"landed locally; '{self.name}' on this rank will not be updated",
                stacklevel=2,
            )
        else:
            kdt = uw.kdtree.KDTree(landed_X)
            self.array[:, 0, :] = kdt.rbf_interpolator_local(
                self.swarm._particle_coordinates.data, landed_D, nnn=1
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

    def _update_proxy_if_stale(self):
        """
        Refresh the level-set proxy variables if they are marked stale.

        Overrides the base implementation (which requires ``self._meshVar``;
        an IndexSwarmVariable keeps its proxies in ``_meshLevelSetVars``
        instead). Used by the lazy ``.sym`` accessor and by the solve-entry
        refresh (``Swarm._sync_before_assembly``).
        """
        if not self._proxy_stale or self._updating_proxy:
            return

        # Fossil-variable contract — see the base implementation: a
        # variable that outlives its (weakly-referenced) parent swarm keeps
        # the last level-set projection and warns instead of raising.
        if self._swarm_ref is None or self._swarm_ref() is None:
            import warnings

            warnings.warn(
                f"IndexSwarmVariable '{self.clean_name}': the parent swarm "
                "no longer exists; the level-set proxies retain their last "
                "projection and cannot be refreshed.",
                stacklevel=2,
            )
            self._proxy_stale = False  # nothing can ever refresh it again
            return

        try:
            self._updating_proxy = True
            self._update_proxy_variables()
            self._proxy_stale = False
        finally:
            self._updating_proxy = False

    # This is the sympy vector interface - it's meaningless if these are not spatial arrays
    @property
    def sym(self):
        """
        Lazy evaluation of symbolic mask array.

        Only updates proxy variables when they're actually needed (when sym is accessed)
        and only if the proxy variables are marked as stale due to data changes.
        This avoids expensive RBF interpolation during data assignment operations.
        """
        self._update_proxy_if_stale()
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

        symo = sympy.S.Zero
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
        # Starved-rank guard (SWARM-07): with <= 1 local particles the
        # nearest-neighbour machinery cannot run — KDTree construction on an
        # empty coordinate array raises IndexError, aborting/hanging the
        # collective proxy update — and there is nothing to project anyway.
        # Leave this rank's level-set nodal values unchanged and warn. Every
        # rank still enters the (collective) access contexts below so ranks
        # holding particles can proceed.
        starved = self.swarm.local_size <= 1
        if starved and self.swarm._population_generation > 0:
            # (silent for a never-populated swarm — creation-time .sym
            # touches are expected; see the equivalent guard in
            # _rbf_to_meshVar)
            import warnings

            warnings.warn(
                f"IndexSwarmVariable proxy update: rank {uw.mpi.rank} holds "
                f"{max(self.swarm.local_size, 0)} particles; level-set "
                f"variables for '{self.clean_name}' left unchanged on this "
                "rank.",
                stacklevel=2,
            )

        if self.update_type == 0:
            if not starved:
                # Use non-dimensional coordinates for internal level set KDTree
                kd = self._meshLevelSetVars[0]._get_kdtree()

                n_distance, n_indices = kd.query(
                    self.swarm._particle_coordinates.data, k=self.nnn, sqr_dists=False
                )
                kd_swarm = self.swarm._get_kdtree()
                # n, d, b = kd_swarm.find_closest_point(self._meshLevelSetVars[0].coords)
                d, n = kd_swarm.query(self._meshLevelSetVars[0].coords, k=1, sqr_dists=False)

                # Which (particle, node) pairs are valid:
                # - node is at same distance as the nearest node
                # - node is within radius_s
                is_nearest = np.isclose(n_distance, n_distance[:, [0]])
                within_radius = n_distance < self.radius_s
                valid = is_nearest & within_radius

                # IDW weights (only for valid pairs; others zeroed)
                weights = 1.0 / (n_distance + 1e-16)
                weights[~valid] = 0.0

                # Total weight per node — material independent, compute once
                n_mesh_nodes = self._meshLevelSetVars[0].data.shape[0]
                w = np.zeros(n_mesh_nodes)
                np.add.at(w, n_indices, weights)

            for ii in range(self.indices):
                meshVar = self._meshLevelSetVars[ii]

                # MeshVariable reads/writes perform collective ghost
                # synchronisation, so every rank must execute exactly the
                # same read-then-write sequence per level set: compute into
                # a LOCAL buffer first, then issue a single symmetric write.
                # (The previous in-context formulation issued a
                # data-dependent number of writes per rank — the deferred
                # sync at access-exit then ran mismatched collectives.)
                final_values = np.array(meshVar.data[:, 0], copy=True)

                if not starved:
                    # Material presence mask: (n_particles, 1) for broadcasting
                    mat_mask = (self.data.flatten() == ii).astype(weights.dtype)[:, None]

                    # Weighted sum per node
                    node_values = np.zeros(final_values.shape[0])
                    np.add.at(node_values, n_indices, weights * mat_mask)

                    # Normalize
                    node_values[w > 0] /= w[w > 0]
                    final_values = node_values

                    # if there is no material found,
                    # impose a near-neighbour hunt for a valid material and set that one
                    ind_w0 = np.where(w == 0.0)[0]
                    if len(ind_w0) > 0:
                        ind_ = np.where(self.data[n[ind_w0]] == ii)[0]
                        if len(ind_) > 0:
                            final_values[ind_w0[ind_]] = 1.0

                # single symmetric write (starved ranks write back their
                # current values, i.e. the proxy is left unchanged there)
                meshVar.data[:, 0] = final_values
        elif self.update_type == 1:
            # NOTE: this branch performs data-dependent MeshVariable writes
            # outside any access context, which is not parallel-safe
            # independently of the starved-rank issue (pre-existing).
            # The guard here only prevents the empty-rank KDTree crash.
            if starved:
                return
            kd = uw.kdtree.KDTree(self.swarm._particle_coordinates.data)
            n_distance, n_indices = kd.query(
                self._meshLevelSetVars[0].coords, k=self.nnn, sqr_dists=False
            )

            # IDW weights and validity mask for all (node, particle) pairs
            valid = n_distance < self.radius_s
            a = 1.0 / (n_distance + 1e-16)
            a[~valid] = 0.0

            # Total weight per node (material-independent)
            w = a.sum(axis=1)

            # Handle boundary nodes: restrict to nnn_bc particles
            if hasattr(self, 'ind_bc') and self.ind_bc is not None:
                bc_idx = np.array(list(self.ind_bc))
                bc_idx = bc_idx[bc_idx < a.shape[0]]
                if len(bc_idx) > 0:
                    valid_bc = n_distance[bc_idx, :self.nnn_bc] < self.radius_s
                    a_bc = 1.0 / (n_distance[bc_idx, :self.nnn_bc] + 1e-16)
                    a_bc[~valid_bc] = 0.0
                    w_bc = a_bc.sum(axis=1)

                    a[bc_idx] = 0.0
                    a[bc_idx, :self.nnn_bc] = a_bc
                    w[bc_idx] = w_bc

            for ii in range(self.indices):
                meshVar = self._meshLevelSetVars[ii]

                # Material presence at each (node, particle) pair
                # self.data has shape (n_particles, 1); flatten to (n_particles,)
                # so fancy indexing yields (n_nodes, nnn) — matching `a`.
                mat_present = (self.data.flatten()[n_indices] == ii).astype(a.dtype)

                # Weighted sum per node, then normalize
                node_values = (a * mat_present).sum(axis=1)
                node_values[w > 0] /= w[w > 0]
                meshVar.data[:, 0] = node_values[...]

                # if there is no material found,
                # impose a near-neighbour hunt for a valid material and set that one
                ind_w0 = np.where(w == 0.0)[0]
                if len(ind_w0) > 0:
                    ind_ = np.where(self.data[n_indices[ind_w0]] == ii)[0]
                    if len(ind_) > 0:
                        meshVar.data[ind_w0[ind_]] = 1.0
        return


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
        Particle recycling (streak swarms) is NOT implemented: values > 1
        raise ``NotImplementedError``. The parameter is retained so that
        existing calls passing the default (0 or 1, meaning no recycling)
        keep working.
    verbose : bool, optional
        Enable verbose output for debugging and monitoring particle operations.
        Default is False.

    Examples
    --------
    Create a basic swarm and populate with particles:

    >>> import underworld3 as uw
    >>> mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(1,1))
    >>> swarm = uw.swarm.Swarm(mesh=mesh)
    >>> swarm.populate(fill_param=2)

    Add custom particle data:

    >>> temperature = swarm.add_variable("temperature", 1)
    >>> velocity = swarm.add_variable("velocity", mesh.dim)

    Particle migration after coordinate updates:

    Note: writing particle coordinates (via ``swarm._particle_coordinates.data``
    or the ``coords`` setter) marks the swarm for migration; the collective
    ``migrate()`` itself is DEFERRED to the next collective point — a
    ``migration_control()`` context exit, an explicit ``swarm.migrate()``,
    or solve entry — never run per-write (uneven writes would deadlock).

    Note: `swarm.populate` uses a the mesh point locations for discontinuous interpolants to
    determine the particle locations.

    """

    instances = 0

    @timing.routine_timer_decorator
    def __init__(self, mesh, recycle_rate=0, verbose=False, clip_to_mesh=True):
        # Particle recycling (streak swarms) was excised in 2026-07: the
        # machinery had been broken (NameError) and untested for some time
        # (audit finding SWARM-08). Refuse rather than crash later.
        if recycle_rate > 1:
            raise NotImplementedError(
                "Particle recycling / streak swarms (recycle_rate > 1) are not "
                "implemented. Construct the Swarm without recycle_rate and manage "
                "particle re-seeding explicitly (e.g. add_particles_with_coordinates)."
            )

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

        # Informational counter incremented at every particle-population
        # mutation site (populate, migrate, add_particles_*, advection
        # remesh). NOT a snapshot-restore invalidation gate — restore
        # rebuilds the local population from the snapshot regardless of
        # what happened in between. Useful for logging, debugging, and
        # any cache that wants to know "did the population change?"
        # See docs/developer/design/in_memory_checkpoint_design.md.
        self._population_generation = 0

        # Register this swarm with the mesh for coordinate change notifications
        mesh.register_swarm(self)

        self.dm = PETSc.DMSwarm().create()
        # Use cdim (embedding dim), not dim (topological). On manifold
        # meshes (e.g. SphericalManifold: dim=2, cdim=3) coordinates are
        # cdim-vectors and the DMSwarmPIC_coor field is registered at
        # cdim blocksize. Setting the swarm's intrinsic dim here keeps
        # PETSc's bookkeeping consistent with the coord-field shape.
        # On volume meshes dim == cdim so this is a no-op.
        self.dm.setDimension(self.cdim)
        self.dm.setType(SwarmType.DMSWARM_BASIC.value)
        self._data = None

        # Add data structure to hold point location information in
        # an array with a callback that resets the relevant parts of the
        # swarm variable stack when the data structure is modified.

        self._coords = None

        ####

        # Retained attribute: always 0 or 1 (no recycling) — see the
        # NotImplementedError guard above.
        self.recycle_rate = recycle_rate

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

        self._X0_uninitialised = True
        self._index = None
        self._nnmapdict = {}
        self._migration_disabled = False

        # Deterministic (SPMD-consistent) creation index — used to order
        # collective per-swarm operations identically on every rank.
        self._instance_number = Swarm.instances

        # Names of variables whose canonical writes were made while
        # _migration_disabled was set: the PETSc pack is deferred (not
        # discarded) and flushed by _flush_pending_petsc_sync() at context
        # exit (SWARM-04).
        self._pending_petsc_sync = set()

        # Set when particle coordinates are written through the modern
        # interface; the actual (collective) migrate() is deferred to the
        # next collective point — migration-context exit or solve entry —
        # never run per-write, which would deadlock when ranks write
        # unevenly (SWARM-03).
        self._needs_migration = False

        # SPMD-consistent guard: while True, _sync_before_assembly() must NOT
        # run the deferred migration. Set by advection() around its substep
        # loop — its velocity evaluations pass through Mesh.update_lvec(), and
        # a migrate() there would reorder particle rows between the coordinate
        # array and the velocity array captured from it. advection() runs its
        # own migrate() at the end.
        self._deferred_migration_suspended = False

        super().__init__()

        # Register with the same model already captured in self._model_ref
        # above (not a fresh ``get_default_model()`` call) so that
        # ``__del__`` deregisters from the same registry it registered with,
        # even if the default model is swapped mid-session.
        model._register_swarm(self)

    def __del__(self):
        """Cleanup swarm: unregister from mesh and model, destroy the DM.

        Three steps: drop the mesh-side weak-set entry, drop the model-side
        registry entry (which also forgets any SwarmVariables that belonged
        to this swarm), and call ``self.dm.destroy()`` to release the PETSc
        DMSwarm and its registered fields.

        Historically the model registry kept a strong reference to every
        swarm and ``__del__`` did not call ``dm.destroy()`` — both leaks
        accumulated quickly inside time-stepping loops that build transient
        swarms (global expression evaluation, checkpoint reads, mesh
        adaptation transfers). The registry is now a ``WeakValueDictionary``
        and ``__del__`` cleans up its own resources.
        """
        try:
            if hasattr(self, "mesh") and self.mesh is not None:
                self.mesh.unregister_swarm(self)
        except (AttributeError, ReferenceError, RuntimeError):
            # Mesh/Model may have already been garbage collected, which is fine
            pass
        try:
            model = self._model_ref() if hasattr(self, "_model_ref") else None
            if model is not None:
                model._unregister_swarm(self)
        except (AttributeError, ReferenceError, RuntimeError):
            pass
        try:
            if hasattr(self, "dm") and self.dm is not None:
                self.dm.destroy()
        except Exception:
            # DM may already have been destroyed (e.g. by Mesh.adapt) or
            # PETSc may be finalising. Either way, nothing more we can do.
            pass

    def _invalidate_canonical_data(self):
        """Drop cached array views on the coordinates and every registered variable.

        Required after any operation that bypasses :meth:`Swarm.migrate` but
        still mutates particle layout — notably the bare ``self.dm.migrate(...)``
        used by round-trip evaluation patterns. Calling :meth:`Swarm.migrate`
        already does this internally; call this directly only when you used the
        underlying DMSwarm migrate.
        """
        if hasattr(self, "_particle_coordinates") and self._particle_coordinates is not None:
            self._particle_coordinates._canonical_data = None
        # ``self._vars`` may be a WeakValueDictionary whose values disappear
        # asynchronously during GC; iterate a snapshot.
        for var in list(self._vars.values()):
            if hasattr(var, "_canonical_data"):
                var._canonical_data = None
            # Mark the proxy stale (lazy) so it re-interpolates on next access.
            # NB: set the flag directly rather than calling var._update() —
            # IndexSwarmVariable._update() is EAGER (_update_proxy_variables),
            # so calling it here re-interpolates the proxy on every invalidation
            # (i.e. every swarm.access write), an O(100 MiB) leak over a time loop
            # (tests/test_0006_memory_leak.py). The proxy still refreshes lazily
            # via .sym / _update_proxy_if_stale().
            if hasattr(var, "_proxy_stale"):
                var._proxy_stale = True

        # Invalidate cached spatial index
        self._kdtree = None

    def _flush_pending_petsc_sync(self):
        """Pack canonical arrays written while migration was suppressed.

        Writes made inside ``migration_control()`` / ``migration_disabled()``
        land in each variable's canonical array but their PETSc pack is
        deferred (see the sync callbacks). This flushes them into the DMSwarm
        fields. Called on migration-context exit and defensively at
        :meth:`migrate` entry; a no-op when nothing is pending (SWARM-04).
        """
        pending, self._pending_petsc_sync = self._pending_petsc_sync, set()
        for name in pending:
            var = self._vars.get(name, None)
            if var is None:
                continue
            canonical = getattr(var, "_canonical_data", None)
            if canonical is None:
                # cache was invalidated after the write; PETSc already holds
                # the authoritative data — nothing left to flush.
                continue
            arr = np.asarray(canonical).reshape(-1, var.num_components)
            if arr.shape[0] != max(self.dm.getLocalSize(), 0):
                raise RuntimeError(
                    f"Cannot flush deferred writes for swarm variable "
                    f"'{name}': cached array has {arr.shape[0]} rows but the "
                    f"DMSwarm holds {self.dm.getLocalSize()} particles. The "
                    "particle layout changed while migration was disabled."
                )
            var.pack_raw_data_to_petsc(arr)
            if self._coord_var is var:
                self._needs_migration = True
            if hasattr(var, "_on_data_changed"):
                var._on_data_changed()

    def _sync_before_assembly(self):
        """Collective: bring PETSc-facing swarm state up to date for a solve.

        Called from ``Mesh.update_lvec()`` — the common entry point where
        assembly pulls variable data — this performs, in order:

        1. any DEFERRED particle migration (coordinates written through the
           modern interface mark ``_needs_migration`` instead of migrating
           per-write, which would deadlock under uneven writes — SWARM-03);
        2. an eager refresh of stale proxy mesh variables, which are
           otherwise refreshed only via the lazy ``.sym`` accessor — solvers
           read the proxy DM directly and previously consumed stale data
           (issue #215 Bug 3 / issue #289).

        Rank-local flags are combined with a global MAX reduction so every
        rank takes the same sequence of collective actions even when writes
        were rank-uneven. Repeated calls are no-ops (flag-guarded).
        """
        if self._migration_disabled:
            # A migration-suppressed context is active (SPMD-consistent by
            # construction); leave everything for its exit to handle.
            return

        # A swarm with no particles anywhere has nothing to migrate or
        # project — leave proxies stale (they refresh after population)
        # rather than issue spurious starved-rank warnings pre-populate.
        global_count = max(self.local_size, 0)
        if uw.mpi.size > 1:
            global_count = uw.mpi.comm.allreduce(global_count, op=uw.MPI.MAX)
        if global_count == 0:
            return

        if not self._deferred_migration_suspended:
            needs_migration = bool(self._needs_migration)
            if uw.mpi.size > 1:
                needs_migration = (
                    uw.mpi.comm.allreduce(int(needs_migration), op=uw.MPI.MAX) > 0
                )
            if needs_migration:
                self.migrate()

        # Deterministic variable order: the refresh performs collective
        # mesh-variable writes, so all ranks must visit variables in the
        # same sequence.
        for name in sorted(self._vars.keys()):
            var = self._vars.get(name)
            if var is None:
                continue
            has_proxy = (
                getattr(var, "_meshVar", None) is not None
                or isinstance(var, IndexSwarmVariable)
            )
            if not has_proxy:
                continue
            stale = bool(getattr(var, "_proxy_stale", False))
            if uw.mpi.size > 1:
                stale = uw.mpi.comm.allreduce(int(stale), op=uw.MPI.MAX) > 0
            if stale:
                var._proxy_stale = True  # align ranks before the collective refresh
                var._update_proxy_if_stale()

    def _get_kdtree(self):
        """
        Return a cached KDTree for the swarm particle coordinates.
        Invalidated automatically whenever particles migrate or positions change.
        """
        # Note: self.data returns unit-aware array if units are active,
        # but kdtree construction expects non-dimensional values.
        # Use _particle_coordinates.data directly.
        if not hasattr(self, "_kdtree") or self._kdtree is None:
            self._kdtree = uw.kdtree.KDTree(self._particle_coordinates.data)

        return self._kdtree

    def _route_by_nearest_centroid(self):
        """Migrate every particle to the rank whose domain-centroid is closest.

        This is a deterministic alternative to :meth:`migrate`: the destination
        is a pure function of the coordinate, computed identically on every
        rank. Two swarms migrated this way are guaranteed to place equal
        coordinates on the same rank — which the standard ``migrate`` does not
        guarantee at partition boundaries (vertex DOFs sitting on a shared face
        can return ``True`` from ``points_in_domain`` on multiple ranks).

        Used by checkpoint readers and any consumer that needs source data and
        query coordinates to converge on the same rank without relying on
        PETSc's DOF distribution.
        """
        centroids = self.mesh._get_domain_centroids()
        centroid_kdt = uw.kdtree.KDTree(centroids)

        coords = self.dm.getField("DMSwarmPIC_coor").reshape(-1, self.cdim).copy()
        self.dm.restoreField("DMSwarmPIC_coor")

        if coords.shape[0] > 0:
            _, owner_rank = centroid_kdt.query(coords, k=1, sqr_dists=False)
            rank_arr = self.dm.getField("DMSwarm_rank")
            # ``DMSwarm_rank`` shape varies by PETSc version: 1-D ``(N,)`` on
            # 3.21, 2-D ``(N, 1)`` on 3.25+. ``reshape(-1)`` flattens either
            # form into a writable view into the same buffer.
            rank_arr.reshape(-1)[:] = owner_rank.astype(rank_arr.dtype, copy=False)
            self.dm.restoreField("DMSwarm_rank")

        self.dm.migrate(remove_sent_points=True)
        uw.mpi.barrier()
        self._invalidate_canonical_data()

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
        model_coords = (self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.cdim))).copy()
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
        coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.cdim))
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

        # Convert to physical units if reference quantities are set
        import underworld3 as uw
        model = uw.get_default_model()

        # If no reference quantities, return raw coordinates (model units = physical units)
        if not model.has_units_active():
            return model_coords

        # Try to convert to physical units; fall back to raw coordinates if
        # length scale is not defined (e.g., only temperature_diff was set)
        try:
            return uw.scaling.dimensionalise(model_coords, uw.units.meter)
        except ValueError:
            # Length scale not available - return raw model-unit coordinates
            return model_coords

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
        Migration is NOT called when exiting the context. Writes made inside
        the context are packed to PETSc at exit (only the migration is
        suppressed — data is never discarded).

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

        Examples
        --------
        Defer migration until end (default)::

            with swarm.migration_control():
                swarm.points[mask1] += delta1
                swarm.points[mask2] *= scale
                # Migration happens HERE on exit

        Completely disable migration::

            with swarm.migration_control(disable=True):
                # Operations where migration should never happen
                # No migration on exit

        In both modes, variable/coordinate writes made inside the context are
        flushed to the underlying DMSwarm at exit — only the migration itself
        is deferred (default) or skipped (``disable=True``).
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

                # Flush writes deferred while the flag was set — suppressing
                # migration must not discard data (SWARM-04). Skipped only
                # when an enclosing context still holds the flag (it flushes
                # on its own exit).
                if not self.swarm._migration_disabled:
                    self.swarm._flush_pending_petsc_sync()

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

        coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.cdim))
        ranks = self.dm.getField("DMSwarm_rank")
        coords[...] = newp_coords[...]
        ranks[...] = uw.mpi.rank
        self.dm.restoreField("DMSwarmPIC_coor")
        self.dm.restoreField("DMSwarm_rank")

        # Invalidate cached data — the swarm was just given its particles.
        # Any canonical `.data` array created before populate() (legitimate:
        # variables must be created first) is sized for the empty swarm and
        # would otherwise hide every particle from reads and corrupt writes
        # (SWARM-17, same stale-cache class as #216).
        self._invalidate_canonical_data()

        # Informational: particle population just changed.
        self._population_generation += 1

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

        # Deferred writes must reach the DMSwarm before we read coordinates
        # from it below (no-op unless a migration-suppressed context left
        # pending packs behind, SWARM-04).
        self._flush_pending_petsc_sync()

        # Informational: migration may move or drop particles. Bump
        # unconditionally; restore is not gated on this counter so a
        # conservative no-op bump is harmless.
        self._population_generation += 1

        from time import time

        if delete_lost_points is None:
            delete_lost_points = self.clip_to_mesh

        mesh_domain_kdtree = self.mesh._get_domain_kdtree()

        # This will only worry about particles that are not already claimed !
        #

        swarm_coord_array = (self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.cdim))).copy()
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
            # No particle needs to change rank, but we were still called
            # because coordinates and/or the population may have changed
            # (in-place coordinate writes, addNPoints, serial advection).
            # The cached canonical arrays, the particle kd-tree, and the
            # proxy variables are stale regardless of whether anything
            # moved between ranks. Skipping this invalidation froze proxy
            # mesh variables after serial advection (issue #289) and left
            # wrong-sized `.data` caches after particle addition
            # (SWARM-01/SWARM-02, 2026-07 audit).
            self._invalidate_canonical_data()
            self._needs_migration = False
            # any explicit/terminal migrate ends an advection suspension
            self._deferred_migration_suspended = False
            return

        # Migrate particles between processes (if there are more than one of them)

        if uw.mpi.size > 1:
            for it in range(0, min(max_its, uw.mpi.size)):

                # Send unclaimed points to next processor in line

                swarm_rank_array = self.dm.getField("DMSwarm_rank")
                swarm_coord_array = self.dm.getField("DMSwarmPIC_coor").reshape(-1, self.cdim)

                if not_my_points.shape[0] > 0:
                    dist, rank = mesh_domain_kdtree.query(
                        swarm_coord_array[not_my_points], k=it + 1, sqr_dists=False
                    )

                    swarm_rank_array.reshape(-1)[not_my_points] = rank.reshape(
                        -1, it + 1
                    )[:, it].flatten()

                self.dm.restoreField("DMSwarm_rank")
                self.dm.restoreField("DMSwarmPIC_coor")

                # Now we send the points (basic migration)
                self.dm.migrate(remove_sent_points=True)
                uw.mpi.barrier()

                swarm_coord_array = self.dm.getField("DMSwarmPIC_coor").reshape(-1, self.cdim)
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

        # Invalidate all cached data after migration.
        # Any particle movement (send, receive, or balanced swap) makes
        # cached arrays stale — both size and values may have changed.
        self._invalidate_canonical_data()
        self._needs_migration = False
        # any explicit/terminal migrate ends an advection suspension
        self._deferred_migration_suspended = False

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
        """Add particles at given coordinates, keeping only locally-owned points.

        Each rank filters the input array and adds only the points that fall
        within its local domain partition. Non-local points are silently
        ignored, so it is safe to pass the same global coordinate array to
        every rank — no duplicates will be created.

        This is the recommended method for user code.  For pre-partitioned
        data where each rank already holds only its own points, this method
        also works correctly.

        Parameters
        ----------
        coordinatesArray : numpy.ndarray
            Coordinates of new particles, shape ``(n, dim)``.

        Returns
        --------
        npoints : int
            Number of points actually added on this rank.
        """

        if not isinstance(coordinatesArray, np.ndarray):
            raise TypeError("'coordinateArray' must be provided as a numpy array")
        if not len(coordinatesArray.shape) == 2:
            raise ValueError("The 'coordinateArray' is expected to be two dimensional.")
        if not coordinatesArray.shape[1] == self.mesh.cdim:
            raise ValueError(
                """The 'coordinateArray' must have shape (n, cdim), where 'n' is the
                              number of particles to add, and 'cdim' is the embedding
                              (coordinate) dimensionality of the supporting mesh ({}).""".format(
                    self.mesh.cdim
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
            coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.cdim))
            ranks = self.dm.getField("DMSwarm_rank")
            coords[swarm_size::, :] = valid_coordinates[:, :]
            ranks[swarm_size::] = uw.mpi.rank
            self.dm.restoreField("DMSwarm_rank")
            self.dm.restoreField("DMSwarmPIC_coor")

        self.dm.migrate(remove_sent_points=True)

        # Invalidate cached data — particle count changed after addNPoints + migrate
        self._invalidate_canonical_data()

        # Informational: addNPoints + direct dm.migrate path doesn't go
        # through Swarm.migrate, so bump explicitly.
        self._population_generation += 1

        return npoints

    @timing.routine_timer_decorator
    def add_particles_with_global_coordinates(
        self,
        globalCoordinatesArray,
        migrate=True,
        delete_lost_points=True,
    ) -> int:
        """Insert the full coordinate array on every rank (low-level primitive).

        Every rank inserts **all** supplied points into its local swarm.
        There is no locality filtering and no deduplication: migration is a
        scatter that routes each inserted particle to the rank owning its
        location, so passing the same array on every rank at ``np`` processes
        yields ``np`` copies of every point.

        Correct usage is one of:

        - ``migrate=False``, where each rank deliberately keeps a full copy
          of the points (the global-evaluation / mesh-transfer pattern), or
        - input that is pre-partitioned across ranks — or supplied on rank 0
          only, with empty ``(0, dim)`` arrays elsewhere — followed by
          migration to route each point to its owner.

        For general use, prefer :meth:`add_particles_with_coordinates`,
        which accepts a rank-identical array and filters non-local points so
        each point is added exactly once.

        Parameters
        ----------
        globalCoordinatesArray : numpy.ndarray
            Coordinates of new particles, shape ``(n, dim)``.
        migrate : bool
            Run PETSc swarm migration after insertion (default True).
        delete_lost_points : bool
            Remove particles that fall outside any rank's domain
            during migration (default True).

        Returns
        --------
        npoints : int
            Number of points added on this rank before migration.
        """

        if not isinstance(globalCoordinatesArray, np.ndarray):
            raise TypeError("'coordinateArray' must be provided as a numpy array")
        if not len(globalCoordinatesArray.shape) == 2:
            raise ValueError("The 'coordinateArray' is expected to be two dimensional.")
        if not globalCoordinatesArray.shape[1] == self.mesh.cdim:
            raise ValueError(
                """The 'coordinateArray' must have shape (n, cdim), where 'n' is the
                                number of particles to add, and 'cdim' is the embedding
                                (coordinate) dimensionality of the supporting mesh ({}).""".format(
                    self.mesh.cdim
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

        # Informational: population changed even if migrate=False is
        # passed (in which case Swarm.migrate's bump wouldn't fire).
        self._population_generation += 1

        # Add new points with provided coords
        # Record the current rank (migration needs to know where we start from !)

        coords = self.dm.getField("DMSwarmPIC_coor").reshape((-1, self.cdim))
        ranks = self.dm.getField("DMSwarm_rank")
        coords[swarm_size::, :] = globalCoordinatesArray[:, :]
        ranks[swarm_size::] = uw.mpi.rank
        self.dm.restoreField("DMSwarm_rank")
        self.dm.restoreField("DMSwarmPIC_coor")

        # Invalidate cached data — the particle count changed via addNPoints
        # (mirrors add_particles_with_coordinates). This must not be left to
        # migrate(): with migrate=False nothing else invalidates, and every
        # cached `.data` array keeps the old particle count (SWARM-01).
        self._invalidate_canonical_data()

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
            # BUGFIX(#151): the previous parallel path called
            #   h5f.create_dataset("coordinates", data=points_data_copy)
            # collectively, but each rank passed its own local-sized array.
            # In parallel HDF5 every rank must specify the *same* dataset
            # shape on a collective create_dataset; passing different shapes
            # leaves HDF5's internal metadata inconsistent so the collective
            # close never synchronises, producing a silent hang.
            #
            # Fix: allgather the per-rank sizes, create the dataset at the
            # global shape, then each rank writes its own slice.
            points_data_copy = self._particle_coordinates.data[:].copy()
            local_n = points_data_copy.shape[0]
            cdim = points_data_copy.shape[1]
            sizes = comm.allgather(local_n)
            total_n = sum(sizes)
            offset = sum(sizes[: comm.rank])

            with h5py.File(f"{filename[:-3]}.h5", "w", driver="mpio", comm=comm) as h5f:
                if compression == True:
                    dset = h5f.create_dataset(
                        "coordinates",
                        shape=(total_n, cdim),
                        dtype=points_data_copy.dtype,
                        chunks=True,
                        compression=compressionType,
                    )
                else:
                    dset = h5f.create_dataset(
                        "coordinates",
                        shape=(total_n, cdim),
                        dtype=points_data_copy.dtype,
                    )
                if local_n > 0:
                    dset[offset : offset + local_n] = points_data_copy

            del points_data_copy

        else:
            # Sequential fallback: rank 0 creates the file and writes its slab,
            # then each higher rank appends in turn.
            #
            # MODEL-UNIT coordinates, exactly like the parallel branch above:
            # the deprecated `self.points` used here previously applied the
            # model length scale, so sequential checkpoints differed from
            # parallel ones by that factor and could not round-trip through
            # read_timestep, which re-inserts raw model-unit coordinates
            # (SWARM-19 / BF-17).
            points_data_copy = self._particle_coordinates.data[:].copy()
            local_n = points_data_copy.shape[0]

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
                if comm.rank == i and local_n > 0:
                    # BUGFIX(#151): the previous version referenced an undefined
                    # ``data_copy`` here; passive swarms with a zero-particle
                    # rank would have raised NameError. Use the local
                    # ``points_data_copy`` we already have.
                    with h5py.File(f"{filename[:-3]}.h5", "a") as h5f:
                        existing_size = h5f["coordinates"].shape[0]
                        h5f["coordinates"].resize((existing_size + local_n), axis=0)
                        h5f["coordinates"][existing_size:] = points_data_copy
                comm.barrier()

            del points_data_copy

        ## Add swarm coordinate unit metadata to the file
        # TODO(BUG): save() returns on non-zero ranks while rank 0 is still
        # appending this metadata; a rank that immediately reopens the file
        # (e.g. read_timestep straight after write_timestep) hits HDF5 file
        # locking (BlockingIOError, errno 35). A trailing barrier would make
        # the file quiescent on return. Found while reproducing issue #324.
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
        """Restore the swarm's particle coordinates from a saved timestep.

        Adds particles at the coordinates saved by
        ``Swarm.write_timestep``; call this on a freshly-built swarm
        *before* restoring variable values with
        ``SwarmVariable.read_timestep``. File read (matching
        ``write_timestep(filename, swarmname, index, outputPath=...)``):

        ``{outputPath}/{base_filename}.{swarm_id}.{index:05}.h5``

        Parameters
        ----------
        base_filename : str
            Base name used when the checkpoint was written
            (``filename`` on ``write_timestep``; note
            ``SwarmVariable.read_timestep`` spells it
            ``data_filename`` — the two signatures predate a common
            convention).
        swarm_id : str
            Swarm identifier used when the checkpoint was written
            (``swarmname`` on ``write_timestep``).
        index : int
            Timestep index (zero-padded to five digits in the
            filename).
        outputPath : str, optional
            Directory holding the checkpoint files (default: current
            directory).
        migrate : bool, optional
            ``True`` (default): every rank reads the saved coordinates
            and keeps only the points it owns — the parallel restore
            path. ``False``: every rank keeps a full copy of the saved
            swarm, including points outside the local (or entire)
            domain — useful for debugging, visualisation, or when the
            mesh has been adapted since the save.
        """
        output_base_name = os.path.join(outputPath, base_filename)
        swarm_file = output_base_name + f".{swarm_id}.{index:05}.h5"

        if migrate:
            # Keep-local restore: every rank reads the saved coordinates
            # (plain read-only h5py) and add_particles_with_coordinates
            # keeps only the points this rank owns — no communication, no
            # rank-0 memory hotspot. The cost is np-fold read amplification,
            # which scalable/striped parallel filesystems absorb, so this is
            # the right default; rank-0-routed reading remains the pattern
            # in SwarmVariable.read_timestep pending its own reconsideration.
            # Reading everywhere and inserting via the *global* method with
            # migration restores one copy of the swarm per rank — migration
            # is a scatter with no deduplication (issue #324).
            with h5py.File(f"{swarm_file}", "r") as h5f:
                coordinates = h5f["coordinates"][:]

            self.add_particles_with_coordinates(coordinates)
        else:
            # No migration: every rank keeps a full copy of the saved swarm.
            # Skipping migration also preserves points that fall outside the
            # mesh, which migration would delete (useful for debugging /
            # visualisation, or when adapting the mesh).
            with h5py.File(f"{swarm_file}", "r") as h5f:
                coordinates = h5f["coordinates"][:]

            self.add_particles_with_global_coordinates(coordinates, migrate=False)

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

    # ----- Unitary snapshot / restore -----
    #
    # See ``src/underworld3/checkpoint/snapshot.py`` and
    # ``docs/developer/design/in_memory_checkpoint_design.md``. Capture
    # records the per-rank particle layout and user-variable arrays.
    # Restore rebuilds the local population from the snapshot rather
    # than refusing on counter mismatch — restore is precisely for the
    # case where particles have moved / migrated / been repopulated.

    def _snapshot_stable_name(self) -> str:
        """Per-process stable name. ``instance_number`` comes from uw_object."""
        return f"swarm_{self.instance_number}"

    def snapshot_payload(self) -> dict:
        """Return a self-contained dict describing this swarm's state.

        Captured: per-rank particle coordinates (from
        ``DMSwarmPIC_coor``) and every user swarm-variable's data
        array. PETSc-internal variables (``DMSwarmPIC_coor``,
        ``DMSwarm_X0``) are excluded — their
        contents either come from the captured coords or are
        regenerated on the next solve.
        """
        coord_field = self.dm.getField("DMSwarmPIC_coor").reshape(
            (-1, self.dim)
        )
        coords = np.asarray(coord_field).copy()
        self.dm.restoreField("DMSwarmPIC_coor")

        var_arrays: dict = {}
        for var in list(self._vars.values()):
            if var.name.startswith("DMSwarm"):
                continue
            var_arrays[var.clean_name] = np.asarray(var.data).copy()

        return {
            "name": self._snapshot_stable_name(),
            "mesh_name": self.mesh.name,
            "population_generation": int(self._population_generation),
            "coords": coords,
            "vars": var_arrays,
        }

    def apply_snapshot_payload(self, payload: dict) -> None:
        """Rebuild this swarm's local particle population from a payload.

        Algorithm:

        1. Drop every current local particle (``dm.removePoint`` from
           the end is O(1) per call, O(N) total).
        2. Add the captured-rank's particles back via the raw PETSc
           primitives — ``addNPoints`` then writing the coord field
           directly. We deliberately bypass
           :meth:`add_particles_with_coordinates` because that method
           filters via ``points_in_domain`` (slow) and triggers
           ``dm.migrate`` (unnecessary — saved coords were already
           local at capture time, and the mesh hasn't changed).
        3. Write captured per-variable data back. The local particle
           count matches the captured count because we just put the
           same particles back in the same order.

        This bumps ``_population_generation`` once (from the addNPoints
        step in restore), which is correct: the population *did* just
        change. Downstream consumers that care can compare against the
        captured value in ``payload['captured_population_generation']``.
        """
        from underworld3.checkpoint.snapshot import SnapshotInvalidatedError

        saved_coords = np.asarray(payload["coords"])

        # Step 1: clear local population. removePoint() removes the last
        # particle, so this is O(N) total.
        while self.dm.getLocalSize() > 0:
            self.dm.removePoint()

        # Step 2: re-add. add raw points, write coords + ranks directly.
        n_saved = int(saved_coords.shape[0])
        if n_saved > 0:
            self.dm.finalizeFieldRegister()
            self.dm.addNPoints(npoints=n_saved)

            coord_field = self.dm.getField("DMSwarmPIC_coor").reshape(
                (-1, self.dim)
            )
            if coord_field.shape != saved_coords.shape:
                self.dm.restoreField("DMSwarmPIC_coor")
                raise SnapshotInvalidatedError(
                    f"swarm {self._snapshot_stable_name()!r}: after "
                    f"addNPoints({n_saved}) the coord field has shape "
                    f"{coord_field.shape}, expected {saved_coords.shape}"
                )
            coord_field[...] = saved_coords
            self.dm.restoreField("DMSwarmPIC_coor")

            rank_field = self.dm.getField("DMSwarm_rank")
            rank_field[...] = uw.mpi.rank
            self.dm.restoreField("DMSwarm_rank")

        # Invalidate canonical-data caches — the underlying arrays
        # have been reallocated by the addNPoints path.
        self._invalidate_canonical_data()

        # The raw PETSc primitives used above (removePoint loop +
        # addNPoints + direct field writes) deliberately bypass
        # Swarm.migrate / add_particles_with_coordinates, so they do
        # not touch _population_generation. Bump it explicitly here
        # for consistency with the other mutation sites — a restore
        # IS a population change, downstream consumers should see it.
        # (Comment rewritten per Copilot review on #195.)
        self._population_generation += 1

        # Step 3: write captured per-variable data. Per Copilot
        # review on #195, also raise on the inverse direction —
        # any user swarm variable on the LIVE swarm that wasn't in
        # the snapshot would retain stale/uninitialised contents
        # after the clear+addNPoints reallocation, which is exactly
        # the silent-incoherence failure we want loud rather than
        # quiet. The contract is symmetric with the mesh-variable
        # restore: same variable set on both sides.
        current_user_vars = {
            var.clean_name: var
            for var in self._vars.values()
            if not var.name.startswith("DMSwarm")
        }
        captured_names = set(payload["vars"].keys())
        live_names = set(current_user_vars.keys())
        extras = live_names - captured_names
        if extras:
            raise SnapshotInvalidatedError(
                f"swarm {self._snapshot_stable_name()!r}: variables "
                f"{sorted(extras)!r} exist on the live swarm but were "
                f"not in the snapshot. Restore would leave them with "
                f"incoherent data after the population rebuild — add "
                f"them before the snapshot was taken, or remove them "
                f"before restoring."
            )

        for var_clean_name, saved in payload["vars"].items():
            var = current_user_vars.get(var_clean_name)
            if var is None:
                raise SnapshotInvalidatedError(
                    f"swarm {self._snapshot_stable_name()!r}: variable "
                    f"{var_clean_name!r} from snapshot is not present"
                )
            current = np.asarray(var.data)
            if current.shape != saved.shape:
                raise SnapshotInvalidatedError(
                    f"swarm {self._snapshot_stable_name()!r}: variable "
                    f"{var_clean_name!r} data shape mismatch — current "
                    f"{current.shape} vs snapshot {saved.shape}"
                )
            # Write THROUGH the canonical array (not the detached
            # np.asarray view above) so the PETSc pack callback fires.
            # Writing into the view mutated only the cached copy: the
            # DMSwarm field kept its post-realloc garbage, and the first
            # cache invalidation after restore (e.g. migrate() at the end
            # of advection) silently replaced the restored values with
            # that garbage (exposed by the SWARM-01 invalidation fix).
            var.data[...] = saved

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
        kd = self._get_kdtree()

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
            self._nnmapdict[digest] = kd.query(meshvar_coords, k=1, sqr_dists=False)[1]
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
        r"""Advect the particle swarm through one timestep of a velocity field.

        Particle positions are updated with an explicit Runge-Kutta step
        of :math:`\dot{\mathbf{x}} = \mathbf{V}(\mathbf{x})`. Velocity
        is sampled with a *global* evaluation (off-rank sample points
        are resolved collectively), so the call is MPI-collective: every
        rank must make it, including ranks holding no particles. On
        completion the swarm is migrated so each particle lands on the
        rank owning its new location; particles that leave the domain
        are removed (unless ``restore_points_to_domain_func`` or the
        mesh's own ``return_coords_to_bounds`` returns them).

        Parameters
        ----------
        V_fn : vector UW function / sympy expression
            Velocity field, evaluated at particle locations. Any
            expression accepted by :func:`uw.function.evaluate`,
            e.g. ``stokes.u.sym``.
        delta_t : float or UWQuantity
            Timestep. Non-dimensionalised internally
            (``uw.scaling.non_dimensionalise``), so a dimensional time
            (e.g. ``uw.quantity(1000, "year")``) is valid when scaling
            is active; a plain float is taken to be in model units,
            consistent with the model-unit velocity.
        order : int, optional
            Runge-Kutta order: ``2`` (default) is the midpoint scheme;
            any other value falls back to first-order forward Euler.
        corrector : bool, optional
            Historical predictor-corrector option; currently inert (the
            corrector block is disabled). Retained for call
            compatibility. Default ``False``.
        restore_points_to_domain_func : callable, optional
            Maps an ``(n, dim)`` coordinate array back into the domain
            (periodic wrap, boundary projection, ...). Applied to the
            updated positions each substep, in addition to the mesh's
            own ``return_coords_to_bounds``.
        evalf : bool, optional
            Historical flag selecting RBF (``evalf``) velocity
            sampling; currently inert — velocity is always sampled with
            ``uw.function.global_evaluate``. Default ``False``.
        step_limit : bool, optional
            If ``True``, split ``delta_t`` into substeps no larger than
            :meth:`estimate_dt` (roughly one element crossing per
            substep). Default ``False`` here; note
            :meth:`NodalPointSwarm.advection` defaults it to ``True``.
        """
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

        # Suspend the deferred (solve-entry) migration for the duration of
        # the substep loop: the velocity evaluations below pass through
        # Mesh.update_lvec(), and a migrate() firing there would reorder
        # particle rows between the coordinate array and the velocity array
        # captured from it. advection() performs its own migrate() at the end.
        self._deferred_migration_suspended = True

        # Wrap this whole thing in sub-stepping loop
        for step in range(0, substeps):

            X0.array[:, 0, :] = self._particle_coordinates.data[...]

            # Mid point algorithm (2nd order)

            if order == 2:
                print(f"Advection (2nd): {self.local_size} - swarm points", flush=True)

                # Use internal model-unit coordinates directly (no conversion needed)
                v_at_Vpts = np.zeros_like(self._particle_coordinates.data[...])

                # First evaluate the velocity at the launch points. This must
                # be a GLOBAL evaluation: no migration happens inside the
                # substep loop (deferred migration is suspended above, so
                # arrays keep a stable row order), which means from substep 2
                # onward a particle can sit outside this rank's domain — a
                # rank-local evaluation silently extrapolates wrong values
                # for it (SWARM-16 / BF-16).

                v_at_Vpts[...] = uw.function.global_evaluate(
                    V_fn_matrix, self._particle_coordinates.data
                )[:, 0, :]

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
        self._deferred_migration_suspended = False

        # Re-route particles to their owning ranks and remove any that
        # have genuinely left the domain. Use the default max_its so that
        # boundary particles whose owner is the 2nd/3rd-closest centroid
        # get reclaimed via the kdtree retry — max_its=1 here was an
        # accidental regression that deleted boundary particles (issue #175,
        # reported by @bknight1).
        self.migrate(
            delete_lost_points=True,
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

        # Ensure vel is a plain numpy array in flat (n_particles, dim) form.
        # evaluate() returns matrix-shaped (n, 1, dim) arrays; indexing that
        # shape as vel[:, 1] hit the size-1 axis and the swallowed IndexError
        # made estimate_dt() return None for every non-trivial velocity —
        # silently disabling advection's step_limit substepping (BF-16).
        vel = np.asarray(vel)
        if vel.ndim == 3:
            vel = vel.reshape(vel.shape[0], -1)

        try:
            magvel_squared = vel[:, 0] ** 2 + vel[:, 1] ** 2
            if self.mesh.dim == 3:
                magvel_squared += vel[:, 2] ** 2

            max_magvel = math.sqrt(magvel_squared.max())

        except (ValueError, IndexError):
            # Sanctioned: a rank holding zero particles has an empty vel
            # array (its .max() raises); it contributes zero to the
            # global maximum below.
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

    .. deprecated:: 2026-07
        ``NodalPointSwarm`` is deprecated and will be removed in the next
        release cycle. The semi-Lagrangian history managers in
        ``uw.systems.ddt`` no longer use it and there are no remaining
        internal callers.

    The swarmVariable `X0` is defined so that the particles can "snap back" to their original locations
    after they have been moved.

    The purpose of this Swarm is to manage sample points for advection schemes based on upstream sampling
    (method of characteristics etc)"""

    def __init__(
        self,
        trackedVariable: uw.discretisation.MeshVariable,
        verbose=False,
    ):
        import warnings

        warnings.warn(
            "NodalPointSwarm is deprecated and will be removed in the next "
            "release cycle. Use the semi-Lagrangian history managers in "
            "uw.systems.ddt, or a plain Swarm populated at the variable's "
            "coordinates.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.trackedVariable = trackedVariable
        self.swarmVariable = None

        mesh = trackedVariable.mesh

        # Keyword-explicit: Swarm.__init__ takes recycle_rate as its second
        # positional parameter, so a positional `verbose` here used to land
        # in recycle_rate and be silently discarded (SWARM-11).
        super().__init__(mesh, verbose=verbose, clip_to_mesh=False)

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
        """Advect the nodal-point swarm one timestep (semi-Lagrangian sweep).

        Records each particle's launch point (its home mesh node) and
        origin rank, then delegates to :meth:`Swarm.advection`. The
        recorded launch data is what lets semi-Lagrangian schemes return
        sampled values to the node the particle departed from after the
        (possibly off-rank) trajectory.

        Parameters are those of :meth:`Swarm.advection`, with one
        difference: ``step_limit`` defaults to ``True`` — trajectories
        are substepped to at most ~one element crossing per substep
        (see :meth:`Swarm.estimate_dt`), which keeps the node-return
        bookkeeping robust for large ``delta_t``.
        """
        self._X0.data[...] = self._nX0.data[...]

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
