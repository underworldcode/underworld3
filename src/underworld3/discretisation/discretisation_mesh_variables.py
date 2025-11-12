from typing import Optional, Tuple, Union
from enum import Enum

import os
import weakref
from mpi4py.MPI import Info
import numpy
import sympy
from sympy.matrices.expressions.blockmatrix import bc_dist
import sympy.vector
from petsc4py import PETSc
import underworld3 as uw

from underworld3.utilities._api_tools import Stateful
from underworld3.utilities._api_tools import uw_object
from underworld3.utilities._utils import gather_data

# Mathematical operations moved to PersistentMeshVariable wrapper
# from underworld3.utilities.mathematical_mixin import MathematicalMixin

from underworld3.coordinates import CoordinateSystem, CoordinateSystemType

# from underworld3.cython import petsc_discretisation
import underworld3.timing as timing

## Introduce these two specific types of coordinate tracking vector objects

from sympy.vector import CoordSys3D

## Add the ability to inherit an Enum, so we can add standard boundary
## types to ones that are supplied by the users / the meshing module
## https://stackoverflow.com/questions/46073413/python-enum-combination


def extend_enum(inherited):
    def wrapper(final):
        joined = {}
        inherited.append(final)
        for i in inherited:
            for j in i:
                joined[j.name] = j.value
        return Enum(final.__name__, joined)

    return wrapper


class _BaseMeshVariable(Stateful, uw_object):
    """
    The MeshVariable class generates a variable supported by a finite element mesh and the
    underlying sympy representation that makes it possible to construct expressions that
    depend on the values of the MeshVariable.

    To set / read nodal values, use the numpy interface via the 'data' property.

    Parameters
    ----------
    varname :
        A text name for this variable. Use an R-string if a latex-expression is used
    mesh :
        The supporting underworld mesh.
    num_components :
        The number of components this variable has.
        For example, scalars will have `num_components=1`,
        while a 2d vector would have `num_components=2`.
    vtype :
        Optional. The underworld variable type for this variable.
        If not defined it will be inferred from `num_components`
        if possible.
    degree :
        The polynomial degree for this variable.
    varsymbol:
        Over-ride the varname with a symbolic form for printing etc (latex). Should be an R-string.

    """

    def __new__(
        cls,
        varname: Union[str, list],
        mesh: "Mesh",
        num_components: Union[int, tuple] = None,
        vtype: Optional["uw.VarType"] = None,
        degree: int = 1,
        continuous: bool = True,
        varsymbol: Union[str, list] = None,
        _register: bool = True,
        units: Optional[str] = None,
        units_backend: Optional[str] = None,
    ):
        """
        Create or return existing MeshVariable instance.

        Handles object uniqueness and mesh DM state management.
        """
        if isinstance(varname, list):
            name = varname[0] + R"+ \dots"
        else:
            name = varname

        ## Check if already defined (return existing object)
        import re

        clean_name = re.sub(r"[^a-zA-Z0-9_]", "", name)

        if clean_name in mesh.vars.keys():
            print(f"Variable with name {name} already exists on the mesh - Skipping.")
            return mesh.vars[clean_name]

        # NOTE: DM reconstruction is now handled in _setup_ds() - no snapshotting needed here

        # Create new instance
        obj = super().__new__(cls)

        # Store parameters for __init__
        obj._init_params = {
            "varname": name,
            "mesh": mesh,
            "num_components": num_components,
            "vtype": vtype,
            "degree": degree,
            "continuous": continuous,
            "varsymbol": varsymbol,
            "_register": _register,
            "units": units,
            "units_backend": units_backend,
        }

        return obj

    @timing.routine_timer_decorator
    def __init__(
        self,
        varname=None,
        mesh=None,
        num_components=None,
        vtype=None,
        degree=1,
        continuous=True,
        varsymbol=None,
        _register=True,
        units=None,
        units_backend=None,
    ):
        """
        Initialize MeshVariable (only called for NEW objects).

        Retrieves initialization parameters from __new__ and handles DM reconstruction.
        """
        # Only initialize if this is a new object (not returned existing)
        if hasattr(self, "_initialized"):
            return  # Already initialized

        # Get parameters - either from __new__ (via _init_params) or direct arguments
        if hasattr(self, "_init_params"):
            # Parameters from __new__ method
            params = self._init_params
            varname = params["varname"]
            mesh = params["mesh"]
            num_components = params["num_components"]
            vtype = params["vtype"]
            degree = params["degree"]
            continuous = params["continuous"]
            varsymbol = params["varsymbol"]
            _register = params["_register"]
            units = params["units"]
            units_backend = params["units_backend"]
        else:
            # Direct initialization (should not happen with __new__ pattern, but for safety)
            pass

        # Variable initialization logic
        import re
        import math

        # Variable naming and symbol handling
        if isinstance(varname, list):
            name = varname[0] + " ... "
            symbol = "{" + varname[0] + R"\cdots" + "}"
        else:
            name = varname
            if varsymbol is not None:
                symbol = "{" + varsymbol + "}"
            else:
                symbol = "{" + varname + "}"

        self._lvec = None
        self._gvec = None
        self._data = None

        self._is_accessed = False
        self._available = True  # Make vectors available by default for solver compatibility

        ## Note sympy needs a unique symbol even across different meshes
        ## or it will get confused when it clones objects. We try this: add
        ## a label to the variable that is not rendered - CHECK this works !!!

        self.name = name
        self.symbol = symbol

        if mesh.instance_number > 1:
            invisible = rf"\hspace{{ {mesh.instance_number/10000}pt }}"
            self.symbol = f"{{ {invisible} {symbol} }}"

        self.clean_name = re.sub(r"[^a-zA-Z0-9_]", "", name)

        # Variable type inference
        if vtype == None:
            if isinstance(num_components, int) and num_components == 1:
                vtype = uw.VarType.SCALAR
            elif isinstance(num_components, int) and num_components == mesh.dim:
                vtype = uw.VarType.VECTOR
            elif isinstance(num_components, tuple):
                if num_components[0] == mesh.dim and num_components[1] == mesh.dim:
                    vtype = uw.VarType.TENSOR
                else:
                    vtype = uw.VarType.MATRIX
            else:
                raise ValueError(
                    "Unable to infer variable type from `num_components`. Please explicitly set the `vtype` parameter."
                )

        if not isinstance(vtype, uw.VarType):
            raise ValueError(
                "'vtype' must be an instance of 'Variable_Type', for example `underworld.VarType.SCALAR`."
            )

        self.vtype = vtype
        self._mesh_ref = weakref.ref(mesh)
        self.shape = num_components
        self.degree = degree
        self.continuous = continuous

        # Store unit metadata for variable and initialize backend
        # Convert string units to pint.Unit using the global uw.units registry
        # This ensures all units use the same registry and can be combined
        if units is not None:
            if isinstance(units, str):
                # Convert string to pint.Unit using uw.units registry
                # uw.units('K') returns a Quantity (1 kelvin), so we extract .units to get the Unit
                self._units = uw.units(units).units
            elif hasattr(units, "dimensionality"):
                # Already a pint.Unit object
                self._units = units
            else:
                # Fallback: store as-is (shouldn't happen)
                self._units = units

            # Initialize units backend properly
            from underworld3.utilities.units_mixin import PintBackend

            if units_backend is None or units_backend == "pint":
                self._units_backend = PintBackend()
            else:
                raise ValueError(
                    f"Unknown units backend: {units_backend}. Only 'pint' is supported."
                )
        else:
            self._units = None
            self._units_backend = None

        # Component and shape handling
        if vtype == uw.VarType.SCALAR:
            self.shape = (1, 1)
            self.num_components = 1
        elif vtype == uw.VarType.VECTOR:
            self.shape = (1, mesh.dim)
            self.num_components = mesh.dim
        elif vtype == uw.VarType.TENSOR:
            self.num_components = mesh.dim * mesh.dim
            self.shape = (mesh.dim, mesh.dim)
        elif vtype == uw.VarType.SYM_TENSOR:
            self.num_components = math.comb(mesh.dim + 1, 2)
            self.shape = (mesh.dim, mesh.dim)
        elif vtype == uw.VarType.MATRIX:
            self.num_components = self.shape[0] * self.shape[1]

        self._data_container = numpy.empty(self.shape, dtype=object)

        # create associated sympy function
        from underworld3.function import UnderworldFunction

        if vtype == uw.VarType.SCALAR:
            self._sym = sympy.Matrix.zeros(1, 1)
            self._sym[0] = UnderworldFunction(
                self.symbol,
                self,
                vtype,
                0,
                0,
            )(*self.mesh.r)
            self._ijk = self._sym[0]

        elif vtype == uw.VarType.VECTOR:
            self._sym = sympy.Matrix.zeros(1, mesh.dim)
            for comp in range(mesh.dim):
                self._sym[0, comp] = UnderworldFunction(
                    self.symbol,
                    self,
                    vtype,
                    comp,
                    comp,
                )(*self.mesh.r)

            self._ijk = sympy.vector.matrix_to_vector(self._sym, self.mesh.N)

        elif vtype == uw.VarType.TENSOR:
            self._sym = sympy.Matrix.zeros(mesh.dim, mesh.dim)

            # Matrix form (any number of components)
            for i in range(mesh.dim):
                for j in range(mesh.dim):
                    self._sym[i, j] = UnderworldFunction(
                        self.symbol,
                        self,
                        vtype,
                        (i, j),
                        self._data_layout(i, j),
                    )(*self.mesh.r)

        elif vtype == uw.VarType.SYM_TENSOR:
            self._sym = sympy.Matrix.zeros(mesh.dim, mesh.dim)

            # Matrix form (any number of components)
            for i in range(mesh.dim):
                for j in range(0, mesh.dim):
                    if j >= i:
                        self._sym[i, j] = UnderworldFunction(
                            self.symbol,
                            self,
                            vtype,
                            (i, j),
                            self._data_layout(i, j),
                        )(*self.mesh.r)

                    else:
                        self._sym[i, j] = self._sym[j, i]

        elif vtype == uw.VarType.MATRIX:
            self._sym = sympy.Matrix.zeros(self.shape[0], self.shape[1])

            # Matrix form (any number of components)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self._sym[i, j] = UnderworldFunction(
                        self.symbol,
                        self,
                        vtype,
                        (i, j),
                        self._data_layout(i, j),
                    )(*self.mesh.r)

        # Set up data container
        from collections import namedtuple

        MeshVariable_ij = namedtuple("MeshVariable_ij", ["data", "sym"])

        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                self._data_container[i, j] = MeshVariable_ij(
                    data=f"MeshVariable[...].data is only available within mesh.access() context",
                    sym=self.sym[i, j],
                )

        super().__init__()

        self.mesh.vars[self.clean_name] = self
        self._setup_ds()

        # Setup public view of data - using NDArray_With_Callback
        self._array_cache = None  # Will be created lazily when first accessed
        self._data_cache = None  # Will be created lazily when first accessed

        # Register with default model for orchestration (only if _register=True)
        if _register:
            uw.get_default_model()._register_variable(self.name, self)

        # Phase 4: Validate reference quantities if variable has units
        if units is not None:
            from underworld3.utilities.nondimensional import validate_variable_reference_quantities

            model = uw.get_default_model()
            is_valid, warning_msg = validate_variable_reference_quantities(
                self.name, str(units), model
            )

            if not is_valid:
                import warnings

                warnings.warn(
                    f"\n{warning_msg}\n"
                    f"Variable will use scaling_coefficient=1.0, which may lead to poor numerical conditioning.\n",
                    UserWarning,
                    stacklevel=2,
                )

        # NOTE: DM reconstruction is now handled in _setup_ds() when fields already exist
        # The old DM rebuild code here has been removed to avoid double rebuilds
        # _setup_ds() rebuilds the DM when num_existing_fields > 0

        # Mark as initialized
        self._initialized = True

        return

    @property
    def units(self):
        """Return the units associated with this variable."""
        return self._units

    @units.setter
    def units(self, value):
        """Set the units for this variable."""
        self._units = value

    @property
    def has_units(self):
        """Check if this variable has units."""
        return self._units is not None

    @property
    def dimensionality(self):
        """Get the dimensionality of this variable."""
        if not self.has_units:
            return None
        if self._units_backend is None:
            return None
        quantity = self._units_backend.create_quantity(1.0, self._units)
        return self._units_backend.get_dimensionality(quantity)

    def _create_variable_array(self, initial_data=None):
        """
        Factory function to create NDArray_With_Callback for variable data.
        Follows the same pattern as mesh.points implementation.

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

        # Create NDArray_With_Callback (following mesh._points pattern)
        array_obj = uw.utilities.NDArray_With_Callback(
            initial_data,
            owner=self,
            disable_inplace_operators=False,  # Allow operations like existing arrays
        )

        # Single callback function (following mesh_update_callback pattern)
        def variable_update_callback(array, change_context):
            """Callback to sync variable changes back to PETSc (like mesh.points)"""
            # Only act on data-changing operations (following mesh.points pattern)
            data_changed = change_context.get("data_has_changed", True)
            if not data_changed:
                return

            # Prevent recursion by checking if we're already in a callback
            if hasattr(self, "_in_callback") and self._in_callback:
                return

            # Set recursion guard
            self._in_callback = True

            try:
                # Skip updates during mesh coordinate changes to prevent corruption
                # Check if mesh is currently being updated
                if hasattr(self.mesh, "_mesh_update_lock"):
                    # Try to acquire lock without blocking - if we can't, skip update
                    if not self.mesh._mesh_update_lock.acquire(blocking=False):
                        return
                    try:
                        # Persist changes to PETSc (like mesh callback updates coordinates)
                        self.pack_uw_data_to_petsc(array, sync=True)
                    finally:
                        self.mesh._mesh_update_lock.release()
                else:
                    # Fallback if no lock exists
                    self.pack_uw_data_to_petsc(array, sync=True)
            finally:
                # Clear recursion guard
                self._in_callback = False

        # Register the callback (following mesh.points pattern)
        array_obj.add_callback(variable_update_callback)
        return array_obj

    def _create_flat_data_array(self, initial_data=None):
        """
        Factory function to create NDArray_With_Callback for backward-compatible flat data.
        Returns data in shape (-1, num_components) using pack_raw/unpack_raw methods.

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
            # Use unpack_raw to get flat format (-1, num_components)
            initial_data = self.unpack_raw_data_from_petsc(squeeze=False, sync=True)

        # Create NDArray_With_Callback for flat data
        array_obj = uw.utilities.NDArray_With_Callback(
            initial_data,
            owner=self,
            disable_inplace_operators=False,  # Allow operations like existing arrays
        )

        # Callback for flat data format
        def flat_data_update_callback(array, change_context):
            """Callback to sync flat data changes back to PETSc"""
            # Only act on data-changing operations
            data_changed = change_context.get("data_has_changed", True)
            if not data_changed:
                return

            # Prevent recursion by checking if we're already in a callback
            if hasattr(self, "_in_flat_callback") and self._in_flat_callback:
                return

            # Set recursion guard
            self._in_flat_callback = True

            try:
                # Skip updates during mesh coordinate changes to prevent corruption
                if hasattr(self.mesh, "_mesh_update_lock"):
                    if not self.mesh._mesh_update_lock.acquire(blocking=False):
                        return
                    try:
                        # Use pack_raw for flat data format
                        self.pack_raw_data_to_petsc(array, sync=True)
                    finally:
                        self.mesh._mesh_update_lock.release()
                else:
                    # Fallback if no lock exists
                    self.pack_raw_data_to_petsc(array, sync=True)
            finally:
                # Clear recursion guard
                self._in_flat_callback = False

        # Register the callback
        array_obj.add_callback(flat_data_update_callback)
        return array_obj

    def _object_viewer(self):
        """This will substitute specific information about this object"""
        from IPython.display import Latex, Markdown, display
        from textwrap import dedent

        # feedback on this instance

        display(
            Markdown(f"**MeshVariable:**"),
            Markdown(
                f"""\
  > symbol:  ${self.symbol}$\n
  > shape:   ${self.shape}$\n
  > degree:  ${self.degree}$\n
  > continuous:  `{self.continuous}`\n
  > type:    `{self.vtype.name}`"""
            ),
            Markdown(f"**FE Data:**"),
            Markdown(
                f"""
  > PETSc field id:  ${self.field_id}$ \n
  > PETSc field name:   `{self.clean_name}` """
            ),
        )

        display(self.array),

        return

    def clone(self, name, varsymbol):
        newMeshVariable = MeshVariable(
            varname=name,
            mesh=self.mesh,
            num_components=self.shape,
            vtype=self.vtype,
            degree=self.degree,
            continuous=self.continuous,
            varsymbol=varsymbol,
        )

        return newMeshVariable

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

        # Direct PETSc access (following mesh.access pattern)
        # Ensure vector is available
        self._set_vec(available=True)

        # Mark mesh DM as initialized (replaces old _accessed flag logic)
        self.mesh._dm_initialized = True

        try:
            # Direct assignment to PETSc vec (like mesh.access does at line 1156)
            vec_data = self.vec.array.reshape(-1, self.num_components)
            vec_data[:] = data_array

            # Increment variable state to track changes
            self._increment()

            # Mark mesh local vector as stale so update_lvec() will rebuild it
            self.mesh._stale_lvec = True

            # Sync parallel operations if requested
            if sync:
                # Sync ghost values (following lines 1191-1203 pattern)
                indexset, subdm = self.mesh.dm.createSubDM(self.field_id)
                subdm.localToGlobal(self.vec, self._gvec, addv=False)
                subdm.globalToLocal(self._gvec, self.vec, addv=False)
                indexset.destroy()
                subdm.destroy()

        finally:
            # Keep vector available for future access
            pass

        return

    def pack_uw_data_to_petsc(self, data_array, sync=True):
        """
        Enhanced pack method that directly accesses mesh data without access() context.
        Designed for the new meshVariable.array interface.

        Parameters
        ----------
        data_array : numpy.ndarray
            Array data to pack into mesh field
        sync : bool
            Whether to sync parallel operations (default True)
        """
        import numpy as np

        shape = self.shape
        data_array_2d = np.atleast_2d(data_array)

        # Direct PETSc access (following mesh.access pattern)
        # Ensure vector is available
        self._set_vec(available=True)

        # Mark mesh DM as initialized (replaces old _accessed flag logic)
        self.mesh._dm_initialized = True

        try:
            # Get data directly from PETSc vec (like mesh.access does at line 1156)
            flat_data = self.vec.array.reshape(-1, self.num_components)

            # Pack data using same layout as original method
            for i in range(shape[0]):
                for j in range(shape[1]):
                    ij = self._data_layout(i, j)
                    flat_data[:, ij] = data_array_2d[:, i, j]

            # Increment variable state to track changes
            self._increment()

            # Mark mesh local vector as stale so update_lvec() will rebuild it
            self.mesh._stale_lvec = True

            # Sync parallel operations if requested
            if sync:
                # Sync ghost values (following lines 1191-1203 pattern)
                indexset, subdm = self.mesh.dm.createSubDM(self.field_id)
                subdm.localToGlobal(self.vec, self._gvec, addv=False)
                subdm.globalToLocal(self._gvec, self.vec, addv=False)
                indexset.destroy()
                subdm.destroy()

        finally:
            # Clean up
            # Keep vector available for future access
            pass

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

        # Direct PETSc access (following mesh.access pattern at line 1156)
        # Ensure vector is available
        self._set_vec(available=True)

        # Mark mesh DM as initialized (replaces old _accessed flag logic)
        self.mesh._dm_initialized = True

        try:
            # Get data directly from PETSc vec (like mesh.access does)
            result = self.vec.array.reshape(-1, self.num_components).copy()

            # Sync parallel operations if requested
            if sync:
                # Sync ghost values (following lines 1191-1203 pattern)
                indexset, subdm = self.mesh.dm.createSubDM(self.field_id)
                subdm.localToGlobal(self.vec, self._gvec, addv=False)
                subdm.globalToLocal(self._gvec, self.vec, addv=False)
                indexset.destroy()
                subdm.destroy()

        finally:
            # Clean up
            # Keep vector available for future access
            pass

        if squeeze:
            return result.squeeze()
        else:
            return result

    def unpack_uw_data_from_petsc(self, squeeze=True, sync=True):
        """
        Enhanced unpack method that directly accesses mesh data without access() context.
        Designed for the new meshVariable.array interface.

        Parameters
        ----------
        squeeze : bool
            Whether to remove singleton dimensions (default True)
        sync : bool
            Whether to sync parallel operations (default True)

        Returns
        -------
        numpy.ndarray
            Array data in correct shape for the variable
        """
        import numpy as np

        shape = self.shape

        # Direct PETSc access (following mesh.access pattern at line 1156)
        # Ensure vector is available
        self._set_vec(available=True)

        # Mark mesh DM as initialized (replaces old _accessed flag logic)
        self.mesh._dm_initialized = True

        try:
            # Get data directly from PETSc vec (like mesh.access does)
            flat_data = self.vec.array.reshape(-1, self.num_components)
            points = flat_data.shape[0]
            data_array_3d = np.empty(shape=(points, *shape), dtype=flat_data.dtype)

            # Unpack data using same layout as original method
            for i in range(shape[0]):
                for j in range(shape[1]):
                    ij = self._data_layout(i, j)
                    data_array_3d[:, i, j] = flat_data[:, ij]

            # Sync parallel operations if requested
            if sync:
                # Sync ghost values (following lines 1191-1203 pattern)
                indexset, subdm = self.mesh.dm.createSubDM(self.field_id)
                subdm.localToGlobal(self.vec, self._gvec, addv=False)
                subdm.globalToLocal(self._gvec, self.vec, addv=False)
                indexset.destroy()
                subdm.destroy()

        finally:
            # Clean up
            # Keep vector available for future access
            pass

        if squeeze:
            return data_array_3d.squeeze()
        else:
            return data_array_3d

    def rbf_interpolate(self, new_coords, meth=0, p=2, verbose=False, nnn=None, rubbish=None):
        # An inverse-distance mapping is quite robust here ... as long
        # as long we take care of the case where some nodes coincide (likely if used mesh2mesh)

        import numpy as np

        if nnn == None:
            if self.mesh.dim == 3:
                nnn = 4
            else:
                nnn = 3

        D = self.data.copy()

        if verbose and uw.mpi.rank == 0:
            print("Building K-D tree", flush=True)

        # Use non-dimensional coordinates for internal RBF interpolation KDTree
        mesh_kdt = uw.kdtree.KDTree(self.coords_nd)
        values = mesh_kdt.rbf_interpolator_local(new_coords, D, nnn, p=p, verbose=verbose)
        del mesh_kdt

        return values

    @timing.routine_timer_decorator
    def save(
        self,
        filename: str,
        name: Optional[str] = None,
        index: Optional[int] = None,
    ):
        """
        Append variable data to the specified mesh hdf5
        data file. The file must already exist.

        Parameters
        ----------
        filename :
            The filename of the mesh checkpoint file. It
            must already exist.
        name :
            Textual name for dataset. In particular, this
            will be used for XDMF generation. If not
            provided, the variable name will be used.
        index :
            Not currently supported. An optional index which
            might correspond to the timestep (for example).
        """

        # Keep vector available for future access
        pass

        viewer = PETSc.ViewerHDF5().create(filename, "a", comm=PETSc.COMM_WORLD)
        if index:
            raise RuntimeError("Recording `index` not currently supported")
            ## JM:To enable timestep recording, the following needs to be called.
            ## I'm unsure if the corresponding xdmf functionality is enabled via
            ## the PETSc xdmf script.
            # PetscViewerHDF5PushTimestepping(cviewer)
            # viewer.setTimestep(index)

        if name:
            oldname = self._gvec.getName()
            self._gvec.setName(name)
        viewer(self._gvec)
        if name:
            self._gvec.setName(oldname)
        viewer.destroy()

        ## Add variable unit metadata to the file
        import h5py, json

        # Use preferred selective_ranks pattern for unit metadata
        with uw.selective_ranks(0) as should_execute:
            if should_execute:
                f = h5py.File(filename, "a")

                # Create or get metadata group
                if "metadata" not in f:
                    g = f.create_group("metadata")
                else:
                    g = f["metadata"]

                # Add variable unit metadata
                var_metadata = {
                    "units": str(self.units) if hasattr(self, "units") and self.units else None,
                    "dimensionality": (
                        str(self.dimensionality) if hasattr(self, "dimensionality") else None
                    ),
                    "units_backend": (
                        type(self._units_backend).__name__
                        if hasattr(self, "_units_backend")
                        else None
                    ),
                    "num_components": self.num_components,
                    "variable_type": str(self.vtype),
                    "variable_name": self.name,
                }

                g.attrs[f"variable_{self.clean_name}_units"] = json.dumps(var_metadata)
                f.close()

        lvec = self.mesh.dm.getCoordinates()

    # ToDo: rename to vertex_checkpoint (or similar)
    @timing.routine_timer_decorator
    @uw.collective_operation
    def write(
        self,
        filename: str,
    ):
        """
        Write variable data to the specified mesh hdf5
        data file. The file will be over-written.

        Note: This is a COLLECTIVE operation - all MPI ranks must call it.

        Parameters
        ----------
        filename :
            The filename of the mesh checkpoint file
        """

        # Keep vector available for future access
        pass

        # Variable coordinates - let's put those in the file to
        # make it a standalone "swarm"

        dmold = self.mesh.dm.getCoordinateDM()
        dmold.createDS()
        dmnew = dmold.clone()

        options = PETSc.Options()
        options["coordinterp_petscspace_degree"] = self.degree
        options["coordinterp_petscdualspace_lagrange_continuity"] = self.continuous
        options["coordinterp_petscdualspace_lagrange_node_endpoints"] = False

        dmfe = PETSc.FE().createDefault(
            self.mesh.dim,
            self.mesh.cdim,
            self.mesh.isSimplex,
            self.mesh.qdegree,
            "coordinterp_",
            PETSc.COMM_SELF,
        )

        dmnew.setField(0, dmfe)
        dmnew.createDS()

        lvec = dmnew.getLocalVec()
        gvec = dmnew.getGlobalVec()

        lvec.array[...] = self.coords.reshape(-1)[...]
        dmnew.localToGlobal(lvec, gvec, addv=False)
        gvec.setName("coordinates")

        viewer = PETSc.ViewerHDF5().create(filename, "w", comm=PETSc.COMM_WORLD)
        viewer(self._gvec)
        viewer(gvec)

        dmnew.restoreGlobalVec(gvec)
        dmnew.restoreLocalVec(lvec)

        uw.mpi.barrier()
        viewer.destroy()
        dmfe.destroy()

        ## Add variable unit metadata to standalone file
        import h5py, json

        # Use preferred selective_ranks pattern for unit metadata
        with uw.selective_ranks(0) as should_execute:
            if should_execute:
                f = h5py.File(filename, "a")

                # Add variable metadata to standalone file
                var_metadata = {
                    "units": str(self.units) if hasattr(self, "units") and self.units else None,
                    "dimensionality": (
                        str(self.dimensionality) if hasattr(self, "dimensionality") else None
                    ),
                    "units_backend": (
                        type(self._units_backend).__name__
                        if hasattr(self, "_units_backend")
                        else None
                    ),
                    "coordinate_units": (
                        str(self.mesh.coordinate_units)
                        if hasattr(self.mesh, "coordinate_units")
                        else None
                    ),
                    "mesh_type": type(self.mesh).__name__,
                    "variable_name": self.name,
                    "num_components": self.num_components,
                    "variable_type": str(self.vtype),
                }

                # Store as root-level attribute for standalone files
                f.attrs["variable_metadata"] = json.dumps(var_metadata)
                f.close()

        return

    @timing.routine_timer_decorator
    def read_timestep(
        self,
        data_filename,
        data_name,
        index,
        outputPath="",
        verbose=False,
    ):
        """
        Read a mesh variable from an arbitrary vertex-based checkpoint file
        and reconstruct/interpolate the data field accordingly. The data sizes / meshes can be
        different and will be matched using a kd-tree / inverse-distance weighting
        to the new mesh.

        """

        # Fix this to match the write_timestep function

        # mesh.write_timestep( "test", meshUpdates=False, meshVars=[X], outputPath="", index=0)
        # swarm.write_timestep("test", "swarm", swarmVars=[var], outputPath="", index=0)

        output_base_name = os.path.join(outputPath, data_filename)
        data_file = output_base_name + f".mesh.{data_name}.{index:05}.h5"

        # check if data_file exists
        if os.path.isfile(os.path.abspath(data_file)):
            pass
        else:
            raise RuntimeError(f"{os.path.abspath(data_file)} does not exist")

        import h5py
        import numpy as np

        # Keep vector available for future access
        pass

        ## Sub functions that are used to read / interpolate the mesh.
        def field_from_checkpoint(
            data_file=None,
            data_name=None,
        ):
            """Read the mesh data as a swarm-like value"""

            if verbose and uw.mpi.rank == 0:
                print(f"Reading data file {data_file}", flush=True)

            h5f = h5py.File(data_file)
            D = h5f["fields"][data_name][()].reshape(-1, self.shape[1])
            X = h5f["fields"]["coordinates"][()].reshape(-1, self.mesh.dim)

            h5f.close()

            if len(D.shape) == 1:
                D = D.reshape(-1, 1)

            return X, D

        def map_to_vertex_values(X, D, nnn=4, p=2, verbose=False):
            # Map from "swarm" of points to nodal points
            # This is a permutation if we building on the checkpointed
            # mesh file

            mesh_kdt = uw.kdtree.KDTree(X)

            return mesh_kdt.rbf_interpolator_local(self.coords, D, nnn, p, verbose)

        def values_to_mesh_var(mesh_variable, Values):
            mesh = mesh_variable.mesh

            # This should be trivial but there may be problems if
            # the kdtree does not have enough neighbours to allocate
            # values for every point. We handle that here.

            mesh_variable.data[...] = Values[...]

            return

        ## Read file information

        X, D = field_from_checkpoint(
            data_file,
            data_name,
        )

        remapped_D = map_to_vertex_values(X, D)

        # This is empty at the moment
        values_to_mesh_var(self, remapped_D)

        return

    @timing.routine_timer_decorator
    def load_from_h5_plex_vector(
        self,
        filename: str,
        data_name: Optional[str] = None,
    ):
        if data_name is None:
            data_name = self.clean_name

        # Ensure vectors are initialized
        if self._lvec is None:
            self._set_vec(available=True)

        indexset, subdm = self.mesh.dm.createSubDM(self.field_id)

        old_name = self._gvec.getName()
        viewer = PETSc.ViewerHDF5().create(filename, "r", comm=PETSc.COMM_WORLD)

        self._gvec.setName(data_name)
        self._gvec.load(viewer)
        self._gvec.setName(old_name)

        subdm.globalToLocal(self._gvec, self._lvec, addv=False)

        viewer.destroy()
        indexset.destroy()
        subdm.destroy()

        return

    @property
    def fn(self) -> sympy.Basic:
        """
        The handle to the (i,j,k) spatial view of this variable if it exists (deprecated)
        """
        return self._ijk

    @property
    def ijk(self) -> sympy.Basic:
        """
        The handle to the (i,j,k) spatial view of this variable if it exists
        """
        return self._ijk

    @property
    def sym(self) -> sympy.Basic:
        """
        The handle to the sympy.Matrix view of this variable
        """
        # Note: Scaling is applied during unwrap(), not here
        return self._sym

    @property
    def sym_1d(self) -> sympy.Basic:
        """
        The handle to a flattened version of the sympy.Matrix view of this variable.
        Assume components are stored in the same order that sympy iterates entries in
        a matrix except for the symmetric tensor case where we store in a Voigt form
        """

        if self.vtype != uw.VarType.SYM_TENSOR:
            return self._sym.reshape(1, len(self._sym))
        else:
            if self.mesh.dim == 2:
                return sympy.Matrix(
                    [
                        self._sym[0, 0],
                        self._sym[1, 1],
                        self._sym[0, 1],
                    ]
                ).T
            else:
                return sympy.Matrix(
                    [
                        self._sym[0, 0],
                        self._sym[1, 1],
                        self._sym[2, 2],
                        self._sym[0, 1],
                        self._sym[0, 2],
                        self._sym[1, 2],
                    ]
                ).T

    def _data_layout(self, i, j=None):
        # mapping

        if self.vtype == uw.VarType.SCALAR:
            return 0
        if self.vtype == uw.VarType.VECTOR:
            if i < 0 or j < 0:
                return self.mesh.dim
            else:
                if j is None:
                    return i
                elif i == 0:
                    return j
                else:
                    raise IndexError(f"Vectors have shape {self.mesh.dim} or {(1, self.mesh.dim)} ")
        if self.vtype == uw.VarType.TENSOR:
            if self.mesh.dim == 2:
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
            if self.mesh.dim == 2:
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

    def _setup_ds(self):
        options = PETSc.Options()
        name0 = "VAR"  # self.clean_name ## Filling up the options database
        options.setValue(f"{name0}_petscspace_degree", self.degree)
        options.setValue(f"{name0}_petscdualspace_lagrange_continuity", self.continuous)
        options.setValue(
            f"{name0}_petscdualspace_lagrange_node_endpoints", False
        )  # only active if discontinuous

        dim = self.mesh.dm.getDimension()
        petsc_fe = PETSc.FE().createDefault(
            dim,
            self.num_components,
            self.mesh.isSimplex,
            self.mesh.qdegree,
            name0 + "_",
            PETSc.COMM_SELF,
        )

        # Check if this is the first field or if we need to rebuild the DM
        # (needed to ensure Section is properly synchronized with field list)
        num_existing_fields = self.mesh.dm.getNumFields()

        if num_existing_fields > 0:
            # DM already has fields - need to rebuild to sync Section
            # This follows the pattern from __init__ (lines 353-383)
            dm_old = self.mesh.dm
            dm_new = dm_old.clone()

            # Copy existing fields to new DM
            dm_old.copyFields(dm_new)

            # Add our new field to the new DM
            field_id = dm_new.getNumFields()
            dm_new.addField(petsc_fe)
            field, _ = dm_new.getField(field_id)
            field.setName(self.clean_name)

            # Create DS on new DM (this builds a fresh Section with all fields)
            dm_new.createDS()

            # CRITICAL FIX: Transfer data from old vectors to new DM
            # When we rebuild the DM, existing variables' vectors must be recreated
            # from the new DM, but we need to preserve their data

            # Save old variable data before destroying vectors
            var_data_backup = {}
            for var in self.mesh.vars.values():
                if var._lvec is not None:
                    # Save the data
                    var_data_backup[var.clean_name] = var._lvec.array.copy()
                    # Destroy old vectors
                    var._lvec.destroy()
                    var._lvec = None
                if var._gvec is not None:
                    var._gvec.destroy()
                    var._gvec = None

            # Also invalidate mesh's local vector if it exists
            if self.mesh._lvec is not None:
                self.mesh._lvec.destroy()
                self.mesh._lvec = None
                self.mesh._stale_lvec = True

            # Replace old DM with new one
            dm_old.destroy()
            self.mesh.dm = dm_new
            self.mesh.dm_hierarchy[-1] = dm_new

            # Restore data to variables (this will create new vectors from new DM)
            for var in self.mesh.vars.values():
                if var.clean_name in var_data_backup:
                    # _set_vec will create new vectors from the new DM
                    var._set_vec(available=True)
                    # Restore the data
                    var._lvec.array[...] = var_data_backup[var.clean_name]

            self.field_id = field_id
        else:
            # First field - normal fast path
            self.field_id = self.mesh.dm.getNumFields()
            self.mesh.dm.addField(petsc_fe)
            field, _ = self.mesh.dm.getField(self.field_id)
            field.setName(self.clean_name)
            self.mesh.dm.createDS()

        return

    def _set_vec(self, available):
        if self._lvec == None:
            indexset, subdm = self.mesh.dm.createSubDM(self.field_id)

            self._lvec = subdm.createLocalVector()
            self._lvec.zeroEntries()  # not sure if required, but to be sure.
            self._gvec = subdm.createGlobalVector()
            self._gvec.setName(self.clean_name)  # This is set for checkpointing.
            self._gvec.zeroEntries()

        self._available = available

    def __del__(self):
        if self._lvec:
            self._lvec.destroy()
        if self._gvec:
            self._gvec.destroy()

    @property
    def mesh(self):
        """
        The mesh this variable belongs to (accessed via weak reference).
        Raises RuntimeError if the mesh has been garbage collected.
        """
        if self._mesh_ref is None:
            raise RuntimeError("MeshVariable has no mesh reference (internal error)")

        mesh = self._mesh_ref()
        if mesh is None:
            raise RuntimeError(
                f"Mesh for variable '{self.clean_name}' has been garbage collected. "
                "Variables cannot outlive their parent mesh."
            )
        return mesh

    @property
    def vec(self) -> PETSc.Vec:
        """
        The corresponding PETSc local vector for this variable.
        """
        if not self._available:
            raise RuntimeError("Vector must be accessed via the mesh `access()` context manager.")

        # Ensure vector is initialized when accessed
        if self._lvec is None:
            self._set_vec(available=self._available)

        return self._lvec

    @property
    def old_data(self) -> numpy.ndarray:
        """
        TESTING: Original data property implementation.
        Numpy proxy array to underlying variable data.
        Note that the returned array is a proxy for all the *local* nodal
        data, and is provided as 1d list.

        For both read and write, this array can only be accessed via the
        mesh `access()` context manager.
        """
        if self._data is None:
            raise RuntimeError("Data must be accessed via the mesh `access()` context manager.")
        return self._data

    @property
    def array(self):
        """
        Array view of canonical data with automatic format conversion.
        Shape: (N, a, b) for tensor shape (a, b).

        This property is ALWAYS a view of the canonical .data property.
        No direct PETSc access - all changes delegate back to canonical storage.
        """
        return self._create_array_view()

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
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        class SimpleMeshArrayView:
            def __init__(self, parent_var):
                self.parent = parent_var

            def _get_array_data(self):
                # Simple reshape: (-1, num_components) -> (N, a, b)
                data = self.parent.data
                # For simple variables, reshape to (N, a, b) format
                reshaped = data.reshape(data.shape[0], *self.parent.shape)

                # Apply ND scaling if active: convert ND → dimensional
                import underworld3 as uw

                if uw.is_nondimensional_scaling_active():
                    if hasattr(self.parent, "scaling_coefficient"):
                        scale = self.parent.scaling_coefficient
                        if scale != 1.0 and scale != 0.0:
                            # Convert ND values to dimensional: T_dim = T_ND * T_scale
                            reshaped = reshaped * scale

                # Wrap in UnitAwareArray if variable has units
                if hasattr(self.parent, "units") and self.parent.units is not None:
                    return UnitAwareArray(reshaped, units=str(self.parent.units))
                else:
                    return reshaped

            def __getitem__(self, key):
                return self._get_array_data()[key]

            def __setitem__(self, key, value):
                # Get current array view
                array_data = self._get_array_data()
                # Create a copy to modify (avoid modifying view directly)
                modified_data = (
                    array_data.copy()
                    if hasattr(array_data, "copy")
                    else np.array(array_data).copy()
                )

                # Extract magnitude if value is a UWQuantity or UnitAwareArray
                if hasattr(value, "magnitude"):
                    # Convert to variable's units if needed
                    if (
                        hasattr(value, "units")
                        and hasattr(self.parent, "units")
                        and self.parent.units
                    ):
                        # TODO: Add unit conversion here if units mismatch
                        value = value.magnitude
                    else:
                        value = value.magnitude

                modified_data[key] = value
                # Reshape back to flat format and assign to canonical data
                # Extract magnitude from UnitAwareArray if needed
                if hasattr(modified_data, "magnitude"):
                    flat_data = modified_data.magnitude.reshape(-1, self.parent.num_components)
                else:
                    flat_data = modified_data.reshape(-1, self.parent.num_components)

                # Apply inverse ND scaling if active: convert dimensional → ND before storing
                import underworld3 as uw

                if uw.is_nondimensional_scaling_active():
                    if hasattr(self.parent, "scaling_coefficient"):
                        scale = self.parent.scaling_coefficient
                        if scale != 1.0 and scale != 0.0:
                            # Convert dimensional values to ND: T_ND = T_dim / T_scale
                            flat_data = flat_data / scale

                self.parent.data[...] = flat_data

            @property
            def shape(self):
                return self._get_array_data().shape

            @property
            def dtype(self):
                return self._get_array_data().dtype

            @property
            def units(self):
                """Get units from parent variable for consistency with evaluate() results"""
                if hasattr(self.parent, "units") and self.parent.units is not None:
                    # Return string representation to match evaluate() results (UnitAwareArray)
                    return str(self.parent.units)
                return None

            @property
            def _units(self):
                """Alias for uw.get_units() compatibility"""
                return self.units

            def to(self, target_units):
                """
                Convert to target units (unified interface).

                Delegates to UnitAwareArray.to() for the actual conversion.

                Parameters
                ----------
                target_units : str
                    Target units to convert to

                Returns
                -------
                UnitAwareArray
                    Converted array with target units
                """
                data = self._get_array_data()
                if hasattr(data, "to"):
                    return data.to(target_units)
                else:
                    raise ValueError(f"Variable '{self.parent.name}' has no units - cannot convert")

            def __repr__(self):
                units_str = f", units='{self.units}'" if self.units else ""
                return f"SimpleMeshArrayView(shape={self.shape}, dtype={self.dtype}{units_str})"

            def __array__(self):
                """Support for numpy functions like np.allclose(), np.isfinite(), etc."""
                return self._get_array_data()

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                """Support for numpy universal functions"""
                # Convert all MeshArrayView inputs to arrays
                converted_inputs = []
                for input in inputs:
                    if hasattr(input, "_get_array_data"):  # Duck typing for array views
                        converted_inputs.append(input._get_array_data())
                    else:
                        converted_inputs.append(input)

                # Apply the ufunc to the converted inputs
                return ufunc(*converted_inputs, **kwargs)

            def max(self):
                """
                Global maximum value.
                Returns scalar for single-component variables, tuple for multi-component.
                """
                data = self._get_array_data()
                if self.parent.num_components == 1:
                    # For single-component: return scalar (reshape to handle (N,1,1) → scalar)
                    return float(np.max(data))
                else:
                    # For multi-component: return tuple of component maxima
                    return tuple(
                        [float(np.max(data[..., i])) for i in range(self.parent.num_components)]
                    )

            def min(self):
                """
                Global minimum value.
                Returns scalar for single-component variables, tuple for multi-component.
                """
                data = self._get_array_data()
                if self.parent.num_components == 1:
                    # For single-component: return scalar
                    return float(np.min(data))
                else:
                    # For multi-component: return tuple of component minima
                    return tuple(
                        [float(np.min(data[..., i])) for i in range(self.parent.num_components)]
                    )

            def mean(self):
                """
                Global mean value.
                Returns scalar for single-component variables, tuple for multi-component.
                """
                data = self._get_array_data()
                if self.parent.num_components == 1:
                    # For single-component: return scalar
                    return float(np.mean(data))
                else:
                    # For multi-component: return tuple of component means
                    return tuple(
                        [float(np.mean(data[..., i])) for i in range(self.parent.num_components)]
                    )

            def sum(self):
                """
                Global sum.
                Returns scalar for single-component variables, tuple for multi-component.
                """
                data = self._get_array_data()
                if self.parent.num_components == 1:
                    # For single-component: return scalar
                    return float(np.sum(data))
                else:
                    # For multi-component: return tuple of component sums
                    return tuple(
                        [float(np.sum(data[..., i])) for i in range(self.parent.num_components)]
                    )

            def std(self):
                """
                Global standard deviation.
                Returns scalar for single-component variables, tuple for multi-component.
                """
                data = self._get_array_data()
                if self.parent.num_components == 1:
                    # For single-component: return scalar
                    return float(np.std(data))
                else:
                    # For multi-component: return tuple of component standard deviations
                    return tuple(
                        [float(np.std(data[..., i])) for i in range(self.parent.num_components)]
                    )

        return SimpleMeshArrayView(self)

    def _create_tensor_array_view(self):
        """Array view for complex tensors using pack/unpack operations"""
        import numpy as np
        from underworld3.utilities.unit_aware_array import UnitAwareArray

        class TensorMeshArrayView:
            def __init__(self, parent_var):
                self.parent = parent_var

            def _get_array_data(self):
                # Use complex pack/unpack for tensor layouts
                unpacked = self.parent.unpack_uw_data_from_petsc(squeeze=False)

                # Apply ND scaling if active: convert ND → dimensional
                import underworld3 as uw

                if uw.is_nondimensional_scaling_active():
                    if hasattr(self.parent, "scaling_coefficient"):
                        scale = self.parent.scaling_coefficient
                        if scale != 1.0 and scale != 0.0:
                            # Convert ND values to dimensional: T_dim = T_ND * T_scale
                            unpacked = unpacked * scale

                # Wrap in UnitAwareArray if variable has units
                if hasattr(self.parent, "units") and self.parent.units is not None:
                    return UnitAwareArray(unpacked, units=str(self.parent.units))
                else:
                    return unpacked

            def __getitem__(self, key):
                return self._get_array_data()[key]

            def __setitem__(self, key, value):
                # Get current array view
                array_data = self._get_array_data()
                # Create a copy to modify (avoid modifying view directly)
                modified_data = (
                    array_data.copy()
                    if hasattr(array_data, "copy")
                    else np.array(array_data).copy()
                )

                # Extract magnitude if value is a UWQuantity or UnitAwareArray
                if hasattr(value, "magnitude"):
                    # Convert to variable's units if needed
                    if (
                        hasattr(value, "units")
                        and hasattr(self.parent, "units")
                        and self.parent.units
                    ):
                        # TODO: Add unit conversion here if units mismatch
                        value = value.magnitude
                    else:
                        value = value.magnitude

                modified_data[key] = value

                # Extract magnitude from UnitAwareArray if needed
                if hasattr(modified_data, "magnitude"):
                    data_to_pack = modified_data.magnitude
                else:
                    data_to_pack = modified_data

                # Apply inverse ND scaling if active: convert dimensional → ND before storing
                import underworld3 as uw

                if uw.is_nondimensional_scaling_active():
                    if hasattr(self.parent, "scaling_coefficient"):
                        scale = self.parent.scaling_coefficient
                        if scale != 1.0 and scale != 0.0:
                            # Convert dimensional values to ND: T_ND = T_dim / T_scale
                            data_to_pack = data_to_pack / scale

                # Pack back to canonical data format
                self.parent.pack_uw_data_to_petsc(data_to_pack)

            @property
            def shape(self):
                return self._get_array_data().shape

            @property
            def dtype(self):
                return self._get_array_data().dtype

            @property
            def units(self):
                """Get units from parent variable for consistency with evaluate() results"""
                if hasattr(self.parent, "units") and self.parent.units is not None:
                    # Return string representation to match evaluate() results (UnitAwareArray)
                    return str(self.parent.units)
                return None

            @property
            def _units(self):
                """Alias for uw.get_units() compatibility"""
                return self.units

            def to(self, target_units):
                """
                Convert to target units (unified interface).

                Delegates to UnitAwareArray.to() for the actual conversion.

                Parameters
                ----------
                target_units : str
                    Target units to convert to

                Returns
                -------
                UnitAwareArray
                    Converted array with target units
                """
                data = self._get_array_data()
                if hasattr(data, "to"):
                    return data.to(target_units)
                else:
                    raise ValueError(f"Variable '{self.parent.name}' has no units - cannot convert")

            def __repr__(self):
                units_str = f", units='{self.units}'" if self.units else ""
                return f"TensorMeshArrayView(shape={self.shape}, dtype={self.dtype}{units_str})"

            def __array__(self):
                """Support for numpy functions like np.allclose(), np.isfinite(), etc."""
                return self._get_array_data()

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                """Support for numpy universal functions"""
                # Convert all MeshArrayView inputs to arrays
                converted_inputs = []
                for input in inputs:
                    if hasattr(input, "_get_array_data"):  # Duck typing for array views
                        converted_inputs.append(input._get_array_data())
                    else:
                        converted_inputs.append(input)

                # Apply the ufunc to the converted inputs
                return ufunc(*converted_inputs, **kwargs)

            def max(self):
                """
                Global maximum value.
                Returns scalar for single-component variables, tuple for multi-component.
                """
                data = self._get_array_data()
                if self.parent.num_components == 1:
                    # For single-component: return scalar
                    return float(np.max(data))
                else:
                    # For multi-component: return tuple of component maxima
                    return tuple(
                        [float(np.max(data[..., i])) for i in range(self.parent.num_components)]
                    )

            def min(self):
                """
                Global minimum value.
                Returns scalar for single-component variables, tuple for multi-component.
                """
                data = self._get_array_data()
                if self.parent.num_components == 1:
                    # For single-component: return scalar
                    return float(np.min(data))
                else:
                    # For multi-component: return tuple of component minima
                    return tuple(
                        [float(np.min(data[..., i])) for i in range(self.parent.num_components)]
                    )

            def mean(self):
                """
                Global mean value.
                Returns scalar for single-component variables, tuple for multi-component.
                """
                data = self._get_array_data()
                if self.parent.num_components == 1:
                    # For single-component: return scalar
                    return float(np.mean(data))
                else:
                    # For multi-component: return tuple of component means
                    return tuple(
                        [float(np.mean(data[..., i])) for i in range(self.parent.num_components)]
                    )

            def sum(self):
                """
                Global sum.
                Returns scalar for single-component variables, tuple for multi-component.
                """
                data = self._get_array_data()
                if self.parent.num_components == 1:
                    # For single-component: return scalar
                    return float(np.sum(data))
                else:
                    # For multi-component: return tuple of component sums
                    return tuple(
                        [float(np.sum(data[..., i])) for i in range(self.parent.num_components)]
                    )

            def std(self):
                """
                Global standard deviation.
                Returns scalar for single-component variables, tuple for multi-component.
                """
                data = self._get_array_data()
                if self.parent.num_components == 1:
                    # For single-component: return scalar
                    return float(np.std(data))
                else:
                    # For multi-component: return tuple of component standard deviations
                    return tuple(
                        [float(np.std(data[..., i])) for i in range(self.parent.num_components)]
                    )

        return TensorMeshArrayView(self)

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
        # Use direct __dict__ check to avoid potential attribute access issues
        if "_canonical_data" not in self.__dict__ or self._canonical_data is None:
            # Create the single canonical data array with PETSc sync
            self._canonical_data = self._create_canonical_data_array()

        return self._canonical_data

    def _create_canonical_data_array(self):
        """
        Create the single canonical data array with PETSc synchronization for MeshVariable.
        This is the ONLY method that creates arrays with PETSc callbacks.

        Handles mesh-specific requirements like locking and ghost value synchronization.

        Returns
        -------
        NDArray_With_Callback
            Canonical array object with callback for automatic PETSc synchronization
        """
        # Ensure PETSc vector is available
        self._set_vec(available=True)
        self.mesh._dm_initialized = True

        # Get direct access to PETSc vector in packed format
        flat_petsc_data = self.vec.array.reshape(-1, self.num_components)

        # Create NDArray_With_Callback with proper shape and data
        from underworld3.utilities import NDArray_With_Callback

        array_obj = NDArray_With_Callback(flat_petsc_data)

        # Single canonical callback for PETSc synchronization
        def canonical_data_callback(array, change_context):
            """ONLY callback that handles PETSc synchronization - prevents conflicts"""
            # Only act on data-changing operations
            data_changed = change_context.get("data_has_changed", True)
            if not data_changed:
                return

            # Check for None array to prevent copy errors
            if array is None:
                return

            # STEP 1: Ensure array has correct canonical shape before PETSc sync
            # The callback might receive wrong-shaped arrays from array view operations
            import numpy as np

            canonical_array = np.atleast_2d(array)

            if canonical_array.shape != (canonical_array.shape[0], self.num_components):
                # Only reshape if we actually need to
                canonical_array = canonical_array.reshape(-1, self.num_components)

            # Skip updates during mesh coordinate changes to prevent corruption
            if hasattr(self.mesh, "_mesh_update_lock"):
                if not self.mesh._mesh_update_lock.acquire(blocking=False):
                    return
                try:
                    # STEP 1: Sync to PETSc using established method with correct shape
                    self.pack_raw_data_to_petsc(canonical_array, sync=True)
                finally:
                    self.mesh._mesh_update_lock.release()
            else:
                # Fallback if no lock exists
                self.pack_raw_data_to_petsc(canonical_array, sync=True)

            # STEP 2: Handle variable-specific updates (extensible like SwarmVariable)
            if hasattr(self, "_on_data_changed"):
                self._on_data_changed()

        array_obj.add_callback(canonical_data_callback)
        return array_obj

    @array.setter
    def array(self, array_value):
        """
        Set variable data using pack method to handle shape transformation.
        """
        # Use pack method to handle proper data transformation and shape conversion
        self.pack_uw_data_to_petsc(array_value, sync=True)

    ## ToDo: We should probably deprecate this in favour of using integrals

    def _dimensionalise_stat(self, value: Union[float, tuple]) -> Union[float, tuple]:
        """
        Helper to dimensionalise statistical values using uw.dimensionalise().

        Takes non-dimensional value(s) from PETSc and converts to dimensional
        form using the variable's units and model reference quantities.

        Parameters
        ----------
        value : float or tuple
            Non-dimensional value(s) from PETSc

        Returns
        -------
        float, tuple, or UWQuantity
            Dimensionalised value(s) if units are enabled, else unchanged
        """
        # Check if units mode is enabled
        model = uw.get_default_model()
        if not model.has_units() or not hasattr(self, 'units') or self.units is None:
            return value  # Backward compatible - no units mode or variable has no units

        # Extract dimensionality from units
        # self.units is already a Pint Unit object with .dimensionality attribute
        try:
            if hasattr(self.units, 'dimensionality'):
                # Pint Unit object - extract dimensionality directly
                dimensionality = dict(self.units.dimensionality)
            else:
                # String or other - parse using Pint
                from ..scaling import units as ureg
                pint_unit = ureg(self.units)
                dimensionality = dict(pint_unit.dimensionality)
        except Exception as e:
            # If extraction fails, fall back to no dimensionality
            import warnings
            warnings.warn(f"Failed to extract dimensionality from units '{self.units}': {e}")
            return value

        # Dimensionalise using proper units system
        if isinstance(value, tuple):
            return tuple(uw.dimensionalise(val, target_dimensionality=dimensionality, model=model) for val in value)
        else:
            return uw.dimensionalise(value, target_dimensionality=dimensionality, model=model)

    def min(self) -> Union[float, tuple]:
        """
        The global variable minimum value.
        Returns the value only (not the rank). For multi-component variables,
        returns a tuple of minimum values for each component.

        When units are enabled (model.has_units() == True), returns UWQuantity
        with proper dimensionality.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        # Sync local→global to ensure global vector has latest data
        indexset, subdm = self.mesh.dm.createSubDM(self.field_id)
        subdm.localToGlobal(self._lvec, self._gvec, addv=False)
        indexset.destroy()
        subdm.destroy()

        # Get raw non-dimensional values from PETSc
        if self.num_components == 1:
            rank, value = self._gvec.min()
            min_vals = value
        else:
            min_vals = tuple([self._gvec.strideMin(i)[1] for i in range(self.num_components)])

        # Dimensionalise using units system
        return self._dimensionalise_stat(min_vals)

    def max(self) -> Union[float, tuple]:
        """
        The global variable maximum value.
        Returns the value only (not the rank). For multi-component variables,
        returns a tuple of maximum values for each component.

        When units are enabled (model.has_units() == True), returns UWQuantity
        with proper dimensionality.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        # Sync local→global to ensure global vector has latest data
        indexset, subdm = self.mesh.dm.createSubDM(self.field_id)
        subdm.localToGlobal(self._lvec, self._gvec, addv=False)
        indexset.destroy()
        subdm.destroy()

        # Get raw non-dimensional values from PETSc
        if self.num_components == 1:
            rank, value = self._gvec.max()
            max_vals = value
        else:
            max_vals = tuple([self._gvec.strideMax(i)[1] for i in range(self.num_components)])

        # Dimensionalise using units system
        return self._dimensionalise_stat(max_vals)

    def sum(self) -> Union[float, tuple]:
        """
        The global variable sum value.

        When units are enabled (model.has_units() == True), returns UWQuantity
        with proper dimensionality.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        # Sync local→global to ensure global vector has latest data
        indexset, subdm = self.mesh.dm.createSubDM(self.field_id)
        subdm.localToGlobal(self._lvec, self._gvec, addv=False)
        indexset.destroy()
        subdm.destroy()

        # Get raw non-dimensional values from PETSc
        if self.num_components == 1:
            sum_vals = self._gvec.sum()
        else:
            cpts = []
            for i in range(0, self.num_components):
                cpts.append(self._gvec.strideSum(i))
            sum_vals = tuple(cpts)

        # Dimensionalise using units system
        return self._dimensionalise_stat(sum_vals)

    def norm(self, norm_type) -> Union[float, tuple]:
        """
        The global variable norm value.

        norm_type: type of norm, one of
            - 0: NORM 1 ||v|| = sum_i | v_i |. ||A|| = max_j || v_*j ||
            - 1: NORM 2 ||v|| = sqrt(sum_i |v_i|^2) (vectors only)
            - 3: NORM INFINITY ||v|| = max_i |v_i|. ||A|| = max_i || v_i* ||, maximum row sum

        When units are enabled (model.has_units() == True), returns UWQuantity
        with proper dimensionality.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components > 1 and norm_type == 2:
            raise RuntimeError("Norm 2 is only available for vectors.")

        # Sync local→global to ensure global vector has latest data
        indexset, subdm = self.mesh.dm.createSubDM(self.field_id)
        subdm.localToGlobal(self._lvec, self._gvec, addv=False)
        indexset.destroy()
        subdm.destroy()

        # Get raw non-dimensional values from PETSc
        if self.num_components == 1:
            norm_vals = self._gvec.norm(norm_type)
        else:
            norm_vals = tuple([self._gvec.strideNorm(i, norm_type) for i in range(self.num_components)])

        # Dimensionalise using units system
        return self._dimensionalise_stat(norm_vals)

    def mean(self) -> Union[float, tuple]:
        """
        The global variable mean value.

        When units are enabled (model.has_units() == True), returns UWQuantity
        with proper dimensionality.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        # Sync local→global to ensure global vector has latest data
        indexset, subdm = self.mesh.dm.createSubDM(self.field_id)
        subdm.localToGlobal(self._lvec, self._gvec, addv=False)
        indexset.destroy()
        subdm.destroy()

        # Get raw non-dimensional values from PETSc
        if self.num_components == 1:
            vecsize = self._gvec.getSize()
            mean_vals = self._gvec.sum() / vecsize
        else:
            vecsize = self._gvec.getSize() / self.num_components
            mean_vals = tuple([self._gvec.strideSum(i) / vecsize for i in range(self.num_components)])

        # Dimensionalise using units system
        return self._dimensionalise_stat(mean_vals)

    def std(self) -> Union[float, tuple]:
        """
        The global variable standard deviation value.

        When units are enabled (model.has_units() == True), returns UWQuantity
        with proper dimensionality.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        # Sync local→global to ensure global vector has latest data
        indexset, subdm = self.mesh.dm.createSubDM(self.field_id)
        subdm.localToGlobal(self._lvec, self._gvec, addv=False)
        indexset.destroy()
        subdm.destroy()

        # Get raw values from PETSc
        if self.num_components == 1:
            # For scalar: std = sqrt((sum(x^2)/n) - (sum(x)/n)^2)
            vecsize = self._gvec.getSize()
            vec_sum = self._gvec.sum()
            vec_mean = vec_sum / vecsize

            # Create a temporary vector for x^2 computation
            vec_squared = self._gvec.duplicate()
            vec_squared.pointwiseMult(self._gvec, self._gvec)
            sum_squared = vec_squared.sum()
            vec_squared.destroy()

            # Calculate variance: E[x^2] - (E[x])^2
            variance = (sum_squared / vecsize) - (vec_mean**2)
            std_vals = float(numpy.sqrt(max(variance, 0.0)))  # max() ensures non-negative
        else:
            vecsize = self._gvec.getSize() / self.num_components
            stds = []
            for i in range(self.num_components):
                component_sum = self._gvec.strideSum(i)
                component_mean = component_sum / vecsize

                # Create temporary for squared values
                vec_squared = self._gvec.duplicate()
                vec_squared.pointwiseMult(self._gvec, self._gvec)
                # Sum only the i-th component
                sum_squared = vec_squared.strideSum(i)
                vec_squared.destroy()

                # Variance for this component
                variance = (sum_squared / vecsize) - (component_mean**2)
                stds.append(float(numpy.sqrt(max(variance, 0.0))))

            std_vals = tuple(stds)

        # Dimensionalise using units system
        return self._dimensionalise_stat(std_vals)

    @uw.collective_operation
    def stats(self):
        """
        Universal statistics method for all variable types.

        Returns various statistical measures appropriate for the variable type.
        For scalars: standard statistical measures.
        For vectors: magnitude-based statistics.
        For tensors: Frobenius norm and invariant-based measures.

        Returns
        -------
        dict
            Dictionary containing statistical measures:
            - 'type': Variable type ('scalar', 'vector', 'tensor')
            - 'components': Number of components
            - 'size': Number of elements
            - 'mean': Mean value (scalar) or magnitude mean (vector/tensor)
            - 'min': Minimum value (scalar) or magnitude min (vector/tensor)
            - 'max': Maximum value (scalar) or magnitude max (vector/tensor)
            - 'sum': Sum of all values
            - 'norm2': L2 norm
            - 'rms': Root mean square

            Additional keys for vectors/tensors:
            - 'magnitude_*': Statistics on vector magnitude
            - 'frobenius_*': Statistics on tensor Frobenius norm (for tensors)

        Note: This is a COLLECTIVE operation - all MPI ranks must call it.
        """

        if self.num_components == 1:
            return self._scalar_stats()
        elif self.num_components <= self.mesh.dim:
            return self._vector_stats()
        else:
            return self._tensor_stats()

        #       This uses a private work MeshVariable and the various norms defined there but
        #       could either be simplified to just use petsc vectors, or extended to
        #       compute integrals over the elements which is in line with uw1 and uw2

        from petsc4py.PETSc import NormType

        vsize = self._gvec.getSize()
        vmean = self.mean()
        vmax = self.max()[1]
        vmin = self.min()[1]
        vsum = self.sum()
        vnorm2 = self.norm(NormType.NORM_2)
        vrms = vnorm2 / numpy.sqrt(vsize)

        return {
            "type": "scalar",
            "components": 1,
            "size": vsize,
            "mean": vmean,
            "min": vmin,
            "max": vmax,
            "sum": vsum,
            "norm2": vnorm2,
            "rms": vrms,
        }

    def _scalar_stats(self):
        """Statistics for scalar variables (original implementation)."""
        from petsc4py.PETSc import NormType

        vsize = self._gvec.getSize()
        vmean = self.mean()
        vmax = self.max()  # Now returns value directly, not tuple
        vmin = self.min()  # Now returns value directly, not tuple
        vsum = self.sum()
        vnorm2 = self.norm(NormType.NORM_2)
        vrms = vnorm2 / numpy.sqrt(vsize)

        return {
            "type": "scalar",
            "components": 1,
            "size": vsize,
            "mean": vmean,
            "min": vmin,
            "max": vmax,
            "sum": vsum,
            "norm2": vnorm2,
            "rms": vrms,
        }

    def _vector_stats(self):
        """Statistics for vector variables using magnitude."""
        import numpy as np

        # Create temporary scalar variable for magnitude
        magnitude_var = _BaseMeshVariable(f"_temp_mag_{id(self)}", self.mesh, 1, degree=self.degree)

        try:
            # Compute magnitude: |v| = sqrt(v·v)
            with uw.synchronised_array_update():
                mag_squared = 0.0
                for i in range(self.num_components):
                    component = self.array[:, 0, i].flatten()
                    mag_squared += component**2
                magnitude_var.array[:, 0, 0] = np.sqrt(mag_squared)

            # Get scalar stats on magnitude
            mag_stats = magnitude_var._scalar_stats()

            # Update with vector-specific info
            mag_stats.update(
                {
                    "type": "vector",
                    "components": self.num_components,
                    "magnitude_mean": mag_stats["mean"],
                    "magnitude_max": mag_stats["max"],
                    "magnitude_min": mag_stats["min"],
                    "magnitude_rms": mag_stats["rms"],
                }
            )

            return mag_stats

        finally:
            # Cleanup temporary variable
            if magnitude_var.name in self.mesh.vars:
                del self.mesh.vars[magnitude_var.name]

    def _tensor_stats(self):
        """Statistics for tensor variables using Frobenius norm."""
        import numpy as np

        # Create temporary scalar variable for Frobenius norm
        frobenius_var = uw.discretisation.MeshVariable(
            f"_temp_frob_{id(self)}", self.mesh, 1, degree=self.degree
        )

        try:
            # Compute Frobenius norm: ||A||_F = sqrt(sum(A_ij^2))
            with uw.synchronised_array_update():
                sum_squares = 0.0
                for i in range(self.num_components):
                    component = self.array[:, 0, i].flatten()
                    sum_squares += component**2
                frobenius_var.array[:, 0, 0] = np.sqrt(sum_squares)

            # Get scalar stats on Frobenius norm
            frob_stats = frobenius_var._scalar_stats()

            # Update with tensor-specific info
            frob_stats.update(
                {
                    "type": "tensor",
                    "components": self.num_components,
                    "frobenius_mean": frob_stats["mean"],
                    "frobenius_max": frob_stats["max"],
                    "frobenius_min": frob_stats["min"],
                    "frobenius_rms": frob_stats["rms"],
                }
            )

            return frob_stats

        finally:
            # Cleanup temporary variable
            if frobenius_var.name in self.mesh.vars:
                del self.mesh.vars[frobenius_var.name]

    @property
    def coords(self) -> numpy.ndarray:
        """
        The array of variable vertex coordinates for this variable's DOF locations.

        Returns coordinates for this variable's specific degree-of-freedom locations,
        which may differ from mesh coordinate variable locations if the degrees differ.

        When mesh has reference quantities set, returns unit-aware coordinates in meters.
        """
        # Get non-dimensional [0-1] model coordinates for this variable's specific DOF locations
        coords_nondim = self.mesh._get_coords_for_var(self)

        # If mesh has units, dimensionalise to physical coordinates
        if self.mesh.units is not None:
            import underworld3 as uw

            # Dimensionalise using the proper units system
            # Specify length dimensionality since coords have dimension [length]
            length_dimensionality = {'[length]': 1}
            coords_dimensional = uw.dimensionalise(
                coords_nondim,
                target_dimensionality=length_dimensionality
            )

            return coords_dimensional
        else:
            # No units - return non-dimensional coordinates
            return coords_nondim

    @property
    def coords_nd(self) -> numpy.ndarray:
        """
        Non-dimensional [0-1] coordinates for this variable's DOF locations.

        Returns raw model coordinates from PETSc without any unit wrapping.
        This is the coordinate system used by internal KDTree indexing, evaluation,
        and other algorithmic operations.

        For user-facing operations with physical units, use `.coords` which returns
        dimensional coordinates when the model has reference quantities set.

        Returns
        -------
        ndarray
            Non-dimensional [0-1] coordinates, shape (N, dim)

        Examples
        --------
        >>> # Internal algorithmic use - KDTree indexing
        >>> kd_tree = uw.kdtree.KDTree(var.coords_nd)
        >>>
        >>> # User-facing display with dimensional units
        >>> print(f"Positions: {var.coords}")  # Shows meters, km, etc.

        Notes
        -----
        This is a zero-copy operation that returns a view of the cached coordinate
        array directly from the mesh. No memory allocation or copying occurs.

        See Also
        --------
        coords : Dimensional coordinates with unit wrapping (user-facing)
        """
        # Direct access to non-dimensional coordinates from mesh cache
        # This is a ZERO-COPY operation - returns cached array directly
        return self.mesh._get_coords_for_var(self)

    # vector calculus routines - the advantage of using these inbuilt routines is
    # that they are tied to the appropriate mesh definition.

    def divergence(self):
        try:
            return self.mesh.vector.divergence(self.sym)
        except:
            return None

    def gradient(self):
        try:
            return self.mesh.vector.gradient(self.sym)
        except:
            return None

    def curl(self):
        try:
            return self.mesh.vector.curl(self.sym)
        except:
            return None

    def jacobian(self):
        ## validate if this is a vector ?
        return self.mesh.vector.jacobian(self.sym)


# Note: EnhancedMeshVariable is imported as MeshVariable in __init__.py to avoid circular imports
