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
from underworld3.utilities.mathematical_mixin import MathematicalMixin

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


def MeshVariable(
    varname: Union[str, list],
    mesh: "Mesh",
    num_components: Union[int, tuple] = None,
    vtype: Optional["uw.VarType"] = None,
    degree: int = 1,
    continuous: bool = True,
    varsymbol: Union[str, list] = None,
):
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

    if isinstance(varname, list):
        name = varname[0] + R"+ \dots"
    else:
        name = varname

    ## Smash if already defined (we need to check this BEFORE the old meshVariable object is destroyed)

    import re

    clean_name = re.sub(r"[^a-zA-Z0-9_]", "", name)

    if clean_name in mesh.vars.keys():
        print(f"Variable with name {name} already exists on the mesh - Skipping.")
        return mesh.vars[clean_name]

    if mesh._dm_initialized:
        ## Before adding a new variable, we first snapshot the data from the mesh.dm
        ## (if not accessed, then this will not be necessary and may break)

        mesh.update_lvec()

        old_gvec = mesh.dm.getGlobalVec()
        mesh.dm.localToGlobal(mesh._lvec, old_gvec, addv=False)

    new_meshVariable = _MeshVariable(
        name, mesh, num_components, vtype, degree, continuous, varsymbol
    )

    if mesh._dm_initialized:
        ## Recreate the mesh variable dm and restore the data

        dm0 = mesh.dm
        dm1 = mesh.dm.clone()
        dm0.copyFields(dm1)
        dm1.createDS()

        mdm_is, subdm = dm1.createSubDM(range(0, dm1.getNumFields() - 1))

        if mesh._lvec is not None:
            mesh._lvec.destroy()
        mesh._lvec = dm1.createLocalVec()
        new_gvec = dm1.getGlobalVec()
        new_gvec_sub = new_gvec.getSubVector(mdm_is)

        # Copy the array data and push to gvec
        new_gvec_sub.array[...] = old_gvec.array[...]
        new_gvec.restoreSubVector(mdm_is, new_gvec_sub)

        # Copy the data to mesh._lvec and delete gvec
        dm1.globalToLocal(new_gvec, mesh._lvec)

        dm1.restoreGlobalVec(new_gvec)
        dm0.restoreGlobalVec(old_gvec)

        # destroy old dm
        dm0.destroy()

        # Set new dm on mesh
        mesh.dm = dm1
        mesh.dm_hierarchy[-1] = dm1

    return new_meshVariable


class _MeshVariable(MathematicalMixin, Stateful, uw_object):
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

    @timing.routine_timer_decorator
    def __init__(
        self,
        varname: Union[str, list],
        mesh: "underworld.mesh.Mesh",
        size: Union[int, tuple],
        vtype: Optional["underworld.VarType"] = None,
        degree: int = 1,
        continuous: bool = True,
        varsymbol: Union[str, list] = None,
    ):
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
        continuous:
            True for continuous element discretisation across element boundaries.
            False for discontinuous values across element boundaries.
        varsymbol :
            Over-ride the varname with a symbolic form for printing etc (latex). Should be an R-string.
        """

        import re
        import math

        # if varsymbol is None and not isinstance(varname, list):
        #     varsymbol = "{" + repr(varname)[1:-1] + "}"

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
        self._available = (
            True  # Make vectors available by default for solver compatibility
        )

        ## Note sympy needs a unique symbol even across different meshes
        ## or it will get confused when it clones objects. We try this: add
        ## a label to the variable that is not rendered - CHECK this works !!!

        self.name = name
        self.symbol = symbol

        if mesh.instance_number > 1:
            invisible = rf"\hspace{{ {mesh.instance_number/100}pt }}"
            self.symbol = f"{{ {invisible} {symbol} }}"

        self.clean_name = re.sub(r"[^a-zA-Z0-9_]", "", name)

        # ToDo: Suggest we deprecate this and require it to be set explicitly
        # The tensor types are hard to infer correctly

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

        if not isinstance(vtype, uw.VarType):
            raise ValueError(
                "'vtype' must be an instance of 'Variable_Type', for example `underworld.VarType.SCALAR`."
            )

        self.vtype = vtype
        self._mesh_ref = weakref.ref(mesh)
        self.shape = size
        self.degree = degree
        self.continuous = continuous

        # First create the petsc FE object of the
        # correct size / dimension to represent the
        # unknowns when used in computations (for tensors)
        # we will need to pack them correctly as well
        # (e.g. T.sym.reshape(1,len(T.sym))))
        # Symmetric tensors ... a bit more work again

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

        # This allows us to define a __getitem__ method
        # to return a view for a given component when
        # the access manager is active

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

        # Setup public view of data - using NDArray_With_Callback (following mesh.points pattern)
        # Use lazy initialization to avoid calling unpack during constructor
        self._array_cache = None  # Will be created lazily when first accessed
        self._data_cache = None  # Will be created lazily when first accessed

        return

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

    def rbf_interpolate(
        self, new_coords, meth=0, p=2, verbose=False, nnn=None, rubbish=None
    ):
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

        mesh_kdt = uw.kdtree.KDTree(self.coords)
        values = mesh_kdt.rbf_interpolator_local(
            new_coords, D, nnn, p=p, verbose=verbose
        )
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

        lvec = self.mesh.dm.getCoordinates()

    # ToDo: rename to vertex_checkpoint (or similar)
    @timing.routine_timer_decorator
    def write(
        self,
        filename: str,
    ):
        """
        Write variable data to the specified mesh hdf5
        data file. The file will be over-written.

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
                    raise IndexError(
                        f"Vectors have shape {self.mesh.dim} or {(1, self.mesh.dim)} "
                    )
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
            raise RuntimeError(
                "Vector must be accessed via the mesh `access()` context manager."
            )

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
            raise RuntimeError(
                "Data must be accessed via the mesh `access()` context manager."
            )
        return self._data

    @property
    def array(self):
        """
        Access variable data as NDArray_With_Callback.
        Follows the simple mesh.points pattern - returns cached object with lazy creation.
        """
        # Lazy creation: create array cache on first access
        if self._array_cache is None:
            self._array_cache = self._create_variable_array()
        return self._array_cache

    @property
    def data(self):
        """
        Backward-compatible data property that returns flat array shape (-1, num_components).
        This property provides the legacy interface for compatibility with existing code.

        Returns
        -------
        NDArray_With_Callback
            Array with shape (-1, num_components) with automatic PETSc synchronization

        Notes
        -----
        This interface is deprecated. Use the `array` property instead for the new
        interface with proper tensor shape (N, a, b).
        """
        # Check if we have a cached data array
        if hasattr(self, "_data_cache") and self._data_cache is not None:
            return self._data_cache

        # For symmetric tensors, we need to access PETSc data directly in packed format
        # rather than reshaping the array (which has different component count)

        # Ensure PETSc vector is available
        self._set_vec(available=True)
        self.mesh._dm_initialized = True

        # Get direct access to PETSc vector in packed format
        flat_petsc_data = self.vec.array.reshape(-1, self.num_components)

        # Create NDArray_With_Callback with proper shape and data
        from underworld3.utilities import NDArray_With_Callback

        flat_view = NDArray_With_Callback(flat_petsc_data)

        def flat_data_update_callback(array, change_context):
            """Callback to sync flat data changes back to PETSc using raw pack method"""
            # Only act on data-changing operations
            data_changed = change_context.get("data_has_changed", True)
            if not data_changed:
                return

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

        flat_view.add_callback(flat_data_update_callback)

        # Cache the data array for future access
        self._data_cache = flat_view
        return flat_view

    @array.setter
    def array(self, array_value):
        """
        Set variable data using pack method to handle shape transformation.
        """
        # Use pack method to handle proper data transformation and shape conversion
        self.pack_uw_data_to_petsc(array_value, sync=True)

    ## ToDo: We should probably deprecate this in favour of using integrals

    def min(self) -> Union[float, tuple]:
        """
        The global variable minimum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            return self._gvec.min()
        else:
            return tuple(
                [self._gvec.strideMin(i)[1] for i in range(self.num_components)]
            )

    def max(self) -> Union[float, tuple]:
        """
        The global variable maximum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            return self._gvec.max()
        else:
            return tuple(
                [self._gvec.strideMax(i)[1] for i in range(self.num_components)]
            )

    def sum(self) -> Union[float, tuple]:
        """
        The global variable sum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            return self._gvec.sum()
        else:
            cpts = []
            for i in range(0, self.num_components):
                cpts.append(self._gvec.strideSum(i))

            return tuple(cpts)

    def norm(self, norm_type) -> Union[float, tuple]:
        """
        The global variable norm value.

        norm_type: type of norm, one of
            - 0: NORM 1 ||v|| = sum_i | v_i |. ||A|| = max_j || v_*j ||
            - 1: NORM 2 ||v|| = sqrt(sum_i |v_i|^2) (vectors only)
            - 3: NORM INFINITY ||v|| = max_i |v_i|. ||A|| = max_i || v_i* ||, maximum row sum
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components > 1 and norm_type == 2:
            raise RuntimeError("Norm 2 is only available for vectors.")

        if self.num_components == 1:
            return self._gvec.norm(norm_type)
        else:
            return tuple(
                [
                    self._gvec.strideNorm(i, norm_type)
                    for i in range(self.num_components)
                ]
            )

    def mean(self) -> Union[float, tuple]:
        """
        The global variable mean value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            vecsize = self._gvec.getSize()
            return self._gvec.sum() / vecsize
        else:
            vecsize = self._gvec.getSize() / self.num_components
            return tuple(
                [self._gvec.strideSum(i) / vecsize for i in range(self.num_components)]
            )

    def stats(self):
        """
        The equivalent of mesh.stats but using the native coordinates for this variable
        Not set up for vector variables so we just skip that for now.

        Returns various norms on the mesh using the native mesh discretisation for this
        variable. It is a wrapper on the various _gvec stats routines for the variable.

          - size
          - mean
          - min
          - max
          - sum
          - L2 norm
          - rms
        """

        if self.num_components > 1:
            raise NotImplementedError(
                "stats not available for multi-component variables"
            )

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

        return vsize, vmean, vmin, vmax, vsum, vnorm2, vrms

    @property
    def coords(self) -> numpy.ndarray:
        """
        The array of variable vertex coordinates.
        """
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
