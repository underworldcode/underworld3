from pykdtree.kdtree import KDTree as _oKDTree
import underworld3 as uw
import numpy as np


## Note we are missing the function rbf_interpolator_local_to_kdtree
#
# Should be used in this swarm proxy function instead of hand-writing the code:
#
#    def _rbf_reduce_to_meshVar(self, meshVar, verbose=False):
#    """
#    This method updates a mesh variable for the current
#    swarm & particle variable state by reducing the swarm to
#    the nearest point for each particle


# inherit from the pykdtree
class KDTree(_oKDTree):
    """
    Unit-aware KDTree for spatial indexing and queries.

    This class extends pykdtree.KDTree to handle coordinate units properly:
    - Stores coordinate units when constructed from unit-aware data
    - Automatically converts query coordinates to match the KD-tree's units
    - Returns distances with appropriate units (when not squared)
    - Compatible with both unit-aware and plain numpy arrays

    Usage:
        # Create KD-tree from mesh with units
        mesh = uw.meshing.StructuredQuadBox(..., units="km")
        kd = uw.kdtree.KDTree(mesh.points)

        # Query with coordinates (units will be converted if needed)
        query_pts = np.array([[100.0, 50.0]])  # Can have units or not
        distances, indices = kd.query(query_pts)
    """

    def __init__(self, data, leafsize=16, **kwargs):
        """
        Construct KD-tree from coordinate data.

        Parameters
        ----------
        data : array-like
            Coordinate data to build tree from. Can be unit-aware (UnitAwareArray)
            or plain numpy array. Shape should be (n_points, n_dimensions).
        leafsize : int, optional
            Number of points at which to switch to brute-force (default 16)
        **kwargs : dict
            Additional arguments passed to pykdtree.KDTree
        """
        # Import here to avoid circular imports
        import underworld3.function.unit_conversion as unit_conv

        # Check if data has units and store them
        self.coord_units = unit_conv.get_units(data) if unit_conv.has_units(data) else None

        # Extract raw numpy array for pykdtree (it doesn't understand units)
        if unit_conv.has_units(data):
            # Get underlying numpy array without unit wrapping
            data_raw = np.asarray(data)
        else:
            data_raw = data

        # Call parent constructor with raw data
        super().__init__(data_raw, leafsize=leafsize, **kwargs)

    def _convert_coords_to_tree_units(self, coords):
        """
        Convert query coordinates to match the KD-tree's coordinate system.

        Parameters
        ----------
        coords : array-like
            Query coordinates (may or may not have units)

        Returns
        -------
        np.ndarray
            Coordinates converted to tree's coordinate system (raw numpy array)
        """
        import underworld3.function.unit_conversion as unit_conv
        import underworld3.scaling

        # If tree has no units, just extract raw array
        if self.coord_units is None:
            if unit_conv.has_units(coords):
                raise ValueError(
                    f"KD-tree was built with dimensionless coordinates, "
                    f"but query coordinates have units '{unit_conv.get_units(coords)}'. "
                    f"Convert to dimensionless first."
                )
            return np.asarray(coords)

        # Tree has units - check query coordinates
        if not unit_conv.has_units(coords):
            raise ValueError(
                f"KD-tree was built with coordinates in '{self.coord_units}', "
                f"but query coordinates have no units. "
                f"Provide coordinates with units or convert tree coordinates to dimensionless."
            )

        query_units = unit_conv.get_units(coords)

        # Same units - just extract raw array
        if query_units == self.coord_units:
            return np.asarray(coords)

        # Different units - convert to tree's coordinate system
        try:
            # Use UnitAwareArray's to method if available
            if hasattr(coords, "to"):
                coords_converted = coords.to(self.coord_units)
                return np.asarray(coords_converted)
            else:
                # Convert using Pint directly
                ureg = underworld3.scaling.units
                coords_qty = ureg.Quantity(np.asarray(coords), query_units)
                coords_converted_qty = coords_qty.to(self.coord_units)
                return coords_converted_qty.magnitude

        except Exception as e:
            raise ValueError(
                f"Cannot convert query coordinates from '{query_units}' "
                f"to KD-tree's coordinate system '{self.coord_units}': {e}"
            )

    def rbf_interpolator_local(
        self,
        coords,
        data,
        nnn=4,
        p=2,
        verbose=False,
    ):

        return self.rbf_interpolator_local_from_kdtree(
            coords,
            data,
            nnn,
            p,
            verbose,
        )

    #   def find_closest_points( self, coords, nnn )

    # NOTE: Override query() method from pykdtree so it has the same interface as ckdtree.
    # another option is to eliminate the sqr_dist argument from both ckdtree and pykdtree.
    def query(self, coords, k=1, sqr_dists=True):
        """
        Find the n points closest to the provided coordinates.

        This method is unit-aware: if the KD-tree was built with unit-aware coordinates,
        it will automatically convert query coordinates to match and return distances
        with appropriate units.

        Parameters
        ----------
        coords : array-like
            An array of coordinates for which the kd-tree index will be searched for nearest
            neighbours. This should be a 2-dimensional array of size (n_coords, dim).
            Can be unit-aware (UnitAwareArray) or plain numpy array.
            If KD-tree has coordinate units, coords must have compatible units.
        k : int, optional
            The number of nearest neighbour points to find for each `coords` (default 1).
        sqr_dists : bool, optional
            Set to True to return the squared distances, set to False to return the actual
            distances (default True).

        Returns
        -------
        d : array
            A float array of the squared (sqr_dists = True) or actual distances (sqr_dists = False)
            between the provided coords and the nearest neighbouring points.
            If KD-tree has coordinate units and sqr_dists=False, distances will be unit-aware.
            Shape is (n_coords,) for k=1, or (n_coords, k) for k>1.
        i : array
            An integer array of indices into the `points` array (passed into the constructor)
            corresponding to the nearest neighbour for the search coordinates.
            Shape is (n_coords,) for k=1, or (n_coords, k) for k>1.
        """
        # Convert coordinates to match tree's coordinate system
        coords_converted = self._convert_coords_to_tree_units(coords)
        coords_contiguous = np.ascontiguousarray(coords_converted)

        # Query parent KD-tree (returns actual distances, not squared)
        distance_k, closest_k = super().query(query_pts=coords_contiguous, k=k)

        # Handle distance units
        if sqr_dists:
            # Squared distances - dimensionless or have squared units
            # For now, return as plain array (squared units are complex to handle)
            return distance_k**2, closest_k
        else:
            # Actual distances - should have same units as coordinates
            if self.coord_units is not None:
                # Wrap with unit-aware array
                from underworld3.utilities.unit_aware_array import UnitAwareArray

                distance_k = UnitAwareArray(distance_k, units=self.coord_units)
            return distance_k, closest_k

    def rbf_interpolator_local_from_kdtree(self, coords, data, nnn, p, verbose):
        """
        Performs an inverse distance (squared) mapping of data to the target `coords`.

        This method is unit-aware: if the KD-tree was built with unit-aware coordinates,
        it will automatically convert query coordinates to match before interpolation.

        Parameters
        ----------
        coords : array-like
            The target spatial coordinates to evaluate the data from.
            Can be unit-aware (UnitAwareArray) or plain numpy array.
            If KD-tree has coordinate units, coords must have compatible units.
            coords.shape[1] == self.ndim
        data : ndarray
            The known data to map from. Must be fully described over kd-tree.
            i.e., data.shape[0] == self.n
        nnn : int
            The number of neighbour points to sample from. If `1`, no distance averaging is done.
        p : int
            The power index to calculate weights, i.e., pow(distance, -p)
        verbose : bool
            Print when mapping occurs

        Returns
        -------
        ndarray
            Interpolated data values at target coordinates
        """
        # Convert coordinates to match tree's coordinate system
        coords_converted = self._convert_coords_to_tree_units(coords)

        if coords_converted.shape[1] != self.ndim:
            raise RuntimeError(
                f"Interpolation coordinates dimensionality ({coords_converted.shape[1]}) is different to kD-tree dimensionality ({self.ndim})."
            )
        if data.shape[0] != self.n:
            raise RuntimeError(
                f"Data does not match kd-tree size array ({data.shape[0]} v ({self.n}))"
            )

        # query nnn points to the coords using wrapped query function
        # distance_n is a list of distance to the nearest neighbours for all coords_contiguous
        # closest_n is the index of the neighbours from ncoords for all coords_contiguous
        # Note: We use the converted coordinates here, and query() will handle them properly
        distance_n, closest_n = self.query(coords, k=nnn, sqr_dists=False)

        if np.any(closest_n > self.n):
            raise RuntimeError(
                "Error in rbf_interpolator_local_from_kdtree - a nearest neighbour wasn't found"
            )

        if verbose and uw.mpi.rank == 0:
            # For Debugging
            # print(f"kd-tree diagnostics: d.shape - {distance_n.shape}, c.shape - {closest_n.shape}")
            print(f"Mapping values with nnn - {nnn} & p {p}  ... start", flush=True)

        if nnn == 1:
            # only use nearest neighbour raw data
            return data[closest_n]

        # Extract raw distance values (in case distance_n is unit-aware)
        distance_values = np.asarray(distance_n)

        # can decompose weighting vectors as IDW is a linear relationship
        # build normalise weight vectors and multiply that with known data
        epsilon = 1e-12
        weights = 1 / np.power(epsilon + distance_values[:], p)
        n_weights = (weights.T / np.sum(weights, axis=1)).T
        kdata = data[closest_n[:]]

        # magic with einstein summation power
        vals = np.einsum("sdc,sd->sc", kdata, n_weights)
        # print(valz)

        if verbose and uw.mpi.rank == 0:
            print(f"Mapping values  ... finished", flush=True)

        return vals


## NB the rbf interpolator TO kdtree is missing (and we need that one that we introduced to do a better job of mapping values from swarms to nodes for proxy variables)
