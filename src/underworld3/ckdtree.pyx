from types import WrapperDescriptorType
import underworld3
import underworld3 as uw
import underworld3.timing as timing
import numpy
import numpy as np

from libcpp cimport bool

cdef extern from "kdtree_interface.hpp" nogil:
    cdef cppclass KDTree_Interface:
        KDTree_Interface()
        KDTree_Interface( const double* points, int numpoints, int dim )
        void build_index()
        void find_closest_point( size_t  num_coords, const double* coords, long unsigned int* indices, double* out_dist_sqr, bool* found )
        size_t knnSearch(const double* query_point, const size_t num_closest, long unsigned int* indices, double* out_dist_sqr )

cdef class KDTree:
    """
    Unit-aware KD-Tree for spatial indexing and queries.

    This class generates a kd-tree index for the provided points and provides
    the necessary methods for finding which points are closest to a given query
    location. It automatically handles coordinate units when provided.

    This class utilises `nanoflann` for kd-tree functionality.

    Parameters
    ----------
    points : array-like
        The points for which the kd-tree index will be built. This
        should be a 2-dimensional array of size (n_points, dim).
        Can be unit-aware (UnitAwareArray) or plain numpy array.

    Example
    -------
    >>> import numpy as np
    >>> import underworld3 as uw

    Generate a random set of points
    >>> pts = np.random.random( size=(100,2) )

    Build the index on the points
    >>> index = uw.kdtree.KDTree(pts)

    Search the index for a coordinate
    >>> coord = np.zeros((1,2))
    >>> coord[0] = (0.5,0.5)
    >>> indices, dist_sqr, found = index.find_closest_point(coord)

    Confirm that a point has been found
    >>> found[0]
    True

    """
    cdef KDTree_Interface* index
    cdef const double[:,::1] points
    cdef public object coord_units  # Store coordinate units

    def __cinit__( self,
                   points_input not None:   numpy.ndarray ) :

        # Check if points have units and store them
        # Import here to avoid circular imports
        import underworld3.function.unit_conversion as unit_conv
        self.coord_units = unit_conv.get_units(points_input) if unit_conv.has_units(points_input) else None

        # Extract raw numpy array for C++ interface
        cdef const double[:,::1] points
        if unit_conv.has_units(points_input):
            points = np.ascontiguousarray(points_input, dtype=np.float64)
        else:
            points = points_input

        if points.shape[1] not in (2,3):
            raise RuntimeError(f"Provided points array dimensionality must be 2 or 3, not {points.shape[1]}.")
        self.points = points
        self.index = new KDTree_Interface(<const double *> &points[0][0], points.shape[0], points.shape[1])

        super().__init__()

    def __dealloc__(self):
        del self.index

    @property
    def n(self):
        return self.points.shape[0]

    @property
    def ndim(self):
        return self.points.shape[1]

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
            return np.asarray(coords, dtype=np.float64)

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
            return np.asarray(coords, dtype=np.float64)

        # Different units - convert to tree's coordinate system
        try:
            # Use UnitAwareArray's to_units method if available
            if hasattr(coords, 'to_units'):
                coords_converted = coords.to_units(self.coord_units)
                return np.asarray(coords_converted, dtype=np.float64)
            else:
                # Convert using Pint directly
                ureg = underworld3.scaling.units
                coords_qty = ureg.Quantity(np.asarray(coords), query_units)
                coords_converted_qty = coords_qty.to(self.coord_units)
                return np.asarray(coords_converted_qty.magnitude, dtype=np.float64)

        except Exception as e:
            raise ValueError(
                f"Cannot convert query coordinates from '{query_units}' "
                f"to KD-tree's coordinate system '{self.coord_units}': {e}"
            )


    @timing.routine_timer_decorator
    def build_index(self):
        """
        Build the kd-tree index.
        """
        self.index.build_index()

    def kdtree_points(self):
        """
        Returns a view of the points used to define the kd-tree
        """

        return np.array(self.points)


    @timing.routine_timer_decorator
    def find_closest_point(self,
                          const double[:,::1] coords not None:   numpy.ndarray):
        """
        Find the points closest to the provided set of coordinates.

        Parameters
        ----------
        coords:
            An array of coordinates for which the kd-tree index will be searched for nearest
            neighbours. This should be a 2-dimensional array of size (n_coords,dim).

        Returns
        -------
        indices:
            An integer array of indices into the `points` array (passed into the constructor) corresponding to
            the nearest neighbour for the search coordinates. It will be of size (n_coords).
        dist_sqr:
            A float array of squared distances between the provided coords and the nearest neighbouring
            points. It will be of size (n_coords).
        found:
            A bool array of flags which signals whether a nearest neighbour has been found for a given
            coordinate. It will be of size (n_coords).



        """
        if coords.shape[1] != self.points.shape[1]:
            raise RuntimeError(f"Provided coords array dimensionality ({coords.shape[1]}) is different to points dimensionality ({self.points.shape[1]}).")

        count = coords.shape[0]
        indices  = np.empty(count, dtype=np.uint64,  order='C')
        dist_sqr = np.empty(count, dtype=np.float64, order='C')
        found    = np.empty(count, dtype=np.bool_,   order='C')

        cdef long unsigned int[::1]  c_indices = indices
        cdef            double[::1] c_dist_sqr = dist_sqr
        cdef              bool[::1]    c_found = found
        self.index.find_closest_point(count,
                                    <    const double *> &coords[0][0],
                                    <long unsigned int*> &c_indices[0],
                                    <           double*> &c_dist_sqr[0],
                                    <             bool*> &c_found[0] )
        return indices, dist_sqr, found

    @timing.routine_timer_decorator
    def find_closest_n_points(self,
                  const int nCount                    :   numpy.int64,
                  const double[: ,::1] coords not None:   numpy.ndarray):
        """
        Find the n points closest to the provided coordinates.

        Parameters
        ----------
        nCount:
            The number of nearest neighbour points to find for each `coords`.

        coords:
            Coordinates of the points for which the kd-tree index will be searched for nearest
            neighbours. This should be a 2-dimensional array of size (n_coords,dim).

        Returns
        -------
        indices:
            An integer array of indices into the `points` array (passed into the constructor) corresponding to
            the nearest neighbour for the search coordinates. It will be of size (n_coords).
        dist_sqr:
            A float array of squared distances between the provided coords and the nearest neighbouring
            points. It will be of size (n_coords).

        """

        if coords.shape[1] != self.points.shape[1]:
            raise RuntimeError(f"Provided coords array dimensionality ({coords.shape[1]}) is different to points dimensionality ({self.points.shape[1]}).")
        nInput = coords.shape[0]

        # allocate numpy arrays -

        n_indices  = np.empty((coords.shape[0], nCount), dtype=np.uint64,  order='C')
        n_dist_sqr = np.empty((coords.shape[0], nCount), dtype=np.float64,  order='C')

        indices  = np.empty(nCount, dtype=np.uint64,  order='C')
        dist_sqr = np.empty(nCount, dtype=np.float64, order='C')

        # allocate memoryviews in C contiguous layout
        cdef long unsigned int[::1] c_indices  = indices
        cdef            double[::1] c_dist_sqr = dist_sqr

        # Build the array one point at a time

        for p in range(coords.shape[0]):
            self.index.knnSearch( <const double *> &coords[p][0],
                                nCount,
                                <long unsigned int*> &c_indices[0],
                                <           double*> &c_dist_sqr[0])

            n_indices[p,:] = indices[:]
            n_dist_sqr[p,:] = dist_sqr[:]

        # return numpy data
        return n_indices, n_dist_sqr


    @timing.routine_timer_decorator
    def query(self,
             coords,
             k=1,
             sqr_dists=True,
    ):
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

        i, d = self.find_closest_n_points(k, coords_contiguous)

        # For consistency with pykdtree
        if k==1:
            i = i.reshape(-1)

        if sqr_dists:
            return d, i
        else:
            distance_actual = numpy.sqrt(d)
            # Wrap with unit-aware array if tree has coordinate units
            if self.coord_units is not None:
                from underworld3.utilities.unit_aware_array import UnitAwareArray
                distance_actual = UnitAwareArray(distance_actual, units=self.coord_units)
            return distance_actual, i


## A general point-to-point rbf interpolator here
## NOTE this is not using cython optimisation for numpy

    # For backward compatibility, default for the rbf_interpolator function
    # is the _from_kdtree version

    def rbf_interpolator_local(self,
            coords,
            data,
            nnn = 4,
            p=2,
            verbose = False,
        ):

        return self.rbf_interpolator_local_from_kdtree(
            coords, data, nnn, p, verbose,
        )

    def old_rbf_interpolator_local_from_kdtree(self,
            coords,
            data,
            nnn = 4,
            verbose = False,
        ):

        '''
        An inverse (squared) distance weighted mapping of a numpy array from the
        set of coordinates defined by the kd-tree to the set of input points specified.
        This assumes all points are local to the same processor.
        If that is not the case, it is best to use a particle swarm
        to manage the distributed data.
        '''

        if coords.shape[1] != self.points.shape[1]:
            raise RuntimeError(f"Interpolation coordinates dimensionality ({coords.shape[1]}) is different to kD-tree dimensionality ({self.points.shape[1]}).")
        nInput = coords.shape[0]

        if data.shape[0] != self.points.shape[0]:
                raise RuntimeError(f"Data does not match kD-tree size array ({data.shape[0]}) v ({self.points.shape[0]}).")

        coords_contiguous = np.ascontiguousarray(coords)

        closest_n, distance_n = self.find_closest_n_points(nnn, coords_contiguous)

        num_local_points = coords.shape[0]
        try:
            data_size = data.shape[1]
        except IndexError:
            data_size = 1
            data = data.reshape(-1,1)
        Values = np.zeros((num_local_points, data_size))
        Weights = np.zeros((num_local_points, 1))

        if verbose and uw.mpi.rank == 0:
            print("Mapping values  ... start", flush=True)

        epsilon = 1.0e-9
        for j in range(nnn):
            j_distance = epsilon + np.sqrt(distance_n[:, j])
            Weights[:, 0] += 1.0 / j_distance[:]

        for d in range(data_size):
            for j in range(nnn):
                j_distance = epsilon + np.sqrt(distance_n[:, j])
                j_nearest = closest_n[:, j]
                Values[:, d] += data[j_nearest, d] / j_distance

        Values[...] /= Weights[:]

        if verbose and uw.mpi.rank == 0:
            print("Mapping values ... done", flush=True)

        del coords_contiguous
        del closest_n
        del distance_n
        del Weights

        return Values

    def old_rbf_interpolator_local_to_kdtree(self,
                    coords,
                    data,
                    nnn = 4,
                    verbose = False,
                    weights = None
                ):

        '''
        An inverse (squared) distance weighted mapping of a numpy array to the
        set of coordinates defined by the kd-tree from the set of input points specified.
        This assumes all points are local to the same processor.
        If that is not the case, it is sensible to use a particle swarm
        to manage the distributed data.
        '''

        if coords.shape[1] != self.points.shape[1]:
            raise RuntimeError(f"Interpolation coordinates dimensionality ({coords.shape[1]}) is different to kD-tree dimensionality ({self.points.shape[1]}).")
        nInput = coords.shape[0]

        if data.shape[0] != coords.shape[0]:
                raise RuntimeError(f"Data does not match coords size array ({data.shape[0]}) v ({coords.shape[0]}).")

        coords_contiguous = np.ascontiguousarray(coords)

        closest_n, distance_n = self.find_closest_n_points(nnn, coords_contiguous)

        num_local_points = self.points.shape[0]
        try:
            data_size = data.shape[1]
        except IndexError:
            data_size = 1
            data = data.reshape(-1,1)

        Values = np.zeros((num_local_points, data_size))
        Weights = np.zeros((num_local_points, 1))

        if verbose and uw.mpi.rank == 0:
            print(f"Mapping values  ... start", flush=True)

        epsilon = 1.0e-9
        for j in range(nnn):
            j_distance = epsilon + np.sqrt(distance_n[:, j])
            Weights[closest_n[:,j], 0] += 1.0 / j_distance[:]

        for d in range(data_size):
            for j in range(nnn):
                j_distance = epsilon + np.sqrt(distance_n[:, j])
                j_nearest = closest_n[:, j]
                Values[j_nearest, d] += data[:, d] / j_distance

        # In this case, weights may be zero
        Values[Weights!=0] /= Weights[Weights!=0]

        if verbose and uw.mpi.rank == 0:
            print("Mapping values ... done", flush=True)

        if isinstance(weights, np.ndarray):
                    weights[...] = Weights[...]

        del coords_contiguous
        del closest_n
        del distance_n
        del Weights

        return Values


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

        coords_contiguous = np.ascontiguousarray(coords_converted)
        # query nnn points to the coords
        # distance_n is a list of distance to the nearest neighbours for all coords_contiguous
        # closest_n is the index of the neighbours from ncoords for all coords_contiguous
        # Note: query() returns sqr_dists=True by default, and we use the converted coords
        distance_n, closest_n = self.query(coords, k=nnn)

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

        # can decompose weighting vecotrs as IDW is a linear relationship
        # build normalise weight vectors and multiply that with known data
        epsilon = 1e-12
        weights = 1 / np.power(epsilon + distance_n[:], p)
        n_weights = (weights.T / np.sum(weights, axis=1)).T
        kdata = data[closest_n[:]]

        # magic with einstein summation power
        vals = np.einsum("sdc,sd->sc", kdata, n_weights)
        # print(valz)

        if verbose and uw.mpi.rank == 0:
            print(f"Mapping values  ... finished", flush=True)

        return vals
