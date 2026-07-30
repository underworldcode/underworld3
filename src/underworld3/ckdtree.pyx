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

# Module-level live-instance counter for memory introspection.
# Incremented in __cinit__, decremented in __dealloc__. CPython refcounting
# calls __dealloc__ promptly when the refcount hits zero, so the count is
# accurate for typical use; it can lag if a KDTree ends up in a reference
# cycle that only the cyclic garbage collector can break — call
# gc.collect() before reading if that matters. Read via
# uw.utilities.memprobe.snapshot() or directly via uw.kdtree.live_count().
cdef long _live_instances = 0
cdef long _total_constructed = 0

def live_count():
    """Number of KDTree instances currently alive on this rank."""
    return _live_instances

def total_constructed():
    """Total KDTree instances ever constructed on this rank."""
    return _total_constructed


def _normalise_monotone(monotone):
    """Resolve the ``monotone`` argument to a bool, using UW3's one vocabulary.

    The canonical spelling lives with the evaluator
    (``function.functions_unit_system._normalize_monotone``) and is reused here
    so there is a single definition of what the word accepts. Only the
    ``"clamp"`` mode has meaning for a local stencil: ``"pick"`` re-evaluates
    out-of-bounds points through the FE path, which a kd-tree knows nothing
    about.
    """
    # Early out before the import: the overwhelmingly common call has no
    # limiter, and ckdtree is a low-level module that should not take a
    # dependency on the evaluator just to be told "no".
    if monotone is False or monotone is None:
        return False

    from underworld3.function.functions_unit_system import _normalize_monotone

    mode = _normalize_monotone(monotone)
    if mode == "pick":
        raise ValueError(
            "monotone='pick' has no meaning for a local kd-tree stencil — it "
            "re-evaluates out-of-bounds points through the finite-element path. "
            "Use monotone='clamp' here, or uw.function.evaluate(..., monotone='pick')."
        )
    return mode == "clamp"


cdef class KDTree:
    """
    Unit-aware KD-Tree for spatial indexing and queries.

    This class generates a kd-tree index for the provided points and provides
    the necessary methods for finding which points are closest to a given query
    location. It automatically handles coordinate units when provided.

    This class utilises `nanoflann` for kd-tree functionality.

    .. note::
        The vendored ``nanoflann.hpp`` is version **1.3.2** (2021).
        Upstream nanoflann is at **1.9.0** as of 2026-02. Consider updating
        for ~20% performance improvement on small point clouds and accumulated
        bug fixes. See https://github.com/jlblancoc/nanoflann/releases
        See planning file: underworld.md (Nice to Have, 2026-02-13)

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

        # A legible error beats the memoryview's "Buffer has wrong number
        # of dimensions" — the common trigger is numpy.array([]) from an
        # empty point list, which is 1-D (issue #399).
        if points_input.ndim != 2:
            raise RuntimeError(
                f"KDTree points must be a 2-D (n_points, dim) array, "
                f"got shape {points_input.shape}. (An empty point set must "
                f"still be shaped (0, dim).)"
            )

        # Extract raw numpy array for C++ interface
        cdef const double[:,::1] points
        if unit_conv.has_units(points_input):
            points = np.ascontiguousarray(points_input, dtype=np.float64)
        else:
            points = points_input

        if points.shape[1] not in (2,3):
            raise RuntimeError(f"Provided points array dimensionality must be 2 or 3, not {points.shape[1]}.")
        self.points = points
        # An empty point cloud is legitimate — a parallel rank can own zero
        # cells (issue #399). No C++ index is built (taking &points[0][0]
        # would be invalid); the query methods return honest empties.
        if points.shape[0] > 0:
            self.index = new KDTree_Interface(<const double *> &points[0][0], points.shape[0], points.shape[1])
        else:
            self.index = NULL

        global _live_instances, _total_constructed
        _live_instances += 1
        _total_constructed += 1

        super().__init__()

    def __dealloc__(self):
        del self.index
        global _live_instances
        _live_instances -= 1

    @property
    def n(self):
        """Number of points in the KD-tree."""
        return self.points.shape[0]

    @property
    def ndim(self):
        """Spatial dimensionality of the KD-tree (2 or 3)."""
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
            # Use UnitAwareArray's to method if available
            if hasattr(coords, 'to'):
                coords_converted = coords.to(self.coord_units)
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
        if self.index is NULL:
            return
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

        # Empty index (a rank owning zero points, issue #399): nothing can
        # be found — report that honestly for every query.
        if self.index is NULL or count == 0:
            indices[...] = 0
            dist_sqr[...] = np.inf
            found[...] = False
            return indices, dist_sqr, found

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

        # Empty index (a rank owning zero points, issue #399): no
        # neighbours exist — infinite distances, sentinel indices.
        if self.index is NULL or nInput == 0:
            n_indices[...] = 0
            n_dist_sqr[...] = np.inf
            return n_indices, n_dist_sqr

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
            d = d.reshape(-1)

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
            nnn = None,
            p=1,
            verbose = False,
            order = 0,
            monotone = False,
        ):
        """
        Interpolate data from the KD-tree points to arbitrary target coordinates.

        Two local schemes are available, selected by ``order`` — the highest
        degree of polynomial the weights reproduce exactly:

        ``order=0`` (default)
            Inverse-distance (Shepard) weighting. Weights are positive and sum
            to one, so a **constant** field is reproduced exactly and the result
            is a convex combination of the neighbouring values — bounded, but a
            field with a gradient is smeared and the error does not vanish as
            the stencil tightens.

        ``order=1``
            Polyharmonic RBF with an affine tail, so **constant and linear**
            fields are reproduced exactly (:math:`\\sum_j w_j = 1` and
            :math:`\\sum_j w_j x_j = x^*`). Weights may be negative, so the
            result can overshoot the neighbouring values; see ``monotone``.

        Both are local: ``nnn`` non-zero weights per target point.

        Parameters
        ----------
        coords : array-like
            Target coordinates where data will be interpolated.
            Shape should be ``(n_coords, dim)``.
        data : ndarray
            Known data values at KD-tree points.
            Shape should be ``(n_points,)`` or ``(n_points, n_components)``.
        nnn : int, optional
            Number of nearest neighbours to use. Defaults to 4 for ``order=0``
            and ``2 * (dim + 1)`` for ``order=1`` — the default has to depend
            on both, because ``order=1`` requires ``nnn >= dim + 2`` and a
            fixed default of 4 would raise in 3D. If 1, returns raw
            nearest-neighbour values without distance weighting.
        p : int, optional
            Power index for distance weighting: ``weight = 1/distance^p``
            (default 1, i.e. inverse distance). Used by ``order=0`` only.
        verbose : bool, optional
            Print progress messages (default False).
        order : int, optional
            Polynomial reproduction order, 0 (default) or 1.
        monotone : bool or str, optional
            ``False`` (default) or ``True`` / ``"clamp"``. Limits the
            **non-affine part** of the interpolant: the local least-squares
            affine trend is preserved exactly, and only the RBF correction on
            top of it is bounded by the correction actually present in the
            stencil. This is the slope-limiter discipline — the linear
            reconstruction is never clipped — so linear reproduction survives
            the limiter and it is a no-op on any field the scheme already
            reproduces exactly. It bounds new oscillation, not absolute range.

        Returns
        -------
        ndarray
            Interpolated data values at target coordinates.

        See Also
        --------
        rbf_interpolator_local_from_kdtree : The underlying implementation.
        query : Find nearest neighbours without interpolation.
        """
        return self.rbf_interpolator_local_from_kdtree(
            coords, data, nnn, p, verbose, order, monotone,
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


    def _resolve_nnn(self, nnn, order):
        """Default stencil size. ``order=1`` needs at least ``dim + 2``, so a
        fixed default cannot serve both schemes (it would raise in 3D)."""
        if nnn is not None:
            return nnn
        return 2 * (self.ndim + 1) if order == 1 else 4

    def _retry_degenerate_stencils(self, coords_converted, tree_points,
                                   degenerate, nnn):
        """Re-solve the failed points on a wider stencil.

        A neighbourhood that cannot support an affine fit is usually a local
        accident of point placement, so reaching further normally escapes it.
        Only the failed points are re-queried, so the cost is set by how many
        actually failed (measured: a handful in tens of thousands).

        Returns ``(rows, indices, weights, remaining)`` — the recovered rows and
        their wider stencils, plus the mask of points that failed even after
        widening and so keep their inverse-distance weights.
        """
        from underworld3.utilities.rbf_stencil import linear_exact_weights

        wide = min(4 * nnn, self.n)
        if wide <= nnn:
            return None, None, None, degenerate

        failed = np.flatnonzero(degenerate)
        targets = np.ascontiguousarray(coords_converted[failed], dtype=np.float64)

        # The raw query: coords_converted is already in tree units, so the
        # unit-aware `query` would reject it when the tree carries units.
        indices, _ = self.find_closest_n_points(wide, targets)
        if np.any(indices >= self.n):
            return None, None, None, degenerate

        weights, still_degenerate = linear_exact_weights(
            targets, tree_points[indices]
        )

        remaining = degenerate.copy()
        remaining[failed] = still_degenerate

        recovered = ~still_degenerate
        if not recovered.any():
            return None, None, None, remaining

        return (failed[recovered], indices[recovered], weights[recovered],
                remaining)

    def _local_stencil(self, coords_converted, nnn, p, order):
        """Neighbour indices and weights for every target point.

        The single source of the interpolation weights: both
        :meth:`rbf_interpolator_local` and :meth:`interpolation_matrix` are
        built on this, so the operator and the values it produces cannot drift
        apart.

        Returns
        -------
        indices, weights : ndarray
            Shape ``(n_targets, nnn)``. The primary stencil.
        wide : tuple or None
            ``(rows, indices, weights)`` for targets re-solved on a wider
            stencil after their primary one proved degenerate. Their primary
            row is superseded and must be discarded by the caller.
        degenerate : ndarray
            Bool, shape ``(n_targets,)``. Targets still degenerate after
            widening; these carry inverse-distance weights, which are bounded
            but not linear-exact.
        """
        # find_closest_n_points returns (indices, dist_sqr) -- the reverse of
        # query()'s (dist, indices). It is used here rather than query()
        # because coords_converted is already in tree units, and query() would
        # convert a second time.
        closest_n, distance_n = self.find_closest_n_points(
            nnn, np.ascontiguousarray(coords_converted, dtype=np.float64)
        )

        # valid indices are 0..n-1; the empty-tree sentinel (0 with n=0)
        # must trip this guard, so the comparison is >= (issue #399).
        if np.any(closest_n >= self.n):
            raise RuntimeError(
                f"Cannot build a {nnn}-point stencil: the kd-tree holds "
                f"{self.n} point(s) and a nearest neighbour wasn't found. "
                "Reduce nnn, or check that the tree is not empty."
            )

        # np.bool_, not bool: this module cimports the C++ `bool` from libcpp,
        # which shadows the Python builtin and will not compile here.
        degenerate = np.zeros(coords_converted.shape[0], dtype=np.bool_)

        if nnn == 1:
            # The guard lives HERE, not in the callers: this early return
            # precedes all `order` handling, so a caller-side check leaves the
            # other entry point silently returning a nearest-neighbour stencil
            # when a linear-exact one was asked for (issue #443).
            if order == 1:
                raise ValueError(
                    "order=1 needs at least dim + 2 neighbours to determine "
                    "the affine tail; nnn=1 selects the raw nearest-neighbour "
                    "path, which reproduces constants only."
                )
            return closest_n, np.ones(closest_n.shape), None, degenerate

        # Inverse distance weighting: w = 1 / (eps + r)^p, normalised.
        # `find_closest_n_points` returns SQUARED distances, so the square root
        # is taken here -- without it the decay is r^(-2p) rather than the
        # r^(-p) the argument names (issue #427). epsilon is a length floor for
        # a target sitting exactly on a source point; it is on r, not r^2, so
        # its scale is the one it reads as.
        epsilon = 1e-12
        distance = np.sqrt(distance_n[:])
        weights = 1 / np.power(epsilon + distance, p)
        n_weights = (weights.T / np.sum(weights, axis=1)).T

        if order == 0:
            return closest_n, n_weights, None, degenerate

        from underworld3.utilities.rbf_stencil import linear_exact_weights

        tree_points = np.asarray(self.points)
        linear_weights, degenerate = linear_exact_weights(
            coords_converted, tree_points[closest_n[:]]
        )
        # A stencil that cannot support an affine fit (collinear in 2D,
        # coplanar in 3D) keeps the inverse-distance weights: finite and
        # bounded, but not linear-exact.
        linear_weights[degenerate] = n_weights[degenerate]

        wide = None
        if degenerate.any():
            # Widen before surrendering. A single non-exact node sitting among
            # exact ones is an isolated SPIKE, far more damaging to derivatives
            # and integrals than a smooth error of the same size.
            rows, wide_indices, wide_weights, degenerate = (
                self._retry_degenerate_stencils(
                    coords_converted, tree_points, degenerate, nnn
                )
            )
            if rows is not None:
                wide = (rows, wide_indices, wide_weights)

        n_degenerate = int(degenerate.sum())
        if n_degenerate:
            import warnings

            warnings.warn(
                f"rbf_interpolator_local(order=1): {n_degenerate} of "
                f"{degenerate.size} stencils could not support an affine fit "
                "(collinear/coplanar neighbours) even after widening, and fell "
                "back to inverse-distance weighting.",
                # 2 = whichever public method called _local_stencil. The two
                # entry points sit at different depths below that, so this is
                # the deepest frame that is correct for both.
                stacklevel=2,
            )

        return closest_n, linear_weights, wide, degenerate

    def interpolation_matrix(self, coords, nnn=None, p=1, order=0):
        """Sparse operator mapping values on the KD-tree points to ``coords``.

        ``T @ data`` is exactly what :meth:`rbf_interpolator_local` returns for
        the same arguments — both are built from the same weights.

        Use this instead of the value API when the same transfer is applied
        more than once, or when the operator itself is the product (a multigrid
        prolongation, for example). The weights depend only on geometry, so one
        build serves every field and every component; the value API re-solves
        each time.

        Parameters
        ----------
        coords : array-like
            Target coordinates, shape ``(n_coords, dim)``.
        nnn : int, optional
            Neighbours per target. Defaults to 4 for ``order=0`` and
            ``2 * (dim + 1)`` for ``order=1``.
        p : int, optional
            Inverse-distance power; used by ``order=0`` only.
        order : int, optional
            Polynomial reproduction order, 0 or 1. See
            :meth:`rbf_interpolator_local`.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape ``(n_coords, self.n)``. Rows carry ``nnn`` non-zeros, except
            those re-solved on a wider stencil.

        Notes
        -----
        Degenerate stencils are handled exactly as in the value path — widened,
        then fall back to inverse-distance weights with a warning. Rows are
        never empty.

        **Row-wise construction gives no column guarantee.** Every row has
        ``nnn`` non-zeros, but a source point that is not among the neighbours
        of any target produces an empty column. Consumers that need full column
        rank -- a Galerkin coarse operator :math:`P^{T} A P`, for instance --
        must check for and repair empty columns themselves (see issue #424).
        """
        import scipy.sparse as sp

        if order not in (0, 1):
            raise ValueError(
                f"order must be 0 (inverse distance) or 1 (linear-exact), got {order!r}."
            )
        coords_converted = self._convert_coords_to_tree_units(coords)
        if coords_converted.shape[1] != self.ndim:
            raise RuntimeError(
                f"Interpolation coordinates dimensionality "
                f"({coords_converted.shape[1]}) is different to kD-tree "
                f"dimensionality ({self.ndim})."
            )
        nnn = self._resolve_nnn(nnn, order)

        indices, weights, wide, _ = self._local_stencil(
            coords_converted, nnn, p, order
        )

        n_targets = coords_converted.shape[0]
        keep = np.ones(n_targets, dtype=np.bool_)
        if wide is not None:
            keep[wide[0]] = False

        rows = np.repeat(np.flatnonzero(keep), nnn)
        cols = indices[keep].ravel()
        vals = weights[keep].ravel()

        if wide is not None:
            wide_rows, wide_indices, wide_weights = wide
            rows = np.concatenate(
                [rows, np.repeat(wide_rows, wide_indices.shape[1])]
            )
            cols = np.concatenate([cols, wide_indices.ravel()])
            vals = np.concatenate([vals, wide_weights.ravel()])

        return sp.csr_matrix(
            (vals, (rows, cols.astype(np.int64))), shape=(n_targets, self.n)
        )

    def rbf_interpolator_local_from_kdtree(self, coords, data, nnn, p, verbose,
                                           order=0, monotone=False):
        """
        Map data held on the KD-tree points onto the target `coords`.

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
            The power index to calculate weights, i.e., pow(distance, -p),
            on the actual distance. Used by ``order=0`` only.
        verbose : bool
            Print when mapping occurs
        order : int, optional
            Polynomial reproduction order: 0 for inverse-distance (constants
            exact), 1 for polyharmonic + affine tail (constants and linears
            exact). See :meth:`rbf_interpolator_local`.
        monotone : bool or str, optional
            ``False``, ``True`` or ``"clamp"`` — bound each result to the
            min/max of its own stencil's source values.

        Returns
        -------
        ndarray
            Interpolated data values at target coordinates
        """
        if order not in (0, 1):
            raise ValueError(
                f"order must be 0 (inverse distance) or 1 (linear-exact), got {order!r}."
            )
        monotone_clamp = _normalise_monotone(monotone)

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

        nnn = self._resolve_nnn(nnn, order)

        if verbose and uw.mpi.rank == 0:
            print(f"Mapping values with nnn - {nnn} & p {p}  ... start", flush=True)

        if nnn == 1:
            # Only the raw nearest-neighbour value; _local_stencil holds the
            # order=1 guard so both entry points share it (issue #443).
            closest_n, _, _, _ = self._local_stencil(
                coords_converted, nnn, p, order
            )
            # (n, 1) -> (n,): this path returns data rows directly, so the
            # stencil axis must not survive into the result.
            return data[closest_n.reshape(-1)]

        closest_n, n_weights, wide, degenerate = self._local_stencil(
            coords_converted, nnn, p, order
        )
        kdata = data[closest_n[:]]

        # magic with einstein summation power
        vals = np.einsum("sdc,sd->sc", kdata, n_weights)

        if wide is not None:
            wide_rows, wide_indices, wide_weights = wide
            vals[wide_rows] = np.einsum(
                "sdc,sd->sc", data[wide_indices], wide_weights
            )

        if monotone_clamp:
            if order == 0:
                # Already a convex combination; the clip is exact-arithmetic
                # redundant and only guards round-off.
                vals = np.clip(vals, kdata.min(axis=1), kdata.max(axis=1))
            else:
                # Limit the CORRECTION, never the linear part -- the same
                # discipline as a slope limiter in a second-order FV scheme.
                #
                # Clipping the total against the stencil's raw min/max would
                # be wrong here: a target outside the convex hull of its own
                # neighbours has a value outside their range even for an
                # exactly linear field, so the naive clip cannot distinguish
                # legitimate extrapolation from ringing and destroys the
                # reproduction guarantee.
                from underworld3.utilities.rbf_stencil import affine_trend

                # The bound comes from the primary stencil even for the few
                # rows re-solved on a wider one: it is still a legitimate local
                # bound, and the affine trend -- the part being preserved -- is
                # what matters here.
                trend_at_target, trend_at_stencil = affine_trend(
                    coords_converted, np.asarray(self.points)[closest_n[:]], kdata
                )
                residual = kdata - trend_at_stencil
                vals = trend_at_target + np.clip(
                    vals - trend_at_target,
                    residual.min(axis=1),
                    residual.max(axis=1),
                )

        if verbose and uw.mpi.rank == 0:
            print(f"Mapping values  ... finished", flush=True)

        return vals
