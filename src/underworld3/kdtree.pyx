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
    KD-Tree indexes are data structures and algorithms for the efficient
    determination of nearest neighbours.  

    This class generates a kd-tree index for the provided points, and provides
    the necessary methods for finding which points are closest to a given query
    location. 
 
    This class utilises `nanoflann` for kd-tree functionality.

    Parameters
    ----------
    points:
        The points for which the kd-tree index will be build. This 
        should be a 2-dimensional array of size (n_points,dim).

    Example
    -------
    >>> import numpy as np
    >>> import underworld3 as uw

    Generate a random set of points
    >>> pts = np.random.random( size=(100,2) )

    Build the index on the points
    >>> index = uw.algorithms.KDTree(pts)
    >>> index.build_index()

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

    def __cinit__( self, 
                   const double[:,::1] points not None:   numpy.ndarray ) :

        if points.shape[1] not in (2,3):
            raise RuntimeError(f"Provided points array dimensionality must be 2 or 3, not {points.shape[1]}.")
        self.points = points
        self.index = new KDTree_Interface(<const double *> &points[0][0], points.shape[0], points.shape[1])
        super().__init__()

    def __dealloc__(self):
        del self.index

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
            A float array of squred distances between the provided coords and the nearest neighbouring
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


## A general point-to-point rbf interpolator here
## NOTE this is not using cython optimisation for numpy

    def rbf_interpolator_local(self, 
            coords,
            data,
            nnn = 4,
            verbose = False,
        ):

        '''
        An inverse (squared) distance weighted mapping of a numpy array from one
        set of coordinates to another. This assumes all points are local to the
        same processor. If that is not the case, it is best to use a particle swarm
        to migrate data.
        '''

        if coords.shape[1] != self.points.shape[1]:
            raise RuntimeError(f"Interpolation coordinates dimensionality ({coords.shape[1]}) is different to kD-tree dimensionality ({self.points.shape[1]}).")
        nInput = coords.shape[0]

        if data.shape[0] != self.points.shape[0]:
                raise RuntimeError(f"Data does not match kD-tree size array ({data.shape[0]}) v ({self.points.shape[0]}).")

        coords_contiguous = np.ascontiguousarray(coords)

        closest_n, distance_n = self.find_closest_n_points(nnn, coords_contiguous)

        num_local_points = coords.shape[0]
        data_size = data.shape[1]

        Values = np.zeros((num_local_points, data_size))
        Weights = np.zeros((num_local_points, 1))

        if verbose and uw.mpi.rank == 0:
            print("Mapping values  ... start", flush=True)

        epsilon = 1.0e-24
        for j in range(nnn):
            j_distance = epsilon + np.sqrt(distance_n[:, j])
            Weights[:, 0] += 1.0 / j_distance[:]


        epsilon = 1.0e-24
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


