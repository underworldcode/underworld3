import underworld3
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
            A float array of squred distances between the provided coords and the nearest neighbouring
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
    def knnSearch(self, 
                  const int nCount                    :   numpy.int,
                  const double[: ,::1] coords not None:   numpy.ndarray):
        """
        Find the n points closest to the provided coordinates. 

        Parameters
        ----------
        coords:
            An array of coordinates for which the kd-tree index will be searched for nearest
            neighbours. This should be a 2-dimensional array of size (n_coords,dim).
        nCount:
            The number of nearest neighbour points to find for each `coords`.

        Returns
        -------
        indices:
            An integer array of indices into the `points` array (passed into the constructor) corresponding to
            the nearest neighbour for the search coordinates. It will be of size (n_coords).
        dist_sqr:
            A float array of squred distances between the provided coords and the nearest neighbouring
            points. It will be of size (n_coords).
        found:
            A bool array of flags which signals whether a nearest neighbour has been found for a given
            coordinate. It will be of size (n_coords).



        """
        if coords.shape[1] != self.points.shape[1]:
            raise RuntimeError(f"Provided coords array dimensionality ({coords.shape[1]}) is different to points dimensionality ({self.points.shape[1]}).")
        nInput = coords.shape[0]

        # allocate numpy arrays
        indices  = np.empty(nCount, dtype=np.uint64,  order='C')
        dist_sqr = np.empty(nCount, dtype=np.float64, order='C')
        # allocate memoryviews in C contiguous layout
        cdef long unsigned int[::1] c_indices  = indices 
        cdef            double[::1] c_dist_sqr = dist_sqr

        # invoke cpp function
        self.index.knnSearch( <const double *> &coords[0][0], 
                              nCount,
                              <long unsigned int*> &c_indices[0], 
                              <           double*> &c_dist_sqr[0]) 

        # return numpy data
        return indices, dist_sqr
