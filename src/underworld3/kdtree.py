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
    def query(self, coords, k = 1, sqr_dists = True):
        """
        Find the n points closest to the provided coordinates.
        Wraps the pykdtree query() method and the only difference is the sqr_dists parameter. 
        
        Parameters
        ----------
        coords:
            An array of coordinates for which the kd-tree index will be searched for nearest
            neighbours. This should be a 2-dimensional array of size (n_coords, dim).
        k:
            The number of nearest neighbour points to find for each `coords`. 
        sqr_dists:
            Set to True to return the squared distances, set to False to return the actual distances. 
            
        Returns
        -------
        d:
            A float array of the squared (sqr_dists = True) or actual distances (sqr_dists = False) between the provided coords and the nearest neighbouring
            points. It will be of size (n_coords).
        i:
            An integer array of indices into the `points` array (passed into the constructor) corresponding to
            the nearest neighbour for the search coordinates. It will be of size (n_coords).
        """

        coords_contiguous = np.ascontiguousarray(coords)

        distance_k, closest_k = super().query(query_pts = coords_contiguous, k = k)

        if sqr_dists:
            return distance_k**2, closest_k
        else:
            return distance_k, closest_k
    

    def rbf_interpolator_local_from_kdtree(self, coords, data, nnn, p, verbose):
        """
        Performs an inverse distance (squared) mapping of data to the target `coords`.
        User can controls the algorithm by altering the number of neighbours used, `nnn` or the
        power factor `p` of the mapping weighting.

        Args:
        coords  : ndarray,
                The target spatial coordinates to evaluate the data from.
                coords.shape[1] == self.ndim
        data    : ndarray
                The known data to map from. Must be full described over kd-tree.
                i.e., data.shape[0] == self.n
        nnn     : int,
                The number of neighbour points to sample from, if `1` no distance averaging is done.
        p       : int,
                The power index to calculate weights, ie. pow(distance, -p)
        verbose : bool,
                Print when mapping occurs
        """

        if coords.shape[1] != self.ndim:
            raise RuntimeError(
                f"Interpolation coordinates dimensionality ({coords.shape[1]}) is different to kD-tree dimensionality ({self.ndim})."
            )
        if data.shape[0] != self.n:
            raise RuntimeError(
                f"Data does not match kd-tree size array ({data.shape[0]} v ({self.n}))"
            )

        
        # query nnn points to the coords using wrapped query function
        # distance_n is a list of distance to the nearest neighbours for all coords_contiguous
        # closest_n is the index of the neighbours from ncoords for all coords_contiguous
        distance_n, closest_n = self.query(coords, k=nnn, sqr_dists = False)

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


## NB the rbf interpolator TO kdtree is missing (and we need that one that we introduced to do a better job of mapping values from swarms to nodes for proxy variables)
