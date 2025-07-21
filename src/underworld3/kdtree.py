from pykdtree.kdtree import KDTree as _oKDTree
import underworld3 as uw
import numpy as np

# inherit from the pykdtree
class KDTree( _oKDTree ):
    def rbf_interpolator_local(self,
        coords,
        data,
        nnn = 4,
        p = 2,
        verbose = False,
    ):

        return self.rbf_interpolator_local_from_kdtree(
            coords, data, nnn, p, verbose,
    )

 #   def find_closest_points( self, coords, nnn )

    def rbf_interpolator_local_from_kdtree(
        self,
        coords,
        data,
        nnn,
        p,
        verbose):

        '''
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
        '''

        if coords.shape[1] != self.ndim:
            raise RuntimeError(f"Interpolation coordinates dimensionality ({coords.shape[1]}) is different to kD-tree dimensionality ({self.ndim}).")
        if data.shape[0] != self.n:
            raise RuntimeError(f"Data does not match kd-tree size array ({data.shape[0]} v ({self.n}))")

        coords_contiguous = np.ascontiguousarray(coords)
        # query nnn points to the coords
        # distance_n is a list of distance to the nearest neighbours for all coords_contiguous
        # closest_n is the index of the neighbours from ncoords for all coords_contiguous
        distance_n, closest_n = self.query(coords_contiguous, k=nnn)

        if np.any(closest_n > self.n):
            raise RuntimeError("Error in rbf_interpolator_local_from_kdtree - a nearest neighbour wasn't found")

        if verbose and uw.mpi.rank == 0:
            # For Debugging
            # print(f"kd-tree diagnostics: d.shape - {distance_n.shape}, c.shape - {closest_n.shape}")
            print(f"Mapping values with nnn - {nnn} & p {p}  ... start", flush=True)

        if nnn == 1:
            # only use nearest neighbour raw data
            return data[closest_n]
            
        # can decompose weighting vecotrs as IDW is a linear relationship
        # build normalise weight vectors and multiply that with known data
        epsilon   = 1e-12
        weights   = 1 / np.pow( epsilon+distance_n[:], p)
        n_weights = (weights.T / np.sum(weights, axis=1)).T
        kdata     = data[closest_n[:]]
        
        # magic with einstein summation power
        vals = np.einsum('sdc,sd->sc', kdata, n_weights)
        # print(valz)

        if verbose and uw.mpi.rank == 0:
            print(f"Mapping values  ... finished", flush=True)
        
        return vals
