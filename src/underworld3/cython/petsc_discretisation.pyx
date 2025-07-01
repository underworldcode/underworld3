from typing import Optional, Tuple, Union
from petsc4py import PETSc
import underworld3 as uw

# from cython cimport view  # comment this line to see what will happen
import numpy as np

from underworld3 import _api_tools
import underworld3.timing as timing

include "petsc_extras.pxi"

## This is currently wrapped in petsc4py but returns zero

def petsc_fvm_get_min_radius(mesh) -> float:
        """
        This method returns the minimum distance from any cell centroid to a face.
        It wraps to the PETSc `DMPlexGetMinRadius` routine.
        """

        ## Note: The petsc4py version of DMPlexComputeGeometryFVM does not compute all cells and
        ## does not obtain the minimum radius for the mesh.

        cdef Vec cellgeom = Vec()
        cdef Vec facegeom = Vec()
        cdef DM dm = mesh.dm

        DMPlexComputeGeometryFVM(dm.dm,&cellgeom.vec,&facegeom.vec)

        min_radius = dm.getMinRadius()
        cellgeom.destroy()
        facegeom.destroy()

        return min_radius

def petsc_fvm_get_local_cell_sizes(mesh) -> np.array:
        """
        This method returns the minimum distance from any cell centroid to a face.
        It wraps to the PETSc `DMPlexGetMinRadius` routine.
        """

        ## Note: The petsc4py version of DMPlexComputeGeometryFVM does not compute all cells and
        ## does not obtain the minimum radius for the mesh.

        cdef Vec cellgeom = Vec()
        cdef Vec facegeom = Vec()
        cdef DM dm = mesh.dm

        DMPlexComputeGeometryFVM(dm.dm,&cellgeom.vec,&facegeom.vec)

        # cell_geom is an array of 3D cell centroids plus the cell volume

        cell_sizes = cellgeom.array.copy().reshape(-1, 4)
        cell_radii = np.power(cell_sizes[:,-1], 1/mesh.dim)
        cell_centroids = cell_sizes[:,0:mesh.dim]

        cellgeom.destroy()
        facegeom.destroy()

        return cell_radii, cell_centroids


def petsc_dm_create_submesh_from_label(incoming_dm, boundary_label_name, boundary_label_value, marked_faces=True) -> float:
        """
        Wraps DMPlexCreateSubmesh
        """


        cdef DM c_dm = incoming_dm
        cdef DM subdm
        cdef PetscDMLabel dmlabel
        cdef PetscInt value = boundary_label_value
        cdef PetscBool markedFaces = marked_faces

        subdm = PETSc.DM()

        DMGetLabel(c_dm.dm, "Boundary", &dmlabel)
        # DMPlexCreateSubmesh(dm.dm, dmlabel, value, markedFaces, &subdm.dm)

        return



# This is not cython, does it need to be here or in discretisation.py ?

def petsc_dm_find_labeled_points_local(dm, label_name, sectionIndex=False, verbose=False):
        '''Identify local points associated with "Label"

        dm -> expects a petscDM object
        label_name -> "String Name for Label"
        sectionIndex -> False: leave points as indexed by the relevant section on the dm
                        True: index into the local coordinate array

        NOTE: Assumes uniform element types
        '''

        pStart, pEnd = dm.getDepthStratum(0)
        eStart, eEnd = dm.getDepthStratum(1)
        fStart, fEnd = dm.getDepthStratum(2)

        label = dm.getLabel(label_name)
        if not label:
                if uw.mpi.rank == 0:
                        print(f"Label {label_name} is not present on the dm")
                return np.array([0])

        pointIS = dm.getStratumIS("depth",0)
        edgeIS = dm.getStratumIS("depth",1)
        faceIS = dm.getStratumIS("depth",2)

        # point_indices = pointIS.getIndices()
        # edge_indices = edgeIS.getIndices()
        # face_indices = faceIS.getIndices()

        _, iset_lab = label.convertToSection()

        IndicesP = np.intersect1d(iset_lab.getIndices(), pointIS.getIndices())
        IndicesE = np.intersect1d(iset_lab.getIndices(), edgeIS.getIndices())
        IndicesF = np.intersect1d(iset_lab.getIndices(), faceIS.getIndices())

        # print(f"Label {label_name}")
        # print(f"P -> {len(IndicesP)}, E->{len(IndicesE)}, F->{len(IndicesF)},")

        if IndicesF.any():
                IndicesFe = np.empty((IndicesF.shape[0], dm.getConeSize(fStart)), dtype=int)
                for f in range(IndicesF.shape[0]):
                        IndicesFe[f] = dm.getCone(IndicesF[f])

                IndicesE = np.union1d(IndicesE, IndicesFe)

        # All faces are now recorded as edges

        if IndicesE.any():
                IndicesEp = np.empty((IndicesE.shape[0], dm.getConeSize(eStart)), dtype=int)
                for e in range(IndicesE.shape[0]):
                        IndicesEp[e] = dm.getCone(IndicesE[e])

                IndicesP = np.union1d(IndicesP, IndicesEp)

        # all faces / edges are now points

        if IndicesP.any() and not sectionIndex:
                IndicesP -= pStart

        return IndicesP


## Todo !

"""
def petsc_dm_get_periodicity(incoming_dm):

        dim = incoming_dm.getDimension()

        cdef PetscInt c_dim = dim
        cdef PetscReal c_maxCell[3]
        cdef PetscReal c_Lstart[3]
        cdef PetscReal c_L[3]
        cdef DM dm = incoming_dm


        ierr = DMGetPeriodicity(dm.dm, &c_maxCell[0], &c_Lstart[0],  &c_L[0]); CHKERRQ(ierr)

        maxCell = np.asarray(c_maxCell)

        maxx = maxCell[0]
        maxy = maxCell[1]
        maxz = maxCell[2]

        Lstartx = c_Lstart[0]
        Lstarty = c_Lstart[1]
        Lstartz = c_Lstart[2]

        Lx = c_L[0]
        Ly = c_L[1]
        Lz = c_L[2]


        print(f"Max x - {maxx}, y - {maxy}, z - {maxz}"  )
        print(f"Ls x - {Lstartx}, y - {Lstarty}, z - {Lstartz}"  )
        print(f"L  x - {Lx}, y - {Ly}, z - {Lz}"  )

        return
"""

def petsc_dm_set_periodicity(incoming_dm, maxCell, Lstart, L):
        """
        Wrapper for PETSc DMSetPeriodicity:
          - maxCell - Over distances greater than this, we can assume a point has crossed over to another sheet, when trying to localize cell coordinates.
                Pass NULL to remove such information.
          - Lstart - If we assume the mesh is a torus, this is the start of each coordinate, or NULL for 0.0
          - L - If we assume the mesh is a torus, this is the length of each coordinate, otherwise it is < 0.0
        """

        dim = incoming_dm.getDimension()

        maxCell3 = np.zeros(3)
        Lstart3 = np.zeros(3)
        L3 = np.zeros(3)

        for i in range(dim):
                maxCell3[i] = maxCell[i]
                Lstart3[i] = Lstart[i]
                L3[i] = L[i]

        cdef DM c_dm = incoming_dm
        cdef PetscInt c_dim = dim
        cdef PetscReal c_maxCell[3]
        cdef PetscReal c_Lstart[3]
        cdef PetscReal c_L[3]

        c_maxCell[:] = maxCell3[:]
        c_Lstart[:] = Lstart3[:]
        c_L[:]      = L3[:]

        ierr = DMSetPeriodicity( c_dm.dm, c_maxCell, c_Lstart , c_L); CHKERRQ(ierr)
        ierr = DMLocalizeCoordinates(c_dm.dm); CHKERRQ(ierr)

        # incoming_dm.localizeCoordinates()

        return


def petsc_vec_concatenate( inputVecs  ):

        inputVecs = list(inputVecs)
        # nx = len(inputVecs)

        outputVec = PETSc.Vec().create()

        cdef Py_ssize_t i, nx = len(inputVecs)
        cdef PetscInt n = <PetscInt> nx

        cdef PetscVec cvecs[100]
        cdef Vec output_cVec = Vec()

        for i from 0 <= i < nx:
                cvecs[i] = (<Vec?>inputVecs[i]).vec

        ierr = VecConcatenate(nx, cvecs, &output_cVec.vec, NULL)

        outputVec = output_cVec

        return outputVec


def petsc_get_swarm_coord_name( sdm ):

    return
