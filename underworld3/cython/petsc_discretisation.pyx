from typing import Optional, Tuple, Union
from petsc4py import PETSc
import underworld3 as uw 

# from cython cimport view  # comment this line to see what will happen
import numpy as np

from underworld3 import _api_tools
import underworld3.timing as timing

include "petsc_extras.pxi"

## This is currently wrapped in petsc4py but returns zero

def petsc_fvm_get_min_radius(mesh) -> double:
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


def petsc_dm_create_submesh_from_label(incoming_dm, boundary_label_name, boundary_label_value, marked_faces=True) -> double:
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

        
def petsc_dm_project_coordinates(incoming_dm, incoming_petsc_fe=None):
        """
        Something hangs in petsc4py version of this in parallel
        """

        cdef DM c_dm = incoming_dm
        cdef FE c_fe = incoming_petsc_fe

        if incoming_petsc_fe is None:
                ierr = DMProjectCoordinates( c_dm.dm, NULL ); CHKERRQ(ierr)

        else:
                ierr = DMProjectCoordinates( c_dm.dm, c_fe.fe ); CHKERRQ(ierr)


        # DM should be updated, no value returned

        return 





def petsc_dm_find_labeled_points_local(dm, label_name, sectionIndex=False):
    '''Identify local points associated with "Label" 
        
        dm -> expects a petscDM object 
        label_name -> "String Name for Label"
        sectionIndex -> False: leave points as indexed by the relevant section on the dm
                        True: index into the local coordinate array
    '''

        
    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)

    label = dm.getLabel(label_name)
    if not label:
        return np.array([0])
        
    pointIS = dm.getStratumIS("celltype",0)
    edgeIS = dm.getStratumIS("celltype",1)

    _, iset_lab = label.convertToSection()

    IndicesP = np.intersect1d(iset_lab.getIndices(), pointIS.getIndices()) 
    IndicesE = np.intersect1d(iset_lab.getIndices(), edgeIS.getIndices()) 
    
    IndicesEp = np.empty((IndicesE.shape[0], 2), dtype=int)

    for e in range(IndicesE.shape[0]):
        IndicesEp[e] = dm.getCone(IndicesE[e])
    
    if sectionIndex:
        Indices = np.union1d(IndicesP, IndicesEp)
    else:
        Indices = np.union1d(IndicesP, IndicesEp) - pStart

    return Indices


## This one seems to work from petsc4py 3.18.3 

# def petsc_fe_create_sub_dm(incoming_dm, field_id):

#         cdef DM subdm = PETSc.DM()
#         cdef DM dm = incoming_dm
#         cdef PetscInt fields = field_id

#         ierr = DMCreateSubDM(dm.dm, 1, &fields, NULL, &subdm.dm);CHKERRQ(ierr)

#         return subdm


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
          - maxCell - Over distances greater than this, we can assume a point has crossed over to another sheet, when trying to localize cell coordinates. Pass NULL to remove such information.
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

        incoming_dm.localizeCoordinates()

        return 
