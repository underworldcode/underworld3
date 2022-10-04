from typing import Optional, Tuple, Union
from petsc4py import PETSc
import underworld3 as uw 

from cython cimport view  # comment this line to see what will happen
import numpy as np

from underworld3 import _api_tools
import underworld3.timing as timing

include "petsc_extras.pxi"

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


def petsc_dm_create_surface_submesh(incoming_dm, boundary_label, boundary_label_value, marked_faces=True) -> double:
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


def petsc_fe_create_sub_dm(incoming_dm, field_id):

        cdef DM subdm = PETSc.DM()
        cdef DM dm = incoming_dm
        cdef PetscInt fields = field_id

        ierr = DMCreateSubDM(dm.dm, 1, &fields, NULL, &subdm.dm);CHKERRQ(ierr)

        return subdm

def petsc_dm_get_periodicity(incoming_dm):

        dim = incoming_dm.getDimension()
        
        cdef PetscInt c_dim = dim
        cdef PetscReal c_maxCell[3]
        cdef PetscReal c_Lstart[3]
        cdef PetscReal c_L[3]
        cdef DM dm = incoming_dm

        # c_maxCell[:] = (-1,-1,-1)
        # c_Lstart[:] = (-2,-2,-2)
        # c_L[:]      = (-3,-3,-3)

        ierr = DMGetPeriodicity(dm.dm, <PetscReal **> &c_maxCell[0], <PetscReal **> &c_Lstart[0], <PetscReal **> &c_L[0]); CHKERRQ(ierr)

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


def petsc_dm_set_periodicity(incoming_dm, maxCell, Lstart, L):

        dim = incoming_dm.getDimension()
        
        cdef DM c_dm = incoming_dm
        cdef PetscInt c_dim = dim
        cdef PetscReal c_maxCell[3]
        cdef PetscReal c_Lstart[3]
        cdef PetscReal c_L[3]


        c_maxCell[:] = maxCell[:]
        c_Lstart[:] = Lstart[:]
        c_L[:]      = L[:]

        ierr = DMSetPeriodicity(c_dm.dm, c_maxCell, c_Lstart , c_L); CHKERRQ(ierr)

        return 

"""
# Todo: add types to this definition to make things more clear.
def petsc_dm_add_boundary_condition(incoming_dm, bc_type, bcName, dmLabel, numVals, vals, field, numComps, comps, bcFunc):

        cdef DM c_dm = incoming_dm
        cdef DMLabel c_dmLabel = dmLabel
        cdef PetscInt c_bd = 0 # return value that we discard

        cdef int [::1] c_comps_view = comps
        cdef int [::1] c_vals_view = vals

        # cdef PetscInt c_numComps = numComps
        # cdef PetscInt [::1] = comps # force view into numpy array 

        # The time derivative function is NULL for the time being
        # We only constrain field id 0

        
        DMAddBoundary(c_dm.dm, bc_type,  str(bcName).encode('utf8'), c_dmLabel, 
                        numVals, &c_vals_view[0], 0, numComps, &c_comps_view[0], <void (*)()> bcFunc, NULL, NULL, &c_bd);


#include "petscdm.h"          
#include "petscdmlabel.h"     
#include "petscds.h"     
        # DMAddBoundary(DM dm, DMBoundaryConditionType type, const char name[], DMLabel label, PetscInt Nv, const PetscInt values[], PetscInt field, PetscInt Nc, const PetscInt comps[], void (*bcFunc)(void), void (*bcFunc_t)(void), void *ctx, PetscInt *bd)

        return

"""