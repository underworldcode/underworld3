from typing import Optional, Tuple, Union
from petsc4py import PETSc
import underworld3 as uw 

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

        ## Note: The petsc4py version of DMPlexComputeGeometryFVM does not compute all cells and 
        ## does not obtain the minimum radius for the mesh.


        cdef DM dm = incoming_dm
        cdef DM subdm
        cdef PetscDMLabel dmlabel
        cdef PetscInt value = boundary_label_value
        cdef PetscBool markedFaces = marked_faces

        subdm = PETSc.DM()
        
        DMGetLabel(dm, "Boundary", &dmlabel)
        # DMPlexCreateSubmesh(dm.dm, dmlabel, value, markedFaces, &subdm.dm)

        return 

        

