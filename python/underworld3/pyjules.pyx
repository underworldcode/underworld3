from petsc4py.PETSc import Error
from petsc4py.PETSc cimport DM, PetscDM

cdef extern from * nogil:
    ctypedef long PetscInt
    ctypedef double PetscReal
    ctypedef double PetscScalar
    ctypedef int PetscErrorCode
    ctypedef int PetscBool

cdef extern from "poisson_jules.h":
    ctypedef struct AppCtx:
        PetscBool simplex
        PetscScalar y0
        PetscScalar y1
        PetscScalar T0
        PetscScalar T1
        PetscScalar k
        PetscScalar h

    PetscErrorCode SetupDiscretization(PetscDM dm, AppCtx *user)
    PetscErrorCode PetscJulesUseDM(AppCtx* crap, PetscDM _dm)
    PetscErrorCode PetscJules(AppCtx *user) 
    #PetscErrorCode PetscJules() 

cdef AppCtx cuser

def pySetupDiscretization( DM dm, user ):
    #cdef AppCtx cuser
    cuser.simplex = user["simplex"]
    cuser.y0 = user["y0"]
    cuser.y1 = user["y1"]
    cuser.T0 = user["T0"]
    cuser.T1 = user["T1"]
    cuser.k = user["k"]
    cuser.h = user["h"]
    SetupDiscretization( dm.dm, &cuser )

def pyJulesUseDM(user, DM dm):
    #cdef AppCtx cuser
    cuser.simplex = user["simplex"]
    cuser.y0 = user["y0"]
    cuser.y1 = user["y1"]
    cuser.T0 = user["T0"]
    cuser.T1 = user["T1"]
    cuser.k = user["k"]
    cuser.h = user["h"]
    cdef int ierr
    ierr = PetscJulesUseDM(&cuser, dm.dm)
    if ierr != 0: raise Error(ierr)
    return

def pyJules(user):
    #cdef AppCtx cuser
    cuser.simplex = user["simplex"]
    cuser.y0 = user["y0"]
    cuser.y1 = user["y1"]
    cuser.T0 = user["T0"]
    cuser.T1 = user["T1"]
    cuser.k = user["k"]
    cuser.h = user["h"]
    cdef int ierr
    ierr = PetscJules(&cuser)
    if ierr != 0: raise Error(ierr)
    return

#def pyJules():
#    cdef int ierr
#    ierr = PetscJules()
#    if ierr != 0: raise Error(ierr)
#    return
