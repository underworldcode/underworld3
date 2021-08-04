# %%
import numpy as np
import os
import math
import underworld3
run_uw2 = False
try:
    import underworld
    run_uw2 = True
except:
    pass


import time
global now_time 
now_time = time.time()
def delta_time():
    global now_time
    old_now_time = now_time
    now_time = time.time()
    return now_time - old_now_time

dim = 2
if "UW_LONGTEST" in os.environ:
    n_els = 64
else:
    n_els = 256
boxLength      = 0.9142
boxHeight      = 1.0
viscosityRatio = 1.0
stokes_inner_tol = 1e-6
stokes_outer_tol = 1e-5
ppcell = 15
amplitude  = 0.02
offset     = 0.2
print_time = 10
model_end_time = 20.
# output
inputPath  = 'input/05_Rayleigh_Taylor/'
outputPath = 'output/'
# Make output directory if necessary.
from mpi4py import MPI
if MPI.COMM_WORLD.rank==0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    
from underworld3.tools import parse_cmd_line_options
parse_cmd_line_options()

# %%
def do_uw3():
    from petsc4py import PETSc
    import underworld3 as uw
    from underworld3.systems import Stokes

    options = PETSc.Options()
    # options["help"] = None
    # options["pc_type"]  = "svd"
    options["ksp_rtol"] =  1.0e-6
    options["ksp_atol"] =  1.0e-6
    # options["log_view"] = None
    # options["ksp_monitor"] = None
    # options["snes_type"]  = "fas"
    options["snes_converged_reason"] = None
    options["snes_monitor_short"] = None
    # options["snes_view"]=None
    # options["snes_test_jacobian"] = None
    # options["snes_rtol"] = 1.0e-2  # set this low to force single SNES it. 
    options["snes_max_it"] = 1
    # options["pc_type"] = "fieldsplit"
    # options["pc_fieldsplit_type"] = "schur"
    # options["pc_fieldsplit_schur_factorization_type"] = "full"
    # # options["fieldsplit_pressure_ksp_rtol"] = 1e-6
    # options["fieldsplit_velocity_pc_type"] = "lu"
    # options["fieldsplit_pressure_pc_type"] = "jacobi" 
    # options["fieldsplit_velocity_ksp_type"] = "gmres"
    sys = PETSc.Sys()
    sys.pushErrorHandler("debugger")


    mesh = uw.mesh.Mesh(elementRes=(    n_els,)*dim, 
                        minCoords =(       0.,)*dim, 
                        maxCoords =(boxLength,1.),
                        simplex=False )
    u_degree = 1
    stokes = Stokes(mesh, u_degree=u_degree )
    
    # Create a variable to store material variable
    # matMeshVar = uw.mesh.MeshVariable("matmeshvar", mesh, 1, uw.VarType.SCALAR, degree=u_degree+1)

    #%%
    # Create swarm
    swarm  = uw.swarm.Swarm(mesh)
    # Add variable for material
    matSwarmVar      = swarm.add_variable(name="matSwarmVar",      num_components=1, dtype=PETSc.IntType)
    # Note that `ppcell` specifies particles per cell per dim.
    swarm.populate(ppcell=ppcell)
    with swarm.access():
        print(f"\nSwarm local population is {len(swarm.particle_coordinates.data)} particles.")
        print(f"Swarm local population per el is {len(swarm.particle_coordinates.data)/(n_els**dim)} particles.\n")

    #%%
    # Add some randomness to the particle distribution
    import numpy as np
    # np.random.seed(0)
    # with swarm.access(swarm.particle_coordinates):
    #     factor = 0.5*boxLength/n_els/ppcell
    #     swarm.particle_coordinates.data[:] += factor*np.random.rand(*swarm.particle_coordinates.data.shape)

    #%%
    # define these for convenience. 
    denseIndex = 0
    lightIndex = 1

    # material perturbation from van Keken et al. 1997
    wavelength = 2.0*boxLength
    k = 2. * np.pi / wavelength

    # init material variable
    with swarm.access(matSwarmVar):
        perturbation = offset + amplitude*np.cos( k*swarm.particle_coordinates.data[:,0] )
        matSwarmVar.data[:,0] = np.where( perturbation>swarm.particle_coordinates.data[:,1], lightIndex, denseIndex )

    from sympy import Piecewise, ceiling, Abs

    density = Piecewise( ( 0., Abs(matSwarmVar.fn - lightIndex)<0.5 ),
                         ( 1., Abs(matSwarmVar.fn - denseIndex)<0.5 ),
                         ( 0.,                                True ) )

    stokes.bodyforce = -density*mesh.N.j

    stokes.viscosity = Piecewise( ( viscosityRatio, Abs(matSwarmVar.fn - lightIndex)<0.5 ),
                                  (             1., Abs(matSwarmVar.fn - denseIndex)<0.5 ),
                                  (             1.,                                True ) )

    # note with petsc we always need to provide a vector of correct cardinality. 
    bnds = mesh.boundary
    stokes.add_dirichlet_bc( (0.,0.), [bnds.TOP,  bnds.BOTTOM], (0,1) )  # top/bottom: function, boundaries, components 
    stokes.add_dirichlet_bc( (0.,0.), [bnds.LEFT, bnds.RIGHT ], 0  )  # left/right: function, boundaries, components

    step = 0
    time = 0.
    nprint = 0.
    volume_int = uw.maths.Integral( mesh, 1. )
    volume = volume_int.evaluate()
    v_dot_v_int = uw.maths.Integral(mesh, stokes.u.fn.dot(stokes.u.fn))
    def vrms():
        import math
        v_dot_v = v_dot_v_int.evaluate()
        return math.sqrt(v_dot_v/volume)

    timeVal     = []
    vrmsVal     = []

    # Solve time
    stokes.solve(zero_init_guess=False, _force_setup=False)
    stime = delta_time()

    if time>=nprint:
        nprint += print_time
        outputFilename = os.path.join(outputPath,f"uw3_image_{str(step).zfill(4)}.png")
    ptime = delta_time()

    dt = stokes.dt()
    with swarm.access():
        vel_on_particles = uw.function.evaluate(stokes.u.fn,swarm.particle_coordinates.data)
    etime = delta_time()

    with swarm.access(swarm.particle_coordinates):
        swarm.particle_coordinates.data[:]+=dt*vel_on_particles
    atime = delta_time()

    vrms_val = vrms()
    if MPI.COMM_WORLD.rank==0:
        print(f"Step {str(step).rjust(3)}, time {time:6.2f}, vrms {vrms_val:.3e}, Time(s): Solve {stime:5.2f}, Plot {ptime:5.2f}, Evaluate {etime:5.2f}, Advect {atime:5.2f}")

    timeVal.append(time)
    vrmsVal.append(vrms_val)
    step+=1
    time+=dt

    return timeVal, vrmsVal


# %%
uw3_time, uw3_vrms = do_uw3()
