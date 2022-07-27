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
    n_els = 32
boxLength      = 0.9142
boxHeight      = 1.0
viscosityRatio = 1.0
stokes_inner_tol = 1e-6
stokes_outer_tol = 1e-5
fill_param = 3
amplitude  = 0.02
offset     = 0.2
print_time = 10
model_end_time = 300.
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
    sys.pushErrorHandler("traceback")


    mesh = uw.discretisation.Box(elementRes=(    n_els,)*dim, 
                        minCoords =(       0.,)*dim, 
                        maxCoords =(boxLength,1.),
                        simplex=False )
    u_degree = 1
    stokes = Stokes(mesh, u_degree=u_degree )
    
    # Create a variable to store material variable
    # matMeshVar = uw.discretisation.MeshVariable("matmeshvar", mesh, 1, uw.VarType.SCALAR, degree=u_degree+1)

    # Create swarm
    swarm  = uw.swarm.Swarm(mesh)
    # Add variable for material
    matSwarmVar      = swarm.add_variable(name="matSwarmVar",      num_components=1, dtype=PETSc.IntType)
    # Note that `fill_param` specifies particles per cell per dim.
    swarm.populate(fill_param=fill_param)

    # Add some randomness to the particle distribution
    import numpy as np
    np.random.seed(0)
    with swarm.access(swarm.particle_coordinates):
        factor = 0.5*boxLength/n_els/fill_param
        swarm.particle_coordinates.data[:] += factor*np.random.rand(*swarm.particle_coordinates.data.shape)

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

    while time<model_end_time:
        # Solve time
        stokes.solve(zero_init_guess=False, _force_setup=False)
        stime = delta_time()

        if time>=nprint:
            nprint += print_time
            import plot
            figs = plot.Plot(rulers=True)
            # fig.edges(mesh)
            with swarm.access(),mesh.access():
                figs.swarm_points(swarm, matSwarmVar.data, pointsize=4, colourmap="blue green", colourbar=False, title=time)
                figs.vector_arrows(mesh, stokes.u.data)
                # fig.nodes(mesh,matMeshVar.data,colourmap="blue green", pointsize=6, pointtype=4)
            outputFilename = os.path.join(outputPath,f"uw3_image_{str(step).zfill(4)}.png")
            figs.image(outputFilename)
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
def do_uw2():
    import underworld as uw
    from underworld import function as fn
    import underworld.visualisation as vis
    import math
    import numpy as np

    mesh = uw.discretisation.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                    elementRes  = (n_els, n_els), 
                                    minCoord    = (0., 0.), 
                                    maxCoord    = (boxLength, boxHeight))

    velocityField = mesh.add_variable(         nodeDofCount=2 )
    pressureField = mesh.subMesh.add_variable( nodeDofCount=1 )

    # initialise 
    velocityField.data[:] = [0.,0.]
    pressureField.data[:] = 0.

    # Create a swarm.
    swarm = uw.swarm.Swarm( mesh=mesh )

    # Create a data variable. It will be used to store the material index of each particle.
    materialIndex = swarm.add_variable( dataType="int", count=1 )

    # Create a layout object, populate the swarm with particles.
    swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=30 )
    swarm.populate_using_layout( layout=swarmLayout )

    # define these for convience. 
    denseIndex = 0
    lightIndex = 1

    # material perturbation from van Keken et al. 1997
    wavelength = 2.0*boxLength
    k = 2. * math.pi / wavelength

    # Create function to return particle's coordinate
    coord = fn.coord()

    # Define the material perturbation, a function of the x coordinate (accessed by `coord[0]`).
    perturbationFn = offset + amplitude*fn.math.cos( k*coord[0] )

    # Setup the conditions list. 
    # If z is less than the perturbation, set to lightIndex.
    conditions = [ ( perturbationFn > coord[1] , lightIndex ),
                (                      True , denseIndex ) ]

    # The swarm is passed as an argument to the evaluation, providing evaluation on each particle.
    # Results are written to the materialIndex swarm variable.
    materialIndex.data[:] = fn.branching.conditional( conditions ).evaluate(swarm)

    # Set a density of '0.' for light material, '1.' for dense material.
    densityMap   = { lightIndex:0., denseIndex:1. }
    densityFn    = fn.branching.map( fn_key = materialIndex, mapping = densityMap )

    # Set a viscosity value of '1.' for both materials.
    viscosityMap = { lightIndex:viscosityRatio, denseIndex:1. }
    fn_viscosity  = fn.branching.map( fn_key = materialIndex, mapping = viscosityMap )

    # Define a vertical unit vector using a python tuple.
    z_hat = ( 0.0, 1.0 )

    # Create buoyancy force vector
    buoyancyFn = -densityFn*z_hat

    # Construct node sets using the mesh specialSets
    iWalls = mesh.specialSets["Left_VertexSet"]   + mesh.specialSets["Right_VertexSet"]
    jWalls = mesh.specialSets["Bottom_VertexSet"] + mesh.specialSets["Top_VertexSet"]
    allWalls = iWalls + jWalls

    # Prescribe degrees of freedom on each node to be considered Dirichlet conditions.
    # In the x direction on allWalls flag as Dirichlet
    # In the y direction on jWalls (horizontal) flag as Dirichlet
    stokesBC = uw.conditions.DirichletCondition( variable      = velocityField, 
                                                indexSetsPerDof = (allWalls, jWalls) )

    stokes = uw.systems.Stokes( velocityField = velocityField, 
                                pressureField = pressureField,
                                conditions    = stokesBC,
                                fn_viscosity  = fn_viscosity, 
                                fn_bodyforce  = buoyancyFn )

    solver = uw.systems.Solver( stokes )

    # Optional solver settings
    if(uw.mpi.size==1):
        solver.set_inner_method("lu")
    solver.set_inner_rtol(stokes_inner_tol) 
    solver.set_outer_rtol(stokes_outer_tol) 

    # Create a system to advect the swarm
    advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=1 )

    # Initialise time and timestep.
    time = 0.
    nprint = 0.
    step = 0

    # parameters for output
    timeVal     = []
    vrmsVal     = []

    # define an update function
    def update():
        dt = advector.get_max_dt() # retrieve the maximum possible timestep from the advection system.
        advector.integrate(dt)     # advect step.
        return time+dt, step+1

    while time < model_end_time:

        # Get solution
        solver.solve()
        
        # Calculate the RMS velocity.
        vrms = stokes.velocity_rms()

        # Record values into arrays
        if(uw.mpi.rank==0):
            vrmsVal.append(vrms)
            timeVal.append(time)

        if time>=nprint:
            nprint += print_time
            import plot
            figs = plot.Plot(rulers=True)
            def uw2points(self,swarm,values,**kwargs):
                ptsobj = self.points('swarm_points', **kwargs)
                ptsobj.vertices(swarm.data)
                ptsobj.values(values)

            uw2points(figs, swarm, materialIndex.data, pointsize=4, colourmap="blue green", colourbar=False, title=time)
            outputFilename = os.path.join(outputPath,f"uw2_image_{str(step).zfill(4)}.png")
            figs.image(outputFilename)

        if(uw.mpi.rank==0):
            print(f"Step {str(step).rjust(3)}, time {time:6.2f}, vrms {vrms:.3e}")

        # We are finished with current timestep, update.
        time, step = update()

    return timeVal, vrmsVal


# %%
uw3_time, uw3_vrms = do_uw3()

# %%
if run_uw2:
    uw2_time, uw2_vrms = do_uw2()



# %%
if MPI.COMM_WORLD.rank==0:
    if   np.isclose(viscosityRatio, 1.00) :
        data = np.loadtxt(os.path.join(inputPath,'VrmsCaseA.txt'), unpack=True )
    elif np.isclose(viscosityRatio, 0.10) :
        data = np.loadtxt(os.path.join(inputPath,'VrmsCaseB.txt'), unpack=True )
    elif np.isclose(viscosityRatio, 0.01) :
        data = np.loadtxt(os.path.join(inputPath,'VrmsCaseC.txt'), unpack=True )
    else :
        print('No specific data found - default to Case A')
        data = np.loadtxt(os.path.join(inputPath,'VrmsCaseA.txt'), unpack=True )

    # Load into data arrays to compare with timevals and vrmsvals from above.
    timeCompare, vrmsCompare = data[0], data[1] 

    import matplotlib.pyplot as pyplot
    fig = pyplot.figure()
    fig.set_size_inches(12, 6)
    ax = fig.add_subplot(1,1,1)
    ax.plot(timeCompare, vrmsCompare, color = 'black')
    ax.plot(uw3_time, uw3_vrms, color = 'blue', marker=".", markersize=10, label="uw3") 
    if run_uw2:
        ax.plot(uw2_time, uw2_vrms, color = 'red', marker=".", markersize=10, label="uw2") 
    ax.set_xlabel('Time')
    ax.set_ylabel('RMS velocity')
    ax.set_xlim([0.0,1000.0])
    ax.legend()
    fig.savefig(os.path.join(outputPath,"vrms.png"))

    # test for max vrms time/value
    # switch for numpy arrays
    uw3_vrms = np.array(uw3_vrms)
    uw3_time = np.array(uw3_time)
    uw3_maxvrms = uw3_vrms.max()
    uw3_maxvrms_time = uw3_time[uw3_vrms.argmax()]

    expected_maxvrms = vrmsCompare.max()
    expected_maxvrms_time = timeCompare[vrmsCompare.argmax()]

    if "UW_LONGTEST" in os.environ:
        rtol = 0.02
    else:
        rtol = 0.1

    if not np.allclose(uw3_maxvrms,expected_maxvrms,rtol=rtol):
        raise RuntimeError(f"Encountered max VRMS ({uw3_maxvrms}) not sufficiently close to expected value ({expected_maxvrms})")
    if not np.allclose(uw3_maxvrms_time,expected_maxvrms_time,rtol=rtol):
        raise RuntimeError(f"Encountered max VRMS time ({uw3_maxvrms_time}) not sufficiently close to expected value ({expected_maxvrms_time})")
