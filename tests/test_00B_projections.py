import underworld3 as uw
import numpy as np
import sympy

from underworld3.util_mesh import UnstructuredSimplexBox


def test_scalar_projection():

    mesh = UnstructuredSimplexBox(minCoords=(0.0,0.0), 
                                  maxCoords=(1.0,1.0), 
                                  cellSize=1.0/32.0)

    # The following test projects scalar values defined on a 
    # swarm using a Sympy function to a mesh variable.

    # Create Sympy Function f(x,y), function of 
    # mesh coordinates.
    x = mesh.N.x
    y = mesh.N.y

    s_fn = sympy.cos(5.0*sympy.pi * x) * \
           sympy.cos(5.0*sympy.pi * y)

    # solution mesh variable
    s_soln  = uw.mesh.MeshVariable("T", mesh, 1, degree=2)

    # Build a swarm and create swarm variable
    swarm     = uw.swarm.Swarm(mesh=mesh)
    s_values  = uw.swarm.SwarmVariable("Ss", swarm, 1, proxy_degree=3)
    swarm.populate(fill_param=3)
    
    # Set the values on the swarm variable
    with swarm.access(s_values):
        s_values.data[:,0]  = uw.function.evaluate(s_fn, swarm.data)

    # Prepare projection of swarm values onto the mesh nodes.
    scalar_projection = uw.systems.Projection(mesh, s_soln)
    scalar_projection.uw_function = s_values.fn
    scalar_projection.smoothing = 1.0e-6
    scalar_projection.solve()

    return


def test_vector_projection():

    mesh = UnstructuredSimplexBox(minCoords=(0.0,0.0), 
                                  maxCoords=(1.0,1.0), 
                                  cellSize=1.0/32.0)
    
    # Create Sympy Function f(x,y), function of 
    # mesh coordinates.
    x = mesh.N.x
    y = mesh.N.y

    s_fn_x = sympy.cos(5.0*sympy.pi * x) * \
             sympy.cos(5.0*sympy.pi * y)
    s_fn_y = sympy.sin(5.0*sympy.pi * x) * \
             sympy.sin(5.0*sympy.pi * y)

    # solution mesh variable
    v_soln  = uw.mesh.MeshVariable('U', mesh, mesh.dim, degree=2)

    # Build a swarm and create swarm variable
    swarm     = uw.swarm.Swarm(mesh=mesh)
    v_values  = uw.swarm.SwarmVariable("Vs", swarm, mesh.dim, proxy_degree=3)
    swarm.populate(fill_param=3)
    
    # Set the values on the swarm variable
    with swarm.access(v_values):
        v_values.data[:,0]  = uw.function.evaluate(s_fn_x, swarm.data)    
        v_values.data[:,1]  = uw.function.evaluate(s_fn_y, swarm.data)

    vector_projection = uw.systems.Vector_Projection(mesh, v_soln)
    vector_projection.uw_function = v_values.fn
    vector_projection.smoothing = 1.0e-3
    
    vector_projection.add_dirichlet_bc( (0.0,), "Right",  (0,) )
    vector_projection.add_dirichlet_bc( (0.0,), "Top",    (1,) )
    vector_projection.add_dirichlet_bc( (0.0,), "Bottom", (1,) )
    
    vector_projection.solve()

    return
