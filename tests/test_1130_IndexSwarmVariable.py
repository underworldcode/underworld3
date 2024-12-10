import underworld3 as uw
import numpy as np
import sympy
import pytest

# ### Test IndexSwarmVariable in gettting the right value on the Symmetrical Points

xmin, xmax = -1,1
ymin, ymax = -1,1
xres,yres = 2,2
dx = (xmax-xmin)/xres
dy = (ymax-ymin)/yres

ppdegree = 1
ppcont = True

fill_params = [2,3,4,5]


meshStructuredQuadBox = uw.meshing.StructuredQuadBox(elementRes=(int(xres), int(yres)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax))
meshUnstructuredSimplexbox_regular = uw.meshing.UnstructuredSimplexBox(cellSize=dx,  minCoords=(xmin, ymin), maxCoords=(xmax, ymax),regular=True,refinement=0)
meshUnstructuredSimplexbox_irregular = uw.meshing.UnstructuredSimplexBox(cellSize=dx,  minCoords=(xmin, ymin), maxCoords=(xmax, ymax),regular=False,refinement=0)

@pytest.mark.parametrize(
    "mesh",
    [
        meshStructuredQuadBox,
        meshUnstructuredSimplexbox_regular,
        meshUnstructuredSimplexbox_irregular,
    ],
)

def test_IndexSwarmVariable(mesh):
    Pmesh = uw.discretisation.MeshVariable("P", mesh, 1, degree=ppdegree,continuous=ppcont)

    for fill_param in fill_params:
        print(fill_param)
        swarm = uw.swarm.Swarm(mesh)
        material = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_degree=ppdegree,proxy_continuous=ppcont) 
        swarm.populate(fill_param=fill_param)
        
        amplitude, offset, wavelength= 0.5, 0., 1
        k = 2.0 * np.pi / wavelength
        interfaceSwarm = uw.swarm.Swarm(mesh)
        npoints = 101
        x = np.linspace(mesh.data[:,0].min(), mesh.data[:,0].max(), npoints)
        y = offset + amplitude * np.cos(k * x)
        interface_coords = np.ascontiguousarray(np.array([x,y]).T)
        interfaceSwarm.add_particles_with_coordinates(interface_coords)
        
        M0Index = 0
        M1Index = 1
        with swarm.access(material):
            perturbation = offset + amplitude * np.cos(k * swarm.particle_coordinates.data[:, 0])+0.01
            material.data[:, 0] = np.where(swarm.particle_coordinates.data[:, 1] <= perturbation, M0Index, M1Index)
        
        P0, P1 = 1,10
        P_fn = material.createMask([P0,P1])
    
        ## compare the value on the Symmetrical Point on the left and riht wall
        with mesh.access(Pmesh):
            Pmesh.data[:,0] = uw.function.evaluate(P_fn,Pmesh.coords)
            assert np.allclose(Pmesh.data[0],Pmesh.data[1], atol=0.01)
            assert np.allclose(Pmesh.data[2],Pmesh.data[3], atol=0.01)
            assert np.allclose(Pmesh.data[6],Pmesh.data[7], atol=0.01)
        del swarm
        del material
            

del meshStructuredQuadBox
del meshUnstructuredSimplexbox_regular
del meshUnstructuredSimplexbox_irregular