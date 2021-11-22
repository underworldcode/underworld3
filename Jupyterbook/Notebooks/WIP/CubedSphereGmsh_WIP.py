from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
options["ksp_rtol"] =  1.0e-3
options["ksp_monitor_short"] = None
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
# options["snes_view"]=None
# options["snes_test_jacobian"] = None
options["snes_max_it"] = 1
options["pc_type"] = "fieldsplit"
options["pc_fieldsplit_type"] = "schur"
options["pc_fieldsplit_schur_factorization_type"] ="full"
options["pc_fieldsplit_schur_precondition"] = "a11"
options["fieldsplit_velocity_pc_type"] = "lu"
options["fieldsplit_pressure_ksp_rtol"] = 1.e-3
options["fieldsplit_pressure_pc_type"] = "lu"


# +
# This (guy) sets up the visualisation defaults

import numpy as np
import pyvista as pv
import vtk

pv.global_theme.background = 'white'
pv.global_theme.window_size = [1000, 500]
pv.global_theme.antialiasing = True
pv.global_theme.jupyter_backend = 'panel'
pv.global_theme.smooth_shading = True
# -

import pygmsh
import meshio
import gmsh

# +
gmsh.initialize()
gmsh.model.add("cubed")

r2 = 1.0 / np.sqrt(3)
res = 5
lc = 1.0/res

# gmsh.model.geo.addPoint(0,0,0,lc, 1)

# The 8 corners of the cubes

cube = np.array([[-1,-1,-1],
                 [ 1,-1,-1],
                 [ 1, 1,-1],
                 [-1, 1,-1],
                 [-1,-1, 1],
                 [ 1,-1, 1],
                 [ 1, 1, 1],
                 [-1, 1, 1] ] )

pt = [-1]*8
for i in range(0,8):
    pt[i] = gmsh.model.geo.add_point(cube[i,0], cube[i,0], cube[i,0], lc, tag=-1)

    
loop = [-1]*6   
face = [-1]*6   


loop[0] = gmsh.model.geo.add_polyline([pt[0], pt[1], pt[2], pt[3], pt[0]])
print(loop[0])


face[0] = gmsh.model.geo.add_curve_loop([loop[0]])
gmsh.model.mesh.setTransfiniteCurve(face[0], 50)
    
print(face[0])

# gmsh.model.geo.add_line(10,11,100)
# gmsh.model.geo.add_line(11,12,101)
# gmsh.model.geo.add_line(12,13,102)
# gmsh.model.geo.add_line(13,10,103)

# gmsh.model.geo.mesh.set_transfinite_curve(100, res+1)
# gmsh.model.geo.mesh.set_transfinite_curve(101, res+1)
# gmsh.model.geo.mesh.set_transfinite_curve(102, res+1)
# gmsh.model.geo.mesh.set_transfinite_curve(103, res+1)

# gmsh.model.geo.add_curve_loop([100,101,102,103], tag=1000)
# gmsh.model.geo.add_plane_surface([1000], tag=10000)
# gmsh.model.geo.mesh.set_transfinite_surface(10000)

# gmsh.model.geo.synchronize()

# sq2 = gmsh.model.geo.copy([(2,10000)])
# gmsh.model.geo.translate(sq2, 0.0,1.0, 0.0)

# gmsh.model.geo.synchronize()

# for dimtag in sq2:
#     if dimtag[0] == 1:
#         gmsh.model.mesh.set_transfinite_curve(dimtag[1], res+1)
#     if dimtag[0] == 2:
#         gmsh.model.mesh.set_transfinite_surface(dimtag[1])

# print(sq2)

# # r1 = gmsh.model.geo.extrude([(1,100)], 0.0, 1.0, 0.0, [res])
# # r2 = gmsh.model.geo.extrude([(1,101)], 0.0, 1.0, 0.0, [res])

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(dim=2)

gmsh.write("ignore_ex.vtk")

# gmsh.finalize()

# -

pvmesh = pv.read("ignore_ex.vtk")
pvmesh.plot(show_edges=True)

sq2











3=1


cubed_sphere_mesh_shell = uw.mesh.StructuredCubeSphereShellMesh(elementRes=(11,5), radius_inner=0.5,
                                        radius_outer=1.0, simplex=False)


cubed_sphere_mesh_ball = uw.mesh.StructuredCubeSphereBallMesh(elementRes=9,
                                        radius_outer=1.0, simplex=True)


cubed_sphere_mesh = cubed_sphere_mesh_ball

# +


pvmesh = cubed_sphere_mesh.mesh2pyvista()

# pvmesh.cell_data['my cell values'] = np.arange(pvmesh.n_cells)
# pvmesh.plot(scalars='my cell values', show_edges=True)


clipped_stack = pvmesh.clip(origin=(0.00001,0.0,0.0), normal=(1, 0, 0), invert=False)

pl = pv.Plotter()

# pl.add_mesh(pvstack,'Blue', 'wireframe' )
pl.add_mesh(clipped_stack, cmap="coolwarm", edge_color="Black", show_edges=True, 
              use_transparency=False)
pl.show()


# +
def cs_build_pygmsh(
                elementRes  = 16, 
                radius_outer =1.0,
                simplex   =False
                ):

            import meshio
            import gmsh

            gmsh.initialize()
            gmsh.model.add("cubed")

            lc = 0.0

            r2 = radius_outer / np.sqrt(3)

            res = elementRes+1

            gmsh.model.geo.addPoint(0,0,0,lc, 1)

            # The 8 corners of the cubes

            gmsh.model.geo.addPoint(-r2, -r2, -r2, lc, 100)
            gmsh.model.geo.addPoint( r2, -r2, -r2, lc, 101)
            gmsh.model.geo.addPoint( r2,  r2, -r2, lc, 102)
            gmsh.model.geo.addPoint(-r2,  r2, -r2, lc, 103)
            gmsh.model.geo.addPoint(-r2, -r2,  r2, lc, 104)
            gmsh.model.geo.addPoint( r2, -r2,  r2, lc, 105)
            gmsh.model.geo.addPoint( r2,  r2,  r2, lc, 106)
            gmsh.model.geo.addPoint(-r2,  r2,  r2, lc, 107)

            # The 12 edges of the cube2

            gmsh.model.geo.addCircleArc(100,1,101, 1000)
            gmsh.model.geo.addCircleArc(101,1,102, 1001)
            gmsh.model.geo.addCircleArc(102,1,103, 1002)
            gmsh.model.geo.addCircleArc(103,1,100, 1003)

            gmsh.model.geo.addCircleArc(101,1,105, 1004)
            gmsh.model.geo.addCircleArc(102,1,106, 1005)
            gmsh.model.geo.addCircleArc(103,1,107, 1006)
            gmsh.model.geo.addCircleArc(100,1,104, 1007)

            gmsh.model.geo.addCircleArc(104,1,105, 1008)
            gmsh.model.geo.addCircleArc(105,1,106, 1009)
            gmsh.model.geo.addCircleArc(106,1,107, 1010)
            gmsh.model.geo.addCircleArc(107,1,104, 1011)

            ## These should all be transfinite lines

            for i in range(1000, 1012):
                gmsh.model.geo.mesh.set_transfinite_curve(i, res)

            # The 6 faces of the cube2

            gmsh.model.geo.addCurveLoop([1000, 1004, 1008, 1007], 10000, reorient=True)
            gmsh.model.geo.addCurveLoop([1001, 1005, 1009, 1004], 10001, reorient=True)
            gmsh.model.geo.addCurveLoop([1002, 1006, 1010, 1005], 10002, reorient=True)
            gmsh.model.geo.addCurveLoop([1003, 1007, 1011, 1006], 10003, reorient=True)
            gmsh.model.geo.addCurveLoop([1000, 1003, 1002, 1001], 10004, reorient=True)
            gmsh.model.geo.addCurveLoop([1008, 1009, 1010, 1011], 10005, reorient=True)

            gmsh.model.geo.add_surface_filling([10000], 10101, sphereCenterTag=1)
            gmsh.model.geo.add_surface_filling([10001], 10102, sphereCenterTag=1)
            gmsh.model.geo.add_surface_filling([10002], 10103, sphereCenterTag=1)
            gmsh.model.geo.add_surface_filling([10003], 10104, sphereCenterTag=1)
            gmsh.model.geo.add_surface_filling([10004], 10105, sphereCenterTag=1)
            gmsh.model.geo.add_surface_filling([10005], 10106, sphereCenterTag=1)

            for i in range(10101, 10107):
                gmsh.model.geo.mesh.setTransfiniteSurface(i, "Left")
                if not simplex:
                    gmsh.model.geo.mesh.setRecombine(2, i)

            gmsh.model.geo.synchronize()

            # outer surface / inner_surface
            gmsh.model.geo.add_surface_loop([10101, 10102, 10103, 10104, 10105, 10106], 10111)
            gmsh.model.geo.add_volume([10111], 100001)

            gmsh.model.geo.synchronize()

            gmsh.model.mesh.set_transfinite_volume(100001)
            if not simplex:
                gmsh.model.geo.mesh.setRecombine(3, 100001)

            # gmsh.model.mesh.set_size([(3,100001)],10.0)

            gmsh.model.geo.remove_all_duplicates()
            gmsh.model.remove_entities([[2,10111]], recursive=True)
            gmsh.model.mesh.generate(dim=3)
            gmsh.model.mesh.removeDuplicateNodes()

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".msh") as tfile:
                gmsh.write(tfile.name)
                
                cubed_sphere_ball_mesh = meshio.read(tfile.name)
                cubed_sphere_ball_mesh.remove_lower_dimensional_cells()


            gmsh.finalize()

 


            return cubed_sphere_ball_mesh

cs_ball = cs_build_pygmsh(8, 1.0, simplex=True)
cs_ball_mesh = uw.mesh.MeshFromMeshIO(dim=3, meshio=cs_ball, cell_size=100.0, simplex=True)


# +
def puffball_build_pygmsh(
            elementRes, 
            radius_outer,
            simplex, 
            ):

        import pygmsh 
        
        r = radius_outer

        minCoords = (-r,-r,-r)
        maxCoords = ( r, r, r)

        xx = maxCoords[0]-minCoords[0]
        yy = maxCoords[1]-minCoords[1]
        zz = maxCoords[2]-minCoords[2]

        x_sep=(maxCoords[0] - minCoords[0])/elementRes

        with pygmsh.geo.Geometry() as geom:
            points = [geom.add_point([x, minCoords[1], minCoords[2]], x_sep) for x in [minCoords[0], maxCoords[0]]]
            line = geom.add_line(*points)

            _, rectangle, _ = geom.extrude(line, translation_axis=[0.0, maxCoords[1]-minCoords[1], 0.0], 
                                           num_layers=elementRes, recombine=(not simplex))

            geom.extrude(
                    rectangle,
                    translation_axis=[0.0, 0.0, maxCoords[2]-minCoords[2]],
                    num_layers=elementRes,
                    recombine=(not simplex),
                )

            hex_box = geom.generate_mesh()
            hex_box.remove_lower_dimensional_cells()

#             # Now adjust the point locations
#             # first make a pyramid that subtends the correct angle at each level

#             hex_box.points[:,0] *= hex_box.points[:,2] * np.tan(theta/2) 
#             hex_box.points[:,1] *= hex_box.points[:,2] * np.tan(phi/2) 

            # second, adjust the distance so each layer forms a spherical cap 
    
            ac = np.abs(hex_box.points)

            targetR = np.maximum(np.maximum(ac[:,0],ac[:,1]), ac[:,2])
            actualR = np.sqrt(hex_box.points[:,0]**2 + hex_box.points[:,1]**2 + hex_box.points[:,2]**2)

            hex_box.points[:,0] *= (targetR / actualR)
            hex_box.points[:,1] *= (targetR / actualR)
            hex_box.points[:,2] *= (targetR / actualR)

            # finalise geom context

        return hex_box
## 



puffball = puffball_build_pygmsh(10,1.0,simplex=True)
puffball_mesh = uw.mesh.MeshFromMeshIO(dim=3, meshio=puffball, cell_size=100.0, simplex=True)
# -

puffball_mesh.mesh2pyvista(elementType=vtk.VTK_TETRA).plot(show_edges=True)





# +
def cs_shell_build_pygmsh(elementRes=(16, 8), radius_outer=1.0, radius_inner =0.5, simplex=False):


        import pygmsh 
        import meshio

        l = 0.0

        inner_radius = radius_inner
        outer_radius = radius_outer
        nodes = elementRes[0]+1  # resolution of the cube laterally
        layers= elementRes[1]

        with pygmsh.geo.Geometry() as geom:
            cpoint = geom.add_point([0.0,0.0,0.0], l)

            genpt = [0,0,0,0,0,0,0,0]

            # 8 corners of the cube 

            r2 = 1.0 / np.sqrt(3.0) # Generate a unit sphere

            genpt[0] = geom.add_point([ -r2, -r2, -r2],  l)
            genpt[1] = geom.add_point([  r2, -r2, -r2],  l)
            genpt[2] = geom.add_point([  r2,  r2, -r2],  l)
            genpt[3] = geom.add_point([ -r2,  r2, -r2],  l)
            genpt[4] = geom.add_point([ -r2, -r2,  r2],  l)
            genpt[5] = geom.add_point([  r2, -r2,  r2],  l)
            genpt[6] = geom.add_point([  r2,  r2,  r2],  l)
            genpt[7] = geom.add_point([ -r2,  r2,  r2],  l)


            # 12 edges of the cube

            b_circ00 = geom.add_line(genpt[0], genpt[1])
            b_circ01 = geom.add_line(genpt[1], genpt[2])
            b_circ02 = geom.add_line(genpt[2], genpt[3])
            b_circ03 = geom.add_line(genpt[0], genpt[3])

            b_circ04 = geom.add_line(genpt[1], genpt[5])
            b_circ05 = geom.add_line(genpt[2], genpt[6])
            b_circ06 = geom.add_line(genpt[3], genpt[7])
            b_circ07 = geom.add_line(genpt[0], genpt[4])

            b_circ08 = geom.add_line(genpt[4], genpt[5])
            b_circ09 = geom.add_line(genpt[5], genpt[6])
            b_circ10 = geom.add_line(genpt[6], genpt[7])
            b_circ11 = geom.add_line(genpt[4], genpt[7])

            for arc in [b_circ00, b_circ01, b_circ02, b_circ03,
                        b_circ04, b_circ05, b_circ06, b_circ07,
                        b_circ08, b_circ09, b_circ10, b_circ11 ]:

                    geom.set_transfinite_curve(arc, num_nodes=nodes, 
                                            mesh_type="Progression", coeff=1.0)

            # 6 Cube faces

            face00_loop = geom.add_curve_loop([b_circ00, b_circ04, -b_circ08, -b_circ07])
            face00 = geom.add_surface(face00_loop) 
            geom.set_transfinite_surface(face00, arrangement="Left",
                                        corner_pts = [genpt[0], genpt[1], genpt[5], genpt[4]])   


            face01_loop = geom.add_curve_loop([b_circ01, b_circ05, -b_circ09, -b_circ04])
            face01 = geom.add_surface(face01_loop) 
            geom.set_transfinite_surface(face01, arrangement="Left",
                                        corner_pts = [genpt[1], genpt[2], genpt[6], genpt[5]])   


            face02_loop = geom.add_curve_loop([b_circ02, b_circ06, -b_circ10, -b_circ05])
            face02 = geom.add_surface(face02_loop) 
            geom.set_transfinite_surface(face02, arrangement="Left",
                                        corner_pts = [genpt[2], genpt[3], genpt[7], genpt[6]])   


            face03_loop = geom.add_curve_loop([-b_circ03, b_circ07, b_circ11, -b_circ06])
            face03 = geom.add_surface(face03_loop) 
            geom.set_transfinite_surface(face03, arrangement="Left",
                                        corner_pts = [genpt[3], genpt[0], genpt[4], genpt[7]])   


            face04_loop = geom.add_curve_loop([-b_circ00, b_circ03, -b_circ02, -b_circ01])
            face04 = geom.add_surface(face04_loop) 
            geom.set_transfinite_surface(face04, arrangement="Left",
                                        corner_pts = [genpt[1], genpt[0], genpt[3], genpt[2]])   


            face05_loop = geom.add_curve_loop([b_circ08, b_circ09,  b_circ10, -b_circ11])
            face05 = geom.add_surface(face05_loop) 
            geom.set_transfinite_surface(face05, arrangement="Left",
                                        corner_pts = [genpt[4], genpt[5], genpt[6], genpt[7]])   


            geom.set_recombined_surfaces([face00, face01, face02, face03, face04, face05])
            shell = geom.add_surface_loop([face00, face01, face02, face03, face04, face05])
            
            geom.dilate()
            
            volume = geom.add_volume(shell)

            two_D_cubed_sphere = geom.generate_mesh(dim=3, verbose=False)
            two_D_cubed_sphere.remove_orphaned_nodes()
            two_D_cubed_sphere.remove_lower_dimensional_cells()
            
            
            return two_D_cubed_sphere
        

cs_shell = cs_shell_build_pygmsh(
            elementRes=(6, 3), 
            radius_outer=1.0,
            radius_inner =0.5,
            simplex=True)
        
cs_shell_mesh = uw.mesh.MeshFromMeshIO(dim=3, meshio=cs_shell, cell_size=100.0, simplex=True)


# +
cs_shell.write("ignore_csh.vtk")
pvmesh = pv.read("ignore_csh.vtk")

clipped_stack = pvmesh.clip(origin=(0.00001,0.0,0.0), normal=(1, 0, 0), invert=False)

pl = pv.Plotter()

# pl.add_mesh(pvmesh,'Blue', 'wireframe' )
pl.add_mesh(clipped_stack, cmap="coolwarm", edge_color="Black", show_edges=True, 
              use_transparency=False)
pl.show()
# -

cubed_sphere_mesh = cs_shell_mesh

# +
# Create Stokes object
stokes = Stokes(cubed_sphere_mesh,u_degree=2,p_degree=1)
# Constant visc
stokes.viscosity = 1.

# Velocity boundary conditions
stokes.add_dirichlet_bc( (0.,0.,0.), cubed_sphere_mesh.boundary.ALL_BOUNDARIES, (0,1,2) )

# +
# Create a density structure

import sympy

dens_ball = 10.
dens_other = 1.
position_ball = 0.75*cubed_sphere_mesh.N.k
radius_ball = 0.5

off_rvec = cubed_sphere_mesh.rvec - position_ball
abs_r = off_rvec.dot(off_rvec)
density = sympy.Piecewise( ( dens_ball,    abs_r < radius_ball**2 ),
                           ( dens_other,                   True ) )
density
# -

# Write density into a variable for saving
densvar = uw.mesh.MeshVariable("density",cubed_sphere_mesh,1)

with cubed_sphere_mesh.access(densvar):
    densvar.data[:,0] = uw.function.evaluate(density,densvar.coords)
    print(densvar.data.max())

unit_rvec = cubed_sphere_mesh.rvec / (1.0e-10+sympy.sqrt(cubed_sphere_mesh.rvec.dot(cubed_sphere_mesh.rvec)))
stokes.bodyforce = -unit_rvec*density
stokes.bodyforce

stokes.solve()

# +
pv_vtkmesh = cubed_sphere_mesh.mesh2pyvista(elementType=vtk.VTK_TETRA)

umag = stokes.u.fn.dot(stokes.u.fn)

pv_vtkmesh.point_data['density'] = uw.function.evaluate(density,cubed_sphere_mesh.data)
# pv_vtkmesh.point_data['umag'] = uw.function.evaluate(umag,cubed_sphere_mesh.data)
# -

clipped = pv_vtkmesh.clip(normal=(1, 0, 0), invert=False)
contours = pv_vtkmesh.contour([1.0,5.0, 10.0], scalars="density")

# +
pl = pv.Plotter()
pl.add_mesh(clipped, cmap="coolwarm", edge_color="Black", show_edges=True, 
            scalars="density",  use_transparency=False)

with cubed_sphere_mesh.access():
    usol = stokes.u.data
    
pl.add_arrows(stokes.u.coords, usol, mag=3.0)
# pl.add_mesh(contours, opacity=0.5)

pl.show()
# -

usol

3=1

# +
# Try the regional mesh approach to puff up a spherical ball

# +



import pygmsh 

r = 1.0
elementRes = 8
simplex=True

minCoords = (-r,-r,-r)
maxCoords = ( r, r, r)

xx = maxCoords[0]-minCoords[0]
yy = maxCoords[1]-minCoords[1]
zz = maxCoords[2]-minCoords[2]

x_sep=(maxCoords[0] - minCoords[0])/elementRes

with pygmsh.geo.Geometry() as geom:
    points = [geom.add_point([x, minCoords[1], minCoords[2]], x_sep) for x in [minCoords[0], maxCoords[0]]]
    line = geom.add_line(*points)

    _, rectangle, _ = geom.extrude(line, translation_axis=[0.0, maxCoords[1]-minCoords[1], 0.0], 
                                   num_layers=elementRes, recombine=(not simplex))
    
    
    

    geom.extrude(
            rectangle,
            translation_axis=[0.0, 0.0, maxCoords[2]-minCoords[2]],
            num_layers=elementRes,
            recombine=(not simplex),
        )
    

    hex_box = geom.generate_mesh()
    hex_box.remove_lower_dimensional_cells()



    # second, adjust the distance so each layer forms a spherical cap 

    ac = np.abs(hex_box.points)

    targetR = np.maximum(np.maximum(ac[:,0],ac[:,1]), ac[:,2])
    actualR = np.sqrt(hex_box.points[:,0]**2 + hex_box.points[:,1]**2 + hex_box.points[:,2]**2)

    hex_box.points[:,0] *= (targetR / actualR)
    hex_box.points[:,1] *= (targetR / actualR)
    hex_box.points[:,2] *= (targetR / actualR)

    # finalise geom context




# +
hex_box.write("ignore_pfb.vtk")
pvmesh = pv.read("ignore_pfb.vtk")

clipped_stack = pvmesh.clip(origin=(0.00001,0.0,0.0), normal=(1, 0, 0), invert=False)

pl = pv.Plotter()

# pl.add_mesh(pvmesh,'Blue', 'wireframe' )
pl.add_mesh(clipped_stack, cmap="coolwarm", edge_color="Black", show_edges=True, 
              use_transparency=False)
pl.show()


# +
def cs_build_pygmsh(
                elementRes  = 16, 
                radius_outer =1.0,
                simplex   =False
                ):

            import meshio
            import gmsh

            gmsh.initialize()
            gmsh.model.add("cubed")

            lc = 0.0

            ro = radius_outer / np.sqrt(3)
            ri = 0.5 * ro

            res = elementRes+1

            gmsh.model.geo.addPoint(0,0,0,lc, 1)

            # The 8 corners of the cubes

            gmsh.model.geo.addPoint(-ro, -ro, -ro, lc, 100)
            gmsh.model.geo.addPoint( ro, -ro, -ro, lc, 101)
            gmsh.model.geo.addPoint( ro,  ro, -ro, lc, 102)
            gmsh.model.geo.addPoint(-ro,  ro, -ro, lc, 103)
            gmsh.model.geo.addPoint(-ro, -ro,  ro, lc, 104)
            gmsh.model.geo.addPoint( ro, -ro,  ro, lc, 105)
            gmsh.model.geo.addPoint( ro,  ro,  ro, lc, 106)
            gmsh.model.geo.addPoint(-ro,  ro,  ro, lc, 107)

            gmsh.model.geo.addPoint(-ri, -ri, -ri, lc, 200)
            gmsh.model.geo.addPoint( ri, -ri, -ri, lc, 201)
            gmsh.model.geo.addPoint( ri,  ri, -ri, lc, 202)
            gmsh.model.geo.addPoint(-ri,  ri, -ri, lc, 203)
            gmsh.model.geo.addPoint(-ri, -ri,  ri, lc, 204)
            gmsh.model.geo.addPoint( ri, -ri,  ri, lc, 205)
            gmsh.model.geo.addPoint( ri,  ri,  ri, lc, 206)
            gmsh.model.geo.addPoint(-ri,  ri,  ri, lc, 207)


            # The 12 edges of the cube2

            gmsh.model.geo.addLine(100, 101, 1000)
            gmsh.model.geo.addLine(101, 102, 1001)
            gmsh.model.geo.addLine(102, 103, 1002)
            gmsh.model.geo.addLine(103, 100, 1003)

            gmsh.model.geo.addLine(101, 105, 1004)
            gmsh.model.geo.addLine(102, 106, 1005)
            gmsh.model.geo.addLine(103, 107, 1006)
            gmsh.model.geo.addLine(100, 104, 1007)

            gmsh.model.geo.addLine(104, 105, 1008)
            gmsh.model.geo.addLine(105, 106, 1009)
            gmsh.model.geo.addLine(106, 107, 1010)
            gmsh.model.geo.addLine(107, 104, 1011)

            gmsh.model.geo.addLine(200, 201, 2000)
            gmsh.model.geo.addLine(201, 202, 2001)
            gmsh.model.geo.addLine(202, 203, 2002)
            gmsh.model.geo.addLine(203, 200, 2003)

            gmsh.model.geo.addLine(201, 205, 2004)
            gmsh.model.geo.addLine(202, 206, 2005)
            gmsh.model.geo.addLine(203, 207, 2006)
            gmsh.model.geo.addLine(200, 204, 2007)

            gmsh.model.geo.addLine(204, 205, 2008)
            gmsh.model.geo.addLine(205, 206, 2009)
            gmsh.model.geo.addLine(206, 207, 2010)
            gmsh.model.geo.addLine(207, 204, 2011)


            ## These should all be transfinite lines

            for i in range(1000, 1012):
                gmsh.model.geo.mesh.set_transfinite_curve(i, res)


            for i in range(2000, 2012):
                gmsh.model.geo.mesh.set_transfinite_curve(i, res)

            # The 6 faces of the cube2

            gmsh.model.geo.addCurveLoop([1000, 1004, 1008, 1007], 10000, reorient=True)
            gmsh.model.geo.addCurveLoop([1001, 1005, 1009, 1004], 10001, reorient=True)
            gmsh.model.geo.addCurveLoop([1002, 1006, 1010, 1005], 10002, reorient=True)
            gmsh.model.geo.addCurveLoop([1003, 1007, 1011, 1006], 10003, reorient=True)
            gmsh.model.geo.addCurveLoop([1000, 1003, 1002, 1001], 10004, reorient=True)
            gmsh.model.geo.addCurveLoop([1008, 1009, 1010, 1011], 10005, reorient=True)

            gmsh.model.geo.addCurveLoop([2000, 2004, 2008, 2007], 20000, reorient=True)
            gmsh.model.geo.addCurveLoop([2001, 2005, 2009, 2004], 20001, reorient=True)
            gmsh.model.geo.addCurveLoop([2002, 2006, 2010, 2005], 20002, reorient=True)
            gmsh.model.geo.addCurveLoop([2003, 2007, 2011, 2006], 20003, reorient=True)
            gmsh.model.geo.addCurveLoop([2000, 2003, 2002, 2001], 20004, reorient=True)
            gmsh.model.geo.addCurveLoop([2008, 2009, 2010, 2011], 20005, reorient=True)


            gmsh.model.geo.add_surface_filling([10000], 10101)
            gmsh.model.geo.add_surface_filling([10001], 10102)
            gmsh.model.geo.add_surface_filling([10002], 10103)
            gmsh.model.geo.add_surface_filling([10003], 10104)
            gmsh.model.geo.add_surface_filling([10004], 10105)
            gmsh.model.geo.add_surface_filling([10005], 10106)

            gmsh.model.geo.add_surface_filling([20000], 20101)
            gmsh.model.geo.add_surface_filling([20001], 20102)
            gmsh.model.geo.add_surface_filling([20002], 20103)
            gmsh.model.geo.add_surface_filling([20003], 20104)
            gmsh.model.geo.add_surface_filling([20004], 20105)
            gmsh.model.geo.add_surface_filling([20005], 20106)


            for i in range(10101, 10107):
                gmsh.model.geo.mesh.setTransfiniteSurface(i, "Left")
                if not simplex:
                    gmsh.model.geo.mesh.setRecombine(2, i)

            for i in range(20101, 20107):
                gmsh.model.geo.mesh.setTransfiniteSurface(i, "Left")
                if not simplex:
                    gmsh.model.geo.mesh.setRecombine(2, i)

            gmsh.model.geo.synchronize()

            # outer surface / inner_surface
            gmsh.model.geo.add_surface_loop([10101, 10102, 10103, 10104, 10105, 10106], 10111)
            gmsh.model.geo.add_surface_loop([20101, 20102, 20103, 20104, 20105, 20106], 20111)

            
                
            gmsh.model.geo.add_volume([10111, 20111], 100001)
            gmsh.model.geo.synchronize()

            # gmsh.model.mesh.set_transfinite_volume(100001)
            if not simplex:
                gmsh.model.geo.mesh.setRecombine(3, 100001)

            gmsh.model.geo.remove_all_duplicates()
            gmsh.model.remove_entities([[2,10111]], recursive=True)
            gmsh.model.mesh.generate(dim=3)
            gmsh.model.mesh.removeDuplicateNodes()
            
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".msh") as tfile:
                gmsh.write(tfile.name)
                
                cubed_sphere_ball_mesh = meshio.read(tfile.name)
                cubed_sphere_ball_mesh.remove_lower_dimensional_cells()


            gmsh.finalize()

 


            return cubed_sphere_ball_mesh

cs_ball = cs_build_pygmsh(8, 1.0, simplex=False)



# -



# +
cs_ball.write("ignore_csb.vtk")
pvmesh = pv.read("ignore_csb.vtk")

clipped_stack = pvmesh.clip(origin=(0.01,0.0,0.0), normal=(1, 0, 0), invert=False)

pl = pv.Plotter()

# pl.add_mesh(pvmesh,'Blue', 'wireframe' )
pl.add_mesh(clipped_stack, cmap="coolwarm", edge_color="Black", show_edges=True, 
              use_transparency=False)
pl.show()


# +
def cs_shell_build_pygmsh(elementRes=(16, 8), radius_outer=1.0, radius_inner =0.5, simplex=False):


        import pygmsh 
        import meshio

        l = 0.0

        inner_radius = radius_inner
        outer_radius = radius_outer
        nodes = elementRes[0]+1  # resolution of the cube laterally
        layers= elementRes[1]

        with pygmsh.geo.Geometry() as geom:
            cpoint = geom.add_point([0.0,0.0,0.0], l)

            genpt = [0,0,0,0,0,0,0,0]

            # 8 corners of the cube 

            r2 = 1.0 / np.sqrt(3.0) # Generate a unit sphere

            genpt[0] = geom.add_point([ -r2, -r2, -r2],  l)
            genpt[1] = geom.add_point([  r2, -r2, -r2],  l)
            genpt[2] = geom.add_point([  r2,  r2, -r2],  l)
            genpt[3] = geom.add_point([ -r2,  r2, -r2],  l)
            genpt[4] = geom.add_point([ -r2, -r2,  r2],  l)
            genpt[5] = geom.add_point([  r2, -r2,  r2],  l)
            genpt[6] = geom.add_point([  r2,  r2,  r2],  l)
            genpt[7] = geom.add_point([ -r2,  r2,  r2],  l)


            # 12 edges of the cube

            b_circ00 = geom.add_circle_arc(genpt[0], cpoint, genpt[1])
            b_circ01 = geom.add_circle_arc(genpt[1], cpoint, genpt[2])
            b_circ02 = geom.add_circle_arc(genpt[2], cpoint, genpt[3])
            b_circ03 = geom.add_circle_arc(genpt[0], cpoint, genpt[3])

            b_circ04 = geom.add_circle_arc(genpt[1], cpoint, genpt[5])
            b_circ05 = geom.add_circle_arc(genpt[2], cpoint, genpt[6])
            b_circ06 = geom.add_circle_arc(genpt[3], cpoint, genpt[7])
            b_circ07 = geom.add_circle_arc(genpt[0], cpoint, genpt[4])

            b_circ08 = geom.add_circle_arc(genpt[4], cpoint, genpt[5])
            b_circ09 = geom.add_circle_arc(genpt[5], cpoint, genpt[6])
            b_circ10 = geom.add_circle_arc(genpt[6], cpoint, genpt[7])
            b_circ11 = geom.add_circle_arc(genpt[4], cpoint, genpt[7])

            for arc in [b_circ00, b_circ01, b_circ02, b_circ03,
                        b_circ04, b_circ05, b_circ06, b_circ07,
                        b_circ08, b_circ09, b_circ10, b_circ11 ]:

                    geom.set_transfinite_curve(arc, num_nodes=nodes, 
                                            mesh_type="Progression", coeff=1.0)

            # 6 Cube faces

            face00_loop = geom.add_curve_loop([b_circ00, b_circ04, -b_circ08, -b_circ07])
            face00 = geom.add_surface(face00_loop) 
            geom.set_transfinite_surface(face00, arrangement="Left",
                                        corner_pts = [genpt[0], genpt[1], genpt[5], genpt[4]])   


            face01_loop = geom.add_curve_loop([b_circ01, b_circ05, -b_circ09, -b_circ04])
            face01 = geom.add_surface(face01_loop) 
            geom.set_transfinite_surface(face01, arrangement="Left",
                                        corner_pts = [genpt[1], genpt[2], genpt[6], genpt[5]])   


            face02_loop = geom.add_curve_loop([b_circ02, b_circ06, -b_circ10, -b_circ05])
            face02 = geom.add_surface(face02_loop) 
            geom.set_transfinite_surface(face02, arrangement="Left",
                                        corner_pts = [genpt[2], genpt[3], genpt[7], genpt[6]])   


            face03_loop = geom.add_curve_loop([-b_circ03, b_circ07, b_circ11, -b_circ06])
            face03 = geom.add_surface(face03_loop) 
            geom.set_transfinite_surface(face03, arrangement="Left",
                                        corner_pts = [genpt[3], genpt[0], genpt[4], genpt[7]])   


            face04_loop = geom.add_curve_loop([-b_circ00, b_circ03, -b_circ02, -b_circ01])
            face04 = geom.add_surface(face04_loop) 
            geom.set_transfinite_surface(face04, arrangement="Left",
                                        corner_pts = [genpt[1], genpt[0], genpt[3], genpt[2]])   


            face05_loop = geom.add_curve_loop([b_circ08, b_circ09,  b_circ10, -b_circ11])
            face05 = geom.add_surface(face05_loop) 
            geom.set_transfinite_surface(face05, arrangement="Left",
                                        corner_pts = [genpt[4], genpt[5], genpt[6], genpt[7]])   


            geom.set_recombined_surfaces([face00, face01, face02, face03, face04, face05])
            shell = geom.add_surface_loop([face00, face01, face02, face03, face04, face05])

            
            
            
           
            two_D_cubed_sphere = geom.generate_mesh(dim=2, verbose=False)
            two_D_cubed_sphere.remove_orphaned_nodes()
            two_D_cubed_sphere.remove_lower_dimensional_cells()

            return two_D_cubed_sphere
        

cs_shell_mesh = cs_shell_build_pygmsh(
            elementRes=(6, 3), 
            radius_outer=1.0,
            radius_inner =0.5,
            simplex=True)
        


# +
cs_shell_mesh.write("ignore_csh.vtk")
pvmesh = pv.read("ignore_csh.vtk")

clipped_stack = pvmesh.clip(origin=(0.00001,0.0,0.0), normal=(1, 0, 0), invert=False)

pl = pv.Plotter()

# pl.add_mesh(pvmesh,'Blue', 'wireframe' )
pl.add_mesh(clipped_stack, cmap="coolwarm", edge_color="Black", show_edges=True, 
              use_transparency=False)
pl.show()
# -


