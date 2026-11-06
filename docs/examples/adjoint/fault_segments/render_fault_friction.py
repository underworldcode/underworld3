"""PyVista renders of the friction example at the true coefficients.

Writes log10 eta_1 (the plane's viscosity), the slip rate, the velocity and
the pressure to ~/+Simulations/adjoint_fault_example/, on the mesh's own
triangulation. The model is the notebook's in its own units: lengths in
units of the depth (10 km), stresses in units of the viscous stress of the
shortening (26 MPa), velocities in units of the convergence rate (8.4 mm/yr).
"""
import math
import os

import numpy as np
import sympy
import pyvista as pv

import underworld3 as uw
import underworld3.visualisation as vis

pv.OFF_SCREEN = True
OUT = os.path.expanduser("~/+Simulations/adjoint_fault_example")

cell_size, surface_dip, flat_depth, surface_x, band_w = 1 / 12, 60.0, 0.3, 1.9, 0.08
true_mu, cohesion, rho_g = [0.05, 0.15, 0.25, 0.4], 0.05, 10.0

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0),
                                         cellSize=cell_size, qdegree=3, refinement=1)
x, y = mesh.X
v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)

phi_top = math.radians(surface_dip) - math.pi / 2
R = (1.0 - flat_depth) / (1.0 + math.sin(phi_top))
yc = flat_depth + R
xc = surface_x - R * math.cos(phi_top)
r = sympy.sqrt((x - xc) ** 2 + (y - yc) ** 2)
phi = sympy.atan2(y - yc, x - xc)
on_flat = x < xc
d = sympy.Piecewise((y - flat_depth, on_flat), (R - r, True))
s = sympy.Piecewise((x, on_flat), (xc + R * (phi + sympy.pi / 2), True))
n_hat = sympy.Matrix([[sympy.Piecewise((0, on_flat), ((x - xc) / r, True)),
                       sympy.Piecewise((1, on_flat), ((y - yc) / r, True))]])
ramp = R * (phi_top + math.pi / 2)
edges_s = [0.0, xc, xc + ramp / 3, xc + 2 * ramp / 3, xc + ramp]
band = sympy.exp(-(d / band_w) ** 2)

def segment(k):
    on = 1 if k == 0 else (1 + sympy.tanh((s - edges_s[k]) / band_w)) / 2
    off = 1 if k == 3 else (1 - sympy.tanh((s - edges_s[k + 1]) / band_w)) / 2
    return on * off

E = mesh.vector.strain_tensor(v.sym)
t_hat = sympy.Matrix([[-n_hat[1], n_hat[0]]])
e_s = sympy.sqrt((t_hat * E * n_hat.T)[0] ** 2 + uw.maths.functions.vanishing)
friction = sum(true_mu[k] * segment(k) for k in range(4))
tau_y = cohesion + friction * sympy.Max(p.sym[0], 0)
eta_plane = tau_y / (tau_y + 2 * e_s)
eta_1 = 1 - band * (1 - eta_plane)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_1
stokes.constitutive_model.Parameters.director = n_hat
stokes.tolerance = 1e-8
stokes.bodyforce = sympy.Matrix([0, -rho_g])
stokes.add_essential_bc((0.0, 0.0), "Bottom")
stokes.add_essential_bc((0.5, None), "Left")
stokes.add_essential_bc((-0.5, None), "Right")
stokes.solve(zero_init_guess=True)

# Nodal values on the mesh's own triangulation (skill rule 5): no grid resampling.
pv_mesh = vis.mesh_to_pv_mesh(mesh)
pts = np.asarray(pv_mesh.points[:, :2])
edges = pv_mesh.extract_all_edges()
pv_mesh.point_data["log10 eta_1"] = np.log10(np.clip(np.asarray(uw.function.evaluate(eta_1, pts)).ravel(), 1e-3, 1.0))
pv_mesh.point_data["slip rate"] = np.asarray(uw.function.evaluate(band * e_s, pts)).ravel()
pv_mesh.point_data["p"] = np.asarray(uw.function.evaluate(p.sym[0], pts)).ravel()
pv_v = vis.meshVariable_to_pv_mesh_object(v)
vdata = np.asarray(v.data)
pv_v.point_data["|v|"] = np.linalg.norm(vdata, axis=1)
pv_v.point_data["v"] = np.column_stack([vdata, np.zeros(len(vdata))])
pv_v.point_data["v_y"] = vdata[:, 1]

points = [(0.3, 0.65), (0.75, 0.12), (1.25, 0.3), (0.9, 0.6), (1.55, 0.85)]

def observations(pl):
    """Where the observations are read: the five points, and the band under the
    surface (two band widths deep, where its Gaussian weight is above 2%)."""
    strip = pv.Rectangle([[0.0, 1 - 2 * band_w, 0.001], [2.0, 1 - 2 * band_w, 0.001], [2.0, 1.0, 0.001]])
    pl.add_mesh(strip, color="white", opacity=0.45, lighting=False)
    pl.add_mesh(pv.Line([0, 1 - 2 * band_w, 0.002], [2, 1 - 2 * band_w, 0.002]), color="white", line_width=2, lighting=False)
    for px, py in points:
        pl.add_mesh(pv.Disc(center=(px, py, 0.003), inner=0.0, outer=0.022, normal=(0, 0, 1)), color="white", lighting=False)
        pl.add_mesh(pv.Disc(center=(px, py, 0.004), inner=0.0, outer=0.013, normal=(0, 0, 1)), color="black", lighting=False)

def frame(name, obj, scalars, cmap, clim, title, arrows=False, observed=False):
    pl = pv.Plotter(off_screen=True, window_size=(1600, 820))
    pl.set_background("white")
    pl.add_mesh(obj, scalars=scalars, cmap=cmap, clim=clim, show_edges=False, lighting=False,
                scalar_bar_args=dict(title=title, color="black", vertical=False, position_x=0.3,
                                     position_y=0.04, width=0.4, height=0.06, title_font_size=26,
                                     label_font_size=22))
    pl.add_mesh(edges, color="black", line_width=0.25, lighting=False, opacity=0.2)
    if arrows:
        # a coarse regular set of points, so the arrows read as a field and not as the mesh
        gx, gy = np.meshgrid(np.linspace(0.05, 1.95, 30), np.linspace(0.05, 0.95, 15))
        seeds = np.column_stack([gx.ravel(), gy.ravel()])
        vals = np.asarray(uw.function.evaluate(v.sym, seeds)).reshape(-1, 2)
        cloud = pv.PolyData(np.column_stack([seeds, np.zeros(len(seeds))]))
        cloud["v"] = np.column_stack([vals, np.zeros(len(vals))])
        cloud["|v|"] = np.linalg.norm(vals, axis=1)
        pl.add_mesh(cloud.glyph(orient="v", scale="|v|", factor=0.09), color="black", lighting=False)
    if observed:
        observations(pl)
    pl.view_xy()
    pl.camera.parallel_projection = True
    pl.camera.focal_point = (1.0, 0.42, 0.0)
    pl.camera.parallel_scale = 0.66
    pl.screenshot(os.path.join(OUT, name))
    pl.close()

frame("fault_eta1.png", pv_mesh, "log10 eta_1", "viridis", (-2.5, 0.0), "log10 plane viscosity", observed=True)
frame("fault_slip.png", pv_mesh, "slip rate", "magma_r", (0.0, 1.0), "shear strain rate on the plane")
frame("fault_vy.png", pv_v, "v_y", "viridis", (0.0, 0.7), "vertical velocity (uplift rate)", arrows=True)
frame("fault_p.png", pv_mesh, "p", "RdBu_r", (-8.0, 8.0), "pressure")
print("wrote", OUT)
