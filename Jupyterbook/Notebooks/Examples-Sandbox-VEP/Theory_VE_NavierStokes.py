# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Viscoelastic Navier-Stokes equation
#
# Here we outline how we combine the numerical NS scheme and the numerical Visco-elastic scheme
#
# ## Navier-Stokes equation
#
# $$
#     \frac{D{\mathbf{u}}}{D t} + \nabla \cdot \mathbf{\tau} - \nabla p = \mathbf{\rho g \hat{\mathbf{z}}}
# $$
#
# In the Lagrangian reference frame, a finite difference expression for the time-derivative can be constructed using, for example, one of the Adams-Moulton methods.  
#
# $$
# \frac{\mathbf{u} - \mathbf{u^*}}{\Delta t} + 
#             \frac{1}{2} \nabla \cdot \mathbf{\tau} + \frac{1}{2} \nabla \cdot \mathbf{\tau^*}
#             - \nabla p = \mathbf{\rho g \hat{\mathbf{z}}}
# $$ 
#
# Here $\mathbf{u}^*$ and $\mathbf{\tau}^*$ are values determined along the characteristics at the time $t-\Delta t$. Numerically, this is the value on a particle at the previous timestep.
#
# In the Navier-stokes problem, it is common to write $\tau=\eta \left(\nabla \mathbf u + (\nabla \mathbf u)^T \right)$ and $\tau^*=\eta \left(\nabla \mathbf u^* + (\nabla \mathbf u^*)^T \right)$ which ignores  rotation and shearing of the stress during the interval $\Delta T$. This simplifies the implementation because only the velocity history is required, not the history of the stress tensor.
#
# A higher order Adams-Moulton scheme yields
#
# $$
# \frac{\mathbf{u} - \mathbf{u^*}}{\Delta t} + 
#             \frac{5}{12} \nabla \cdot \mathbf{\tau} + \frac{2}{3} \nabla \cdot \mathbf{\tau^*}
#             + \frac{1}{12} \nabla \cdot \mathbf{\tau^{**}}
#             - \nabla p = \mathbf{\rho g \hat{\mathbf{z}}}
# $$ 
#
# ## Viscoelastic Stokes equation
#
# In viscoelasticity, the elastic part of the deformation is related to the stress rate. If we approach this problem as a perturbation to the viscous Stokes equation, we first consider the constitutive behaviour 
#
# $$
#  \frac{1}{2\mu}\frac{D{\tau}}{Dt} + \frac{\tau}{2\eta} = \dot\varepsilon
# $$
#
# A first order difference form for ${D \tau}/{D t}$ then gives
#
# $$
#     \frac{\tau - \tau^*}{\Delta t} = \dot\varepsilon
# $$
#
# where $\tau^*$ is the stress history along the characteristics associated with the current computational points. Rearranging to find an expression for the current stress in terms of the strain rate:
#
# $$
#     \tau = 2 \eta_\textrm{eff} \dot\varepsilon + \frac{ t_{r}}{\Delta t} \tau^{*}
# $$
#
#
# $$
# \nabla \cdot \mathbf{\tau} - \nabla p = \mathbf{\rho g \hat{\mathbf{z}}}
# $$
#
#
#

# +
import numpy as np
import sympy




# +
# Adams / Bashforth & Adams Moulton ... 

s = sympy.Symbol(r"\tau")
s1 = sympy.Symbol(r"\tau^*")
s2 = sympy.Symbol(r"\tau^**")
dt = sympy.Symbol(r"\Delta t")
mu = sympy.Symbol(r"\mu")
eta = sympy.Symbol(r"\eta")
eta_eff = sympy.Symbol(r"\eta_\textrm{eff}")
edot = sympy.Symbol(r"\dot\varepsilon")
tr = sympy.Symbol(r"t_r")

sdot1 = (s - s1)/dt
sdot2 = (3 * s - 4 * s1 + s2) / (2 * dt)
# -


display( sdot1 )
Seq1 = sympy.Equality(sympy.simplify(sdot1 / (2 * mu) + s / (2 * eta)),edot)
display(Seq1)
a = sympy.simplify(sympy.solve(Seq1, s)[0]).expand()
display(a)

eta_eff_1 = sympy.simplify(eta * mu * dt / (mu * dt + eta))
display(eta_eff_1)
a = sympy.solve(Seq1, s)[0].expand()
b = a.subs(eta_eff_1, eta_eff)
b



(2 * edot * eta_eff_1 + eta_eff_1 / (mu*dt) * s1) 



Seq2 = sympy.Equality(sympy.simplify(sdot2 / (2 * mu) + s / (2 * eta)), edot)
display(Seq2)
sympy.simplify(sympy.solve(Seq2, s)[0])

eta_eff_2 = sympy.simplify(2 * eta * mu * dt / ( 2 * mu * dt + 3 * eta))
display(eta_eff_2)
sympy.simplify(2 * eta * sympy.solve(Seq2, s)[0]/ (2*eta_eff_2)) 

a = sympy.simplify(2 * eta * sympy.solve(Seq2, s)[0] / (2*eta_eff_2))
tau_2 = a.expand().subs(eta/mu, tr)

tau_2

0/0


stokes.constitutive_model.Parameters.yield_stress.subs(((strain.sym[0],0.25), (y,0.0)))

stokes.constitutive_model.Parameters.viscosity


def return_points_to_domain(coords):
    new_coords = coords.copy()
    new_coords[:,0] = (coords[:,0] + 1.5)%3 - 1.5
    return new_coords


ts = 0

# + tags=[]
delta_t = stokes.estimate_dt()

expt_name = f"output/shear_band_sw_nonp_{mu}"

for step in range(0, 10):
    
    stokes.solve(zero_init_guess=False)
    
    nodal_strain_rate_inv2.uw_function = (sympy.Max(0.0, stokes._Einv2 - 
                       0.5 * stokes.constitutive_model.Parameters.yield_stress / stokes.constitutive_model.Parameters.bg_viscosity))
    nodal_strain_rate_inv2.solve()

    with mesh1.access(strain_rate_inv2_p):
        strain_rate_inv2_p.data[...] = strain_rate_inv2.data.copy()
        
    nodal_strain_rate_inv2.uw_function = stokes._Einv2
    nodal_strain_rate_inv2.solve()
    
    with swarm.access(strain), mesh1.access():
        XX = swarm.particle_coordinates.data[:,0]
        YY = swarm.particle_coordinates.data[:,1]
        mask =  (2*XX/3)**4 # * 1.0 - (YY * 2)**8 
        strain.data[:,0] +=  delta_t * mask * strain_rate_inv2_p.rbf_interpolate(swarm.data)[:,0] - 0.1 * delta_t
        strain_dat = delta_t * mask *  strain_rate_inv2_p.rbf_interpolate(swarm.data)[:,0]
        print(f"dStrain / dt = {delta_t * (mask * strain_rate_inv2_p.rbf_interpolate(swarm.data)[:,0]).mean()}, {delta_t}")
        
    mesh1.write_timestep_xdmf(f"{expt_name}", 
                         meshUpdates=False,
                         meshVars=[p_soln,v_soln,strain_rate_inv2_p], 
                         swarmVars=[strain],
                         index=ts)
    
    swarm.save(f"{expt_name}.swarm.{ts}.h5")
    strain.save(f"{expt_name}.strain.{ts}.h5")

    # Update the swarm locations
    swarm.advection(v_soln.sym, delta_t=delta_t, 
                 restore_points_to_domain_func=None) 
    
    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))
        
    ts += 1

# -




# +
nodal_visc_calc.uw_function = sympy.log(stokes.constitutive_model.Parameters.viscosity)
nodal_visc_calc.solve()

yield_stress_calc.uw_function = stokes.constitutive_model.Parameters.yield_stress
yield_stress_calc.solve()

nodal_tau_inv2.uw_function = 2 * stokes.constitutive_model.Parameters.viscosity * stokes._Einv2
nodal_tau_inv2.solve()

# +
# check it - NOTE - for the periodic mesh, points which have crossed the coordinate sheet are plotted somewhere
# unexpected. This is a limitation we are stuck with for the moment.

if uw.mpi.size == 1:
    
    mesh1.vtk("tmp_shear_inclusion.vtk")
    pvmesh = pv.read("tmp_shear_inclusion.vtk")

    pvpoints = pvmesh.points[:, 0:2]
    usol = v_soln.rbf_interpolate(pvpoints)

    pvmesh.point_data["P"] = p_soln.rbf_interpolate(pvpoints)
    pvmesh.point_data["Edot"] = strain_rate_inv2.rbf_interpolate(pvpoints)
    pvmesh.point_data["Visc"] = np.exp(node_viscosity.rbf_interpolate(pvpoints))
    pvmesh.point_data["Edotp"] = strain_rate_inv2_p.rbf_interpolate(pvpoints)
    pvmesh.point_data["Strs"] = dev_stress_inv2.rbf_interpolate(pvpoints)
    pvmesh.point_data["StrY"] =  yield_stress.rbf_interpolate(pvpoints)
    pvmesh.point_data["dStrY"] = pvmesh.point_data["StrY"] - 2 *  pvmesh.point_data["Visc"] * pvmesh.point_data["Edot"] 
    pvmesh.point_data["Mat"] = material.rbf_interpolate(pvpoints)
    pvmesh.point_data["Strn"] = strain._meshVar.rbf_interpolate(pvpoints)

    # Velocity arrows
    
    v_vectors = np.zeros_like(pvmesh.points)
    v_vectors[:, 0:2] = v_soln.rbf_interpolate(pvpoints)
        
    # Points (swarm)
    
    with swarm.access():
        plot_points = np.where(strain.data > 0.0001)
        strain_data = strain.data.copy()

        points = np.zeros((swarm.data[plot_points].shape[0], 3))
        points[:, 0] = swarm.data[plot_points[0],0]
        points[:, 1] = swarm.data[plot_points[0],1]
        point_cloud = pv.PolyData(points)
        point_cloud.point_data["strain"] = strain.data[plot_points]

    pl = pv.Plotter(window_size=(500, 500))

    # pl.add_arrows(pvmesh.points, v_vectors, mag=0.1, opacity=0.75)
    # pl.camera_position = "xy"

    pl.add_mesh(
        pvmesh,
        cmap="Blues",
        edge_color="Grey",
        show_edges=True,
        # clim=[-1.0,1.0],
        scalars="Edotp",
        use_transparency=False,
        opacity=0.5,
    )
    
 
    pl.add_points(point_cloud, 
                  colormap="Oranges", scalars="strain",
                  point_size=10.0, 
                  opacity=0.0,
                  # clim=[0.0,0.2],
                 )
    
    pl.camera.SetPosition(0.0, 0.0, 3.0)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)
        
    pl.show()
# -

strain_dat.max()



with swarm.access():
    print(strain.data.max())

strain_rate_inv2_p.rbf_interpolate(mesh1.data).max()



# ## 

mesh1._search_lengths


