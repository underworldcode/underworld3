"""A Coulomb fault in a Maxwell thin sheet dragged by a shearing substrate —
on a flat sheet or on the surface of a sphere.

The sheet (shear modulus mu, Maxwell time eta/mu) rests on a viscous channel
of drag coefficient c whose far side moves at a prescribed substrate velocity
u_s: the force per unit area on the sheet is c (u_s - u), so ``basal_drag = c``
and ``bodyforce = c u_s``. On the flat sheet u_s is the simple shear (y, 0);
on the sphere it is the toroidal flow rhat x grad(z^2), whose shear is largest
on the equator. The sheet is dragged towards the substrate motion and its
stress rises as eta gamma_dot (1 - exp(-t mu/eta)) towards the viscous value;
an interior fault with regularised Coulomb friction (strength mu_f sigma_n)
holds until the shear traction on it reaches strength, then slides, and the
stress drop it imposes diffuses into the sheet with D = mu L / c.

Snapshots show the Coulomb failure change on receiver faults parallel to the
fault, Delta CFF = Delta tau + mu_f Delta sigma_n (shear signed in the loading
sense, tension positive), against a REFERENCE STATE: the same sheet with the fault
welded, on the same mesh and at the same time, solved alongside it. Red is moved towards failure, blue
is the stress shadow. The second panel is the implied surface uplift rate. The
fault traction and slip rate are recorded against time as the run goes
(history.npz is rewritten at every step, so a run that stops early still has its
record); checkpoints are written beside the snapshots.
"""
import os

import numpy as np
import sympy

import underworld3 as uw
from underworld3.utilities import fault_contact
from underworld3.utilities.fault_contact import add_coulomb_fault_bc, fault_pair_jumps

params = uw.Params(
    uw_geometry="flat",          # flat | sphere
    uw_output_dir=os.path.expanduser("~/+Simulations/thin_sheet_faults/cycle_v2"),
    uw_cell_size=0.05,
    uw_dt=0.02,
    uw_t_end=3.0,
    uw_mu_f=0.6,                 # friction coefficient
    uw_sigma_n=1.0,              # prescribed effective normal stress
    uw_V0=1.0e-4,                # regularisation velocity of the Coulomb law
    uw_eta=1.0,                  # Maxwell viscosity: t_r = eta / mu = 1
    uw_shear_modulus=1.0,
    uw_basal_drag=1.0,           # D = mu L / c = 1
    uw_half_length=0.5,          # flat: half-length; sphere: half-angle of the equatorial arc
    uw_snapshots="0.5,1.0,1.5,3.0",
    uw_isostatic_factor=0.15,    # 1 - rho_c / rho_m: surface uplift per unit thickening (Airy)
    uw_weld_viscosity=1.0e6,     # interface viscosity that welds the reference sheet's fault
    uw_tolerance=1.0e-8,         # sheet solver tolerance
    uw_eta_v=0.0,                # viscous term eta_v V added to the Coulomb law (0 = pure Coulomb)
    uw_transport="",             # stress-history transport; "" = the solver's default
    uw_order=1,                  # Maxwell time-integration order
    uw_integrator="etd",         # etd (exponential) | bdf
)
out = os.path.join(params.uw_output_dir, params.uw_geometry)
os.makedirs(out, exist_ok=True)
a = params.uw_half_length

if params.uw_geometry == "flat":
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-2.0, -1.0), maxCoords=(2.0, 1.0), cellSize=params.uw_cell_size, qdegree=3)
    fault_pts = np.array([[-a, 0.0], [a, 0.0]])
else:
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=params.uw_cell_size, degree=1, qdegree=5)
    phi = np.linspace(-a, a, 9)
    fault_pts = np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
split = mesh.add_fault(("Flt", fault_pts))
c = sympy.nsimplify(params.uw_basal_drag, rational=True)


def substrate_velocity(m):
    """The substrate motion in mesh ``m``'s coordinates: simple shear (y, 0) on
    the flat sheet, the toroidal flow rhat x grad(z^2) on the sphere (its shear
    peaks on the equator)."""
    if params.uw_geometry == "flat":
        x, y = m.X
        return sympy.Matrix([[y, 0]])
    x, y, z = m.X[0], m.X[1], m.X[2]
    return sympy.Matrix([[2 * y * z, -2 * x * z, 0]])


def dragged_sheet(m):
    """A Maxwell sheet on mesh ``m`` dragged by the substrate."""
    s = uw.systems.ThinSheet(m)
    if params.uw_transport:
        s.stress_transport = params.uw_transport      # before the model creates the history
    s.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        s.Unknowns, order=params.uw_order, integrator=params.uw_integrator)
    s.constitutive_model.Parameters.shear_viscosity_0 = params.uw_eta
    s.constitutive_model.Parameters.shear_modulus = params.uw_shear_modulus
    s.basal_drag = c
    s.bodyforce = list(c * substrate_velocity(m))
    s.tolerance = params.uw_tolerance
    return s


sheet = dragged_sheet(split)
if params.uw_eta_v > 0.0:
    # Saturated regularised Coulomb is nearly flat in V (dtau/dV ~ mu sigma_n
    # V0 / V^2), which leaves the slip rate of a sliding fault barely determined
    # by the per-step problem; a viscous term eta_v V gives it a slope.
    # TODO(FEATURE): no public way to register a composite SymbolicFaultLaw;
    # this uses fault_contact's private _register_law.
    fault_contact.add_frictionless_fault_bc(sheet, "Flt")
    coulomb = fault_contact.CoulombFaultLaw(params.uw_mu_f, params.uw_sigma_n, params.uw_V0)
    fault_contact._register_law(sheet, "Flt", fault_contact.SymbolicFaultLaw(
        coulomb.expr + params.uw_eta_v * fault_contact.slip_rate))
else:
    add_coulomb_fault_bc(sheet, params.uw_mu_f, "Flt", sigma_n=params.uw_sigma_n, V0=params.uw_V0)
# The reference state for Delta CFF: the same sheet on the SAME split mesh with
# the fault welded (a stiff viscous interface, slip ~ tau / eta_f). Same nodes,
# same discretisation, so Delta CFF is the fault's effect alone — a reference on
# the unsplit mesh differed by interpolation at the cut's moved vertices (1e-4
# speckle before any slip, worse at the free edges).
reference = dragged_sheet(split)
reference.add_fault_bc(params.uw_weld_viscosity, "Flt")
strength = params.uw_mu_f * params.uw_sigma_n

# The sheet thins at -L div_s u; under Airy isostasy the surface rises at
# (1 - rho_c / rho_m) times the thickening rate. The divergence of the P2
# velocity is cell-discontinuous, so it is projected onto a continuous P1 field
# for the picture (the slit's coincident nodes keep their own values).
thinning = uw.discretisation.MeshVariable("thinning_rate", split, 1, degree=1)
thinning_projection = uw.systems.Projection(split, thinning)
thinning_projection.uw_function = -sheet.thickness.sym * split.vector.divergence(sheet.u.sym)
thinning_projection.smoothing = 0.0


def receiver_frame(coords):
    """Unit fault-parallel direction t and in-surface fault normal n at each
    node, for receivers parallel to the fault: the strike (x) on the flat sheet;
    small circles parallel to the equatorial fault on the sphere (t east,
    n north)."""
    if coords.shape[1] == 2:
        t = np.tile([1.0, 0.0], (len(coords), 1))
        n = np.tile([0.0, 1.0], (len(coords), 1))
        return t, n
    rho = np.maximum(np.hypot(coords[:, 0], coords[:, 1]), 1e-12)
    t = np.column_stack([-coords[:, 1] / rho, coords[:, 0] / rho, np.zeros(len(coords))])
    rhat = coords / np.linalg.norm(coords, axis=1)[:, None]
    n = np.cross(rhat, t)
    return t, n


def sheet_stress(coords, tau):
    """sigma = tau + tr(tau) P at each node (plane-stress closure)."""
    if coords.shape[1] == 2:
        P = np.broadcast_to(np.eye(2), tau.shape)
    else:
        rhat = coords / np.linalg.norm(coords, axis=1)[:, None]
        P = np.eye(3)[None] - rhat[:, :, None] * rhat[:, None, :]
    return tau + np.trace(tau, axis1=1, axis2=2)[:, None, None] * P


def resolved(coords, tau):
    """(shear t.sigma.n, normal n.sigma.n) on the receiver planes, tension positive."""
    t, n = receiver_frame(coords)
    sigma = sheet_stress(coords, tau)
    return np.einsum("ni,nij,nj->n", t, sigma, n), np.einsum("ni,nij,nj->n", n, sigma, n)


def render(k, t, tau_field, uplift_field, slip_rate_max):
    """Two panels: Delta CFF on fault-parallel receivers against the unfaulted
    reference (``tau_field``), and the implied surface uplift rate (Airy
    isostasy on the thinning rate)."""
    import pyvista as pv
    import underworld3.visualisation as vis
    pv.OFF_SCREEN = True
    trace = pv.lines_from_points(np.column_stack([fault_pts, np.zeros(len(fault_pts))])
                                 if fault_pts.shape[1] == 2 else fault_pts * 1.001)
    height = 900 if params.uw_geometry == "sphere" else 720
    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2400, height), border=False)
    for col, (name, field, title) in enumerate((("dcff", tau_field, "Delta CFF"),
                                                ("uplift", uplift_field, "uplift rate"))):
        pl.subplot(0, col)
        pvm = vis.mesh_to_pv_mesh(split)
        pvm.point_data[name] = field
        pl.set_background("white")
        vmax = max(float(np.percentile(np.abs(field), 99)), 1e-9)
        pl.add_mesh(pvm, scalars=name, cmap="RdBu_r", clim=(-vmax, vmax), show_edges=False,
                    lighting=False, scalar_bar_args={"title": title, "color": "black"})
        pl.add_mesh(trace, color="black", line_width=4, lighting=False)
        pl.add_text(f"t = {t:.2f}   Delta CFF (mu = {params.uw_mu_f}) vs welded   "
                    f"max slip rate {slip_rate_max:.3f}"
                    if col == 0 else "uplift rate = (1 - rho_c/rho_m) (-L div u)",
                    position="upper_left", font_size=11, color="black")
        if params.uw_geometry == "flat":
            pl.view_xy()
            pl.camera.parallel_projection = True
            pl.camera.parallel_scale = 1.05
        else:
            pl.camera_position = [(3.2, -1.2, 1.4), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    pl.screenshot(os.path.join(out, f"tau_{k:02d}.png"))
    pl.close()


snapshots = [float(s) for s in params.uw_snapshots.split(",")]
history, t, step, k = [], 0.0, 0, 0
nodes_on_fault = None
while t < params.uw_t_end - 0.5 * params.uw_dt:
    sheet.solve(timestep=params.uw_dt, zero_init_guess=False)
    reference.solve(timestep=params.uw_dt, zero_init_guess=False)
    t += params.uw_dt
    step += 1
    info = sheet._rotated_freeslip_info
    coords, jumps, normals = fault_pair_jumps(sheet, "Flt", info)
    slip = np.linalg.norm(jumps - normals * np.einsum("ij,ij->i", jumps, normals)[:, None], axis=1)
    leak = np.abs(np.einsum("ij,ij->i", jumps, normals)).max() if len(jumps) else 0.0
    tau_coords = sheet.tau.coords
    tau = np.asarray(sheet.tau.array)
    if nodes_on_fault is None:
        if params.uw_geometry == "flat":
            nodes_on_fault = (np.abs(tau_coords[:, 1]) < 1e-9) & (np.abs(tau_coords[:, 0]) < 0.5 * a)
            # on strike well beyond the tip and away from the traction-free
            # edges (the stress falls over the drag length sqrt(eta / c) there)
            far = np.argmin(np.hypot(tau_coords[:, 0] - 1.6, tau_coords[:, 1]))
        else:
            phi_n = np.arctan2(tau_coords[:, 1], tau_coords[:, 0])
            nodes_on_fault = (np.abs(tau_coords[:, 2]) < 1e-9) & (np.abs(phi_n) < 0.5 * a)
            far = np.argmin(np.linalg.norm(tau_coords - np.array([np.cos(np.pi), np.sin(np.pi), 0.0]), axis=1))
    shear, normal = resolved(tau_coords, tau)
    history.append((t, shear[nodes_on_fault].mean(), shear[far], slip.max() if len(slip) else 0.0,
                    slip.mean() if len(slip) else 0.0, info["nonlinear_iterations"]))
    uw.pprint(f"t={t:.3f} tau_fault={history[-1][1]:+.4f} tau_far={history[-1][2]:+.4f} "
              f"slip_max={history[-1][3]:.3e} leak={leak:.1e} newton={info['nonlinear_iterations']}", flush=True)
    if uw.mpi.rank == 0:
        np.savez(os.path.join(out, "history.npz"), history=np.array(history), strength=strength)
    if snapshots and abs(t - snapshots[0]) < 0.5 * params.uw_dt:
        snapshots.pop(0)
        # petsc_reload: a split mesh duplicates the fault nodes, and only the
        # PETSc-native payload lets a later read tell the two sides apart (#640)
        thinning_projection.solve()
        split.write_timestep("cycle", k, outputPath=out, meshVars=[sheet.u, sheet.tau, thinning],
                             petsc_reload=True)
        shear_ref, normal_ref = resolved(tau_coords, np.asarray(reference.tau.array))
        loading = np.sign(np.mean(shear_ref[nodes_on_fault])) or 1.0
        dcff = loading * (shear - shear_ref) + params.uw_mu_f * (normal - normal_ref)
        if uw.mpi.rank == 0:
            uplift = params.uw_isostatic_factor * np.asarray(thinning.array[:, 0, 0])
            render(k, t, dcff, uplift, history[-1][3])
        k += 1

if uw.mpi.rank == 0:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.array(history)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True)
    ax1.plot(h[:, 0], h[:, 1], color="#1f5fa8", lw=2.0, label="shear traction on the fault")
    ax1.plot(h[:, 0], h[:, 2], color="#1f5fa8", lw=1.2, ls=":", label="shear stress far from the fault")
    ax1.axhline(strength, color="0.25", lw=1.0, ls="--", label="Coulomb strength")
    ax1.set_ylabel("shear stress")
    ax1.legend(frameon=False, fontsize=9)
    ax2.plot(h[:, 0], h[:, 3], color="#c0392b", lw=2.0, label="max slip rate")
    ax2.plot(h[:, 0], h[:, 4], color="#c0392b", lw=1.2, ls=":", label="mean slip rate")
    ax2.set_xlabel("time  t / t_r")
    ax2.set_ylabel("slip rate")
    ax2.legend(frameon=False, fontsize=9)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.9", lw=0.6)
    fig.suptitle(f"Coulomb fault in a Maxwell sheet dragged by a shearing substrate ({params.uw_geometry})",
                 fontsize=10, x=0.12, ha="left")
    fig.savefig(os.path.join(out, "fault-cycle-history.png"), dpi=160, bbox_inches="tight")
    uw.pprint(f"wrote figures and checkpoints to {out}", flush=True)
