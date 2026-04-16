# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # W5a: Single-Azimuth Visualisation
#
# Loads checkpointed fields for one stress azimuth and provides
# cross-section views of all pipeline stages.
#
# **Requires**: Products from W1 (shared), W2, W3, W4b (per azimuth)
#
# **Sections**: Fault geometry, Stokes velocity magnitude, strain rate
# anomaly, permeability, Darcy velocity, concentration

# %% [markdown]
# ## Configuration

# %%
STRESS_AZIMUTH = 0
RES = 28
CLIP_FRACTION = 0.075  # depth fraction from surface for default cross-section

# %% [markdown]
# ## 1. Load Data

# %%
import numpy as np
import time
from pathlib import Path
import underworld3 as uw
from underworld3.workflows import WorkflowProducts
import pyvista as pv
import underworld3.visualisation as vis

uw.reset_default_model()
model = uw.Model()
model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    viscosity=uw.quantity(1e21, "Pa.s"),
    diffusivity=uw.quantity(1e-6, "m**2/s"),
    verbose=False,
)

SHARED_DIR = Path(f"output/h2ex/res_{RES}/shared")
RUN_DIR = Path(f"output/h2ex/res_{RES}/azimuth_{STRESS_AZIMUTH:03d}")

shared = WorkflowProducts(products_dir=SHARED_DIR / "products")
run = WorkflowProducts(products_dir=RUN_DIR / "products")

print(f"=== H2Ex Visualisation: azimuth={STRESS_AZIMUTH} ===\n")

# %%
t0 = time.time()
mesh = shared.load("adapted_mesh")
faults = shared.load("fault_surfaces", mesh=mesh)

strain_rate = uw.discretisation.MeshVariable("eps", mesh, 1, degree=1)
run.load("strain_rate", mesh_variable=strain_rate)

strain_rate_ref = uw.discretisation.MeshVariable("eps_ref", mesh, 1, degree=1)
run.load("strain_rate_ref", mesh_variable=strain_rate_ref)

# Stokes velocity and pressure (if checkpointed)
v_stokes = None
if run.exists("stokes_velocity"):
    v_stokes = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2,
                                               varsymbol=r"\mathbf{v}", units="cm/yr")
    run.load("stokes_velocity", mesh_variable=v_stokes)
    print("Stokes velocity: loaded")

permeability = uw.discretisation.MeshVariable("perm", mesh, 1, degree=1)
run.load("permeability", mesh_variable=permeability)

v_darcy = uw.discretisation.MeshVariable("v_darcy", mesh, mesh.dim, degree=1,
                                          continuous=True, units="m/s")
run.load("darcy_velocity", mesh_variable=v_darcy)

C = None
if run.exists("concentration"):
    C = uw.discretisation.MeshVariable("C", mesh, 1, degree=1)
    run.load("concentration", mesh_variable=C)

print(f"Mesh: {mesh.X.coords.shape[0]} nodes, {len(faults.surfaces)} faults")
print(f"Concentration: {'loaded' if C else 'not available'}")
print(f"Loaded in {time.time()-t0:.1f}s")

# %% [markdown]
# ## 2. Build PyVista Mesh

# %%
pvmesh = vis.mesh_to_pv_mesh(mesh)

# All P1 fields — direct .data assignment (avoids evaluate + dimensional coords issue)
pvmesh.point_data["eps"] = np.asarray(strain_rate.data).ravel()
pvmesh.point_data["eps_ref"] = np.asarray(strain_rate_ref.data).ravel()
pvmesh.point_data["delta_eps"] = pvmesh.point_data["eps"] - pvmesh.point_data["eps_ref"]

# Stokes velocity (P2 → evaluate at P1 mesh vertices for pyvista)
if v_stokes is not None:
    # v_stokes is P2 with more DOFs than mesh vertices. Use .data which
    # is indexed by DOF, not by mesh vertex. For visualisation on the P1
    # pyvista mesh, evaluate at mesh vertex coords (nondimensional).
    dm_coords = mesh.dm.getCoordinatesLocal().array.reshape(-1, 3)
    v_stokes_at_verts = np.asarray(
        uw.function.evaluate(v_stokes.sym, dm_coords, mode="fast")
    ).reshape(-1, 3)
    pvmesh.point_data["V_stokes"] = v_stokes_at_verts
    pvmesh.point_data["V_stokes_mag"] = np.linalg.norm(v_stokes_at_verts, axis=1)
pvmesh.point_data["log_perm"] = np.log10(np.maximum(np.asarray(permeability.data).ravel(), 1e-10))
pvmesh.point_data["Vd"] = np.asarray(v_darcy.data)
pvmesh.point_data["Vd_mag"] = np.linalg.norm(pvmesh.point_data["Vd"], axis=1)

fields = faults.compute_nearest_fields(mesh, fault_width=uw.quantity(10.0, "km"))
pvmesh.point_data["fault_dist"] = np.asarray(fields["distance"].data).ravel()

if C is not None:
    pvmesh.point_data["C"] = np.asarray(C.data).ravel()

# Radius for clipping
r = np.sqrt(np.sum(pvmesh.points**2, axis=1))
pvmesh.point_data["radius"] = r
r_surface = r.max()
r_bottom = r.min()
depth_range = r_surface - r_bottom

print(f"Fields: {list(pvmesh.point_data.keys())}")

# %% [markdown]
# ## Helpers

# %%
FAULT_COLORS = [
    "Red", "Orange", "Blue", "Green", "Purple",
    "Cyan", "Yellow", "Magenta", "Lime", "Pink",
]

L_ref = float(model.get_fundamental_scales()["length"].magnitude)

def add_faults(plotter, opacity=0.7, show_edges=True):
    """Add fault surfaces, scaling from nondimensional to meters."""
    for i, (name, surf) in enumerate(faults.surfaces.items()):
        if hasattr(surf, "_pv_mesh") and surf._pv_mesh is not None:
            scaled = surf._pv_mesh.copy()
            scaled.points = scaled.points * L_ref
            plotter.add_mesh(
                scaled, style="surface",
                color=FAULT_COLORS[i % len(FAULT_COLORS)],
                opacity=opacity, show_edges=show_edges,
            )

def clip_at_depth(mesh_pv, fraction=CLIP_FRACTION):
    """Clip mesh below a depth fraction for plan-view cross-section."""
    r_cut = r_surface - fraction * depth_range
    return mesh_pv.clip_scalar(scalars="radius", value=r_cut)

clipped = clip_at_depth(pvmesh)

# %% [markdown]
# ## 3. Fault Geometry and Mesh Refinement

# %%
pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="fault_dist", cmap="viridis_r",
            show_edges=True,
            scalar_bar_args={"title": "Distance to nearest fault"})
add_faults(pl, opacity=1)
pl.add_title(f"Adapted Mesh + Fault Distance")
pl.show()

# %% [markdown]
# ## 4. Stokes Velocity Magnitude
#
# Shows the fault-block motions from the TI Stokes solve.
# The velocity magnitude highlights which blocks move fastest
# under the applied stress orientation.

# %%
if "V_stokes_mag" in pvmesh.point_data:
    pl = pv.Plotter(window_size=(1000, 800))
    vs_mag = clipped.point_data["V_stokes_mag"]
    pl.add_mesh(clipped, scalars="V_stokes_mag", cmap="plasma",
                show_edges=False, opacity=1,
                clim=[0, np.percentile(vs_mag, 98)],
                scalar_bar_args={"title": "|v_stokes|"})

    Vs = clipped.point_data["V_stokes"]
    vs_max = np.linalg.norm(Vs, axis=1).max()
    if vs_max > 0:
        pl.add_arrows(clipped.points, Vs, mag=3000 / vs_max,
                      color="White", opacity=0.5)

    add_faults(pl, opacity=0.5)
    pl.add_title(f"Stokes Velocity — azimuth={STRESS_AZIMUTH}")
    pl.show()
else:
    print("Stokes velocity not checkpointed — re-run W2 to save it")

# %% [markdown]
# ## 5. Strain Rate (Faulted)

# %%
eps_data = clipped.point_data["eps"]
pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="eps", cmap="inferno",
            show_edges=False,
            clim=(eps_data.min(), np.percentile(eps_data, 98)),
            scalar_bar_args={"title": "Strain rate inv2"})
add_faults(pl, opacity=0.3)
pl.add_title(f"Strain Rate — azimuth={STRESS_AZIMUTH}")
pl.show()

# %% [markdown]
# ## 6. Strain Rate Anomaly
#
# Delta = faulted minus reference.
# Positive = strain concentrated by faults.
# Negative = strain shadow.

# %%
delta = clipped.point_data["delta_eps"]
delta_absmax = np.percentile(np.abs(delta), 95)

pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="delta_eps", cmap="inferno",
            show_edges=False,
            clim=[-delta_absmax, delta_absmax],
            scalar_bar_args={"title": "Strain rate anomaly"})
add_faults(pl, opacity=0.5)
pl.add_title(f"Strain Rate Anomaly — azimuth={STRESS_AZIMUTH}")
pl.show()

# %% [markdown]
# ## 7. Permeability

# %%
pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="log_perm", cmap="RdYlBu_r",
            show_edges=False,
            scalar_bar_args={"title": "log10(k)"})
add_faults(pl, opacity=0.5)
pl.add_title(f"Permeability — azimuth={STRESS_AZIMUTH}")
pl.show()

# %% [markdown]
# ## 8. Concentration (Flow Accumulation)

# %%
if C is not None:
    pl = pv.Plotter(window_size=(1000, 800))
    pl.add_mesh(clipped, scalars="C", cmap="Blues",
                show_edges=False, clim=(0, 0.5), opacity=1,
                scalar_bar_args={"title": "Concentration C"})
    add_faults(pl, opacity=1)
    pl.add_title(f"Flow Accumulation — azimuth={STRESS_AZIMUTH}")
    pl.show()
else:
    print("Concentration not available — run W4b-Concentration.py")

# %% [markdown]
# ## 9. Depth Variation

# %%
field = "C" if C is not None else "log_perm"
cmap = "Blues" if field == "C" else "RdYlBu_r"

fractions = [0.05, 0.15, 0.35, 0.65]
pl = pv.Plotter(shape=(2, 2), window_size=(1400, 1100))

for i, frac in enumerate(fractions):
    row, col = divmod(i, 2)
    pl.subplot(row, col)
    cut = clip_at_depth(pvmesh, fraction=frac)
    pl.add_mesh(cut, scalars=field, cmap=cmap, show_edges=False,
                scalar_bar_args={"title": field})
    add_faults(pl, opacity=0.3)
    pl.add_title(f"~{frac*100:.0f}% depth")

pl.link_views()
pl.show()

# %%
