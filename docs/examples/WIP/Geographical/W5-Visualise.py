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
# # W5: Visualisation
#
# Loads checkpointed fields from the W-series pipeline and provides
# cross-section views of the fault-controlled flow system.
#
# **Requires**: Products from W1 (shared), W2, W3, W4b (per azimuth)
#
# Cross-sections are clipped by radius to give plan views at chosen
# depths — avoids opacity issues in pyvista.

# %% [markdown]
# ## Configuration

# %%
STRESS_AZIMUTH = 30
CLIP_FRACTION = 0.075  # depth fraction from surface for default cross-section

# %% [markdown]
# ## 1. Load Data from Checkpoints

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

SHARED_DIR = Path("output/h2ex/shared")
RUN_DIR = Path(f"output/h2ex/azimuth_{STRESS_AZIMUTH:03d}")

shared = WorkflowProducts(products_dir=SHARED_DIR / "products")
run = WorkflowProducts(products_dir=RUN_DIR / "products")

print(f"=== H2Ex Visualisation: azimuth={STRESS_AZIMUTH} ===\n")

# %%
# Load mesh
t0 = time.time()
mesh = shared.load("adapted_mesh")
print(f"Mesh: {mesh.X.coords.shape[0]} P1 nodes")

# Load fault surfaces
faults = shared.load("fault_surfaces", mesh=mesh)
print(f"Faults: {len(faults.surfaces)} surfaces")

# Load per-run fields
strain_rate = uw.discretisation.MeshVariable("eps", mesh, 1, degree=1)
run.load("strain_rate", mesh_variable=strain_rate)

strain_rate_ref = uw.discretisation.MeshVariable("eps_ref", mesh, 1, degree=1)
run.load("strain_rate_ref", mesh_variable=strain_rate_ref)

permeability = uw.discretisation.MeshVariable("perm", mesh, 1, degree=1, varsymbol=r"\kappa")
run.load("permeability", mesh_variable=permeability)

v_darcy = uw.discretisation.MeshVariable("v_darcy", mesh, mesh.dim, degree=1,
                                          continuous=True, units="m/s")
run.load("darcy_velocity", mesh_variable=v_darcy)

# Concentration (if available)
has_concentration = run.exists("concentration")
if has_concentration:
    C = uw.discretisation.MeshVariable("C", mesh, 1, degree=1)
    run.load("concentration", mesh_variable=C)
    print("Concentration: loaded")
else:
    C = None
    print("Concentration: not available (run W4b)")

print(f"\nLoaded in {time.time()-t0:.1f}s")

# %% [markdown]
# ## 2. Build PyVista Mesh with Fields

# %%
pvmesh = vis.mesh_to_pv_mesh(mesh)

# NOTE: vis.*_fn_to_pv_points uses evaluate with pyvista points (plain ndarray
# in meters), which evaluate treats as nondimensional — giving wrong results
# on geographic meshes with units. Workaround: assign .data directly for P1
# variables (same DOFs as mesh vertices).

# Strain rates (P1 — direct assignment)
pvmesh.point_data["eps"] = np.asarray(strain_rate.data).ravel()
pvmesh.point_data["eps_ref"] = np.asarray(strain_rate_ref.data).ravel()
pvmesh.point_data["delta_eps"] = pvmesh.point_data["eps"] - pvmesh.point_data["eps_ref"]

# Permeability (P1 — direct assignment)
pvmesh.point_data["log_perm"] = np.log10(np.maximum(np.asarray(permeability.data).ravel(), 1e-10))

# Darcy velocity (P1 — direct assignment)
pvmesh.point_data["Vd"] = np.asarray(v_darcy.data)
pvmesh.point_data["Vd_mag"] = np.linalg.norm(pvmesh.point_data["Vd"], axis=1)

# Fault distance and weight (P1 — computed fresh, then direct)
fields = faults.compute_nearest_fields(mesh, fault_width=uw.quantity(10.0, "km"))
pvmesh.point_data["fault_dist"] = np.asarray(fields["distance"].data).ravel()
pvmesh.point_data["fault_weight"] = np.asarray(fields["weight"].data).ravel()

# Concentration (P1 — direct assignment)
if C is not None:
    pvmesh.point_data["C"] = np.asarray(C.data).ravel()

# Radius for clipping
pts = pvmesh.points
r = np.sqrt(np.sum(pts**2, axis=1))
pvmesh.point_data["radius"] = r

r_surface = r.max()
r_bottom = r.min()
depth_range = r_surface - r_bottom

print(f"PyVista mesh: {pvmesh.n_points} points, {pvmesh.n_cells} cells")
print(f"Fields: {list(pvmesh.point_data.keys())}")

# %% [markdown]
# ## Helpers

# %%
FAULT_COLORS = [
    "Red", "Orange", "Blue", "Green", "Purple",
    "Cyan", "Yellow", "Magenta", "Lime", "Pink",
]

def add_faults(plotter, opacity=0.7, show_edges=True):
    """Add all fault surfaces to a plotter.

    Fault VTK coordinates are nondimensional; the pyvista mesh is in meters.
    Scale fault points by L_ref to match.
    """
    L_ref = float(model.get_fundamental_scales()["length"].magnitude)  # meters

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
    """Clip mesh below a depth fraction, returning a plan-view cross-section."""
    r_cut = r_surface - fraction * depth_range
    return mesh_pv.clip_scalar(scalars="radius", value=r_cut)

# %% [markdown]
# ## 3. Mesh and Fault Geometry

# %%
clipped = clip_at_depth(pvmesh)

pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="fault_dist", cmap="viridis_r",
            show_edges=True,
            scalar_bar_args={"title": "Distance to nearest fault"})
add_faults(pl)
pl.add_title(f"Fault Distance (clipped at {CLIP_FRACTION*100:.0f}% depth)")
pl.show()

# %% [markdown]
# ## 4. Strain Rate (Faulted vs Reference)

# %%
# Strain rate invariant — faulted
pl = pv.Plotter(window_size=(1000, 800))
eps_data = clipped.point_data["eps"]
pl.add_mesh(clipped, scalars="eps", cmap="inferno",
            show_edges=False,
            clim=(eps_data.min(), np.percentile(eps_data, 98)),
            scalar_bar_args={"title": "Strain rate (faulted)"})
add_faults(pl, opacity=0.3)
pl.add_title(f"Strain Rate — azimuth={STRESS_AZIMUTH}")
pl.show()

# %% [markdown]
# ### Strain Rate Anomaly
#
# Delta = faulted minus reference.  Positive = strain concentrated by faults.

# %%
delta = clipped.point_data["delta_eps"]
delta_absmax = np.percentile(np.abs(delta), 95)

pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="delta_eps", cmap="RdBu_r",
            show_edges=False,
            clim=[-delta_absmax, delta_absmax],
            scalar_bar_args={"title": "Strain rate anomaly"})
add_faults(pl, opacity=0.5)
pl.add_title(f"Strain Rate Anomaly — azimuth={STRESS_AZIMUTH}")
pl.show()

# %% [markdown]
# ## 5. Permeability

# %%
pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="log_perm", cmap="RdYlBu_r",
            show_edges=False,
            scalar_bar_args={"title": "log10(k)"})
add_faults(pl, opacity=0.5)
pl.add_title("Permeability (fault-localised)")

pl.show()

# %% [markdown]
# ## 6. Darcy Velocity

# %%
pl = pv.Plotter(window_size=(1000, 800))
pl.add_mesh(clipped, scalars="Vd_mag", cmap="viridis",
            show_edges=False, opacity=0.6,
            scalar_bar_args={"title": "|v_darcy|"})

Vd = clipped.point_data["Vd"]
vd_max = np.linalg.norm(Vd, axis=1).max()
if vd_max > 0:
    pl.add_arrows(clipped.points, Vd, mag=10000 / vd_max,
                  color="White", opacity=0.5)

add_faults(pl, opacity=0.5)
pl.add_title("Darcy Velocity (upward through faults)")
pl.show()

# %% [markdown]
# ## 7. Concentration (Flow Accumulation)

# %%
if C is not None:
    pl = pv.Plotter(window_size=(1000, 800))
    pl.add_mesh(clipped, scalars="C", cmap="magma",
                show_edges=False, clim=(0.0,1.0),
                scalar_bar_args={"title": "Concentration C"})
    add_faults(pl, opacity=1)
    pl.add_title("Flow Accumulation (steady-state concentration)")
    pl.show()
else:
    print("Concentration field not available — run W4b-Concentration.py")

# %% [markdown]
# ## 8. Depth Variation
#
# Same field at different depth fractions.

# %%
field = "C" if C is not None else "log_perm"
cmap = "magma" if field == "C" else "RdYlBu_r"

fractions = [0.05, 0.15, 0.35, 0.65]
pl = pv.Plotter(shape=(2, 2), window_size=(1400, 1100))

for i, frac in enumerate(fractions):
    row, col = divmod(i, 2)
    pl.subplot(row, col)

    depth_km = frac * depth_range / 1000
    cut = clip_at_depth(pvmesh, fraction=frac)

    if field == "C":
        pl.add_mesh(cut, scalars=field, cmap=cmap, show_edges=False,
                    scalar_bar_args={"title": "C"})
    else:
        pl.add_mesh(cut, scalars=field, cmap=cmap, show_edges=False,
                    scalar_bar_args={"title": "log10(k)"})

    add_faults(pl, opacity=0.3)
    pl.add_title(f"~{frac*100:.0f}% depth")

pl.link_views()
pl.show()

# %% [markdown]
# ## 9. Combined Overview

# %%
pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

pl.subplot(0, 0)
pl.add_mesh(clipped, scalars="log_perm", cmap="RdYlBu_r",
            show_edges=False,
            scalar_bar_args={"title": "log10(k)"})
add_faults(pl, opacity=0.5)
pl.add_title("Permeability")

pl.subplot(0, 1)
if C is not None:
    pl.add_mesh(clipped, scalars="C", cmap="magma",
                show_edges=False,
                scalar_bar_args={"title": "C"})
else:
    pl.add_mesh(clipped, scalars="Vd_mag", cmap="viridis",
                show_edges=False,
                scalar_bar_args={"title": "|v_darcy|"})
if vd_max > 0:
    pl.add_arrows(clipped.points, Vd, mag=0.02 / vd_max,
                  color="White", opacity=0.4)
add_faults(pl, opacity=0.5)
pl.add_title("Flow Accumulation" if C is not None else "Darcy Velocity")

pl.link_views()
pl.show()

# %% [markdown]
# ## 10. Two-Azimuth Comparison
#
# Load a second azimuth run and compare strain rate anomaly,
# Darcy velocity magnitude, and concentration side by side.

# %%
AZIMUTH_B = 0 if STRESS_AZIMUTH != 0 else 30
RUN_DIR_B = Path(f"output/h2ex/azimuth_{AZIMUTH_B:03d}")
run_b = WorkflowProducts(products_dir=RUN_DIR_B / "products")

if run_b.exists("strain_rate") and run_b.exists("darcy_velocity"):
    # Load second azimuth fields onto same mesh
    eps_b = uw.discretisation.MeshVariable("eps_b", mesh, 1, degree=1)
    run_b.load("strain_rate", mesh_variable=eps_b)

    eps_ref_b = uw.discretisation.MeshVariable("eps_ref_b", mesh, 1, degree=1)
    run_b.load("strain_rate_ref", mesh_variable=eps_ref_b)

    vd_b = uw.discretisation.MeshVariable("v_darcy_b", mesh, mesh.dim, degree=1,
                                           continuous=True, units="m/s")
    # Need matching var_name for read_timestep
    vd_b.read_timestep("darcy_velocity", "v_darcy", index=0,
                        outputPath=str(RUN_DIR_B / "products"))

    C_b = None
    if run_b.exists("concentration"):
        C_b = uw.discretisation.MeshVariable("C_b", mesh, 1, degree=1)
        run_b.load("concentration", mesh_variable=C_b)

    # Build pyvista data for run B
    pvmesh.point_data["delta_eps_B"] = np.asarray(eps_b.data).ravel() - np.asarray(eps_ref_b.data).ravel()
    pvmesh.point_data["Vd_mag_B"] = np.linalg.norm(np.asarray(vd_b.data), axis=1)
    if C_b is not None:
        pvmesh.point_data["C_B"] = np.asarray(C_b.data).ravel()

    clipped = clip_at_depth(pvmesh)
    print(f"Loaded azimuth {AZIMUTH_B} for comparison")
else:
    print(f"Azimuth {AZIMUTH_B} products not available — run W2+W3 for that azimuth")

# %% [markdown]
# ### Strain Rate Anomaly: two azimuths

# %%
if "delta_eps_B" in pvmesh.point_data:
    delta_A = clipped.point_data["delta_eps"]
    delta_B = clipped.point_data["delta_eps_B"]
    clim = max(np.percentile(np.abs(delta_A), 95), np.percentile(np.abs(delta_B), 95))

    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

    pl.subplot(0, 0)
    pl.add_mesh(clipped, scalars="delta_eps", cmap="RdBu_r",
                show_edges=False, clim=[-clim, clim],
                scalar_bar_args={"title": "delta eps"})
    add_faults(pl, opacity=0.3)
    pl.add_title(f"Azimuth {STRESS_AZIMUTH}")

    pl.subplot(0, 1)
    pl.add_mesh(clipped, scalars="delta_eps_B", cmap="RdBu_r",
                show_edges=False, clim=[-clim, clim],
                scalar_bar_args={"title": "delta eps"})
    add_faults(pl, opacity=0.3)
    pl.add_title(f"Azimuth {AZIMUTH_B}")

    pl.link_views()
    pl.show()

# %% [markdown]
# ### Darcy Velocity Magnitude: two azimuths

# %%
if "Vd_mag_B" in pvmesh.point_data:
    vmax = max(clipped.point_data["Vd_mag"].max(), clipped.point_data["Vd_mag_B"].max())

    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

    pl.subplot(0, 0)
    pl.add_mesh(clipped, scalars="Vd_mag", cmap="viridis",
                show_edges=False, clim=[0, vmax],
                scalar_bar_args={"title": "|v_darcy|"})
    add_faults(pl, opacity=0.3)
    pl.add_title(f"Azimuth {STRESS_AZIMUTH}")

    pl.subplot(0, 1)
    pl.add_mesh(clipped, scalars="Vd_mag_B", cmap="viridis",
                show_edges=False, clim=[0, vmax],
                scalar_bar_args={"title": "|v_darcy|"})
    add_faults(pl, opacity=0.3)
    pl.add_title(f"Azimuth {AZIMUTH_B}")

    pl.link_views()
    pl.show()

# %% [markdown]
# ### Concentration: two azimuths

# %%
if C_b is not None and "C_B" in pvmesh.point_data:
    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

    pl.subplot(0, 0)
    pl.add_mesh(clipped, scalars="C", cmap="magma",
                show_edges=False, clim=[0, 1],
                scalar_bar_args={"title": "C"})
    add_faults(pl, opacity=0.5)
    pl.add_title(f"Azimuth {STRESS_AZIMUTH}")

    pl.subplot(0, 1)
    pl.add_mesh(clipped, scalars="C_B", cmap="magma",
                show_edges=False, clim=[0, 1],
                scalar_bar_args={"title": "C"})
    add_faults(pl, opacity=0.5)
    pl.add_title(f"Azimuth {AZIMUTH_B}")

    pl.link_views()
    pl.show()

# %%
