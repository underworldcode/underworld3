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
# # W5b: Two-Azimuth Comparison
#
# Side-by-side comparison of two stress azimuth runs.  Shows how
# the orientation of the driving stress changes which faults are
# activated, the strain rate pattern, and the flow accumulation.
#
# **Requires**: Products from W2, W3, W4b for both azimuths

# %% [markdown]
# ## Configuration

# %%
AZIMUTH_A = 0
AZIMUTH_B = 30
RES = 28
CLIP_FRACTION = 0.075

# %% [markdown]
# ## 1. Load Data for Both Azimuths

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
shared = WorkflowProducts(products_dir=SHARED_DIR / "products")

run_a = WorkflowProducts(products_dir=Path(f"output/h2ex/res_{RES}/azimuth_{AZIMUTH_A:03d}/products"))
run_b = WorkflowProducts(products_dir=Path(f"output/h2ex/res_{RES}/azimuth_{AZIMUTH_B:03d}/products"))


print(f"=== Azimuth Comparison: {AZIMUTH_A} vs {AZIMUTH_B} ===\n")
# Check both runs exist
for label, run_store in [("A", run_a), ("B", run_b)]:
    azimuth = AZIMUTH_A if label == "A" else AZIMUTH_B
    for dep in ["strain_rate", "darcy_velocity"]:
        if not run_store.exists(dep):
            raise RuntimeError(f"Azimuth {azimuth}: missing '{dep}'. Run W2+W3 first.")
    print(f"  Azimuth {azimuth}: all products available")

# %%
t0 = time.time()
mesh = shared.load("adapted_mesh")
faults = shared.load("fault_surfaces", mesh=mesh)

def load_run(run_store, suffix):
    """Load fields for one azimuth into uniquely-named variables."""
    eps = uw.discretisation.MeshVariable(f"eps_{suffix}", mesh, 1, degree=1)
    run_store.load("strain_rate", mesh_variable=eps)

    eps_ref = uw.discretisation.MeshVariable(f"eps_ref_{suffix}", mesh, 1, degree=1)
    run_store.load("strain_rate_ref", mesh_variable=eps_ref)

    vd = uw.discretisation.MeshVariable(f"v_darcy_{suffix}", mesh, mesh.dim, degree=1,
                                         continuous=True, units="m/s")
    run_store.load("darcy_velocity", mesh_variable=vd)

    vs = None
    if run_store.exists("stokes_velocity"):
        vs = uw.discretisation.MeshVariable(f"v_{suffix}", mesh, mesh.dim, degree=2,
                                             units="cm/yr")
        run_store.load("stokes_velocity", mesh_variable=vs)

    C = None
    if run_store.exists("concentration"):
        C = uw.discretisation.MeshVariable(f"C_{suffix}", mesh, 1, degree=1)
        run_store.load("concentration", mesh_variable=C)

    return eps, eps_ref, vd, vs, C

eps_a, eps_ref_a, vd_a, vs_a, C_a = load_run(run_a, "a")
eps_b, eps_ref_b, vd_b, vs_b, C_b = load_run(run_b, "b")

print(f"\nLoaded in {time.time()-t0:.1f}s")

# %% [markdown]
# ## 2. Build PyVista Mesh

# %%
pvmesh = vis.mesh_to_pv_mesh(mesh)

# Run A
pvmesh.point_data["eps_A"] = np.asarray(eps_a.data).ravel()
pvmesh.point_data["delta_eps_A"] = np.asarray(eps_a.data).ravel() - np.asarray(eps_ref_a.data).ravel()
pvmesh.point_data["Vd_A"] = np.asarray(vd_a.data)
pvmesh.point_data["Vd_mag_A"] = np.linalg.norm(pvmesh.point_data["Vd_A"], axis=1)
if C_a is not None:
    pvmesh.point_data["C_A"] = np.asarray(C_a.data).ravel()

# Run B
pvmesh.point_data["eps_B"] = np.asarray(eps_b.data).ravel()
pvmesh.point_data["delta_eps_B"] = np.asarray(eps_b.data).ravel() - np.asarray(eps_ref_b.data).ravel()
pvmesh.point_data["Vd_B"] = np.asarray(vd_b.data)
pvmesh.point_data["Vd_mag_B"] = np.linalg.norm(pvmesh.point_data["Vd_B"], axis=1)
if C_b is not None:
    pvmesh.point_data["C_B"] = np.asarray(C_b.data).ravel()

# Stokes velocity (P2 → evaluate at P1 vertices)
dm_coords = mesh.dm.getCoordinatesLocal().array.reshape(-1, 3)
if vs_a is not None:
    vs_at_a = np.asarray(uw.function.evaluate(vs_a.sym, dm_coords, mode="fast")).reshape(-1, 3)
    pvmesh.point_data["Vs_mag_A"] = np.linalg.norm(vs_at_a, axis=1)
if vs_b is not None:
    vs_at_b = np.asarray(uw.function.evaluate(vs_b.sym, dm_coords, mode="fast")).reshape(-1, 3)
    pvmesh.point_data["Vs_mag_B"] = np.linalg.norm(vs_at_b, axis=1)

# Radius for clipping
r = np.sqrt(np.sum(pvmesh.points**2, axis=1))
pvmesh.point_data["radius"] = r
r_surface = r.max()
depth_range = r_surface - r.min()

# %% [markdown]
# ## Helpers

# %%
FAULT_COLORS = [
    "Red", "Orange", "Blue", "Green", "Purple",
    "Cyan", "Yellow", "Magenta", "Lime", "Pink",
]
L_ref = float(model.get_fundamental_scales()["length"].magnitude)

def add_faults(plotter, opacity=0.7, show_edges=True):
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
    r_cut = r_surface - fraction * depth_range
    return mesh_pv.clip_scalar(scalars="radius", value=r_cut)

clipped = clip_at_depth(pvmesh)

# %% [markdown]
# ## 3. Stokes Velocity Magnitude: A vs B
#
# Shows fault-block motions — which blocks move fastest under each
# stress orientation.

# %%
if "Vs_mag_A" in pvmesh.point_data and "Vs_mag_B" in pvmesh.point_data:
    vs_max = max(np.percentile(clipped.point_data["Vs_mag_A"], 98),
                 np.percentile(clipped.point_data["Vs_mag_B"], 98))

    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

    pl.subplot(0, 0)
    pl.add_mesh(clipped, scalars="Vs_mag_A", cmap="plasma",
                show_edges=False, clim=[0, vs_max],
                scalar_bar_args={"title": "|v_stokes|"})
    add_faults(pl, opacity=0.3)
    pl.add_title(f"Stokes |v| — azimuth {AZIMUTH_A}")

    pl.subplot(0, 1)
    pl.add_mesh(clipped, scalars="Vs_mag_B", cmap="plasma",
                show_edges=False, clim=[0, vs_max],
                scalar_bar_args={"title": "|v_stokes|"})
    add_faults(pl, opacity=0.3)
    pl.add_title(f"Stokes |v| — azimuth {AZIMUTH_B}")

    pl.link_views()
    pl.show()
else:
    print("Stokes velocity not checkpointed — re-run W2 to save it")

# %% [markdown]
# ## 4. Strain Rate: A vs B

# %%
eps_A = clipped.point_data["eps_A"]
eps_B = clipped.point_data["eps_B"]
eps_max = max(np.percentile(eps_A, 98), np.percentile(eps_B, 98))
eps_min = min(eps_A.min(), eps_B.min())

pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

pl.subplot(0, 0)
pl.add_mesh(clipped, scalars="eps_A", cmap="inferno",
            show_edges=False, clim=[eps_min, eps_max],
            scalar_bar_args={"title": "Strain rate inv2"})
add_faults(pl, opacity=0.3)
pl.add_title(f"Azimuth {AZIMUTH_A}")

pl.subplot(0, 1)
pl.add_mesh(clipped, scalars="eps_B", cmap="inferno",
            show_edges=False, clim=[eps_min, eps_max],
            scalar_bar_args={"title": "Strain rate inv2"})
add_faults(pl, opacity=0.3)
pl.add_title(f"Azimuth {AZIMUTH_B}")

pl.link_views()
pl.show()

# %% [markdown]
# ## 5. Strain Rate Anomaly: A vs B

# %%
delta_A = clipped.point_data["delta_eps_A"]
delta_B = clipped.point_data["delta_eps_B"]
clim = max(np.percentile(np.abs(delta_A), 95), np.percentile(np.abs(delta_B), 95))

pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

pl.subplot(0, 0)
pl.add_mesh(clipped, scalars="delta_eps_A", cmap="inferno",
            show_edges=False, clim=[-clim, clim],
            scalar_bar_args={"title": "delta eps"})
add_faults(pl, opacity=0.3)
pl.add_title(f"Anomaly — azimuth {AZIMUTH_A}")

pl.subplot(0, 1)
pl.add_mesh(clipped, scalars="delta_eps_B", cmap="inferno",
            show_edges=False, clim=[-clim, clim],
            scalar_bar_args={"title": "delta eps"})
add_faults(pl, opacity=0.3)
pl.add_title(f"Anomaly — azimuth {AZIMUTH_B}")

pl.link_views()
pl.show()

# %% [markdown]
# ## 6. Darcy Velocity Magnitude: A vs B

# %%
vmax = max(clipped.point_data["Vd_mag_A"].max(), clipped.point_data["Vd_mag_B"].max())

pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

pl.subplot(0, 0)
pl.add_mesh(clipped, scalars="Vd_mag_A", cmap="viridis",
            show_edges=False, clim=[0, vmax],
            scalar_bar_args={"title": "|v_darcy|"})
add_faults(pl, opacity=0.3)
pl.add_title(f"Darcy |v| — azimuth {AZIMUTH_A}")

pl.subplot(0, 1)
pl.add_mesh(clipped, scalars="Vd_mag_B", cmap="viridis",
            show_edges=False, clim=[0, vmax],
            scalar_bar_args={"title": "|v_darcy|"})
add_faults(pl, opacity=0.3)
pl.add_title(f"Darcy |v| — azimuth {AZIMUTH_B}")

pl.link_views()
pl.show()

# %% [markdown]
# ## 7. Concentration: A vs B

# %%
if C_a is not None and C_b is not None:
    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

    pl.subplot(0, 0)
    pl.add_mesh(clipped, scalars="C_A", cmap="Blues",
                show_edges=False, clim=[0, 0.5],
                scalar_bar_args={"title": "C"})
    add_faults(pl, opacity=0.5)
    pl.add_title(f"Accumulation — azimuth {AZIMUTH_A}")

    pl.subplot(0, 1)
    pl.add_mesh(clipped, scalars="C_B", cmap="Blues",
                show_edges=False, clim=[0, 0.5],
                scalar_bar_args={"title": "C"})
    add_faults(pl, opacity=0.5)
    pl.add_title(f"Accumulation — azimuth {AZIMUTH_B}")

    pl.link_views()
    pl.show()
else:
    missing = []
    if C_a is None: missing.append(str(AZIMUTH_A))
    if C_b is None: missing.append(str(AZIMUTH_B))
    print(f"Concentration not available for azimuth(s): {', '.join(missing)}")

# %%
