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
# # W1: Mesh and Fault Surfaces (Shared)
#
# Creates the geographic mesh, loads fault traces from CSV, builds fault
# surfaces, and adapts the mesh near faults.  These products are
# **shared** across all parameter variations (stress azimuth, etc.).
#
# **Produces**: `adapted_mesh`, `fault_surfaces` → `output/h2ex/shared/`
#
# **Next**: `W2-Stokes.py`

# %% [markdown]
# ## Configuration

# %%
RES = 16  # elements per horizontal dimension (16=dev, 80=publication)

# %% [markdown]
# ## Workflow Status

# %%
import numpy as np
import csv
import time
from pathlib import Path
import underworld3 as uw
from underworld3.workflows import WorkflowProducts

# Units setup
uw.reset_default_model()
model = uw.Model()
model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    viscosity=uw.quantity(1e21, "Pa.s"),
    diffusivity=uw.quantity(1e-6, "m**2/s"),
    verbose=False,
)

# Product storage — resolution-keyed
SHARED_DIR = Path(f"output/h2ex/res_{RES}/shared")
shared = WorkflowProducts(products_dir=SHARED_DIR / "products")

print("=== H2Ex Workflow: Stage 1 (Mesh and Faults) ===")
print(f"Resolution: {RES}x{RES}x{RES//2}")
print(f"Output: {SHARED_DIR}")
print()

# Check if already done
for product in ["adapted_mesh", "fault_surfaces"]:
    status = "ready" if shared.exists(product) else "not built"
    print(f"  {product}: {status}")

if shared.exists("adapted_mesh") and shared.exists("fault_surfaces"):
    print("\nAll products already exist. Re-run cells below to rebuild.")

# %% [markdown]
# ## 1. Load Fault Traces from CSV

# %%
_csv_name = "CombinedInferredFaults_2025_04_22.csv"
_search_paths = [
    Path("Structures") / _csv_name,
    Path.home() / "Library/CloudStorage/Box-Box/+Claude/H2Ex-Data/Structures" / _csv_name,
    Path.home() / "Box/+Claude/H2Ex-Data/Structures" / _csv_name,
]
_csv_path = next((p for p in _search_paths if p.exists()), None)
if _csv_path is None:
    raise FileNotFoundError(
        f"Cannot find {_csv_name} in any of:\n" +
        "\n".join(f"  {p}" for p in _search_paths)
    )

_rows = []
with open(_csv_path) as f:
    for row in csv.DictReader(f):
        _rows.append(row)

fault_lon   = np.array([float(r["Lon"]) for r in _rows])
fault_lat   = np.array([float(r["Lat"]) for r in _rows])
fault_depth = np.array([float(r["Depth"]) for r in _rows])
fault_dip   = np.array([float(r["Dip Angle at depth"]) for r in _rows])
fault_name  = np.array([float(r["Name"]) for r in _rows])
fault_desc  = [r["Description"].strip() for r in _rows]
unique_fault_ids = np.unique(fault_name)

print(f"Loaded {len(fault_lon)} trace points, {len(unique_fault_ids)} segments from {_csv_path.name}")

# %% [markdown]
# ## 2. Create Geographic Mesh

# %%
mesh = uw.meshing.RegionalGeographicBox(
    lon_range=(135.5, 137.5),
    lat_range=(-34.5, -33.0),
    depth_range=(uw.quantity(0, "km"), uw.quantity(50, "km")),
    ellipsoid="WGS84",
    numElements=(RES, RES, RES // 2),
    simplex=True,
)
print(f"Initial mesh: {mesh.X.coords.shape[0]} P1 nodes")

# %% [markdown]
# ## 3. Build Fault Surfaces

# %%
fault_surfaces = {}

for fid in unique_fault_ids:
    mask = fault_name == fid
    if mask.sum() < 2:
        continue

    trace = np.column_stack([fault_lon[mask], fault_lat[mask]])
    depth_km = float(fault_depth[mask][0])
    segment_dip = float(fault_dip[mask][0])
    description = fault_desc[np.argmax(mask)]

    name = f"fault_{fid}"
    s = uw.meshing.Surface.from_trace(
        name, mesh, trace,
        depth_range=(uw.quantity(0, "km"), uw.quantity(depth_km, "km")),
        depth_spacing=uw.quantity(5, "km"),
        trace_resolution=uw.quantity(3, "km"),
        dip=segment_dip,
        dip_direction="right",
        symbol=f"F{fid}",
    )
    fault_surfaces[fid] = s

faults = uw.meshing.SurfaceCollection()
for s in fault_surfaces.values():
    faults.add(s)

print(f"Created {len(fault_surfaces)} fault surfaces")

# %% [markdown]
# ## 4. Adapt Mesh Near Faults

# %%
combined_metric = faults.refinement_metric(
    mesh,
    h_near=uw.quantity(2.0, "km"),
    h_far=uw.quantity(20.0, "km"),
    width=uw.quantity(10.0, "km"),
    profile="smoothstep",
)

n_before = mesh.X.coords.shape[0]
mesh.adapt(combined_metric, verbose=False)
n_after = mesh.X.coords.shape[0]
print(f"Adapted: {n_before} -> {n_after} P1 nodes")

# %% [markdown]
# ## 5. Checkpoint

# %%
t0 = time.time()

shared.save("adapted_mesh", mesh)
shared.save("fault_surfaces", faults)

print(f"Checkpointed in {time.time()-t0:.1f}s")
print()
shared.list()

# %% [markdown]
# ## Done
#
# **Next step**: `W2-Stokes.py` (set `STRESS_AZIMUTH` for orientation)

# %%
print("\n=== Stage 1 Complete ===")
print("Products saved to:", SHARED_DIR / "products")
print("\nNext: W2-Stokes.py")

# %%
