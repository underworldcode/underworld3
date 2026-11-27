"""GWB input, geometry, and diagnostic helpers for the UW3 subduction example.

These functions do not import Underworld3. They implement the same numerical
conventions as the archived November 2026 proof of concept, including the
GWB v1.1.0 two-dimensional composition-column indexing workaround.

Coordinates passed from UW3 are nondimensional in a domain of width 8.0 and
height 1.6. GWB queries are specified in metres; slab profiles are in km.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


COLUMNS = (
    "x", "z", "d", "T", "vx", "vz",
    "c0", "c1", "c2", "c3", "c4", "tag",
)

MATERIAL_NAMES = (
    "Mantle", "Incoming plate", "Descending slab", "Overriding plate",
)


def make_model(source_path: Path, output_path: Path) -> tuple[dict, Path]:
    """Derive the four-material configuration from GWB's original cookbook.

    Assign composition IDs 1 and 2 to the incoming plate and slab, and
    composition ID 3 to the overriding plate. The original mantle remains 0.
    All source temperature and geometrical features are retained.

    Parameters
    ----------
    source_path : Path
        Official GWB v1.1.0 simple_subduction_2d_cartesian cookbook input.
    output_path : Path
        Destination for the modified GWB JSON input.

    Returns
    -------
    tuple[dict, Path]
        Modified model dictionary and output path.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)

    if source_path.resolve() == output_path.resolve():
        raise ValueError("The generated model must not overwrite its input")

    source = json.loads(source_path.read_text(encoding="utf-8"))
    matches = {name: [] for name in ("Subducting", "Slab", "Overriding")}

    for feature in source["features"]:
        name = feature.get("name")
        if name in matches:
            matches[name].append(feature)

    if any(len(features) != 1 for features in matches.values()):
        raise RuntimeError("Unexpected source cookbook feature layout")

    for name, material_id in (("Subducting", 1), ("Slab", 2)):
        models = matches[name][0].get("composition models", [])
        if (
            len(models) != 1
            or models[0].get("model") != "uniform"
            or models[0].get("compositions") != [0]
        ):
            raise RuntimeError(
                f"Unexpected {name} composition configuration; "
                "use the unmodified GWB v1.1.0 cookbook"
            )
        models[0]["compositions"] = [material_id]

    overriding = matches["Overriding"][0]
    if overriding.get("composition models"):
        raise RuntimeError("Overriding composition is already defined")

    overriding["composition models"] = [
        {
            "model": "uniform",
            "compositions": [3],
            "min depth": 0.0,
            "max depth": 100000.0,
        }
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(source, indent=2) + "\n", encoding="utf-8"
    )
    return source, output_path


def make_slab_top(
    source: dict,
    output_path: Path | None = None,
    depth_limit_km: float = 200.0,
) -> tuple[np.ndarray, float, float]:
    """Reconstruct the slab-top polyline and truncate it at the depth limit.

    Integrates dip angles on 0.25-km or smaller arc-length increments, using
    the midpoint angle of each increment. This reproduces the baseline
    workflow's segmented-arc approximation rather than changing geometry.

    Returns
    -------
    tuple
        Polyline (x_km, depth_km), trench x coordinate (km), and cumulative
        arc length to the depth cutoff (km).
    """
    slab = next(
        (feature for feature in source["features"]
         if feature.get("name") == "Slab"), None
    )
    if slab is None:
        raise RuntimeError("The GWB model has no Slab feature")

    trench = np.asarray(slab["coordinates"], dtype=float)
    if trench.ndim != 2 or trench.shape[1] != 2:
        raise ValueError("Expected two-dimensional slab trench coordinates")

    trench_x = float(trench[0, 0]) / 1000.0
    if not np.allclose(trench[:, 0], trench[0, 0]):
        raise RuntimeError("This demonstration requires a vertical trench line")

    dip_x = float(slab["dip point"][0]) / 1000.0
    direction = float(np.sign(dip_x - trench_x))
    if direction == 0.0:
        raise RuntimeError("Cannot determine slab dip direction")

    arc = [0.0]
    xs = [trench_x]
    depths = [0.0]
    previous_angle = None

    for segment in slab["segments"]:
        length_km = float(segment["length"]) / 1000.0
        initial, final = map(
            float, (segment["angle"][0], segment["angle"][-1])
        )
        if (
            length_km <= 0.0
            or min(initial, final) < 0
            or max(initial, final) > 90
        ):
            raise RuntimeError("Invalid slab segment")
        if previous_angle is not None and abs(previous_angle - initial) > 1e-8:
            raise RuntimeError("Discontinuous dip angles")
        previous_angle = final

        count = max(1, math.ceil(length_km / 0.25))
        ds = length_km / count
        for j in range(count):
            angle = math.radians(
                initial + (final - initial) * (j + 0.5) / count
            )
            arc.append(arc[-1] + ds)
            xs.append(xs[-1] + direction * ds * math.cos(angle))
            depths.append(depths[-1] + ds * math.sin(angle))

    arc = np.asarray(arc)
    xs = np.asarray(xs)
    depths = np.asarray(depths)

    cut = int(np.searchsorted(depths, depth_limit_km))
    if not 0 < cut < len(depths):
        raise RuntimeError("Slab does not reach interface depth limit")

    fraction = (
        (depth_limit_km - depths[cut - 1])
        / (depths[cut] - depths[cut - 1])
    )
    last_x = xs[cut - 1] + fraction * (xs[cut] - xs[cut - 1])
    profile = np.column_stack(
        (np.append(xs[:cut], last_x),
         np.append(depths[:cut], depth_limit_km))
    )
    arc_length = arc[cut - 1] + fraction * (arc[cut] - arc[cut - 1])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            output_path, profile, delimiter=",",
            header="x_km,depth_km", comments="",
        )
    return profile, trench_x, float(arc_length)


def query_gwb(
    coords: np.ndarray,
    model: Path,
    name: str,
    executable: Path,
    output_dir: Path,
    height: float = 1.6,
    length_scale_m: float = 1.0e6,
) -> dict[str, np.ndarray]:
    """Query the standalone GWB executable at nondimensional UW3 positions.

    For the two-dimensional Cartesian cookbook, GWB input points use (x,0,d)
    in metres. The d coordinate is depth measured downward from the top.
    The twelve returned columns are checked against the archived GWB v1.1.0
    output contract; a different column layout is rejected explicitly.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise RuntimeError("Expected two-dimensional UW3 coordinates")
    if not np.isfinite(coords).all():
        raise ValueError("Non-finite UW3 coordinates")
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("GWB query name must be a simple file stem")
    if not len(coords):
        raise ValueError("GWB queries require at least one coordinate")

    points = np.column_stack(
        (
            coords[:, 0] * length_scale_m,
            np.zeros(len(coords)),
            (height - coords[:, 1]) * length_scale_m,
        )
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = output_dir / f"{name}_query.dat"
    result_path = output_dir / f"{name}_result.dat"

    with query_path.open("w", encoding="utf-8") as handle:
        handle.write("# dim = 2\n# compositions = 5\n")
        np.savetxt(handle, points, fmt="%.10f")

    with result_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            [str(executable), str(model), str(query_path)],
            stdout=handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    (output_dir / f"{name}_stderr.txt").write_text(
        process.stderr, encoding="utf-8"
    )
    if process.returncode != 0:
        raise RuntimeError(f"GWB query failed: {name}\n{process.stderr}")

    with result_path.open(encoding="utf-8") as handle:
        headers = [
            line.lstrip("#").split()
            for line in handle
            if line.startswith("#")
        ]
    if list(COLUMNS) not in headers:
        raise RuntimeError(f"Unexpected GWB fields for {name}: {headers}")

    values = np.loadtxt(result_path, comments="#", ndmin=2)
    if values.shape != (len(points), len(COLUMNS)):
        raise RuntimeError(
            f"Unexpected GWB result dimensions: {values.shape}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite GWB output")
    if not np.allclose(values[:, :3], points, rtol=0, atol=10.0):
        raise RuntimeError("GWB query coordinates were not preserved")

    return {key: values[:, index] for index, key in enumerate(COLUMNS)}


def assign_materials(fields: dict[str, np.ndarray]) -> np.ndarray:
    """Map the verified GWB v1.1.0 2D composition columns to four labels.

    The 2D gwb-dat query format places the intended material compositions in
    columns c2, c3, c4 for IDs 1, 2, 3. This is not a general GWB API rule.
    Raise if compositions overlap or any of the four materials is absent.
    """
    masks = [fields[name] > 0.5 for name in ("c2", "c3", "c4")]
    occupied = sum(mask.astype(int) for mask in masks)
    if np.any(occupied > 1):
        raise RuntimeError("Overlapping geological compositions")
    labels = np.zeros(len(masks[0]), dtype=np.int32)
    for index, mask in enumerate(masks, 1):
        labels[mask] = index
    if any(not np.any(labels == index) for index in range(4)):
        raise RuntimeError("At least one geological material is missing")
    return labels


def assign_interface(
    coords: np.ndarray,
    profile: np.ndarray,
    height: float = 1.6,
    width_km: float = 40.0,
    max_depth_km: float = 200.0,
) -> np.ndarray:
    """Tag the prescribed finite-width slab-top weak band at swarm particles.

    The distance is to polyline sample vertices, as in the archived workflow.
    With a nominal 0.25 km profile spacing, its approximation is much finer
    than the 40 km band. This is not a discrete fault or contact condition.
    """
    coords = np.asarray(coords, dtype=float)
    locations = np.column_stack(
        (coords[:, 0] * 1000.0, (height - coords[:, 1]) * 1000.0)
    )
    distance, nearest = cKDTree(profile).query(locations)
    weak = (
        (distance <= width_km / 2.0)
        & (locations[:, 1] >= 0.0)
        & (locations[:, 1] <= max_depth_km)
        & (profile[nearest, 1] <= max_depth_km)
    )
    if int(weak.sum()) < 10:
        raise RuntimeError("Insufficient interface particles")
    return weak.astype(np.int32)


def boundary_check(
    coords: np.ndarray,
    velocity: np.ndarray,
    width: float = 8.0,
    height: float = 1.6,
) -> dict[str, float]:
    """Measure normal flow through side/top boundaries and bottom speed."""
    left = np.isclose(coords[:, 0], 0)
    right = np.isclose(coords[:, 0], width)
    top = np.isclose(coords[:, 1], height)
    bottom = np.isclose(coords[:, 1], 0)
    for label, mask in (("left", left), ("right", right),
                        ("top", top), ("bottom", bottom)):
        if not np.any(mask):
            raise RuntimeError(f"No nodes found on {label} boundary")
    result = {
        "left_normal": float(np.abs(velocity[left, 0]).max()),
        "right_normal": float(np.abs(velocity[right, 0]).max()),
        "top_normal": float(np.abs(velocity[top, 1]).max()),
        "bottom_speed": float(np.linalg.norm(
            velocity[bottom], axis=1,
        ).max()),
    }
    if max(result.values()) > 1.0e-7:
        raise RuntimeError(f"Velocity boundary condition failed: {result}")
    return result


def strain_rms(
    coords: np.ndarray,
    velocity: np.ndarray,
    profile: np.ndarray,
    speed_scale_mps: float,
    fine_res: tuple[int, int] = (160, 80),
    width: float = 8.0,
    height: float = 1.6,
    length_scale_m: float = 1.0e6,
) -> dict[str, float | int]:
    """Compute an *interpolated nodal* strain proxy in the interface/collar.

    Uses finite differences of velocity at a structured Q2 node grid, not
    integration of UW3's finite-element strain rate. It is a comparative
    kinematic diagnostic, not a slip-rate or convergence measurement.
    """
    nx = 2 * fine_res[0]
    ny = 2 * fine_res[1]
    dx = width / nx
    dy = height / ny
    expected = (nx + 1) * (ny + 1)
    if len(coords) != expected:
        raise RuntimeError("Unexpected velocity node count for Q2 grid")
    ix = np.rint(coords[:, 0] / dx).astype(int)
    iy = np.rint(coords[:, 1] / dy).astype(int)
    reconstructed = np.column_stack((ix * dx, iy * dy))
    if not np.allclose(coords, reconstructed, atol=1.0e-8, rtol=0):
        raise RuntimeError("Velocity nodes do not form a regular Q2 grid")
    indices = iy * (nx + 1) + ix
    if np.unique(indices).size != expected:
        raise RuntimeError("Duplicate or missing velocity coordinates")
    grid = np.empty((ny + 1, nx + 1, 2))
    grid[iy, ix] = velocity * speed_scale_mps
    exx = np.gradient(grid[:, :, 0], dx * length_scale_m, axis=1)
    eyy = np.gradient(grid[:, :, 1], dy * length_scale_m, axis=0)
    exy = 0.5 * (
        np.gradient(grid[:, :, 0], dy * length_scale_m, axis=0)
        + np.gradient(grid[:, :, 1], dx * length_scale_m, axis=1)
    )
    strain = np.sqrt(0.5 * (exx**2 + eyy**2 + 2.0 * exy**2))
    nodal_positions = np.column_stack(
        (coords[:, 0] * 1000.0, (height - coords[:, 1]) * 1000.0)
    )
    distances, _ = cKDTree(profile).query(nodal_positions)
    depth = nodal_positions[:, 1]
    band = (distances <= 20.0) & (depth > 0) & (depth <= 200)
    collar = (
        (distances > 20.0)
        & (distances <= 80.0)
        & (depth > 0)
        & (depth <= 200)
    )
    if band.sum() < 10 or collar.sum() < 10:
        raise RuntimeError("Insufficient interface sampling for strain audit")
    nodal_strain = strain[iy, ix]

    def rms(values: np.ndarray) -> float:
        """Return the root mean square of selected nodal strain values."""
        return float(np.sqrt(np.mean(values**2)))

    return {
        "band_rms_per_s": rms(nodal_strain[band]),
        "collar_rms_per_s": rms(nodal_strain[collar]),
        "band_nodes": int(band.sum()),
        "collar_nodes": int(collar.sum()),
    }
