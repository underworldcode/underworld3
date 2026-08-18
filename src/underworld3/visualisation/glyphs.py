"""Principal-stress glyphs and stress trajectories.

The stress tensor is sampled at seed points, the way velocity arrows
sample the velocity — not drawn everywhere. Each seed carries one bar
per principal axis: bar length proportional to the principal-value
magnitude, colour by sign (blue compressive, red tensile, matching the
RdBu_r field convention). In 2-D that is a cross; in 3-D, three
orthogonal bars. Stress trajectories integrate the principal
*direction* field into evenly spaced curves (2-D only; the 3-D
analogue is a trajectory surface, which we do not attempt — draw
glyphs on section planes instead).

Sign conventions
----------------
Principal values are of the stress tensor as supplied. For the full
stress :math:`\\sigma = \\tau - p I` in an incompressible model, the
pressure datum is a gauge: the compressive/tensile split (and so the
bar colours) is relative to that datum, while principal *directions*
and the ordering of principal values are gauge-invariant. State the
gauge in the caption (e.g. "demeaned pressure"), or add the
lithostatic reference before plotting if absolute signs matter.
"""


def tensor_fn_to_pv_points(pv_mesh, uw_fn):
    """Evaluate an Underworld tensor function at PyVista mesh points.

    Parameters
    ----------
    pv_mesh : pyvista.DataSet
        PyVista mesh or point cloud to evaluate at.
    uw_fn : sympy.Matrix
        Square (``dim x dim``) Underworld tensor function, e.g. a
        stress built from recovered component variables.

    Returns
    -------
    numpy.ndarray
        Tensor values at mesh points, shape ``(n_points, dim, dim)``.
        Units are stripped; the units string is stored as
        ``pv_mesh._last_tensor_units``.
    """
    import numpy as np
    import underworld3 as uw

    dim = uw_fn.shape[0]
    if uw_fn.shape != (dim, dim):
        raise ValueError(f"Expected a square tensor, got shape {uw_fn.shape}")

    if hasattr(pv_mesh, "_coord_array"):
        coords = pv_mesh._coord_array[:, 0:dim]
    else:
        coords = pv_mesh.points[:, 0:dim]

    tensor_values = uw.function.evaluate(uw_fn, coords)

    tensor_units = None
    if hasattr(tensor_values, "units") and tensor_values.units is not None:
        tensor_units = str(tensor_values.units)
    pv_mesh._last_tensor_units = tensor_units

    if hasattr(tensor_values, "magnitude"):
        tensor_values = tensor_values.magnitude

    return np.asarray(tensor_values).reshape(-1, dim, dim)


def principal_stress_glyphs(coords, stress, scale):
    """Principal-axis bar glyphs for symmetric tensors at seed points.

    Each seed point contributes ``dim`` line segments, one per
    principal axis, centred on the seed. The half-length of the bar
    for principal value :math:`\\lambda_k` along unit eigenvector
    :math:`\\hat{e}_k` is :math:`\\mathrm{scale} \\cdot |\\lambda_k|`.

    Parameters
    ----------
    coords : numpy.ndarray
        Seed coordinates, shape ``(n, 2)`` or ``(n, 3)``.
    stress : numpy.ndarray
        Symmetric tensors at the seeds, shape ``(n, dim, dim)``.
        The input is symmetrised (``eigh`` reads only one triangle,
        so a small asymmetry from recovered components would
        otherwise be ignored silently).
    scale : float
        Bar half-length per unit principal value, in mesh
        coordinates. A good default is
        ``0.45 * seed_spacing / |lambda|_max``.

    Returns
    -------
    pyvista.PolyData
        One line cell per bar, with cell array ``"tensile"``
        (1.0 where :math:`\\lambda_k \\ge 0`, else 0.0). Split with
        ``glyphs.threshold(0.5, scalars="tensile")`` to colour the
        two signs separately.
    """
    import numpy as np
    import pyvista as pv

    coords = np.asarray(coords, dtype=float)
    n, dim = coords.shape
    stress = 0.5 * (stress + np.transpose(stress, (0, 2, 1)))

    lam, vec = np.linalg.eigh(stress)

    segments = []
    tensile = []
    for k in range(dim):
        half = (scale * np.abs(lam[:, k]))[:, None] * vec[:, :, k]
        segments.append(np.stack([coords - half, coords + half], axis=1))
        tensile.append(lam[:, k] >= 0)
    segments = np.vstack(segments).reshape(-1, dim)
    tensile = np.concatenate(tensile).astype(float)

    if dim == 2:
        segments = np.column_stack([segments, np.zeros(len(segments))])

    n_bars = n * dim
    lines = np.column_stack(
        [
            np.full(n_bars, 2),
            np.arange(0, 2 * n_bars, 2),
            np.arange(1, 2 * n_bars, 2),
        ]
    ).ravel()
    glyphs = pv.PolyData(segments, lines=lines)
    glyphs.cell_data["tensile"] = tensile
    return glyphs


def direction_trajectories(
    direction_at, seeds, inside, step, separation, max_steps=2000
):
    """Evenly spaced trajectories of a direction field (2-D).

    A principal-stress direction is defined only modulo 180 degrees,
    so ordinary streamline tools cannot integrate it: the integrator
    here carries orientation continuity, sign-aligning each evaluated
    eigenvector with the previous heading before the RK2 midpoint
    step. Line placement follows Jobard & Lehmann: an occupancy grid
    at ``separation`` keeps a new seed at least one spacing from
    existing lines, while a running line stops only when it enters a
    cell another line has actually traversed — two different tests,
    because using the wide corridor for both chops lines into stubs.

    Parameters
    ----------
    direction_at : callable
        ``direction_at(p)`` with ``p`` shape ``(2,)`` returning a unit
        direction vector, or ``None`` at isotropic points and outside
        the field. The returned sign is arbitrary (mod-180 field).
    seeds : numpy.ndarray
        Candidate seed points, shape ``(m, 2)``. Offer more than you
        expect to be used; the occupancy grid thins them.
    inside : callable
        ``inside(p) -> bool`` gating the domain (bounding box, keep
        clear of fault zones, ...).
    step : float
        RK2 step length in mesh coordinates.
    separation : float
        Target spacing between neighbouring trajectories.
    max_steps : int, optional
        Cap on integration steps per direction from a seed.

    Returns
    -------
    list of numpy.ndarray
        Polylines, each of shape ``(n_points, 2)``. Lines shorter
        than about four separations are dropped. Render with
        :func:`trajectories_to_pv_lines`.
    """
    import numpy as np

    occupied = set()  # cells a line traversed: stops a converging line
    corridor = set()  # widened by one ring: blocks new seeds only

    def cell(p):
        return (
            int(np.floor(p[0] / separation)),
            int(np.floor(p[1] / separation)),
        )

    lines = []
    for seed in seeds:
        if cell(seed) in corridor:
            continue
        u0 = direction_at(np.asarray(seed, dtype=float))
        if u0 is None:
            continue
        branches = []
        for sense in (+1.0, -1.0):
            p = np.array(seed, dtype=float)
            u_prev = sense * u0
            path = [p.copy()]
            for _ in range(max_steps):
                u1 = direction_at(p)
                if u1 is None:
                    break
                if np.dot(u1, u_prev) < 0:
                    u1 = -u1
                p_mid = p + 0.5 * step * u1
                if not inside(p_mid):
                    break
                u2 = direction_at(p_mid)
                if u2 is None:
                    break
                if np.dot(u2, u1) < 0:
                    u2 = -u2
                p = p + step * u2
                if not inside(p) or cell(p) in occupied:
                    break
                u_prev = u2
                path.append(p.copy())
            branches.append(np.array(path))
        line = np.vstack([branches[0][::-1], branches[1][1:]])
        if (len(line) - 1) * step >= 4.0 * separation:
            lines.append(line)
            # A line claims its cells only AFTER integrating, so it
            # never blocks itself; the ring keeps future seeds away.
            for q in line:
                ci, cj = cell(q)
                occupied.add((ci, cj))
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        corridor.add((ci + di, cj + dj))
    return lines


def trajectories_to_pv_lines(lines):
    """Bundle polylines from :func:`direction_trajectories` into PolyData.

    Parameters
    ----------
    lines : list of numpy.ndarray
        Polylines of shape ``(n_points, 2)`` or ``(n_points, 3)``.

    Returns
    -------
    pyvista.PolyData or None
        One polyline cell per input line (``None`` for an empty list).
    """
    import numpy as np
    import pyvista as pv

    points, cells, offset = [], [], 0
    for line in lines:
        if line.shape[1] == 2:
            line = np.column_stack([line, np.zeros(len(line))])
        points.append(line)
        cells.extend([len(line), *range(offset, offset + len(line))])
        offset += len(line)
    if not points:
        return None
    return pv.PolyData(np.vstack(points), lines=np.asarray(cells))


def plot_stress_glyphs(
    mesh,
    stress,
    seeds=None,
    num_seeds=24,
    scale=None,
    compressive_colour="#2166ac",
    tensile_colour="#b2182b",
    line_width=2.5,
    show_edges=False,
    save_png=False,
    dir_fname="",
    title="",
    cpos="xy",
    window_size=(750, 750),
    show=True,
):
    """Plot principal-stress crosses sampled at seed points.

    Parameters
    ----------
    mesh : uw.discretisation.Mesh
        The mesh the stress lives on.
    stress : sympy.Matrix
        Square (``dim x dim``) stress function. Use recovered
        (projected) component variables rather than raw solver
        derivative expressions — see the visualisation guide.
    seeds : numpy.ndarray, optional
        Seed coordinates, shape ``(n, dim)``. Default is a regular
        grid over the mesh bounding box, filtered to points inside
        the mesh (so an annulus hole seeds nothing). In 3-D, prefer
        passing seeds on one or two section planes; a filled volume
        grid of three-bar glyphs is hard to read.
    num_seeds : int, optional
        Grid resolution across the longest box edge for the default
        seeding.
    scale : float, optional
        Bar half-length per unit principal value. Default scales the
        largest bar to 0.45 of the seed spacing.
    compressive_colour, tensile_colour : str, optional
        Bar colours for negative / non-negative principal values.
    line_width : float, optional
        Bar line width in pixels.
    show_edges : bool, optional
        Draw the mesh edge skeleton faintly beneath the glyphs.
    save_png : bool, optional
        Save a screenshot to ``dir_fname``.
    dir_fname : str, optional
        Path for the screenshot when ``save_png`` is True.
    title : str, optional
        Text placed on the figure.
    cpos : str or list, optional
        PyVista camera position (``"xy"`` for 2-D sections).
    window_size : tuple of int, optional
        Render window size in pixels.
    show : bool, optional
        Call ``show()`` before returning. Pass ``False`` to add
        overlays to the returned plotter and screenshot it yourself —
        actors added after ``show()`` are ignored by a finalized
        scene.

    Returns
    -------
    pyvista.Plotter
        The plotter. With ``show=False`` it is still open: add
        overlays, then ``show()`` or ``screenshot()``.
    """
    import numpy as np
    import pyvista as pv

    from .visualisation import mesh_to_pv_mesh

    dim = mesh.dim
    pvmesh = mesh_to_pv_mesh(mesh)

    if seeds is None:
        bounds = np.asarray(pvmesh.bounds).reshape(3, 2)
        extents = bounds[:, 1] - bounds[:, 0]
        spacing = extents[:dim].max() / num_seeds
        axes = [
            np.arange(bounds[k, 0] + 0.5 * spacing, bounds[k, 1], spacing)
            for k in range(dim)
        ]
        grids = np.meshgrid(*axes, indexing="ij")
        seeds = np.column_stack([g.ravel() for g in grids])
        # Keep only seeds inside the mesh: an interior point is its own
        # closest point in the containing cell, an exterior point is not.
        probe = seeds
        if dim == 2:
            probe = np.column_stack([seeds, np.zeros(len(seeds))])
        _cells, closest = pvmesh.find_closest_cell(
            probe, return_closest_point=True
        )
        distance = np.linalg.norm(closest - probe, axis=1)
        seeds = seeds[distance < 1.0e-6 * extents.max()]
    else:
        seeds = np.asarray(seeds, dtype=float)
        # The auto-scale needs the ACTUAL seed spacing — user seeds
        # owe nothing to num_seeds. Mean nearest-neighbour distance,
        # subsampled so the pairwise matrix stays small.
        sample = seeds
        if len(sample) > 2048:
            sample = sample[:: len(sample) // 2048 + 1]
        offsets = sample[:, None, :] - sample[None, :, :]
        distance2 = np.sum(offsets**2, axis=2)
        np.fill_diagonal(distance2, np.inf)
        spacing = float(np.sqrt(distance2.min(axis=1)).mean())

    cloud = pv.PolyData(
        np.column_stack([seeds, np.zeros(len(seeds))])
        if dim == 2
        else seeds
    )
    stress_values = tensor_fn_to_pv_points(cloud, stress)

    if scale is None:
        lam = np.linalg.eigvalsh(
            0.5 * (stress_values + np.transpose(stress_values, (0, 2, 1)))
        )
        scale = 0.45 * spacing / np.abs(lam).max()

    glyphs = principal_stress_glyphs(seeds, stress_values, scale)

    pl = pv.Plotter(window_size=window_size)
    pl.set_background("white")
    if show_edges:
        pl.add_mesh(
            pvmesh.extract_all_edges(),
            color="#d9d9d9",
            line_width=0.5,
            lighting=False,
        )
    for threshold_value, invert, colour in (
        (0.5, True, compressive_colour),
        (0.5, False, tensile_colour),
    ):
        part = glyphs.threshold(
            threshold_value, scalars="tensile", invert=invert
        )
        if part.n_cells:
            pl.add_mesh(
                part, color=colour, line_width=line_width, lighting=False
            )
    if len(title):
        pl.add_text(title, font_size=11, color="black")

    pl.camera_position = cpos
    if show:
        pl.show()
    if save_png:
        pl.screenshot(dir_fname)

    return pl
