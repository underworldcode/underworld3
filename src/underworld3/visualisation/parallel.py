"""
Parallel-safe visualization utilities for Underworld3.

These functions enable matplotlib plotting in parallel notebooks by using
the asymmetric global_evaluate pattern where rank 0 requests data and 
other ranks participate with empty arrays.
"""

import numpy as np
import underworld3 as uw


def _evaluable(field):
    """The form ``global_evaluate`` takes: a MeshVariable is evaluated through
    its symbolic form, anything else (a sympy expression) as given."""
    if isinstance(field, uw.discretisation.MeshVariable):
        return field.sym
    return field


def _points_on_this_rank(sample_points, viz_rank):
    """The points this rank asks ``global_evaluate`` for: the sample points on
    ``viz_rank``, and an empty array of the same dimension on every other rank.

    Only ``viz_rank``'s points are read. Their dimension is broadcast from it, so
    the other ranks may pass anything, ``None`` included. Points on ``viz_rank``
    that are not a finite ``(n, dim)`` array are refused on every rank together,
    before the collective, instead of leaving the other ranks waiting in it."""
    points, dim = None, None
    if uw.mpi.rank == viz_rank:
        try:
            points = np.asarray(sample_points, dtype=float)
        except (TypeError, ValueError):
            # Not convertible to numbers: refused below, on every rank.
            points = None
        # A non-finite coordinate cannot be located, and global_evaluate does not
        # refuse it the same way on every rank.
        if points is not None and points.ndim == 2 and np.isfinite(points).all():
            dim = points.shape[1]
    dim = uw.mpi.comm.bcast(dim, root=viz_rank)
    if dim is None:
        raise ValueError(
            f"the sample points on the plotting rank (viz_rank={viz_rank}) must be "
            f"a finite array of shape (n, dim)")
    if uw.mpi.rank == viz_rank:
        return points
    return np.zeros((0, dim))


def parallel_line_plot(
    field, sample_points, title="Field Profile", xlabel="Position", ylabel="Field Value", viz_rank=0
):
    """
    Create a 1D line plot in parallel notebooks.

    Args:
        field: a MeshVariable or a sympy expression to plot
        sample_points: numpy array of coordinates to sample
        title: Plot title
        xlabel, ylabel: Axis labels
        viz_rank: Which rank creates the visualization (default: 0)

    Every rank calls this; only ``viz_rank``'s ``sample_points`` are used.
    """
    points = _points_on_this_rank(sample_points, viz_rank)
    data = uw.function.global_evaluate(_evaluable(field), points)
    if uw.mpi.rank == viz_rank:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(points[:, 0], data.flatten())
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.show()


def parallel_scatter_plot(
    field_x, field_y, sample_points, labels=("Field X", "Field Y"), title="Scatter Plot", viz_rank=0
):
    """
    Create scatter plot of two fields at sample points.

    Args:
        field_x, field_y: MeshVariables or sympy expressions to plot
        sample_points: numpy array of coordinates to sample
        labels: Tuple of (xlabel, ylabel)
        title: Plot title
        viz_rank: Which rank creates the visualization

    Every rank calls this; only ``viz_rank``'s ``sample_points`` are used.
    """
    points = _points_on_this_rank(sample_points, viz_rank)
    data_x = uw.function.global_evaluate(_evaluable(field_x), points)
    data_y = uw.function.global_evaluate(_evaluable(field_y), points)
    if uw.mpi.rank == viz_rank:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.scatter(data_x.flatten(), data_y.flatten(), alpha=0.6)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()


def parallel_profile_comparison(
    fields, sample_points, labels=None, title="Field Profiles", viz_rank=0
):
    """
    Plot multiple field profiles on the same axes for comparison.

    Args:
        fields: List of MeshVariables or sympy expressions
        sample_points: numpy array of coordinates to sample
        labels: List of field names (optional)
        title: Plot title
        viz_rank: Which rank creates the visualization

    Every rank calls this; only ``viz_rank``'s ``sample_points`` are used.
    """
    if labels is None:
        labels = [f"Field {i}" for i in range(len(fields))]

    points = _points_on_this_rank(sample_points, viz_rank)
    field_data = [uw.function.global_evaluate(_evaluable(field), points).flatten()
                  for field in fields]
    if uw.mpi.rank == viz_rank:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        for i, (data, label) in enumerate(zip(field_data, labels)):
            plt.plot(points[:, 0], data, label=label, marker="o", markersize=3)

        plt.xlabel("Position")
        plt.ylabel("Field Value")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def parallel_custom_plot(plot_function, field_data_specs, viz_rank=0, **plot_kwargs):
    """
    Generic helper for parallel plotting with custom plot functions.

    Args:
        plot_function: Function that creates the plot, receives gathered data as dict
        field_data_specs: List of (field, points, name) tuples
        viz_rank: Which rank creates the visualization
        **plot_kwargs: Additional keyword arguments passed to plot_function

    Every rank calls this with the same fields and names; only ``viz_rank``'s
    points are used.

    Example:
        def my_custom_plot(velocity_data, velocity_points, temperature_data, temperature_points):
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(velocity_points[:, 1], velocity_data.flatten())
            ax2.plot(temperature_points[:, 0], temperature_data.flatten())
            plt.show()

        parallel_custom_plot(
            my_custom_plot,
            [(velocity[1], y_line, "velocity"),
             (temperature, x_line, "temperature")]
        )
    """
    plot_data = {}
    for field, sample_points, name in field_data_specs:
        points = _points_on_this_rank(sample_points, viz_rank)
        plot_data[f"{name}_data"] = uw.function.global_evaluate(_evaluable(field), points)
        plot_data[f"{name}_points"] = points
    if uw.mpi.rank == viz_rank:
        plot_function(**plot_data, **plot_kwargs)


# Convenience functions for common sampling patterns


def create_line_sample(start, end, num_points=50):
    """Create evenly spaced points along a line."""
    start, end = np.array(start), np.array(end)
    t = np.linspace(0, 1, num_points).reshape(-1, 1)
    return start + t * (end - start)


def create_vertical_line(x, y_range=(0, 1), num_points=50):
    """Create vertical line at fixed x coordinate."""
    y_vals = np.linspace(y_range[0], y_range[1], num_points)
    return np.column_stack([np.full(num_points, x), y_vals])


def create_horizontal_line(y, x_range=(0, 1), num_points=50):
    """Create horizontal line at fixed y coordinate."""
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    return np.column_stack([x_vals, np.full(num_points, y)])


def create_diagonal_sample(domain_bounds=((0, 1), (0, 1)), num_points=50):
    """Create diagonal line sampling across domain."""
    x_bounds, y_bounds = domain_bounds
    x_vals = np.linspace(x_bounds[0], x_bounds[1], num_points)
    y_vals = np.linspace(y_bounds[0], y_bounds[1], num_points)
    return np.column_stack([x_vals, y_vals])
