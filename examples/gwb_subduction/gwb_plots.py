"""Headless plotting utilities for the GWB to UW3 subduction demonstration."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from gwb_core import MATERIAL_NAMES


def make_figures(
    particle_coords,
    labels,
    profile,
    solutions,
    scale,
    records,
    output_dir,
    width=8.0,
    height=1.6,
):
    """Render geology, velocity, and comparative nodal strain diagnostics.

    These plots use the same plotting and scaling conventions as the
    archived reference demonstration. The strain bars are based on the
    nodal finite-difference proxy, not integrated FE strain or fault slip.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Serif"

    colors = [
        "#a8adba",
        "#12a5b9",
        "#dd75b8",
        "#d93434",
    ]

    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(
        figsize=(12, 4.5)
    )

    ax.scatter(
        particle_coords[:, 0] * 1000.0,
        (height - particle_coords[:, 1]) * 1000.0,
        c=labels,
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        s=0.7,
        linewidths=0,
        rasterized=True,
    )

    ax.plot(
        profile[:, 0],
        profile[:, 1],
        color="black",
        linewidth=1.2,
    )

    ax.set(
        xlim=(0, width * 1000),
        ylim=(height * 1000, 0),
        xlabel="Horizontal distance (km)",
        ylabel="Depth (km)",
    )

    ax.legend(
        handles=[
            Patch(
                facecolor=colors[i],
                label=name,
            )
            for i, name in enumerate(MATERIAL_NAMES)
        ],
        loc="lower left",
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "gwb_four_material_geometry.png",
        dpi=200,
    )

    plt.close(fig)

    maximum = max(
        np.linalg.norm(
            value[1],
            axis=1,
        ).max() * scale
        for value in solutions.values()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.3),
        sharex=True,
        sharey=True,
    )

    image = None

    for ax, (
        name,
        (coords, velocity),
    ) in zip(axes, solutions.items()):

        image = ax.scatter(
            coords[:, 0] * 1000.0,
            (height - coords[:, 1]) * 1000.0,
            c=np.linalg.norm(
                velocity,
                axis=1,
            ) * scale,
            cmap="viridis",
            vmin=0,
            vmax=maximum,
            s=1.3,
            linewidths=0,
            rasterized=True,
        )

        ax.set_title(
            "Coupled"
            if name == "coupled"
            else "Weak interface"
        )

        ax.set_xlim(
            0,
            width * 1000,
        )

        ax.set_ylim(
            height * 1000,
            0,
        )

        ax.set_xlabel(
            "Horizontal distance (km)"
        )

    axes[0].set_ylabel(
        "Depth (km)"
    )

    fig.colorbar(
        image,
        ax=axes,
        label="Speed (cm/year)",
        shrink=0.82,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.87,
        bottom=0.15,
        top=0.9,
        wspace=0.12,
    )

    fig.savefig(
        output_dir / "stokes_velocity_comparison.png",
        dpi=200,
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(7.5, 4.0)
    )

    x = np.arange(2)
    width = 0.32

    for j, key in enumerate(
        ("coupled", "weak_interface")
    ):
        record = records[key]["strain"]

        ax.bar(
            x + (j - 0.5) * width,
            [
                record["band_rms_per_s"] * 1e14,
                record["collar_rms_per_s"] * 1e14,
            ],
            width=width,
            label=(
                "Coupled"
                if j == 0
                else "Weak interface"
            ),
        )

    ax.set_xticks(
        x,
        [
            "Interface band",
            "Surrounding collar",
        ],
    )

    ax.set_ylabel(
        "RMS strain rate (10⁻¹⁴ s⁻¹)"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "interface_strain_comparison.png",
        dpi=200,
    )

    plt.close(fig)
