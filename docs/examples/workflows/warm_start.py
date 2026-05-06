"""Recipe: warm-start a fresh convection run from an existing run.

This is a *recipe* — an example script that composes the run-directory
primitives (``Run.open``, ``Run.load_field``, ``Run.create``,
``Run.append_timeseries_row``) with the convection workflow's own
config and diagnostic helpers.  It is not part of the ``uw.workflow``
API surface — if 3+ apps end up writing nearly-identical versions, we
promote the pattern to API at that point.

Use cases
---------

* Bump ``T_degree`` (e.g. 3 → 5) on a converged run to refine the
  thermal field without redoing the long transient.
* Bump ``qdegree`` to test integration accuracy at fixed FE space.
* Branch ensembles from a single steady state (perturb the seeded T
  before extending).

How it works
------------

The kd-tree interpolation inside :meth:`MeshVariable.read_timestep`
makes cross-mesh / cross-degree projection cheap: read the source
run's last T checkpoint into the target's freshly-built T MeshVariable
and the FE-space mismatch is handled automatically.  The recipe then
solves Stokes once for an initial v field and persists everything as
step 0 of the target run, so the convection workflow's resume branch
picks up cleanly when the user calls ``evolve`` (or
``WorkflowRunner.build("run_summary")``).

Example
-------

>>> from warm_start import warm_start
>>> from underworld3.workflows import WorkflowRunner
>>> import convection_config as cc
>>> target_cfg = warm_start(
...     "output/test_Ra1e5",
...     "output/test_Ra1e5_T5",
...     T_degree=5,
... )
>>> WorkflowRunner(cc, target_cfg).build("run_summary")
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sympy

import convection_config as cc
import underworld3 as uw
from underworld3.workflows import Run


def warm_start(
    source_dir,
    target_dir,
    **target_overrides: Any,
) -> cc.ConvectionConfig:
    """Seed *target_dir* with the last T from *source_dir* and return its config.

    Parameters
    ----------
    source_dir : str | Path
        An existing run directory with at least one saved checkpoint.
    target_dir : str | Path
        Where the warm-started run should live.  Created if missing;
        if it already has a manifest with a different ``config_hash``,
        the convection workflow's normal ``restart_policy`` applies on
        the next ``evolve`` call.
    **target_overrides
        Identity-field overrides applied to the source's config snapshot
        when constructing the target ConvectionConfig (e.g.
        ``T_degree=5``, ``qdegree=5``, ``rayleigh=2e5``).

    Returns
    -------
    target_config : ConvectionConfig
        Ready to drive ``WorkflowRunner(cc, target_config).build("run_summary")``.
    """
    source = Run.open(source_dir)
    if source.manifest is None:
        raise ValueError(f"No manifest in {source_dir}")
    if not source.steps:
        raise ValueError(f"No saved checkpoints in {source_dir}")

    last = source.steps[-1]
    src_snap = dict(source.manifest.config_snapshot)

    # Build the target config — inherit identity from source, then apply
    # overrides.  Strip any snapshot fields the config no longer knows
    # about (older manifests can carry retired keys).
    valid_fields = set(cc.ConvectionConfig.model_fields.keys())
    target_kwargs = {k: v for k, v in src_snap.items() if k in valid_fields}
    target_kwargs["output_dir"] = str(target_dir)
    target_kwargs.update(target_overrides)
    target_config = cc.ConvectionConfig(**target_kwargs)

    # Build a fresh target mesh + MeshVariables at the target's degrees.
    target_mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(target_config.aspect_ratio, 1.0),
        cellSize=target_config.cellsize,
        regular=target_config.regular,
        qdegree=target_config.qdegree,
    )
    v = uw.discretisation.MeshVariable(
        "U", target_mesh, target_mesh.dim, degree=2,
    )
    p = uw.discretisation.MeshVariable("P", target_mesh, 1, degree=1)
    T = uw.discretisation.MeshVariable(
        "T", target_mesh, 1, degree=target_config.T_degree,
    )

    # Warm-start: kd-tree projection from source's last T into the
    # target's T (handles different mesh / different polynomial degree).
    source.load_field(T, last, data_name="T")

    # Initial Stokes solve to populate v consistently with the seeded T.
    stokes = uw.systems.Stokes(target_mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = target_config.viscosity
    stokes.tolerance = 1.0e-6
    stokes.penalty = 0.0
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")
    stokes.bodyforce = sympy.Matrix([0, target_config.rayleigh * T.sym[0]])
    stokes.solve(zero_init_guess=True)

    # Persist the seeded state as step 0 of the target.  The convection
    # workflow's evolve() will see saved_steps == [0] and resume from
    # this checkpoint when invoked next.
    target = Run.create(target_dir)
    target.write_manifest({
        "workflow": "rayleigh_benard",
        "config_hash": cc._config_hash(target_config),
        "config_snapshot": cc._identity_snapshot(target_config),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rayleigh": target_config.rayleigh,
        "aspect_ratio": target_config.aspect_ratio,
        "warm_start_source": str(Path(source_dir).resolve()),
        "warm_start_step": last,
    })
    diag = cc._compute_diagnostics(target_mesh, T, v, target_config)
    target.append_step(
        step=0, t=0.0, dt=0.0,
        mesh=target_mesh, mesh_vars=[v, T],
        diags=diag, fields=cc._TS_FIELDS,
    )

    src_T_degree = src_snap.get("T_degree", 3)
    src_qdegree = src_snap.get("qdegree", 3)
    uw.pprint(
        f"[warm_start] {source_dir} step={last} -> {target_dir}\n"
        f"  T_degree {src_T_degree} -> {target_config.T_degree}, "
        f"qdegree {src_qdegree} -> {target_config.qdegree}, "
        f"Ra {src_snap.get('rayleigh')} -> {target_config.rayleigh}",
        flush=True,
    )
    return target_config
