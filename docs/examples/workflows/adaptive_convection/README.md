# Adaptive-mesh convection — an `underworld3.workflows` example

A worked example that drives **adaptive-mesh thermal convection** with the
`underworld3.workflows` infrastructure (config → `@workflow_step` DAG →
`Run` directory), on the fixed mmpde mover base (PRs #259 + #264 + #266:
deformed-mesh point-location + monotone RBF metric bake + SPD floor).

It is the clean re-implementation of an earlier monolithic driver. Two
variants share one config base and one diagnostics module:

| File | What it is |
|------|------------|
| `config.py` | `AdaptiveConvectionConfig` + the no-fault DAG (`create_mesh`, `create_solvers`, `evolve`, `summarise_run`) |
| `simulate.py` | auto-CLI for the no-fault workflow (`cli_from_config`) |
| `fault_config.py` | `FaultConvectionConfig` + fault DAG (weak zone + anisotropic-tensor metric, **uniform base**) |
| `fault_simulate.py` | auto-CLI for the fault workflow |
| `diagnostics.py` | reusable: `mesh_quality` (folded / area-ratio / aspect), `nn_spacing_ratios` (BL & fault/bulk), `NusseltSurface`, `vrms`, `History` |
| `render.py` | PyVista T + mesh + streamlines (`--all` frames) |
| `compare.py` | matched-physical-time comparison of N runs (Nu, vrms, mesh quality) |

## The adaptation recipe

(see the `adaptive-meshing` skill and `project_mmpde_holes_real_root_cause`)

- **mover** — `smooth_mesh_interior(method="mmpde", step_frac=0.2,
  accel="cg", momentum=0.0, slip_surfaces=True)`. Variational,
  non-folding; **owns field transfer** (remaps T + SLCN history).
- **metric** — `metric_density_from_gradient(refinement=R,
  metric_choice="front-following")`, R≈5 grading from |∇T|.
- **Stokes** — isotropic Frank-Kamenetskii `η = exp(θ(1−T))` (θ=ln Δη),
  buoyancy `Ra·(T−T_cond)·r̂`, no-slip inner, free-slip outer (penalty),
  `snes_type=ksponly` (linear ⇒ one KSP).
- **advect** — `AdvDiffusionSLCN(theta=1, monotone_mode='clamp')`,
  `dt = estimate_dt(percentile=50)·dt_mult`.

Adapt every step is fine on the fixed base (folded=0 throughout); the
mesh-quality columns in `timeseries.csv` are the definitive health check
(NOT the render alone).

## Run

```bash
# no-fault baseline to steady state (or max_steps)
python simulate.py --output-dir ~/+Simulations/AdaptiveConvection/baseline \
    --rayleigh 1e6 --delta-eta 1e3 --cellsize 0.0417 \
    --resolution-ratio 5 --adapt-every 1 --dt-mult 4 --max-steps 80 --max-t 0.06

# render all frames
python render.py --tag baseline --sim-dir ~/+Simulations/AdaptiveConvection --all

# fault variant (uniform base, anisotropic-tensor metric)
python fault_simulate.py --output-dir ~/+Simulations/FaultConvection/fault \
    --rayleigh 1e6 --delta-eta 1e3 --resolution-ratio 5 --fault-anisotropy 8 \
    --adapt-every 1 --dt-mult 4 --max-steps 60 --max-t 0.06
```

From Python:

```python
import config as ac
from underworld3.workflows import WorkflowRunner
cfg = ac.AdaptiveConvectionConfig(output_dir="...", rayleigh=1e6, max_steps=80)
WorkflowRunner(ac, cfg).build("run_summary")   # resolves the whole DAG
```

Each run is an append-only, resumable `Run` directory: `manifest.yaml`
(config hash + `workflow_api`), `timeseries.csv` (per-step diagnostics),
`run.mesh.NNNNN.{h5,xdmf}` checkpoints, and `run_summary.yaml` once
steady.

## Validating an adaptive run — the resolved arbiter

Compare an adaptive run against a **resolved arbiter** (a uniform mesh
finer in BOTH space and time) at **matched physical time** (runs on
different meshes take different dt, so never compare by step number):

```bash
# arbiter: uniform, no adaptation, fine in time
python simulate.py --output-dir ~/+Simulations/AdaptiveConvection/arbiter \
    --cellsize 0.0208 --resolution-ratio 0 --dt-mult 1.5 --max-steps 250 --max-t 0.06

python compare.py --sim-dir ~/+Simulations/AdaptiveConvection \
    --runs baseline arbiter --labels "adaptive" "arbiter" --out compare.png
```

**Finding (Ra=1e6, Δη=1e3).** Three runs — adaptive R5 at `dt_mult=4`,
adaptive R5 at `dt_mult=1.5`, and a res48 uniform arbiter at `dt_mult=1.5`
— compared at matched physical time:

- **Mesh robustness:** folded=0 for *all three* across the whole
  evolution; adaptive area-ratio flat ~14. The fixed mover does not
  tangle under forced every-step adaptation.
- **Faithfulness:** the adaptive vrms(t)/Nu(t) track the resolved arbiter.
  The dominant discrepancy is **timestep accuracy, not adaptation** —
  `dt_mult=4` with backward-Euler (θ=1) is over-diffusive and *delays the
  convective onset in physical time*; dropping to `dt_mult=1.5` recovers
  most of the arbiter's trajectory (the residual gap is the still-larger
  adaptive dt and res24-vs-res48 spatial resolution). SLCN's unconditional
  *stability* does not buy transient *accuracy*.

So: large `dt_mult` is a sound speed/accuracy trade for reaching the
quasi-steady regime quickly, but for faithful transients use a smaller
`dt_mult` (or θ=0.5). See `compare_dt_faithfulness.png`.

## Fault variant — measured refinement (uniform base)

`fault_config.py` adds a static dipping weak fault with an anisotropic SPD
tensor metric (thin across the fault normal). On a **uniform base**, the
mmpde mover refines the fault to an equilibrium **fault/bulk
nearest-neighbour ratio ≈ 0.55 (~1.8× finer)** under live convection
(Ra=1e6, Rf=8, res24) — monotone descent from ~0.95, `n_fault` 44→96,
folded=0 throughout, convection unharmed (vrms→21). This is the
mover's *creation cap* from uniform — see
`fault_refine_res24_Rf8.png` and the zoomed mesh render
`TVfocus_*.png` (the finer corridor aligned along the fault).

### gmsh fault base (`fault_base_smin`) — resolved, maintained fault

Set `fault_base_smin > 0` to place small cells **along the fault trace at
construction** (via `Annulus(refine_lines=..., refine_size_min=...)`, a
gmsh Distance+Threshold field with the fault points embedded). The mover
then only has to *maintain* the refinement — and because the extra gmsh
nodes lift it off its budget cap, it actually compounds it. Measured
(Ra=1e6, Rf=8, res24, to t=0.06), all **folded=0, T bounded**:

| base | `fault_base_smin` | fault/bulk | finer | n_fault |
|------|-------------------|------------|-------|---------|
| uniform (mover only) | 0 | 0.55 | ~1.8× | 95 |
| gmsh factor 2 | cellsize/2 ≈ 0.0208 | 0.29 | ~3.4× | 210 |
| gmsh factor 3 | cellsize/3 ≈ 0.0139 | 0.19 | ~5.0× | 350 |

Convection stays vigorous (vrms→21–23) and stable at every level — no
blow-up. See `fault_refine_gmsh_compare.png` and the zoomed mesh renders
(`MESHfocus_*.png`: the fault corridor is obvious at f3).

**Keeping the refinement ON the fault.** The fault density must *not* be
multiplied by the |∇T| density: with `metric_combine="product"` the cold
surface boundary layer (ρ_T ~ R^d ~ 20–25) out-competes the fault near its
top and the refinement drifts *above* the fault, stripping the deep fault
of nodes (the mover walks the gmsh nodes off the trace). Use
`metric_combine="max"` (default) **and** a `fault_refine_amp` larger than
ρ_T (~25 at R=5) so the fault density wins along its whole length. Verify
with the fault overlay: `render.py --mesh-only --fault --focus-fault 0.32`
— the fine cells should straddle the red trace from tip to surface.

The fault pull and the surface thermal-BL pull still compete for the coarse
cells in the **wedge between them**. `fault_wedge=True` (default) gmsh-fills
that radial sliver (fault→surface) with nodes too, so both pulls have their
own budget and merge into one coherent fine wedge rather than colliding.

**One-sided fault (the cleanest control).** Even so, the *symmetric* metric
demands refinement on both flanks while the realized nodes drift to the
hanging-wall (upper) side. `fault_metric_side` and `fault_rheology_side`
(`both` | `upper` | `lower`) place the refinement and the weak zone on a
chosen side of the fault *plane* (smooth `tanh` gate on the **signed**
distance; `dfac` stores the signed value — the gaussians square it, so only
the gate sees the sign). The physically-faithful recipe is **both = upper**:
a one-sided hanging-wall damage zone with the mesh refined on the same side,
so refinement and weakening coincide (gmsh f3: fault/bulk ~0.19 (~5.3×),
folded=0, vrms 15.8). `metric_side=lower` instead pulls refinement onto the
footwall to counter the upward drift. Verify weak-zone vs refinement
coincidence by rendering the `eta_fac` influence field with the fault
overlay.

```bash
# uniform base (creation cap)
python fault_simulate.py --output-dir <dir> --fault-anisotropy 8 ...
# gmsh-refined base, factor 2 / 3
python fault_simulate.py --output-dir <dir> --fault-base-smin 0.0208 --fault-anisotropy 8 ...

python fault_refine_plot.py --sim-dir ~/+Simulations/FaultConvection \
    --runs wf_fault_res24_Rf8 wf_fault_gmsh_f2 wf_fault_gmsh_f3 \
    --labels "uniform" "gmsh f2" "gmsh f3" --out fault_refine_gmsh_compare.png
python render.py --run <fault run dir> --index 40 --mesh-only --focus -0.2 0.88 0.32
```
