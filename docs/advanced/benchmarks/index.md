---
title: "Benchmarks"
---

# Solver Benchmarks

Validation benchmarks comparing Underworld3 solvers against
closed-form analytical solutions.  Each benchmark has three pieces:

* a `bench_*.py` runner that solves the problem and writes a
  self-contained `.npz` log to `output/benchmarks/`,
* `plot_benchmarks.py` that reads the logs and produces consistent-style
  figures in `docs/advanced/figures/`,
* a Markdown page (this section) that documents the governing
  equation, the closed-form solution, the test setup, and the result.

The runner and the plotter are deliberately decoupled: each runner
saves the per-step trace, both BDF orders, the analytical reference,
and the parameter dict in one self-contained file; re-running the
plot script to tweak style does not re-run the (slow) simulation.
A separate `bench_convergence.py` runs each case at a sweep of
timestep sizes (and both BDF orders) and saves all per-run traces so
the convergence figure and any per-(order, dt) replot are equally
reproducible from saved data.

## Workflow

```bash
# Run a single per-case benchmark (both BDF orders, ~3-6 min each)
pixi run -e amr-dev python docs/advanced/benchmarks/bench_ve_harmonic.py
pixi run -e amr-dev python docs/advanced/benchmarks/bench_ve_square.py
pixi run -e amr-dev python docs/advanced/benchmarks/bench_vep_square.py

# Run the convergence sweep (~30 min, all dts × both orders × all cases)
pixi run -e amr-dev python docs/advanced/benchmarks/bench_convergence.py

# Replot from saved data — does NOT re-run simulations
pixi run -e amr-dev python docs/advanced/benchmarks/plot_benchmarks.py

# Verify the on-disk data is complete (used as a sanity check before
# claiming a benchmark suite is "done")
pixi run -e amr-dev python docs/advanced/benchmarks/check_saved_data.py
```

## Cases

```{toctree}
:maxdepth: 1

ve-harmonic
ve-square
vep-square
vardt-square
```

| Case | Driving | Closed form | What it tests |
|---|---|---|---|
| `ve-harmonic` | $V_{\mathrm{top}} = V_0\cos(\omega t + \varphi)$ | $A_\infty\cos\omega t$ | amplitude attenuation, phase lag, peak-start IC |
| `ve-square` | square-wave $V_{\mathrm{top}}$ | piecewise exponential | BDF history at BC discontinuities |
| `vep-square` | square-wave with yield | clipped Maxwell square-wave | Min-mode plasticity, projection-snapshot fix |
| `vardt-square` | square-wave + reduced $\Delta t$ near flips | same as `ve-square` / `vep-square` | snapshot machinery under variable timestep |
