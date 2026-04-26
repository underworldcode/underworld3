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

The runner and the plotter are deliberately decoupled: re-running the
plot to tweak style does not re-run the (slow) simulation.

## Cases

```{toctree}
:maxdepth: 1

ve-harmonic
ve-square
vep-square
```

| Case | Driving | Closed form | What it tests |
|---|---|---|---|
| `ve-harmonic` | $V_{\mathrm{top}} = V_0\sin\omega t$ | full Maxwell oscillatory | amplitude attenuation, phase lag |
| `ve-square` | square-wave $V_{\mathrm{top}}$ | piecewise exponential | BDF-2 history at BC discontinuities |
| `vep-square` | square-wave with yield | clipped Maxwell square-wave | Min-mode plasticity, projection-snapshot fix |
