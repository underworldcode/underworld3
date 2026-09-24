"""
Benchmark: Compare projection-based vs current evaluation.

This script tests whether projection-based evaluation is a viable
simpler alternative to the current complex expression parsing approach.

Questions to answer:
1. Is projection solve cost acceptable vs current DMInterpolation?
2. Does code simplification justify any performance difference?
3. How does performance scale with mesh resolution?
"""

import numpy as np
import underworld3 as uw
import time
import sys


def run_benchmark(resolutions=None, n_points=1000, verbose=True):
    """
    Compare evaluation approaches across resolutions and expression types.

    Parameters
    ----------
    resolutions : list, optional
        List of mesh resolutions to test (cell sizes = 1/res)
    n_points : int
        Number of sample points for evaluation
    verbose : bool
        Print results as we go

    Returns
    -------
    list : All benchmark results
    """
    from underworld3.function.evaluate_prototype import evaluate_via_projection, benchmark_single

    if resolutions is None:
        resolutions = [16, 32, 64]

    results = []

    for res in resolutions:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Resolution: {res}x{res} (cellSize={1/res:.4f})")
            print(f"{'='*60}")

        # Create mesh
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=1.0/res,
            regular=True,
        )
        x, y = mesh.X

        # Create test field
        T = uw.discretisation.MeshVariable("T", mesh, num_components=1, degree=1)
        T.array[:, 0, 0] = np.sin(np.pi * T.coords[:, 0]) * np.sin(np.pi * T.coords[:, 1])

        n_nodes = mesh.data.shape[0]
        if verbose:
            print(f"Mesh nodes: {n_nodes}")

        # Sample points (away from boundaries)
        np.random.seed(42)
        coords = np.random.uniform(0.1, 0.9, (n_points, 2))

        # Test expressions of increasing complexity
        expressions = {
            'field': T.sym[0, 0],
            'derivative_x': T.sym.diff(x)[0, 0],
            'derivative_y': T.sym.diff(y)[0, 0],
            'field_squared': T.sym[0, 0] * T.sym[0, 0],
            'grad_product': T.sym.diff(x)[0, 0] * T.sym.diff(y)[0, 0],
        }

        for name, expr in expressions.items():
            try:
                result = benchmark_single(expr, coords, mesh, name=name)
                result['resolution'] = res
                result['n_nodes'] = n_nodes
                result['n_points'] = n_points
                results.append(result)

                if verbose:
                    tc = result['time_current'] * 1000
                    tp = result['time_projection'] * 1000
                    speedup = result['speedup']
                    err = result['max_error']

                    # Timing breakdown
                    breakdown = result['timing_breakdown']
                    solve_pct = 100 * breakdown['solve'] / sum(breakdown.values())
                    interp_pct = 100 * breakdown['interpolate'] / sum(breakdown.values())

                    status = "FASTER" if speedup > 1 else "SLOWER"
                    print(f"  {name:15s}: current={tc:7.2f}ms, proj={tp:7.2f}ms, "
                          f"{speedup:5.2f}x [{status}], err={err:.2e}")
                    print(f"                   (solve={solve_pct:.0f}%, interp={interp_pct:.0f}%)")

            except Exception as e:
                if verbose:
                    print(f"  {name:15s}: FAILED - {e}")

        # Cleanup
        del mesh, T

    return results


def print_summary(results):
    """Print summary table of results."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Expression':<20} {'Res':<6} {'Current':<12} {'Projection':<12} {'Speedup':<10} {'Error':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<20} {r['resolution']:<6} "
              f"{r['time_current']*1000:>8.2f}ms  {r['time_projection']*1000:>8.2f}ms  "
              f"{r['speedup']:>6.2f}x    {r['max_error']:.2e}")

    # Overall statistics
    speedups = [r['speedup'] for r in results]
    print(f"\n{'='*80}")
    print(f"Average speedup: {np.mean(speedups):.2f}x")
    print(f"Min speedup: {np.min(speedups):.2f}x (worst case)")
    print(f"Max speedup: {np.max(speedups):.2f}x (best case)")

    # Where does time go in projection?
    solve_times = [r['timing_breakdown']['solve'] for r in results]
    interp_times = [r['timing_breakdown']['interpolate'] for r in results]
    total_times = [sum(r['timing_breakdown'].values()) for r in results]

    avg_solve_pct = 100 * np.mean(solve_times) / np.mean(total_times)
    avg_interp_pct = 100 * np.mean(interp_times) / np.mean(total_times)

    print(f"\nProjection time breakdown (average):")
    print(f"  Solve: {avg_solve_pct:.1f}%")
    print(f"  Interpolate: {avg_interp_pct:.1f}%")

    # Accuracy
    errors = [r['max_error'] for r in results]
    print(f"\nMax error across all tests: {np.max(errors):.2e}")


def quick_test():
    """Quick sanity test."""
    print("Quick sanity test...")

    from underworld3.function.evaluate_prototype import evaluate_via_projection

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.1,
    )
    x, y = mesh.X

    T = uw.discretisation.MeshVariable("T", mesh, num_components=1, degree=1)
    T.array[:, 0, 0] = T.coords[:, 0]**2 + T.coords[:, 1]

    coords = np.array([[0.5, 0.5], [0.25, 0.75]])

    # Test field evaluation
    result_current = uw.function.evaluate(T.sym[0, 0], coords)
    result_proj, timing = evaluate_via_projection(T.sym[0, 0], coords, mesh, verbose=True)

    print(f"\nField evaluation:")
    print(f"  Current: {result_current.flatten()}")
    print(f"  Projection: {result_proj.flatten()}")
    print(f"  Error: {np.max(np.abs(result_current - result_proj)):.2e}")

    # Test derivative evaluation
    result_current = uw.function.evaluate(T.sym.diff(x), coords)
    result_proj, timing = evaluate_via_projection(T.sym.diff(x)[0, 0], coords, mesh, verbose=True)

    print(f"\nDerivative evaluation (dT/dx):")
    print(f"  Current: {result_current.flatten()}")
    print(f"  Projection: {result_proj.flatten()}")
    print(f"  Exact (2x): {2 * coords[:, 0]}")
    print(f"  Error: {np.max(np.abs(result_current - result_proj)):.2e}")

    print("\nQuick test PASSED!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        results = run_benchmark(resolutions=[16, 32, 64], n_points=1000)
        print_summary(results)
