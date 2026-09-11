"""
Benchmark the vectorized IndexSwarmVariable._update_proxy_variables
against the original loop-based reference.

Uses a larger mesh to make the speedup measurable.
Run with: python tests/benchmark_index_swarm_vectorized.py
"""
import numpy as np
import underworld3 as uw
import time

from test_0115_index_swarm_vectorized import (
    _reference_update_type0,
    _reference_update_type1,
)


def benchmark(update_type=1, mesh_res=(40, 40), fill_param=5, nnn=10, radius=0.1):
    uw.reset_default_model()
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=mesh_res,
        minCoords=(0.0, 0.0),
        maxCoords=(10.0, 10.0),
    )
    swarm = uw.swarm.Swarm(mesh)
    material = uw.swarm.IndexSwarmVariable(
        "M_bench", swarm, indices=4, proxy_degree=1, proxy_continuous=True,
        update_type=update_type, npoints=nnn, radius=radius,
    )
    swarm.populate(fill_param=fill_param)
    with swarm.access():
        # 4-layer material assignment
        y = swarm.data[:, 1]
        material.data[:, 0] = np.where(
            y < 2.5, 0,
            np.where(y < 5.0, 1,
                     np.where(y < 7.5, 2, 3))
        ).astype(int)

    print(f"\n=== update_type={update_type}, mesh={mesh_res}, fill={fill_param}, "
          f"nnn={nnn}, radius={radius} ===")
    print(f"  Particles (local): {swarm.local_size}")
    print(f"  Mesh nodes: {material._meshLevelSetVars[0].data.shape[0]}")
    print(f"  Materials: {material.indices}")

    # Benchmark vectorized (in-class)
    material._proxy_stale = True
    t0 = time.perf_counter()
    material._update_proxy_if_stale()
    t_vec = time.perf_counter() - t0
    print(f"  Vectorized: {t_vec*1000:.2f} ms")

    # Read vectorized result
    vec = np.zeros((material._meshLevelSetVars[0].data.shape[0], material.indices))
    for ii in range(material.indices):
        vec[:, ii] = material._meshLevelSetVars[ii].data[:, 0]

    # Benchmark reference (original loop)
    t0 = time.perf_counter()
    if update_type == 0:
        ref = _reference_update_type0(swarm, material, nnn, radius)
    else:
        ref = _reference_update_type1(swarm, material, nnn, radius)
    t_ref = time.perf_counter() - t0
    print(f"  Reference:  {t_ref*1000:.2f} ms")

    speedup = t_ref / t_vec if t_vec > 0 else float('inf')
    print(f"  Speedup:    {speedup:.1f}x")

    # Verify correctness
    max_diff = np.max(np.abs(ref - vec))
    print(f"  Max abs diff: {max_diff:.2e}")
    assert np.allclose(ref, vec, atol=1e-10, rtol=1e-7), (
        f"Vectorized output does not match reference! Max diff: {max_diff}"
    )
    print(f"  ✓ Output matches reference")
    return speedup


if __name__ == "__main__":
    print("Benchmarking vectorized IndexSwarmVariable._update_proxy_variables")
    print("=" * 70)

    speedups = []
    for res in [(10, 10), (20, 20), (40, 40), (60, 60)]:
        for ut in [0, 1]:
            s = benchmark(update_type=ut, mesh_res=res, nnn=10, radius=0.5)
            speedups.append((ut, res, s))

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"{'update_type':<15} {'mesh':<15} {'speedup':<10}")
    print("-" * 40)
    for ut, res, s in speedups:
        print(f"{ut:<15} {str(res):<15} {s:.1f}x")
