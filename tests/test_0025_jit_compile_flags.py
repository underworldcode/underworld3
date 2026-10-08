"""The flags the JIT compiles its kernels with (#834).

gcc defaults to ``-fmath-errno``: ``sqrt``, ``pow`` and ``exp`` may set ``errno``, so it
treats each call as a side effect and cannot merge repeated calls. The kernels never read
``errno``; with ``-fno-math-errno`` the viscoplastic box's Jacobian callbacks cost a third
as much on gcc 14, and every assembled value is bit-identical. The flag must survive a
``UW3_JIT_CFLAGS`` override, which replaces only the optimisation flags.
"""
import ast
import re

import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _compile_args(monkeypatch):
    """The ``extra_compile_args`` of the next JIT bundle's generated setup.py."""
    import underworld3.utilities._jitextension as jx

    bundles = []
    generate = jx.generate_c_source

    def capture(*args, **kwargs):
        result = generate(*args, **kwargs)
        bundles.append(dict(result[1]))
        return result

    monkeypatch.setattr(jx, "generate_c_source", capture)
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    # PETSc integrates only on a mesh that carries a field
    uw.discretisation.MeshVariable("U0025", mesh, 1, degree=1)
    uw.maths.Integral(mesh, 1.0 + mesh.X[0]).evaluate()
    assert bundles, "no JIT bundle was generated"
    match = re.search(r"extra_compile_args=(\[.*?\])", bundles[-1]["setup.py"])
    assert match, bundles[-1]["setup.py"]
    return ast.literal_eval(match.group(1))


def test_the_default_flags(monkeypatch):
    monkeypatch.delenv("UW3_JIT_CFLAGS", raising=False)
    assert _compile_args(monkeypatch) == ["-std=c99", "-fno-math-errno", "-O3", "-g0"]


def test_an_override_replaces_only_the_optimisation_flags(monkeypatch):
    monkeypatch.setenv("UW3_JIT_CFLAGS", "-O1 -g0")
    assert _compile_args(monkeypatch) == ["-std=c99", "-fno-math-errno", "-O1", "-g0"]
