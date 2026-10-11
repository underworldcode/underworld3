"""The flags the JIT compiles its kernels with (#834).

gcc, and clang on Linux, default to ``-fmath-errno``: ``sqrt``, ``pow`` and ``exp`` may
set ``errno``, so each call is a side effect that cannot be merged with a repeat of the
same call. The kernels never read ``errno``; with ``-fno-math-errno`` the viscoplastic
box's Jacobian callbacks cost a third as much on gcc 14, and every assembled value is
bit-identical. ``UW3_JIT_CFLAGS`` replaces the default flags, this one included, so a
compiler that rejects it can still be used.
"""
import ast
import re

import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_c]


def _compile_args(bundles):
    """The ``extra_compile_args`` of the bundle a small integral generates."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    # PETSc integrates only on a mesh that carries a field
    uw.discretisation.MeshVariable("U0025", mesh, 1, degree=1)
    uw.maths.Integral(mesh, 1.0 + mesh.X[0]).evaluate()
    assert bundles, "no JIT bundle was generated"
    match = re.search(r"extra_compile_args=(\[.*?\])", bundles[-1]["setup.py"])
    assert match, bundles[-1]["setup.py"]
    return ast.literal_eval(match.group(1))


def test_the_default_flags(jit_bundles, monkeypatch):
    monkeypatch.delenv("UW3_JIT_CFLAGS", raising=False)
    assert _compile_args(jit_bundles) == ["-std=c99", "-O3", "-g0", "-fno-math-errno"]


def test_an_override_replaces_every_flag_but_the_language_standard(jit_bundles, monkeypatch):
    monkeypatch.setenv("UW3_JIT_CFLAGS", "-O1")
    assert _compile_args(jit_bundles) == ["-std=c99", "-O1"]
