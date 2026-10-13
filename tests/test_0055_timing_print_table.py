"""
Regression tests for uw.timing.print_table legacy keywords.

Issue #499: print_table was reduced to (filename, format) when timing moved
to the PETSc log backend, but 18 shipped examples still pass the legacy
display_fraction / group_by / output_file keywords and aborted with
TypeError. The keywords are accepted again: output_file is a working alias
for filename; display_fraction and group_by are accepted with a visible
warning (the PETSc log table cannot honour them).
"""

import os

import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture(scope="module", autouse=True)
def petsc_logging():
    """print_table needs PETSc logging active to produce any output."""
    uw.timing.start()
    yield


def test_legacy_kwargs_accepted_together(tmp_path):
    """The exact call pattern from Ex_Stokes_*_Benchmark_Kramer."""
    out = tmp_path / "mesh_create_time.txt"
    with pytest.warns((FutureWarning, DeprecationWarning)):
        uw.timing.print_table(
            group_by="line_routine",
            output_file=str(out),
            display_fraction=1.00,
        )
    if uw.mpi.rank == 0:
        assert out.exists()
        assert os.path.getsize(out) > 0


def test_display_fraction_alone_warns_not_raises(capsys):
    """Ex_Sheared_Layer_Elastic et al.: print_table(display_fraction=1)."""
    with pytest.warns(FutureWarning, match="display_fraction"):
        uw.timing.print_table(display_fraction=1)


def test_output_file_writes(tmp_path):
    """output_file alone must behave exactly like filename."""
    out = tmp_path / "timing.txt"
    with pytest.warns(DeprecationWarning, match="output_file"):
        uw.timing.print_table(output_file=str(out))
    if uw.mpi.rank == 0:
        assert out.exists()
        assert os.path.getsize(out) > 0


def test_filename_and_output_file_conflict(tmp_path):
    with pytest.raises(TypeError, match="output_file"):
        uw.timing.print_table(
            filename=str(tmp_path / "a.txt"),
            output_file=str(tmp_path / "b.txt"),
        )


def test_group_by_warns(capsys):
    with pytest.warns(FutureWarning, match="group_by"):
        uw.timing.print_table(group_by="routine")


def test_new_signature_still_works(tmp_path):
    """The current (filename, format) interface is unchanged."""
    out = tmp_path / "timing.csv"
    uw.timing.print_table(str(out), format="csv")
    if uw.mpi.rank == 0:
        assert out.exists()
        assert os.path.getsize(out) > 0
