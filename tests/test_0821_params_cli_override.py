#!/usr/bin/env python3
"""Regression: uw.Params picks up -uw_* command-line overrides (#111).

On some platforms (e.g. Gadi) petsc4py does NOT auto-populate the PETSc options
database from sys.argv, so uw.Params silently fell back to its defaults even
when the CLI argument was present. Params now calls parse_cmd_line_options()
during construction, so a `-uw_<name> <value>` override is applied regardless of
platform.
"""
import sys

import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _clear(name):
    from petsc4py import PETSc

    opts = PETSc.Options()
    if opts.hasName(name):
        opts.delValue(name)


def test_params_applies_cli_override():
    """A -uw_* CLI argument overrides the Params default even when the options
    DB was not pre-populated (the Gadi failure mode in #111)."""
    saved = sys.argv
    try:
        # Simulate Gadi: CLI arg present, options DB not yet populated.
        _clear("uw_cellsize")
        sys.argv = ["prog", "-uw_cellsize", "1/16"]
        params = uw.Params(uw_cellsize=uw.Param("1/8", type=uw.ParamType.STRING))
        assert params.uw_cellsize == "1/16", (
            f"CLI override not applied: got {params.uw_cellsize!r}, expected '1/16'"
        )
    finally:
        sys.argv = saved
        _clear("uw_cellsize")


def test_params_uses_default_without_cli():
    """With no CLI override, Params returns its default."""
    saved = sys.argv
    try:
        _clear("uw_testparam_111")
        sys.argv = ["prog"]
        params = uw.Params(uw_testparam_111=uw.Param("default_value", type=uw.ParamType.STRING))
        assert params.uw_testparam_111 == "default_value"
    finally:
        sys.argv = saved


def test_a_negative_value_reaches_the_parameter():
    """A negative CLI value must arrive, not fall back to the default (#642).

    `parse_cmd_line_options` decided what was an option NAME with
    `item[0] == "-" and item[1] != "-"`, which accepts `-2`. So `-uw_sense -2`
    stored `sense` with no value and registered a stray option `2`, and Params
    then used its default — silently. It once ran half of a 26-run parameter
    ladder at the wrong sign while reporting it under the requested label.

    PETSc's own rule (`PetscOptionsValidKey`) is a hyphen followed by a LETTER,
    which is exactly what distinguishes a key from a negative number.
    """
    saved = sys.argv
    try:
        for name, given, expected in (
                ("uw_sense_642", "-2", -2.0),        # negative integer
                ("uw_scale_642", "-2.5", -2.5),      # negative float
                ("uw_tiny_642", "-1e-5", -1.0e-5),   # negative exponent, inner hyphen
        ):
            _clear(name[3:])
            sys.argv = ["prog", f"-{name}", given]
            params = uw.Params(**{name: uw.Param(1.0, "probe")})
            actual = float(getattr(params, name))
            assert actual == pytest.approx(expected), (
                f"-{name} {given} gave {actual}, not {expected} — a negative "
                "value was read as the next option name again"
            )
            _clear(name[3:])
    finally:
        sys.argv = saved


def test_a_positive_value_still_reaches_the_parameter():
    """Negative control: the fix narrows what counts as an option name, so the
    ordinary positive path has to be shown still working."""
    saved = sys.argv
    try:
        _clear("sense_642_pos")
        sys.argv = ["prog", "-uw_sense_642_pos", "2.5"]
        params = uw.Params(uw_sense_642_pos=uw.Param(1.0, "probe"))
        assert float(params.uw_sense_642_pos) == pytest.approx(2.5)
    finally:
        sys.argv = saved
        _clear("sense_642_pos")
