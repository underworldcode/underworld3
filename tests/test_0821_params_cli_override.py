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
