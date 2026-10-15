from petsc4py import PETSc

## I think we should deprecate this since PETSc already does this for us

options = PETSc.Options("uw_")
"""Expose petsc options on the command line as -uw_XXX"""


def require_dirs(ListOfDirs):
    """
    List of directories required by this run
    """
    import os

    for dir in ListOfDirs:
        os.makedirs(dir, exist_ok=True)


def parse_cmd_line_options():
    """Hand the command line to PETSc's own options parser.

    UW parameters are namespaced `-uw_*` precisely so they can sit in the PETSc
    options database alongside PETSc's own without clashing, and `uw.options` is
    a `PETSc.Options("uw_")` view onto it. So there is nothing here that PETSc
    does not already do: `PetscOptionsInsertString` (exposed by petsc4py as
    `Options.insertString`) applies the same parsing rules PETSc applies to its
    own arguments.

    This used to re-implement that parsing, and got it wrong. Its test for an
    option NAME was `item[0] == "-" and item[1] != "-"`, which accepts `-2`, so
    `-uw_sense -2` stored `uw_sense` with no value and registered a stray option
    `2` -- the negative silently never arrived (#642). PETSc's own rule
    (`PetscOptionsValidKey`) requires a hyphen followed by a letter, which is
    exactly what distinguishes an option from a negative number. Deferring to it
    fixes that class of bug rather than the one instance of it.

    It exists at all because petsc4py does NOT populate the options database
    from `sys.argv` on every platform (Gadi being the case in #111), so
    something has to do the insertion explicitly. It is idempotent -- re-inserting
    the same arguments rewrites the same values -- so it is safe to call on every
    `Params` construction.
    """
    from petsc4py import PETSc
    import sys

    arguments = sys.argv[1:]
    if not arguments:
        return

    # PetscOptionsInsertString reads a single string, so an argument carrying
    # whitespace has to be quoted back up; PETSc understands double quotes.
    def requote(argument):
        return f'"{argument}"' if any(c.isspace() for c in argument) else argument

    PETSc.Options().insertString(" ".join(requote(a) for a in arguments))


import os as _os


def fieldsplit_sub_ksps(pc):
    """The block solvers of a field-split ``pc``, or ``[]`` when it has none yet.

    The blocks exist only once ``PCSetUp`` has run, and asking an un-set-up split
    for them is a PETSc error that PETSc prints as a full banner even when the
    exception is caught -- so a healthy solve looks broken (#827). The banner is
    silenced for this one call only: the ``"return"`` handler still raises, it just
    does not print, and it is popped on every path. (The ``"ignore"`` handler is
    different -- it turns errors into success -- and pushed broadly it broke 23
    unrelated tests.)

    Anything that is not a field split also has no blocks, so it gives ``[]``.
    """
    if pc.getType() != "fieldsplit":
        return []
    PETSc.Sys.pushErrorHandler("return")
    try:
        return list(pc.getFieldSplitSubKSP())
    except PETSc.Error:
        return []                         # not set up yet: no blocks to report
    finally:
        PETSc.Sys.popErrorHandler()
