"""Regression test for issue #137 — bare-variable composition asymmetry.

MathematicalMixin advertises that mesh / swarm variables can be used directly
in sympy arithmetic without explicit ``.sym`` access. Pre-fix this only worked
when the bare variable was the *innermost* operand; as soon as a bare variable
appeared on the right of an already-sympified subexpression (e.g. inside a
sympy ``exp()`` of a product), composition raised:

    TypeError: Incompatible classes
        <MutableDenseMatrix>, <EnhancedMeshVariable>

Cause: ``.sym`` on a scalar returns a 1×1 sympy ``Matrix``. SymPy's
``Matrix.__mul__`` raises TypeError directly instead of returning
NotImplemented, so Python's normal fall-through to the right operand's
``__rmul__`` never fires.

Fix: ``MathematicalMixin._op_priority = 11.5`` (above sympy's
``Matrix._op_priority = 10.01``) makes sympy delegate the operation to our
reverse dunder, which then sympifies ``self`` via ``.sym`` and re-runs the
multiplication cleanly as ``Matrix * Matrix``.
"""

import pytest
import sympy

import underworld3 as uw


pytestmark = pytest.mark.level_1


@pytest.fixture(scope="module")
def vars_TC():
    """Two scalar variables of different kinds: a MeshVariable and a SwarmVariable."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5,
    )
    T = uw.discretisation.MeshVariable("T_137", mesh, 1, degree=1)

    swarm = uw.swarm.Swarm(mesh)
    C = uw.swarm.SwarmVariable("C_137", swarm, size=1, proxy_degree=1)
    swarm.populate(fill_param=2)
    return T, C


@pytest.mark.parametrize(
    "label,build",
    [
        ("A_sym_both",      lambda T, C: sympy.exp(-C.sym * T.sym)),
        ("B_bareC_symT",    lambda T, C: sympy.exp(-C     * T.sym)),
        ("C_symC_bareT",    lambda T, C: sympy.exp(-C.sym * T    )),
        ("D_bare_both",     lambda T, C: sympy.exp(-C     * T    )),
    ],
)
def test_bare_variable_composition_under_sympy_function(vars_TC, label, build):
    """All four mixed-form combinations must compose cleanly under sympy.exp.

    Pre-fix, cases C and D raised TypeError. Post-fix all four return a
    sympy expression of identical structure.
    """
    T, C = vars_TC
    eta_0 = sympy.symbols("eta_0_137")
    result = eta_0 * build(T, C)
    # Result should be a sympy object (Matrix or Expr) — type may differ
    # between cases but the structure is equivalent.
    assert result is not None
    # The four results should all simplify to the same canonical form.
    # We can't easily compare across the parametrize boundary in a
    # parametrised test, but we can at least check it's sympifiable.
    assert hasattr(result, "free_symbols") or hasattr(result, "shape")


def test_bare_variable_composition_all_forms_agree(vars_TC):
    """The four mixed-form expressions should be mathematically equivalent."""
    T, C = vars_TC
    eta_0 = sympy.symbols("eta_0_137")

    forms = [
        eta_0 * sympy.exp(-C.sym * T.sym),
        eta_0 * sympy.exp(-C     * T.sym),
        eta_0 * sympy.exp(-C.sym * T    ),
        eta_0 * sympy.exp(-C     * T    ),
    ]

    # Reduce to a comparable scalar by extracting the [0,0] element if needed.
    def scalarise(x):
        if hasattr(x, "shape") and x.shape == (1, 1):
            return x[0, 0]
        return x

    canonical = [sympy.simplify(scalarise(f) - scalarise(forms[0])) for f in forms]
    for i, diff in enumerate(canonical):
        assert diff == 0, (
            f"form {i} differs from form 0 after simplification: residual={diff}"
        )
