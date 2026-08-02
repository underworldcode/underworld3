r"""Turn a machine-generated C kernel into SymPy, preserving its grouping.

The classical analytic solutions were published as Maple output: long runs of
straight-line single assignments, ``t125 = 0.4e1 * t81 * t83 + ...``, with a
branch or two. This module reads that C and rebuilds the same expression tree in
SymPy.

**The grouping is the point.** These expansions contain products like
:math:`\sinh(k)e^{-k}` that are numerically stable only in the arrangement the
generator produced; a re-derivation is a different arrangement and can lose eight
digits at large wavenumber or large viscosity contrast — exactly the regime the
benchmarks exist to probe. So nothing here simplifies, expands, collects or
reorders: each statement is substituted into the next verbatim, and the resulting
tree evaluates term for term as the C does.

Transcribing at run time rather than generating a checked-in module is deliberate.
It costs a fraction of a second, and it means the SymPy form cannot drift from the
C it came from — they are the same artefact, not two copies of one.

Numeric literals become exact :class:`sympy.Rational`\s. Maple writes them as
``0.4e1``, i.e. exact small values, so this loses nothing and lets a solution be
evaluated at arbitrary precision — which is how a transcription error is told
apart from the C's own double-precision cancellation.

See ``docs/developer/subsystems/analytic-solutions.md`` for the validation every
transcription must pass before it is used.
"""

import re

import sympy

# The only functions the Velic kernels call.
_C_FUNCTIONS = {
    "exp": sympy.exp,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "sqrt": sympy.sqrt,
    "pow": lambda base, exponent: base**exponent,
    "M_PI": sympy.pi,
}

# The assignment target keeps any `struct.` prefix. Without it, `out.x = ...`
# reads as an assignment to `x` and silently overwrites the coordinate symbol —
# every later statement referring to x then gets the wrong thing, and the result
# looks plausible rather than broken.
_STATEMENT = re.compile(r"((?:\w+\.)?\w+)\s*=\s*([^;]+);")
_FLOAT_LITERAL = re.compile(r"\b\d+\.\d*(?:[eE][+-]?\d+)?")


def _strip_comments(source):
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    return re.sub(r"//[^\n]*", " ", source)


def _matching_brace(source, opening):
    """Index just past the ``}`` closing the ``{`` at *opening*."""

    depth, index = 0, opening
    while True:
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1


def _as_python(expression):
    """Prepare one C expression for evaluation as Python.

    Two rewrites. Float literals become exact ``Rational``\\s — ``0.4e1`` is the
    generator's way of writing 4, and reading it as a float would make every
    downstream comparison approximate for no reason. And the statement is folded
    onto one line: C statements wrap freely, but a wrapped Python expression with
    indented continuations is a syntax error.
    """

    expression = " ".join(expression.split())
    return _FLOAT_LITERAL.sub(lambda m: f"Rational('{m.group(0)}')", expression)


class CSource:
    """A vendored C kernel, addressable by function and by block.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.c`` file to read.
    """

    def __init__(self, path):
        self.path = str(path)
        self.text = _strip_comments(open(self.path).read())

    def function(self, name, returns="void"):
        """The body of ``<returns> <name>(...)``, braces excluded."""

        signature = self.text.index(f"{returns} {name}(")
        opening = self.text.index("{", self.text.index(")", signature))
        return self.text[opening + 1 : _matching_brace(self.text, opening) - 1]

    @staticmethod
    def returned(body):
        """The expression a body returns, as text."""

        marker = body.index("return")
        return body[marker + len("return") : body.index(";", marker)]

    @staticmethod
    def branches(body, condition, tail_ends_at=None):
        """Split ``if (<condition>) { A } else { B }`` into ``(A, B, tail)``.

        The tail is what follows the ``else`` block — in these kernels, the shared
        arithmetic turning the branch's integration constants into the output
        fields.

        Parameters
        ----------
        tail_ends_at : str, optional
            Cut the tail at the first occurrence of this text. Needed because the
            kernels end with an output section that accumulates into ``sum`` variables
            with ``+=`` and writes through pointers — statements this reader is not
            meant to interpret, and whose operands it has never bound.
        """

        marker = body.index(condition)
        opening = body.index("{", marker)
        then_end = _matching_brace(body, opening)
        then_block = body[opening + 1 : then_end - 1]

        else_opening = body.index("{", body.index("else", then_end - 1))
        else_end = _matching_brace(body, else_opening)
        else_block = body[else_opening + 1 : else_end - 1]

        tail = body[else_end:]
        if tail_ends_at is not None:
            tail = tail[: tail.index(tail_ends_at)]

        return then_block, else_block, tail


def evaluate_expression(text, environment):
    """Evaluate a single C expression against names already in scope.

    For kernels that end in ``return <expression>;`` rather than assigning the
    result to a variable.
    """

    namespace = {**_C_FUNCTIONS, "Rational": sympy.Rational}
    return sympy.sympify(
        eval(_as_python(text), {"__builtins__": {}}, {**namespace, **environment})
    )


def evaluate_block(block, environment):
    """Substitute a straight-line block of C assignments into SymPy.

    Parameters
    ----------
    block : str
        C statements of the form ``name = expression;``. Anything else — a
        declaration, an ``if``, an array write — is ignored, so a block may be
        handed a whole function body and only its assignments are read.
    environment : dict
        Symbol names already in scope, mapped to SymPy expressions. Not modified.

    Returns
    -------
    dict
        *environment* extended with every name the block assigns. A name assigned
        more than once holds its final value, matching C.

    Notes
    -----
    The generated statements are already valid Python once ``pow``, ``exp``,
    ``sin`` and ``cos`` are in scope, so they are evaluated directly against a
    namespace holding no builtins. The input is a kernel vendored in this
    package, not anything a caller supplies.

    Temporaries are substituted as they are read, so the returned expressions are
    full trees rather than a chain of definitions. Measured on ``solCx.c``, the
    largest is a few thousand operations — the sharing is recovered by common
    subexpression elimination when the expression is compiled or lambdified.
    """

    scope = dict(environment)
    namespace = {**_C_FUNCTIONS, "Rational": sympy.Rational}

    for target, expression in _STATEMENT.findall(block):
        value = eval(
            _as_python(expression), {"__builtins__": {}}, {**namespace, **scope}
        )
        scope[target] = sympy.sympify(value)

    return scope
