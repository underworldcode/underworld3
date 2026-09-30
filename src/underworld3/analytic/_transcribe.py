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

# The only functions these kernels call, in both the plain-C and PETSc spellings
# — the same solutions are published in both forms.
_C_FUNCTIONS = {
    "exp": sympy.exp,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "sqrt": sympy.sqrt,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "tanh": sympy.tanh,
    "pow": lambda base, exponent: base**exponent,
    "M_PI": sympy.pi,
    "PetscExpReal": sympy.exp,
    "PetscSinReal": sympy.sin,
    "PetscCosReal": sympy.cos,
    "PetscSqrtReal": sympy.sqrt,
    "PetscPowReal": lambda base, exponent: base**exponent,
    "PETSC_PI": sympy.pi,
}

# C casts, which are juxtaposition in Python and so a syntax error.
_CAST = re.compile(
    r"\(\s*(?:PetscReal|PetscScalar|PetscInt|double|float|int|unsigned)\s*\)\s*"
)

# C identifiers that Python reserves. `in` is the one that actually occurs —
# these kernels take their coordinates as `const double* in`. Renamed in the
# source text and in the caller's environment together, so a caller still writes
# the name the C uses.
_RESERVED = {"in": "_c_in", "lambda": "_c_lambda", "is": "_c_is", "not": "_c_not"}
_RESERVED_PATTERN = re.compile(r"\b(" + "|".join(_RESERVED) + r")\b")


def _rewrite_ternary(text):
    """Rewrite C conditional expressions as Python ones.

    ``c ? a : b`` becomes ``(a) if (c) else (b)``, and the C logical operators
    become their Python spellings. ``!=`` is left alone — the only bare ``!``
    these kernels use is part of it.

    Parenthesised groups are rewritten first, so a conditional nested inside one
    is resolved before the enclosing scan runs; the outer scan then only has to
    find a ``?`` at depth zero. In practice the conditions are on loop indices,
    bound to integers before evaluation, so the whole expression collapses to a
    single branch.
    """

    text = text.replace("&&", " and ").replace("||", " or ")

    # Inner groups first.
    pieces, index = [], 0
    while index < len(text):
        if text[index] == "(":
            close = _matching_paren(text, index)
            pieces.append("(" + _rewrite_ternary(text[index + 1 : close]) + ")")
            index = close + 1
        else:
            pieces.append(text[index])
            index += 1
    text = "".join(pieces)

    depth = 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "?" and depth == 0:
            inner = 0
            for offset, following in enumerate(text[index + 1 :]):
                if following == "(":
                    inner += 1
                elif following == ")":
                    inner -= 1
                elif following == ":" and inner == 0:
                    colon = index + 1 + offset
                    return (
                        f"(({text[index + 1 : colon]})"
                        f" if ({text[:index]})"
                        f" else ({text[colon + 1 :]}))"
                    )
            break

    return text


def _matching_paren(text, opening):
    """Index of the ``)`` closing the ``(`` at *opening*."""

    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return len(text) - 1


def _rename_reserved(text):
    return _RESERVED_PATTERN.sub(lambda m: _RESERVED[m.group(1)], text)


_CHAIN = re.compile(r"\b(\w+)\s*=\s*(?=\w+\s*=(?!=))")


def _expand_chains(block):
    """Give each target of a chained assignment its own statement.

    ``a = b = c;`` assigns c to both. Read as one statement its *value* is
    ``b = c``, which is not an expression and raises. Rewriting it as
    ``b = c; a = b;`` keeps the order the C has — the rightmost target is bound
    first, and the others follow from it.
    """

    out = []
    for statement in block.split(";"):
        targets = _CHAIN.findall(statement)
        if not targets:
            out.append(statement)
            continue

        remainder = _CHAIN.sub("", statement)
        # Innermost first, so each target is defined before the next uses it.
        pieces = [remainder]
        previous = remainder.split("=")[0].strip()
        for target in reversed(targets):
            pieces.append(f" {target} = {previous}")
            previous = target
        out.append(";".join(pieces))

    return ";".join(out)


def _split_declarators(block):
    """Give each declarator in a C declaration its own statement.

    ``double x=in[0], y=in[1], z=in[2];`` is one statement to the reader below,
    whose value would run past the first comma and fail to parse. Splitting on
    top-level commas that introduce a new ``name =`` turns it into three.

    Only commas outside brackets count, so function arguments and array
    subscripts are left alone.
    """

    out = []
    depth = 0
    for index, character in enumerate(block):
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif character == "," and depth == 0:
            if re.match(r"\s*\w+\s*=(?!=)", block[index + 1 :]):
                out.append(";")
                continue
        out.append(character)

    return "".join(out)

# The assignment target keeps whatever it is written through: a `struct.` prefix
# or an `[index]` suffix. Both occur — these kernels return their results through
# `out.xx = ...` or `out[0] = ...` depending on their vintage.
#
# The prefix is not cosmetic. Without it, `out.x = ...` reads as an assignment to
# `x` and silently overwrites the coordinate symbol; every later statement using x
# then gets the wrong thing, and the result looks plausible rather than broken.
# Without the suffix, `out[0] = ...` matches nothing at all — a loud failure
# rather than a quiet one, but a failure.
_STATEMENT = re.compile(r"((?:\w+\.)?\w+(?:\[\d+\])?)\s*=\s*([^;]+);")
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

    Three rewrites. Float literals become exact ``Rational``\\s — ``0.4e1`` is
    the generator's way of writing 4, and reading it as a float would make every
    downstream comparison approximate for no reason. The statement is folded onto
    one line, since C wraps freely but a wrapped Python expression with indented
    continuations is a syntax error. And C casts are dropped: ``(PetscReal)n`` is
    juxtaposition in Python, which does not parse.
    """

    expression = _rewrite_ternary(
        _rename_reserved(_CAST.sub("", " ".join(expression.split())))
    )
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
    def loop_body(body, header):
        """The body of a ``for`` loop, braces excluded.

        The series solutions accumulate over modes. The reader does not
        interpret the loop — the caller evaluates this body once per mode with
        the index bound, and sums — because the accumulation uses ``+=``, which
        is not an assignment this reader recognises, and because the summation
        has to happen in SymPy anyway.
        """

        marker = body.index(header)
        opening = body.index("{", marker)
        return body[opening + 1 : _matching_brace(body, opening) - 1]

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


def resolve_branches(block, bindings):
    """Replace ``if``/``else`` whose condition is already decided by its branch.

    The series kernels guard their zero modes with tests on the loop indices —
    ``if (n != 0 && m != 0)`` and so on. Those indices are bound to integers
    before anything is evaluated, so the condition has an answer at transcription
    time and the construct collapses to whichever branch the C would take.

    That matters because :func:`evaluate_block` does not interpret ``if``: it
    reads every assignment in order, so an unresolved guard leaves each guarded
    variable holding the *last* branch's value. In SolH that silently zeroes two
    velocity components, which looks like a plausible solution rather than a
    broken one.

    Nested guards resolve outermost first, repeatedly, until none remain.

    Parameters
    ----------
    block : str
    bindings : dict
        Names the condition may use, mapped to Python values — normally the loop
        indices as integers.
    """

    namespace = {**_C_FUNCTIONS, "Rational": sympy.Rational}

    while True:
        marker = re.search(r"\bif\s*\(", block)
        if marker is None:
            return block

        condition_start = block.index("(", marker.start())
        condition_end = _matching_paren(block, condition_start)
        condition = block[condition_start + 1 : condition_end]

        opening = block.index("{", condition_end)
        then_end = _matching_brace(block, opening)
        taken = block[opening + 1 : then_end - 1]
        after = block[then_end:]

        otherwise = ""
        stripped = after.lstrip()
        if stripped.startswith("else"):
            else_opening = block.index("{", then_end)
            else_end = _matching_brace(block, else_opening)
            otherwise = block[else_opening + 1 : else_end - 1]
            after = block[else_end:]

        chosen = taken if eval(
            _as_python(condition), {"__builtins__": {}}, {**namespace, **bindings}
        ) else otherwise

        block = block[: marker.start()] + chosen + after


def evaluate_expression(text, environment):
    """Evaluate a single C expression against names already in scope.

    For kernels that end in ``return <expression>;`` rather than assigning the
    result to a variable.
    """

    namespace = {**_C_FUNCTIONS, "Rational": sympy.Rational}
    scope = {_RESERVED.get(name, name): value for name, value in environment.items()}
    return sympy.sympify(
        eval(_as_python(text), {"__builtins__": {}}, {**namespace, **scope})
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

    scope = {_RESERVED.get(name, name): value for name, value in environment.items()}
    namespace = {**_C_FUNCTIONS, "Rational": sympy.Rational}

    for target, expression in _STATEMENT.findall(_expand_chains(_split_declarators(block))):
        value = eval(
            _as_python(expression), {"__builtins__": {}}, {**namespace, **scope}
        )
        scope[target] = sympy.sympify(value)

    return scope
