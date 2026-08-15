r"""Checks a transcribed analytic solution must pass before it is trusted.

The published kernels are careful; the risk is in our conversion of them. These
are the standing checks that catch a conversion error, and they are deliberately
of two kinds:

**Against the source.** :func:`reference_agreement` compares the transcription
with the kernel it came from, sampled where the solution is hard rather than
uniformly.

**Against the physics.** :func:`incompressibility_residual` and
:func:`momentum_residual` put the fields back into the equations they claim to
solve. These need no oracle at all, which is what makes them the strongest
checks here: they catch an error that a comparison would miss because the
comparison and the error share an assumption, and they catch the failure a
convergence test structurally cannot — if a transcription and the solver share a
mistaken sign, the solve converges beautifully to the wrong answer.

:func:`strainrate_consistency` sits between the two. The kernels return velocity
and stress as separately derived quantities, so differentiating one and
comparing against the other is a genuine cross-check rather than a tautology.

A check is only worth as much as its ability to fail. Every solution's test
should include a negative control — perturb one coefficient of the transcription
and confirm these report it — because a check that passes a deliberately broken
input is measuring nothing.

See ``docs/developer/subsystems/analytic-solutions.md``.
"""

import itertools

import numpy as np
import sympy

import underworld3 as uw


def adversarial_points(x_c=None, count=40, seed=20260802, dim=2):
    """Sample points that stress a solution rather than flatter it.

    Stratified over the unit box or cube, then loaded with the places these
    solutions are hard: either side of a material interface, the boundary faces,
    and the corners.

    Parameters
    ----------
    x_c : float, optional
        Position of a material interface normal to x, to sample across.
    count : int
        Number of interior points.
    seed : int
        Fixed, so a failure is reproducible.
    dim : int
        2 or 3. A 3D solution needs 3D points; sampling it on a plane would
        leave any error in the third direction unseen.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, dim)``.
    """

    rng = np.random.default_rng(seed)
    points = list(map(tuple, rng.uniform(0.0, 1.0, size=(count, dim))))

    interior = (0.37,) * (dim - 1)
    if x_c is not None:
        points += [
            (x_c - 1.0e-9,) + interior,
            (x_c + 1.0e-9,) + interior,
            (x_c,) + interior,
        ]

    # One point on each face, and every corner.
    for axis in range(dim):
        for value in (0.0, 1.0):
            face = [0.31] * dim
            face[axis] = value
            points.append(tuple(face))

    points += list(itertools.product((0.0, 1.0), repeat=dim))

    return np.array(points)


def sample(solution, expression, points):
    """Evaluate a solution's expression at *points*, without the JIT.

    These expressions are pure SymPy in the mesh coordinates, so they can be
    lambdified straight to NumPy. Routing them through
    :func:`underworld3.function.evaluate` instead would compile C for each one,
    and for a viscosity-jump solution that means a large ``Piecewise`` inside a
    stress derivative — a known combination that takes minutes to generate and
    build, turning a validation run into something nobody will wait for.

    ``cse=True`` recovers the sharing that back-substitution flattened, so the
    generated function is about the size of the original kernel.
    """

    coordinates = tuple(solution.mesh.X)

    # Mesh coordinates are not plain Symbols, and lambdify cannot bind them as
    # arguments, so swap in ordinary symbols first.
    plain = sympy.symbols(f"_c0:{len(coordinates)}")
    expression = sympy.sympify(expression).subs(dict(zip(coordinates, plain)))

    points = np.asarray(points, dtype=float)
    values = np.asarray(
        # SciPy first: NumPy has no erfc, and lambdify then silently falls back
        # to the scalar `math.erfc`, which fails only once an array reaches it.
        sympy.lambdify(plain, expression, ["scipy", "numpy"], cse=True)(
            *(points[:, i] for i in range(len(coordinates)))
        )
    )
    values = np.broadcast_to(values, (len(points),))

    if np.iscomplexobj(values):
        # A solution built from complex potentials is real-valued, but SymPy
        # cannot prove it, so the generated code returns complex. Take the real
        # part — after checking the imaginary one is round-off, because a
        # genuinely complex result would mean the construction is wrong and
        # discarding it silently would hide exactly that.
        #
        # The relative test alone is not enough. These same functions are used on
        # residuals, which are meant to vanish: there the real part is round-off
        # too, and comparing one round-off with another reports a large
        # "imaginary fraction" for a perfectly good result. So an absolute floor
        # comes first — an imaginary part at 1e-12 is noise whatever it is being
        # compared with.
        largest_imaginary = float(np.max(np.abs(values.imag)))
        largest_real = float(np.max(np.abs(values.real)))

        if largest_imaginary > 1.0e-12 and largest_imaginary > 1.0e-8 * largest_real:
            raise ValueError(
                f"expression evaluated complex: imaginary part reaches "
                f"{largest_imaginary:.3e} against a real part of "
                f"{largest_real:.3e}. It should be real-valued."
            )

        values = values.real

    return np.asarray(values, dtype=float)


def _worst_normalised(mine, theirs):
    """Largest difference, scaled by the field's own magnitude.

    Not a pointwise relative error: these fields pass through zero, so dividing
    by the true value reports a huge error wherever the solution is small and
    makes a correct transcription look catastrophic.
    """

    scale = max(float(np.max(np.abs(theirs))), 1.0e-300)
    return float(np.max(np.abs(mine - theirs)) / scale)


def reference_agreement(solution, fields, points):
    """Compare a transcription against the kernel it was transcribed from.

    Parameters
    ----------
    solution : AnalyticSolution
        The transcription under test.
    fields : dict
        Name -> ``(sympy expression, callable)``. The callable takes the sample
        coordinates and returns the reference value there.
    points : numpy.ndarray
        Sample coordinates, shape ``(N, 2)``.

    Returns
    -------
    dict
        Field name -> worst normalised error over the sample.
    """

    worst = {}
    for name, (expression, kernel) in fields.items():
        mine = sample(solution, expression, points)
        theirs = np.array([float(kernel(x, z)) for x, z in points])
        worst[name] = _worst_normalised(mine, theirs)

    return worst


def incompressibility_residual(solution, points):
    r""":math:`\max|\nabla\cdot\mathbf u|`, from the transcribed velocity alone.

    Needs no reference. A transcription that has dropped or corrupted a term in
    the velocity will generally stop being divergence-free.
    """

    coordinates = solution.mesh.X
    velocity = solution.fn_velocity

    divergence = sum(
        sympy.diff(velocity[0, i], coordinates[i]) for i in range(solution.mesh.dim)
    )
    values = sample(solution, divergence, points)

    return float(np.max(np.abs(values)))


def momentum_residual(solution, points):
    r"""Largest :math:`|\nabla\cdot\sigma + \mathbf f|`, relative to its terms.

    Uses the solution's own total (Cauchy) stress and body force, so it needs no
    reference and does not consult the solver. This is the check that catches a
    shared mistaken convention: the original SolCx port had the body-force sign
    the opposite way round from Underworld2's documentation, and a residual like
    this is what settles which is right.
    """

    coordinates = solution.mesh.X
    stress = solution.fn_stress
    bodyforce = solution.fn_bodyforce
    dim = solution.mesh.dim

    scale = 0.0
    worst = 0.0
    for i in range(dim):
        terms = [sympy.diff(stress[i, j], coordinates[j]) for j in range(dim)]
        terms.append(bodyforce[0, i])

        residual = sum(terms)
        worst = max(worst, float(np.max(np.abs(sample(solution, residual, points)))))

        # Scale by the largest term being cancelled, not by the body force. A
        # solution driven entirely by its boundary has no body force at all — the
        # elliptical inclusion is one — and normalising by it divides by zero.
        # The size of the terms is also the right yardstick for a cancellation:
        # it says how many digits actually had to cancel.
        for term in terms:
            scale = max(scale, float(np.max(np.abs(sample(solution, term, points)))))

    return worst / max(scale, 1.0e-300)


def transport_residual(solution, points):
    r"""Largest :math:`|\nabla\cdot(k\nabla u) + f|`, relative to its terms.

    The scalar counterpart of :func:`momentum_residual`, and the same argument
    for it: the coefficient, the source and the field all come from the solution
    itself, so this consults nothing external and cannot be fooled by a mistake
    the solution shares with whatever it might be compared against.
    """

    coordinates = solution.mesh.X
    dim = solution.mesh.dim

    flux = [
        solution.fn_coefficient * sympy.diff(solution.fn_solution, coordinates[i])
        for i in range(dim)
    ]
    terms = [sympy.diff(flux[i], coordinates[i]) for i in range(dim)]
    terms.append(solution.fn_source)

    worst = float(np.max(np.abs(sample(solution, sum(terms), points))))
    scale = max(
        float(np.max(np.abs(sample(solution, term, points)))) for term in terms
    )

    return worst / max(scale, 1.0e-300)


def diffusion_residual(solution, points, time):
    r"""Transient scalar residual at *time*.

    .. math::
        \partial_t u + \mathbf v\cdot\nabla u - \nabla\cdot(k\nabla u)

    The advection term is included because it is the general case: a purely
    diffusive solution declares no advecting velocity and it drops out. Leaving
    it off instead reports an order-one residual for a perfectly good
    advection-diffusion solution, which reads like a broken solution rather than
    a check applied to the wrong equation.

    Needs no reference — every term comes from the solution itself.
    """

    coordinates = solution.mesh.X
    dim = solution.mesh.dim

    rate = sympy.diff(solution.fn_solution, solution.t)
    carried = sum(
        solution.fn_advection[0, i] * sympy.diff(solution.fn_solution, coordinates[i])
        for i in range(dim)
    )
    spread = sum(
        sympy.diff(
            solution.fn_coefficient * sympy.diff(solution.fn_solution, coordinates[i]),
            coordinates[i],
        )
        for i in range(dim)
    )

    at_time = {solution.t: time}
    residual = (rate + carried - spread).subs(at_time)
    worst = float(np.max(np.abs(sample(solution, residual, points))))
    scale = max(
        float(np.max(np.abs(sample(solution, term.subs(at_time), points))))
        for term in (rate, carried, spread)
        if term != 0
    )

    return worst / max(scale, 1.0e-300)


def strainrate_consistency(solution, points):
    r"""Compare :math:`\tfrac12(\nabla\mathbf u + \nabla\mathbf u^{T})` with
    the solution's own strain rate, scaled by its magnitude.

    Not a tautology: the kernels derive velocity and stress independently, and
    the strain rate here comes from the stress and pressure. So this checks two
    separately derived quantities against each other, and it exercises the
    *derivatives* — which is what a solver consumes, and where a transcription
    can be wrong while still matching pointwise.
    """

    coordinates = solution.mesh.X
    velocity = solution.fn_velocity
    dim = solution.mesh.dim

    worst = 0.0
    scale = 0.0
    for i in range(dim):
        for j in range(dim):
            from_velocity = (
                sympy.diff(velocity[0, i], coordinates[j])
                + sympy.diff(velocity[0, j], coordinates[i])
            ) / 2
            difference = from_velocity - solution.fn_strainrate[i, j]

            values = sample(solution, difference, points)
            reference = sample(solution, solution.fn_strainrate[i, j], points)

            worst = max(worst, float(np.max(np.abs(values))))
            scale = max(scale, float(np.max(np.abs(reference))))

    return worst / max(scale, 1.0e-300)


def high_precision_value(expression, substitutions, digits=50):
    """Evaluate an expression at high precision, to separate two kinds of error.

    When a transcription and its source differ near the agreement threshold, the
    difference is either ours or the kernel's own double-precision cancellation.
    Evaluating the transcription at 50 digits tells them apart — which is what
    makes "the generator's grouping is numerically stable" a measured claim
    rather than an assumption.

    Parameters
    ----------
    expression : sympy expression
    substitutions : dict
        Values to substitute. Pass exact numbers (``sympy.Rational``,
        ``sympy.Integer``); floats defeat the purpose.
    digits : int
    """

    return sympy.N(expression.subs(substitutions), digits)


def richards_residual(solution, points, time=None):
    r"""Residual of the Richards equation for an unsaturated-flow solution.

    .. math::
        C(\psi)\,\partial_t \psi - \nabla\cdot\!\left[K(\psi)(\nabla\psi + \hat y)\right]

    The :math:`\hat y` is gravity, and leaving it out is the same class of
    mistake as leaving advection out of :func:`diffusion_residual` — it turns a
    correct solution into an order-one failure.

    *time* is ``None`` for a steady solution, which drops the capacity term.
    Needs no oracle: the conductivity and capacity are the solution's own.
    """

    coordinates = solution.mesh.X
    dim = solution.mesh.dim
    vertical = dim - 1

    gravity = [0] * dim
    gravity[vertical] = 1

    # The conducted and gravitational parts of the flux divergence are kept
    # apart because they are what has to cancel, and so they are the honest
    # scale. Normalising by the divergence itself divides the residual by the
    # residual, which reports 1.0 for a flawless solution — that is not a
    # tolerance failure, it is a meaningless number.
    terms = []
    for i in range(dim):
        conducted = solution.fn_conductivity * sympy.diff(
            solution.fn_solution, coordinates[i]
        )
        terms.append(sympy.diff(conducted, coordinates[i]))
        if gravity[i]:
            terms.append(
                sympy.diff(solution.fn_conductivity * gravity[i], coordinates[i])
            )

    divergence = sum(terms)

    if time is None:
        residual = -divergence
        at_time = {}
    else:
        storage = solution.fn_capacity * sympy.diff(solution.fn_solution, solution.t)
        terms.append(storage)
        residual = storage - divergence
        at_time = {solution.t: time}

    terms = [term for term in terms if term != 0]

    worst = float(np.max(np.abs(sample(solution, residual.subs(at_time), points))))
    scale = max(
        float(np.max(np.abs(sample(solution, term.subs(at_time), points))))
        for term in terms
    )

    return worst / max(scale, 1.0e-300)
