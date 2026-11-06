# Function and Expressions

## Expressions

### UWexpression

```{eval-rst}
.. autoclass:: underworld3.function.expressions.UWexpression
   :members:
   :show-inheritance:
```

### expression

Factory function for creating UWexpression objects.

```{eval-rst}
.. autofunction:: underworld3.function.expression
```

## Mesh Variable Functions

### UnderworldFunction

Symbolic representation of mesh variable fields used in expressions and equations.

```{eval-rst}
.. autoclass:: underworld3.function.UnderworldFunction
   :members:
   :show-inheritance:
```

### unwrap

Unwrap UWexpressions to their underlying SymPy expressions for compilation.

```{eval-rst}
.. autofunction:: underworld3.function.unwrap
```

### diff_wrt_field

Differentiate with respect to a field value or gradient, keeping the field's realness.
Use it, not `sympy.diff`, for a hand-written tangent or `flux_jacobian`.

```{eval-rst}
.. autofunction:: underworld3.function.diff_wrt_field
```

### derive_by_array_wrt_field

`sympy.derive_by_array` through `diff_wrt_field`.

```{eval-rst}
.. autofunction:: underworld3.function.derive_by_array_wrt_field
```

## Quantities and Units

### UWQuantity

```{eval-rst}
.. autoclass:: underworld3.function.UWQuantity
   :members:
   :show-inheritance:
```

### quantity

Factory function for creating UWQuantity objects with units.

```{eval-rst}
.. autofunction:: underworld3.function.quantity
```

## Evaluation

### evaluate

```{eval-rst}
.. autofunction:: underworld3.function.evaluate
```

### global_evaluate

```{eval-rst}
.. autofunction:: underworld3.function.global_evaluate
```

### evaluate_gradient

```{eval-rst}
.. autofunction:: underworld3.function.evaluate_gradient
```

## Analytic Functions

The analytic solutions have moved to {doc}`analytic` — `underworld3.function.analytic`
is a deprecation shim. Use `uw.analytic.SolCx(mesh, ...)`.
