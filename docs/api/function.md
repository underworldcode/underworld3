# Function and Expressions

```{eval-rst}
.. automodule:: underworld3.function
   :no-members:
```

## Expressions

### UWexpression

```{eval-rst}
.. autoclass:: underworld3.function.UWexpression
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

### evalf

```{eval-rst}
.. autofunction:: underworld3.function.evalf
```

### evaluate_gradient

```{eval-rst}
.. autofunction:: underworld3.function.evaluate_gradient
```

## Analytic Functions

```{eval-rst}
.. automodule:: underworld3.function.analytic
   :members:
   :show-inheritance:
```
