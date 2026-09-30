# Analytic Solutions

```{eval-rst}
.. automodule:: underworld3.analytic
   :members:
   :show-inheritance:
```

## The contract

Every solution satisfies the same contract, so a validation run reads the same way
whichever one you use.

```{eval-rst}
.. automodule:: underworld3.analytic._base
   :members:
   :show-inheritance:
```

## Zhong spherical-shell response oracle

`Zhong2008` is a mesh-independent propagator-matrix oracle. It returns the
boundary response coefficients used by the Zhong et al. (2008) benchmark; it
does not implement the symbolic mesh-field contract above.

```{eval-rst}
.. automodule:: underworld3.analytic.zhong2008
   :members:
   :show-inheritance:
```

## See also

- {doc}`solvers` — the solvers these solutions validate.
- `docs/developer/subsystems/analytic-solutions.md` — the implementation form,
  the validation protocol every transcription must pass, and the provenance of
  each vendored reference kernel.
