# Constitutive Models

## Viscous Flow Models

### ViscousFlowModel

```{eval-rst}
.. autoclass:: underworld3.constitutive_models.ViscousFlowModel
```

## Diffusion Models

### DiffusionModel

```{eval-rst}
.. autoclass:: underworld3.constitutive_models.DiffusionModel
```

### TransverseIsotropicFlowModel

```{eval-rst}
.. autoclass:: underworld3.constitutive_models.TransverseIsotropicFlowModel
```

## Viscoelastic Models

### ViscoElasticPlasticFlowModel

```{eval-rst}
.. autoclass:: underworld3.constitutive_models.ViscoElasticPlasticFlowModel
```

## Yield-law regularisation

### Micromorphic

Set on a yielding model as ``cm.Parameters.plastic_regularisation``.

```{eval-rst}
.. autoclass:: underworld3.constitutive_models.Micromorphic
   :members: length, modulus, mask, chi, shift, coupled_field_request
```

## Darcy Flow

### DarcyFlowModel

```{eval-rst}
.. autoclass:: underworld3.constitutive_models.DarcyFlowModel
```
