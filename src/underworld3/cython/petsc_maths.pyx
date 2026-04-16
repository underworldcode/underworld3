from typing import Union
import sympy

import underworld3
import underworld3.timing as timing
from   underworld3.utilities._jitextension import getext, JITCallbackSet

from petsc4py import PETSc

include "petsc_extras.pxi"

cdef extern from "petsc.h" nogil:
    PetscErrorCode PetscDSSetObjective( PetscDS, PetscInt, PetscDSResidualFn )
    PetscErrorCode DMPlexComputeIntegralFEM( PetscDM, PetscVec, PetscScalar*, void* )
    PetscErrorCode DMPlexComputeCellwiseIntegralFEM( PetscDM, PetscVec, PetscVec, void* )


class Integral:
    """
    The `Integral` class constructs the volume integral

    .. math:: F_{i}  =   \int_V \, f(\mathbf{x}) \, \mathrm{d} V

    for some scalar function :math:`f` over the mesh domain :math:`V`.

    Parameters
    ----------
    mesh :
        The mesh over which integration is performed.
    fn :
        Function to be integrated.

    Example
    -------
    Calculate volume of mesh:

    >>> import underworld3 as uw
    >>> import numpy as np
    >>> mesh = uw.discretisation.Box()
    >>> volumeIntegral = uw.maths.Integral(mesh=mesh, fn=1.)
    >>> np.allclose( 1., volumeIntegral.evaluate(), rtol=1e-8)
    True
    """

    @timing.routine_timer_decorator
    def __init__( self,
                  mesh:  underworld3.discretisation.Mesh,
                  fn:    Union[float, int, sympy.Basic] ):

        self.mesh = mesh
        self.fn = sympy.sympify(fn)
        super().__init__()

    @timing.routine_timer_decorator
    def evaluate(self, verbose=False):
        """
        Evaluate the integral and return the result with units (if applicable).

        Returns
        -------
        float or UWQuantity
            The integral value. If the integrand has units AND the mesh coordinates
            have units, returns a UWQuantity with proper dimensional analysis
            (integrand_units * volume_units). Otherwise returns a plain float.
        """
        if len(self.mesh.vars)==0:
            raise RuntimeError("The mesh requires at least a single variable for integration to function correctly.\n"
                               "This is a PETSc limitation.")

        # Create JIT extension.
        #
        # Note that - we pass in the mesh variables as primary variables, as this
        # is how they are represented on the mesh DM.

        # Note that -  (at this time) PETSc does not support vector integrands, so
        # if we wish to do vector integrals we'll need to split out the components
        # and calculate them individually. Let's support only scalars for now.

        # Note that - DMPlexComputeIntegralFEM returns an array even though we have set only
        # one objective function and only expect one non-zero value to be returned
        # Temporary workaround for this is to over-allocate the array we collect.

        if isinstance(self.fn, sympy.vector.Vector):
            raise RuntimeError("Integral evaluation for Vector integrands not supported.")
        elif isinstance(self.fn, sympy.vector.Dyadic):
            raise RuntimeError("Integral evaluation for Dyadic integrands not supported.")


        self.dm = self.mesh.dm  # .clone()
        mesh=self.mesh

        _getext_result = getext(self.mesh, JITCallbackSet(residual=(self.fn,)),
                                self.mesh.vars.values(), verbose=verbose)
        cdef PtrContainer ext = _getext_result.ptrobj

        # Pull out vec for variables, and go ahead with the integral

        self.mesh.update_lvec()
        a_global = self.dm.getGlobalVec()
        self.dm.localToGlobal(self.mesh.lvec, a_global)

        cdef Vec cgvec
        cgvec = a_global

        cdef DM dm = self.dm
        cdef DS ds = self.dm.getDS()
        cdef PetscScalar val_array[256]

        # Now set callback...
        ierr = PetscDSSetObjective(ds.ds, 0, ext.fns_residual[0]); CHKERRQ(ierr)
        ierr = DMPlexComputeIntegralFEM(dm.dm, cgvec.vec, &(val_array[0]), NULL); CHKERRQ(ierr)

        self.dm.restoreGlobalVec(a_global)

        # We're making an assumption here that PetscScalar is same as double.
        # Need to check where this may not be the case.
        cdef double vald = <double> val_array[0]

        # Unit propagation: compute result units from integrand and mesh volume
        # Result units = integrand_units * (coordinate_units)^dim
        #
        # IMPORTANT: The integral is computed in ND space by PETSc, so the raw value
        # is nondimensional. We need to re-dimensionalize it using the reference scales.
        try:
            integrand_units = underworld3.get_units(self.fn)
            coord_units = underworld3.get_units(self.mesh.X[0])

            # Import ureg early for dimensionless check
            from underworld3.scaling import units as ureg

            # Helper: check if units are "meaningful" (not None, not dimensionless)
            def has_meaningful_units(u):
                if u is None:
                    return False
                # Check if dimensionless - treat as "no units" for backward compatibility
                try:
                    if u == ureg.dimensionless:
                        return False
                except:
                    pass
                return True

            integrand_has_units = has_meaningful_units(integrand_units)
            coord_has_units = has_meaningful_units(coord_units)

            # Only attach units if BOTH integrand and coordinates have meaningful units,
            # OR if just one has meaningful units (then we use dimensionless for the other).
            # But if NEITHER has meaningful units, return plain float for backward compatibility.
            if integrand_has_units or coord_has_units:

                if coord_has_units:
                    volume_units = coord_units ** self.mesh.dim
                else:
                    # No coordinate units - don't add dimensionless volume units
                    # Just use integrand units if present
                    volume_units = None

                if integrand_has_units and volume_units is not None:
                    result_units = integrand_units * volume_units
                elif integrand_has_units:
                    # Integrand has units but coordinates don't - just use integrand units
                    result_units = integrand_units
                elif volume_units is not None:
                    # Coordinates have units but integrand doesn't - use volume units
                    result_units = volume_units
                else:
                    # Neither has meaningful units - return float
                    return vald

                # The raw value (vald) is computed in ND coordinates by PETSc.
                # Both the integrand and the domain are nondimensionalized:
                #   - Integrand 500 K → 0.5 (T_scale = 1000 K)
                #   - Domain 1 km² → 1 (L_scale = 1 km)
                #   - ND result = 0.5 × 1 = 0.5
                #
                # To get physical result in result_units (K·km²):
                #   - Need to multiply ND result by integrand scale
                #   - Volume scale is already encoded in coord_units
                #   - physical_value = vald × integrand_scale (in the units of integrand_units)
                #
                # Example: vald=0.5, integrand_scale=1000 K (in K), result_units=K·km²
                #   - physical_value = 0.5 × 1000 = 500
                #   - Result = 500 K·km² ✓

                physical_value = vald

                # Scale by integrand reference if ND scaling is active and integrand has meaningful units
                if underworld3.is_nondimensional_scaling_active() and integrand_has_units:
                    try:
                        model = underworld3.get_default_model()
                        if model.has_units():
                            # Get scale for integrand dimensionality only
                            integrand_dimensionality = (1 * integrand_units).dimensionality
                            integrand_scale = model.get_scale_for_dimensionality(integrand_dimensionality)
                            # Convert scale to the target integrand units
                            integrand_scale_converted = integrand_scale.to(integrand_units)
                            physical_value = vald * float(integrand_scale_converted.magnitude)
                    except Exception:
                        # If scaling fails, use raw value (will be ND)
                        pass

                # Return UWQuantity with computed units
                return underworld3.quantity(physical_value, result_units)
        except Exception:
            # If unit computation fails for any reason, fall back to plain float
            pass

        return vald


class CellWiseIntegral:
    """
    Compute volume integrals over each mesh cell individually.

    The ``CellWiseIntegral`` class constructs cell-by-cell volume integrals:

    .. math:: F_c  =   \\int_{V_c} \\, f(\\mathbf{x}) \\, \\mathrm{d} V

    for some scalar function :math:`f` over each cell volume :math:`V_c`.

    Unlike :class:`Integral` which returns a single scalar over the entire
    mesh domain, this class returns an array with one value per mesh cell.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        The mesh over which integration is performed.
    fn : float, int, or sympy.Basic
        Function to be integrated.

    See Also
    --------
    Integral : For domain-wide (global) volume integrals.
    """

    @timing.routine_timer_decorator
    def __init__( self,
                  mesh:  underworld3.discretisation.Mesh,
                  fn:    Union[float, int, sympy.Basic] ):

        self.mesh = mesh
        self.fn = sympy.sympify(fn)
        super().__init__()

    @timing.routine_timer_decorator
    def evaluate(self) -> float:
        """
        Evaluate the cell-wise integral and return results per cell.

        Returns
        -------
        ndarray
            Array of integral values, one per mesh cell.

        Raises
        ------
        RuntimeError
            If the mesh has no variables (PETSc limitation).
            If the integrand is a Vector or Dyadic (not supported).
        """
        if len(self.mesh.vars)==0:
            raise RuntimeError("The mesh requires at least a single variable for integration to function correctly.\n"
                               "This is a PETSc limitation.")

        # Create JIT extension.
        # Note that we pass in the mesh variables as primary variables, as this
        # is how they are represented on the mesh DM.

        # Note that (at this time) PETSc does not support vector integrands, so
        # if we wish to do vector integrals we'll need to split out the components
        # and calculate them individually. Let's support only scalars for now.
        if isinstance(self.fn, sympy.vector.Vector):
            raise RuntimeError("Integral evaluation for Vector integrands not supported.")
        elif isinstance(self.fn, sympy.vector.Dyadic):
            raise RuntimeError("Integral evaluation for Dyadic integrands not supported.")

        cdef PtrContainer ext = getext(self.mesh, JITCallbackSet(residual=(self.fn,)),
                                       self.mesh.vars.values()).ptrobj

        # Pull out vec for variables, and go ahead with the integral
        self.mesh.update_lvec()
        a_global = self.mesh.dm.getGlobalVec()
        self.mesh.dm.localToGlobal(self.mesh.lvec, a_global)
        cdef Vec cgvec
        cgvec = a_global

        ## Does this need to be consistent with everything else ?

        cdef DM dmc = self.mesh.dm.clone()
        cdef FE fec = FE().createDefault(self.dim, 1, False, -1)
        dmc.setField(0, fec)
        dmc.createDS()

        cdef DS ds = dmc.getDS()
        CHKERRQ( PetscDSSetObjective(ds.ds, 0, ext.fns_residual[0]) )

        cdef Vec rvec = dmc.createGlobalVec()
        CHKERRQ( DMPlexComputeCellwiseIntegralFEM(dmc.dm, cgvec.vec, rvec.vec, NULL) )
        self.mesh.dm.restoreGlobalVec(a_global)

        results = rvec.array.copy()
        rvec.destroy()

        return results


class BdIntegral:
    r"""
    The ``BdIntegral`` class constructs the boundary (surface/line) integral

    .. math:: F  =  \int_{\partial \Omega} \, f(\mathbf{x}, \mathbf{n}) \, \mathrm{d} S

    for some scalar function :math:`f` over a named boundary :math:`\partial \Omega`
    of the mesh. In 2D this is a line integral; in 3D a surface integral.

    The integrand may reference the outward unit normal via ``mesh.Gamma_N``
    (components map to ``petsc_n[0], petsc_n[1], ...`` in the generated C code).

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        The mesh whose boundary is integrated over.
    fn : float, int, or sympy.Basic
        Scalar function to be integrated.
    boundary : str
        Name of the boundary label (e.g. ``"Top"``, ``"Bottom"``, ``"Internal"``).
        Must be present in ``mesh.boundaries``.

    Example
    -------
    Calculate length of the top boundary of a unit box:

    >>> import underworld3 as uw
    >>> import numpy as np
    >>> mesh = uw.discretisation.Box()
    >>> bdIntegral = uw.maths.BdIntegral(mesh=mesh, fn=1., boundary="Top")
    >>> np.allclose( 1., bdIntegral.evaluate(), rtol=1e-8)
    True

    See Also
    --------
    Integral : For volume integrals over the entire mesh domain.
    """

    @timing.routine_timer_decorator
    def __init__( self,
                  mesh:     underworld3.discretisation.Mesh,
                  fn:       Union[float, int, sympy.Basic],
                  boundary: str ):

        self.mesh = mesh
        self.fn = sympy.sympify(fn)
        self.boundary = boundary

        # Validate boundary name
        if boundary not in [b.name for b in mesh.boundaries]:
            available = [b.name for b in mesh.boundaries]
            raise ValueError(
                f"Boundary '{boundary}' not found on mesh. "
                f"Available boundaries: {available}"
            )

        super().__init__()

    @timing.routine_timer_decorator
    def evaluate(self, verbose=False):
        """
        Evaluate the boundary integral and return the result.

        Returns
        -------
        float or UWQuantity
            The boundary integral value. If the integrand has units AND the
            mesh coordinates have units, returns a UWQuantity with proper
            dimensional analysis (integrand_units * coord_units^(dim-1)).
            Otherwise returns a plain float.
        """
        if len(self.mesh.vars) == 0:
            raise RuntimeError(
                "The mesh requires at least a single variable for integration "
                "to function correctly.\nThis is a PETSc limitation."
            )

        if isinstance(self.fn, sympy.vector.Vector):
            raise RuntimeError("BdIntegral evaluation for Vector integrands not supported.")
        elif isinstance(self.fn, sympy.vector.Dyadic):
            raise RuntimeError("BdIntegral evaluation for Dyadic integrands not supported.")

        mesh = self.mesh

        # Compile integrand using the boundary residual slot (includes petsc_n[] in signature)
        _getext_result = getext(
            self.mesh, JITCallbackSet(bd_residual=(self.fn,)),
            self.mesh.vars.values(), verbose=verbose)
        cdef PtrContainer ext = _getext_result.ptrobj

        # Prepare the solution vector
        self.mesh.update_lvec()
        a_global = mesh.dm.getGlobalVec()
        mesh.dm.localToGlobal(self.mesh.lvec, a_global)

        cdef Vec cgvec = a_global
        cdef DM dm_c = mesh.dm

        # Get the DMLabel and label value for the named boundary
        boundary_enum = mesh.boundaries[self.boundary]
        cdef PetscInt label_val = boundary_enum.value
        cdef PetscInt num_vals = 1

        c_label = mesh.dm.getLabel(self.boundary)

        # Output value
        cdef PetscScalar result = 0.0

        # Label can be None if this boundary has no facets on this process
        # (normal in parallel) or if the DM doesn't carry the label at all.
        # The C wrapper handles NULL labels gracefully (contributes 0 to the
        # MPI reduction so all ranks still participate).
        cdef PetscDMLabel c_dmlabel = NULL
        cdef DMLabel dmlabel
        if c_label:
            dmlabel = c_label
            c_dmlabel = dmlabel.dmlabel

        # Call the boundary integral
        ierr = UW_DMPlexComputeBdIntegral(
            dm_c.dm, cgvec.vec,
            c_dmlabel, num_vals, &label_val,
            ext.fns_bd_residual[0],
            &result, NULL
        )
        CHKERRQ(ierr)

        mesh.dm.restoreGlobalVec(a_global)

        cdef double vald = <double> result

        # Unit propagation: result_units = integrand_units * coord_units^(dim-1)
        # Boundary integral is over a codimension-1 manifold.
        try:
            integrand_units = underworld3.get_units(self.fn)
            coord_units = underworld3.get_units(self.mesh.X[0])

            from underworld3.scaling import units as ureg

            def has_meaningful_units(u):
                if u is None:
                    return False
                try:
                    if u == ureg.dimensionless:
                        return False
                except:
                    pass
                return True

            integrand_has_units = has_meaningful_units(integrand_units)
            coord_has_units = has_meaningful_units(coord_units)

            if integrand_has_units or coord_has_units:

                if coord_has_units:
                    surface_units = coord_units ** (self.mesh.dim - 1)
                else:
                    surface_units = None

                if integrand_has_units and surface_units is not None:
                    result_units = integrand_units * surface_units
                elif integrand_has_units:
                    result_units = integrand_units
                elif surface_units is not None:
                    result_units = surface_units
                else:
                    return vald

                physical_value = vald

                if underworld3.is_nondimensional_scaling_active() and integrand_has_units:
                    try:
                        orchestration_model = underworld3.get_default_model()
                        if orchestration_model.has_units():
                            integrand_dimensionality = (1 * integrand_units).dimensionality
                            integrand_scale = orchestration_model.get_scale_for_dimensionality(integrand_dimensionality)
                            integrand_scale_converted = integrand_scale.to(integrand_units)
                            physical_value = vald * float(integrand_scale_converted.magnitude)
                    except Exception:
                        pass

                return underworld3.quantity(physical_value, result_units)
        except Exception:
            pass

        return vald
