r"""Materials carried by particles.

The particle half of the material system: a :class:`MaterialSwarm` is a
:class:`~underworld3.swarm.Swarm` that carries a
:class:`~underworld3.materials.MaterialRegistry`, so the materials advect with
the flow. Use it when the material moves; use
:class:`~underworld3.materials.MaterialRegions` when it does not.

    materials = uw.swarm.MaterialSwarm(mesh, fill_param=3)

    materials.add("mantle", shear_viscosity_0=1.0,   density=3300)
    materials.add("slab",   shear_viscosity_0=1.0e3, density=3400)

    materials["slab"] = mesh.X[1] > 0.53

    stokes.materials = materials
    stokes.bodyforce = -materials.density * mesh.CoordinateSystem.unit_e_1

Everything about *what* a material is — declaring it, its properties, blending
them into the symbol a solver reads — lives in
:mod:`underworld3.materials` and is shared with the mesh-region distribution.
What is here is *where*: the particles, the integer label they carry, and the
level sets built from it.

Why a partition of unity, when the masks are 0 or 1
---------------------------------------------------
If every material's property were a *number*, the level sets would be
redundant: exactly one mask is 1 at each integration point, so
:math:`\sum_i \phi_i \eta_i` is a select, and one stored coefficient field
would do the same job. The masks earn their place the moment a property is a
*law* —

    materials.add("crust", shear_viscosity_0=eta_0 * sympy.exp(-T.sym[0]))

— because there is then no number to store, and the only way to combine N
expressions into one symbol the assembler can compile is the weighted sum.
That is also why a material property must not be evaluated on the particles
and sampled: the solver needs the law, not an answer to it.
"""

import numpy as np
import sympy

import underworld3 as uw
from underworld3.materials import MaterialDistribution, _condition_mask
from underworld3.swarm import IndexSwarmVariable, Swarm

__all__ = ["MaterialSwarm"]


class MaterialSwarm(MaterialDistribution, Swarm):
    """A swarm that carries materials.

    Parameters
    ----------
    mesh : Mesh
        The mesh the particles live on.
    registry : MaterialRegistry, optional
        Where the materials are defined. A fresh one is made if omitted, and
        :meth:`add` writes into it; pass one to share definitions with another
        model or with a :class:`~underworld3.materials.MaterialRegions`.
    fill_param : int, default 3
        Particles per cell at population. The swarm is populated lazily, the
        first time anything needs particles, so that every ``add`` can happen
        first; call :meth:`populate` to force it.
    proxy_sampling : {"nearest", "share"}, default "nearest"
        How each integration point reads the particles. ``"nearest"`` gives
        each point the material of its nearest particle — masks are exactly 0
        or 1, nothing is mixed and no mixing rule is implied. ``"share"``
        gives fractional masks in a cell the interface crosses; use it only
        when the material genuinely *is* a sub-cell mixture, and set the
        mixing rule deliberately (see
        :meth:`~underworld3.materials.MaterialDistribution.mixing`).
    proxy_location : {"integration_points", "cells", "nodes"}
        Where the level sets live. The default reads the material where the
        assembler evaluates the weak form. ``"cells"`` is the one to choose if
        a solve needs the *gradient* of a material property.
    name : str, optional
        Base name for the underlying index variable and its level sets.
        Defaults to ``material``, then ``material_1`` and so on, so two
        distributions on one mesh do not collide.

    Examples
    --------
    >>> materials = uw.swarm.MaterialSwarm(mesh, fill_param=3)
    >>> materials.add("mantle", shear_viscosity_0=1.0,   density=3300)
    >>> materials.add("slab",   shear_viscosity_0=1.0e3, density=3400)
    >>> materials["slab"] = mesh.X[1] > 0.53
    >>> stokes.materials = materials

    It is a swarm, so it also advects and repopulates:

    >>> materials.population_control = dict(min_per_cell=8)
    >>> materials.advection(v.sym, dt)

    See Also
    --------
    underworld3.materials.MaterialRegions : the same interface, tied to the mesh.
    underworld3.materials.MaterialRegistry : where the definitions live.
    """

    def __init__(
        self,
        mesh,
        registry=None,
        fill_param=3,
        proxy_sampling="nearest",
        proxy_location="integration_points",
        name=None,
        recycle_rate=0,
        verbose=False,
        clip_to_mesh=True,
    ):
        # Material bookkeeping BEFORE the Swarm constructor: Swarm touches
        # attributes that __getattr__ would otherwise try to resolve as a
        # material property.
        self._material_pending = []
        self._material_fill_param = fill_param
        self._material_proxy_sampling = proxy_sampling
        self._material_proxy_location = proxy_location
        self._index_var = None
        self._init_distribution(registry=registry, name=name)

        Swarm.__init__(
            self, mesh, recycle_rate=recycle_rate, verbose=verbose,
            clip_to_mesh=clip_to_mesh,
        )

    # -- the index variable, and the particles ----------------------------

    @property
    def index(self):
        """The underlying :class:`~underworld3.swarm.IndexSwarmVariable`.

        The machinery, exposed for the cases this interface does not cover.
        A model should not normally need it.
        """
        self._ensure_built()
        return self._index_var

    def _check_can_declare(self, name):
        if self._index_var is not None:
            raise RuntimeError(
                f"cannot add material {name!r}: the swarm is already populated "
                f"with {len(self._registry)} materials and their level sets. "
                "Declare every material before the first read (or before "
                "calling populate)."
            )

    def populate(self, fill_param=None):
        """Create the particles and the level sets, and apply any painting.

        Called automatically the first time anything reads the materials; call
        it explicitly to fix the moment, or to override ``fill_param``.
        """
        if fill_param is not None:
            self._material_fill_param = fill_param
        self._ensure_built()
        return self

    def _ensure_built(self):
        if self._index_var is not None:
            return
        if len(self._registry) == 0:
            raise RuntimeError(
                "no materials have been declared — call add() before reading "
                "a material property"
            )

        _vars_before = len(self.mesh.vars)
        self._index_var = IndexSwarmVariable(
            self._distribution_name,
            self,
            indices=len(self._registry),
            proxy_location=self._material_proxy_location,
            proxy_sampling=(
                self._material_proxy_sampling
                if self._material_proxy_location == "integration_points"
                else None
            ),
        )
        # local_size is -1, not 0, on a swarm that has never been populated.
        if self.local_size <= 0:
            Swarm.populate(self, fill_param=self._material_fill_param)

        self._check_level_sets_are_new(
            self.mesh, _vars_before, len(self._registry))
        self._registry._built.append(self)

        pending, self._material_pending = self._material_pending, []
        for definition, region in pending:
            self._paint(definition, region)

    def _level_sets(self):
        return self._index_var._meshLevelSetVars

    # -- painting ---------------------------------------------------------

    def __setitem__(self, name, region):
        """Put a material wherever ``region`` is true.

        ``region`` may be a symbolic condition on the mesh coordinates
        (``mesh.X[1] > 0.53``, with ``&``, ``|``, ``~``), a boolean array over
        the particles, or a callable of the particle coordinate array.
        Painting is ordered and cumulative: a later region overwrites an
        earlier one where they overlap.
        """
        definition = self._registry[name]
        if self._index_var is None:
            # Deferred so that every add() can still happen; applied in order
            # at population time.
            self._material_pending.append((definition, region))
        else:
            self._paint(definition, region)

    def _paint(self, definition, region):
        """Give ``definition`` exactly ``region``, and nothing else — see
        ``MaterialRegions._paint``; assignment replaces rather than unions."""
        selected = self._region_mask(region)
        labels = np.asarray(self._index_var.data).reshape(-1)
        released = (labels == definition.index) & ~selected
        with uw.synchronised_array_update():
            self._index_var.data[selected, 0] = definition.index
            if released.any():
                self._index_var.data[released, 0] = 0

    def _region_mask(self, region):
        """A boolean array over the particles."""
        coords = np.asarray(self._particle_coordinates.data)
        n = coords.shape[0]

        if isinstance(region, np.ndarray):
            selected = np.asarray(region).reshape(-1)
            if selected.dtype != bool or selected.shape[0] != n:
                raise ValueError(
                    f"a region given as an array must be a boolean array of "
                    f"length {n} (the local particle count), not "
                    f"{selected.dtype} of length {selected.shape[0]}"
                )
            return selected

        if isinstance(region, sympy.Basic):
            return _condition_mask(region, coords)

        if callable(region):
            return np.asarray(region(coords), dtype=bool).reshape(-1)

        raise TypeError(
            "a region must be a symbolic condition on the mesh coordinates "
            "(e.g. mesh.X[1] > 0.53), a boolean array over the particles, or "
            f"a callable of the coordinate array — not {type(region).__name__}"
        )
