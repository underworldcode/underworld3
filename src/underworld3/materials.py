r"""Materials: what they are, and where they are.

Two questions, kept apart on purpose.

**What a material is** is a :class:`MaterialRegistry` entry — a name and a table
of properties, where a property may be a number, a dimensional quantity, or a
*law* (``eta_0 * sympy.exp(-T.sym[0])``). Definitions carry description and
reference metadata, export and import as plain dictionaries, and say nothing
about geometry, so one registry can serve several models.

**Where a material is** is a *distribution*. Underworld3 has two:

- :class:`~underworld3.swarm.MaterialSwarm` — carried by particles, so it
  advects with the flow. The one to use when the material moves.
- :class:`MaterialRegions` — tied to the mesh, from gmsh physical groups or
  from a geometric condition. Exact, needs no particles and no population
  control. The one to use when the material does not move.

Both present the same face to a solver::

    stokes.materials = materials

which sets every constitutive-model parameter the materials declare and the
model recognises, by name. Both build that from the same partition of unity —
one level set per material — because a property that is a law cannot be stored
as a value and has to be combined symbolically.

Example
-------
>>> rocks = uw.MaterialRegistry()
>>> rocks.add("mantle", shear_viscosity_0=1.0,   density=3300)
>>> rocks.add("slab",   shear_viscosity_0=1.0e3, density=3400)
>>>
>>> materials = uw.swarm.MaterialSwarm(mesh, registry=rocks, fill_param=3)
>>> materials["slab"] = mesh.X[1] > 0.53
>>> stokes.materials = materials

or, for a mesh that carries the geometry itself,

>>> materials = uw.MaterialRegions(mesh, registry=rocks)
>>> materials["slab"] = "Slab"            # a gmsh physical group
>>> stokes.materials = materials
"""

import warnings
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import sympy

import underworld3 as uw

__all__ = [
    "MaterialProperty",
    "MaterialDefinition",
    "MaterialRegistry",
    "MaterialDistribution",
    "BoundMaterial",
    "MaterialRegions",
    "create_standard_mantle_material",
    "create_standard_crust_material",
    "create_high_viscosity_material",
]

_MIXING_RULES = ("arithmetic", "harmonic")


class MaterialProperty(Enum):
    """Names for the properties a geodynamic material usually carries.

    The *value* of each member is the property name, and it is the name a
    constitutive model knows it by — ``MaterialProperty.VISCOSITY`` is
    ``"viscosity"``, which is the established alias for
    ``shear_viscosity_0``. Using the enum is optional; a plain keyword is
    equivalent and is what most models do.
    """

    # Mechanical
    VISCOSITY = "viscosity"
    DENSITY = "density"
    YIELD_STRESS = "yield_stress"
    COHESION = "cohesion"
    FRICTION_ANGLE = "friction_angle"

    # Thermal
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    THERMAL_DIFFUSIVITY = "thermal_diffusivity"
    HEAT_CAPACITY = "heat_capacity"
    THERMAL_EXPANSION = "thermal_expansion"

    # Elastic
    YOUNGS_MODULUS = "youngs_modulus"
    POISSONS_RATIO = "poissons_ratio"
    SHEAR_MODULUS = "shear_modulus"

    # Flow
    PERMEABILITY = "permeability"
    POROSITY = "porosity"


def _property_name(prop):
    return prop.value if isinstance(prop, MaterialProperty) else str(prop)


def _as_symbolic(value, name, material):
    """Coerce a declared property value into something sympy can carry."""
    if isinstance(value, sympy.Basic):
        return value
    try:
        return sympy.sympify(value)
    except (TypeError, ValueError, AttributeError):
        pass
    try:                                    # a dimensional quantity
        return sympy.sympify(uw.scaling.non_dimensionalise(value))
    except Exception as exc:                # noqa: BLE001 - name the value
        raise TypeError(
            f"material {material!r}: property {name!r} = {value!r} is neither a "
            "number, a sympy expression, nor a quantity that can be "
            "non-dimensionalised"
        ) from exc


class MaterialDefinition:
    """What a material is: a name, a property table, and provenance.

    Created by :meth:`MaterialRegistry.add`. Says nothing about where the
    material is — that is a distribution's job.

    Attributes
    ----------
    name : str
    index : int
        Position in the registry, and therefore which level set is this
        material's in any distribution built from it.
    properties : dict
        Property name -> value. Values are sympy objects: a number, or a law.
    description, reference : str
        Free text, carried through export/import.
    """

    def __init__(self, registry, name, index, properties=None,
                 description="", reference=""):
        self._registry = registry
        self.name = name
        self.index = index
        self.description = description
        self.reference = reference
        self.properties = {}
        if properties:
            self.set(**properties)

    def set(self, **properties):
        """Set or replace properties, and re-push them to attached solvers."""
        for key, value in properties.items():
            key = _property_name(key)
            self.properties[key] = _as_symbolic(value, key, self.name)
        self._registry._changed()
        return self

    # -- the pre-2026 spelling, kept working ------------------------------
    def set_property(self, prop, value):
        """Set one property. ``set(**{name: value})`` is the shorter form."""
        return self.set(**{_property_name(prop): value})

    def get_property(self, prop, default=None):
        return self.properties.get(_property_name(prop), default)

    def has_property(self, prop) -> bool:
        return _property_name(prop) in self.properties

    def __repr__(self):
        props = ", ".join(f"{k}={v}" for k, v in self.properties.items())
        return f"<Material {self.name!r} [{self.index}] {props}>"


class MaterialRegistry:
    """A set of material definitions, independent of any mesh or swarm.

    A registry is the *what*: it can be built before there is a mesh, shared
    between models, exported to a dictionary and read back. Attach it to a
    distribution — a :class:`~underworld3.swarm.MaterialSwarm` or a
    :class:`MaterialRegions` — to say where each material is.

    Examples
    --------
    >>> rocks = uw.MaterialRegistry()
    >>> rocks.add("mantle", shear_viscosity_0=1.0, density=3300)
    >>> rocks.add("slab", shear_viscosity_0=1.0e3, density=3400)
    >>> rocks.list_materials()
    ['mantle', 'slab']
    """

    def __init__(self, materials=None):
        self._materials: Dict[str, MaterialDefinition] = {}
        self._order: List[str] = []
        self._callbacks = []
        self._version = 0
        if materials:
            for name, properties in dict(materials).items():
                self.add(name, **properties)

    # -- declaring ---------------------------------------------------------

    def add(self, name, description="", reference="", **properties):
        """Declare a material.

        Parameters
        ----------
        name : str
        description, reference : str, optional
            Free text carried through export/import.
        **properties
            Property values, by the name a constitutive model knows them by
            (``shear_viscosity_0``, ``density``, ...). A value may be a number,
            a quantity, or a symbolic law.

        Returns
        -------
        MaterialDefinition
        """
        if name in self._materials:
            raise ValueError(f"material {name!r} has already been declared")
        material = MaterialDefinition(
            self, name, len(self._order), properties, description, reference
        )
        self._materials[name] = material
        self._order.append(name)
        self._changed()
        return material

    def create_material(self, name, description="", reference=""):
        """Declare a material with no properties yet (the pre-2026 spelling)."""
        return self.add(name, description=description, reference=reference)

    def get_material(self, name) -> Optional[MaterialDefinition]:
        return self._materials.get(name)

    def list_materials(self) -> List[str]:
        """Material names, in declaration order — which is level-set order."""
        return list(self._order)

    def delete_material(self, name):
        """Remove a material, and reindex the rest.

        Only safe before a distribution has been built from the registry: the
        index of a material IS which level set is its, so removing one
        renumbers the others.
        """
        if name not in self._materials:
            return
        del self._materials[name]
        self._order.remove(name)
        for i, key in enumerate(self._order):
            self._materials[key].index = i
        self._changed()

    # -- reading -----------------------------------------------------------

    @property
    def materials(self):
        """The definitions, in declaration order."""
        return tuple(self._materials[name] for name in self._order)

    def declared_properties(self):
        """Every property name declared by any material."""
        names = set()
        for material in self.materials:
            names.update(material.properties)
        return names

    def __len__(self):
        return len(self._order)

    def __iter__(self):
        return iter(self.materials)

    def __contains__(self, name):
        return name in self._materials

    def __getitem__(self, name):
        try:
            return self._materials[name]
        except KeyError:
            raise KeyError(
                f"no material {name!r}; declared: {self.list_materials()}"
            ) from None

    def __repr__(self):
        return f"MaterialRegistry({self.list_materials()})"

    # -- change notification ----------------------------------------------

    def add_callback(self, callback):
        """Register ``callback()`` to run whenever a definition changes."""
        self._callbacks.append(callback)

    def _changed(self):
        self._version += 1
        for callback in list(self._callbacks):
            try:
                callback()
            except Exception as exc:                     # noqa: BLE001
                warnings.warn(f"material registry callback failed: {exc}")

    # -- serialisation -----------------------------------------------------

    def export_config(self) -> Dict[str, Any]:
        """Export as plain data. Property values are stringified sympy."""
        return {
            "materials": {
                m.name: {
                    "properties": {k: str(v) for k, v in m.properties.items()},
                    "description": m.description,
                    "reference": m.reference,
                }
                for m in self.materials
            },
            "version": self._version,
        }

    def import_config(self, config: Dict[str, Any]):
        """Read back an :meth:`export_config` dictionary."""
        for name, entry in config.get("materials", {}).items():
            material = self.add(
                name,
                description=entry.get("description", ""),
                reference=entry.get("reference", ""),
            )
            material.set(**{
                key: sympy.sympify(value)
                for key, value in entry.get("properties", {}).items()
            })
        return self


class BoundMaterial:
    """A material seen through one distribution.

    What :meth:`MaterialDistribution.__getitem__` returns: the definition plus
    the things that only mean something once you know *where* the material is —
    its level set, and how to paint it.
    """

    def __init__(self, distribution, definition):
        self._distribution = distribution
        self._definition = definition

    @property
    def name(self):
        return self._definition.name

    @property
    def index(self):
        return self._definition.index

    @property
    def properties(self):
        return self._definition.properties

    @property
    def mask(self):
        r"""This material's level set, :math:`\phi_i`, as a symbol.

        1 where the material is, 0 where it is not. Rarely needed — a property
        blend is what a model usually wants — but it is the right thing for a
        per-material diagnostic, e.g. ``uw.maths.Integral(mesh, slab.mask)``
        for the area the material occupies.
        """
        return self._distribution._mask_of(self._definition)

    def occupies(self, region):
        """Put this material wherever ``region`` is true.

        What ``region`` may be depends on the distribution — see
        :meth:`MaterialDistribution.__setitem__`.
        """
        self._distribution[self.name] = region
        return self

    def set(self, **properties):
        """Change properties, and re-push them to attached solvers."""
        self._definition.set(**properties)
        return self

    def __repr__(self):
        return f"<{type(self._distribution).__name__} {self._definition!r}>"


class MaterialDistribution:
    """Shared behaviour of everything that says *where* materials are.

    A distribution owns a :class:`MaterialRegistry` and supplies one level set
    per material. Everything else — blending a property into a symbol, pushing
    it to a solver, the mixing rule, the unused-property check — is the same
    whether the materials ride on particles or on mesh regions, and lives here.

    Subclasses provide ``_ensure_built()`` and ``_level_sets()``.
    """

    #: How many distributions have taken the default name, so that two on one
    #: mesh do not collide over their level-set variable names.
    _default_name_count = 0

    def _init_distribution(self, registry=None, name=None):
        self._registry = registry if registry is not None else MaterialRegistry()
        if name is None:
            n = MaterialDistribution._default_name_count
            MaterialDistribution._default_name_count += 1
            name = "material" if n == 0 else f"material_{n}"
        self._distribution_name = name
        self._solvers = []
        self._mixing_rules = {}
        self._read_properties = set()
        self._registry.add_callback(self._push_all)

    # -- declaring (shorthand onto the registry) --------------------------

    @property
    def registry(self):
        """The :class:`MaterialRegistry` these materials are defined in."""
        return self._registry

    @property
    def materials(self):
        """The declared materials, in order, bound to this distribution."""
        return tuple(BoundMaterial(self, m) for m in self._registry.materials)

    def add(self, name, where=None, description="", reference="", **properties):
        """Declare a material here — shorthand for ``registry.add(...)``.

        Returns a :class:`BoundMaterial`, so the definition and its place in
        this distribution are reachable from one handle. ``where`` is the same
        as assigning a region afterwards.
        """
        self._check_can_declare(name)
        self._registry.add(name, description=description, reference=reference,
                           **properties)
        bound = self[name]
        if where is not None:
            self[name] = where
        return bound

    def _check_can_declare(self, name):
        """Subclass hook: raise if it is too late to add a material."""

    def __getitem__(self, name):
        return BoundMaterial(self, self._registry[name])

    def __len__(self):
        return len(self._registry)

    # -- level sets (subclass) --------------------------------------------

    def _ensure_built(self):
        raise NotImplementedError

    def _level_sets(self):
        """The per-material level-set variables, in registry order."""
        raise NotImplementedError

    def _mask_of(self, definition):
        self._ensure_built()
        return self._level_sets()[definition.index].sym[0]

    # -- properties --------------------------------------------------------

    def mixing(self, **rules):
        """Choose how a property is blended where masks are fractional.

        ``materials.mixing(shear_viscosity_0="harmonic")``. Only meaningful
        where a mask can take a value strictly between 0 and 1; where exactly
        one mask is 1 every rule gives the same answer.

        ``"arithmetic"`` (the default) is a Voigt average: for a flux at a
        common gradient it is the flux blend. On a sharp contrast it does not
        converge with sampling density. ``"harmonic"`` is the Reuss average,
        which does. Pick on physical grounds.
        """
        for name, rule in rules.items():
            name = _property_name(name)
            if rule not in _MIXING_RULES:
                raise ValueError(
                    f"mixing rule for {name!r} must be one of {_MIXING_RULES}, "
                    f"not {rule!r}"
                )
            self._mixing_rules[name] = rule
        self._push_all()
        return self

    def blend(self, name, mixing=None):
        """The material-weighted symbol for property ``name``.

        Usually reached as ``materials.<name>``; call it directly to override
        the mixing rule for one use.
        """
        name = _property_name(name)
        missing = [m.name for m in self._registry.materials
                   if name not in m.properties]
        if missing:
            raise KeyError(
                f"property {name!r} is not declared by {missing}. Every "
                "material must declare a property that is blended — a material "
                "with no viscosity is not a material with zero viscosity."
            )

        self._ensure_built()
        self._read_properties.add(name)
        values = [m.properties[name] for m in self._registry.materials]
        masks = self._level_sets()
        rule = mixing or self._mixing_rules.get(name, "arithmetic")

        if rule == "arithmetic":
            return sum(masks[i].sym[0] * value for i, value in enumerate(values))
        return 1 / sum(masks[i].sym[0] / value for i, value in enumerate(values))

    def __getattr__(self, name):
        # Only reached when normal lookup fails, so this cannot shadow a real
        # attribute. Private names never resolve to a material property.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            registry = object.__getattribute__(self, "_registry")
        except AttributeError:
            raise AttributeError(name) from None
        if not any(name in m.properties for m in registry.materials):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}, and "
                "no material declares a property of that name"
            )
        return self.blend(name)

    # -- the handoff to a solver ------------------------------------------

    def _attach(self, solver):
        """Called by ``solver.materials = ...``."""
        if not any(s is solver for s in self._solvers):
            self._solvers.append(solver)
        self._push_to(solver)

    def _push_all(self):
        for solver in self._solvers:
            self._push_to(solver)

    def _push_to(self, solver):
        """Set every property the solver's constitutive model recognises."""
        model = getattr(solver, "_constitutive_model", None)
        if model is None:
            return                       # pushed again when the model is set
        if len(self._registry) == 0:
            return
        parameters = model.Parameters
        recognised = set(type(parameters)._list_valid_parameters(type(parameters)))
        for name in sorted(self._registry.declared_properties() & recognised):
            setattr(parameters, name, self.blend(name))

    def _recognised(self):
        names = set()
        for solver in self._solvers:
            model = getattr(solver, "_constitutive_model", None)
            if model is None:
                continue
            parameters = model.Parameters
            names |= set(
                type(parameters)._list_valid_parameters(type(parameters))
            )
        return names

    def unclaimed(self):
        """Declared properties no attached model recognises and nobody read.

        Not in itself a problem: ``density`` is unclaimed until the model
        script reads ``materials.density`` for its body force. It is the list
        :meth:`check` looks through for misspellings.
        """
        return sorted(
            self._registry.declared_properties()
            - self._recognised()
            - self._read_properties
        )

    def check(self):
        """Report a declared property that looks like a misspelled parameter.

        Called from the solver build. A property no constitutive model
        recognises is perfectly normal — that is how ``density`` reaches a body
        force. What is not normal is one that *nearly* matches a parameter the
        model does have and that nothing has read: ``viscocity`` is not
        ``viscosity``, it is silently the default viscosity, and nothing else
        will ever say so.
        """
        import difflib

        recognised = sorted(self._recognised())
        suspicious = []
        for name in self.unclaimed():
            close = difflib.get_close_matches(name, recognised, n=1, cutoff=0.8)
            if close:
                suspicious.append((name, close[0]))
        for name, suggestion in suspicious:
            warnings.warn(
                f"material property {name!r} is declared but unused, and the "
                f"constitutive model has a parameter {suggestion!r} — did you "
                f"mean that? As it stands {suggestion!r} keeps its default "
                f"value and nothing reads {name!r}.",
                stacklevel=3,
            )
        return suspicious

    # -- display -----------------------------------------------------------

    def _object_viewer(self):
        uw.pprint(f"{type(self).__name__}: {len(self._registry)} materials")
        for material in self._registry.materials:
            uw.pprint(f"  [{material.index}] {material.name}")
            for key, value in material.properties.items():
                uw.pprint(f"        {key} = {value}")


class MaterialRegions(MaterialDistribution):
    """Materials tied to the mesh rather than to particles.

    For a model whose materials do not move: layers, inclusions, a basin, a
    gmsh model with physical groups. The level sets are exact — a material
    either owns an integration point or it does not — and there are no
    particles to populate, advect or repopulate.

    A region may be given as a mesh label (a gmsh physical group, reaching
    Underworld3 as ``mesh.regions``), or as a geometric condition, which is
    resolved at the integration points and so keeps sub-cell position.

    Parameters
    ----------
    mesh : Mesh
    registry : MaterialRegistry, optional
        Where the materials are defined. A fresh one is made if omitted, and
        :meth:`~MaterialDistribution.add` writes into it.
    name : str, optional
        Base name for the level-set variables. Defaults to ``material``, then
        ``material_1`` and so on, so two distributions on one mesh do not
        collide.

    Examples
    --------
    >>> materials = uw.MaterialRegions(mesh)
    >>> materials.add("mantle", shear_viscosity_0=1.0)
    >>> materials.add("crust",  shear_viscosity_0=1.0e3)
    >>> materials["crust"] = "Crust"            # a gmsh physical group
    >>> materials["crust"] = mesh.X[1] > 0.8    # or a condition
    >>> stokes.materials = materials

    See Also
    --------
    underworld3.swarm.MaterialSwarm : the same interface, carried by particles.
    """

    def __init__(self, mesh, registry=None, name=None):
        self.mesh = mesh
        self._level_set_vars = None
        self._pending = []
        self._init_distribution(registry=registry, name=name)

    # -- building ----------------------------------------------------------

    def _check_can_declare(self, name):
        if self._level_set_vars is not None:
            raise RuntimeError(
                f"cannot add material {name!r}: the level sets are already "
                "built. Declare every material before the first read."
            )

    def _ensure_built(self):
        if self._level_set_vars is not None:
            return
        if len(self._registry) == 0:
            raise RuntimeError(
                "no materials have been declared — call add() before reading "
                "a material property"
            )
        self._level_set_vars = [
            uw.discretisation.IntegrationPointVariable(
                f"{self._distribution_name}^{{[{i}]}}", self.mesh,
            )
            for i in range(len(self._registry))
        ]
        # material 0 owns everything not claimed by anyone else
        self._level_set_vars[0].data[...] = 1.0
        for i in range(1, len(self._level_set_vars)):
            self._level_set_vars[i].data[...] = 0.0

        pending, self._pending = self._pending, []
        for definition, region in pending:
            self._paint(definition, region)

    def _level_sets(self):
        return self._level_set_vars

    # -- painting ----------------------------------------------------------

    def __setitem__(self, name, region):
        """Assign a region to a material.

        ``region`` may be

        - a mesh label name (``"Crust"``), optionally ``(name, value)`` — the
          cells in that label, and every integration point in them;
        - a symbolic condition on the mesh coordinates (``mesh.X[1] > 0.8``,
          with ``&``, ``|``, ``~``), resolved at each integration point;
        - a boolean array over the integration points, or a callable of their
          coordinates.

        Assignment is ordered and cumulative: a later region overwrites an
        earlier one where they overlap.
        """
        definition = self._registry[name]
        if self._level_set_vars is None:
            self._pending.append((definition, region))
        else:
            self._paint(definition, region)

    def _paint(self, definition, region):
        selected = self._region_mask(region)
        for i, var in enumerate(self._level_set_vars):
            values = np.asarray(var.data).reshape(-1).copy()
            values[selected] = 1.0 if i == definition.index else 0.0
            var.data[:, 0] = values

    def _integration_points(self):
        self._ensure_built()
        return np.asarray(self._level_set_vars[0].integration_points)

    def _region_mask(self, region):
        """A boolean array over the flattened integration points."""
        points = self._integration_points()
        ncells, nq, _ = points.shape
        flat = points.reshape(-1, points.shape[-1])

        if isinstance(region, str):
            return self._label_mask(region, None, ncells, nq)
        if (isinstance(region, tuple) and len(region) == 2
                and isinstance(region[0], str)):
            return self._label_mask(region[0], region[1], ncells, nq)
        if isinstance(region, np.ndarray):
            selected = np.asarray(region).reshape(-1)
            if selected.dtype != bool or selected.shape[0] != flat.shape[0]:
                raise ValueError(
                    "a region given as an array must be a boolean array of "
                    f"length {flat.shape[0]} (the integration points), not "
                    f"{selected.dtype} of length {selected.shape[0]}"
                )
            return selected
        if isinstance(region, sympy.Basic):
            return _condition_mask(region, flat)
        if callable(region):
            return np.asarray(region(flat), dtype=bool).reshape(-1)

        raise TypeError(
            "a region must be a mesh label name, a symbolic condition on the "
            "mesh coordinates, a boolean array over the integration points, or "
            f"a callable of their coordinates — not {type(region).__name__}"
        )

    def _label_mask(self, label_name, label_value, ncells, nq):
        """Integration points of the cells carried by a mesh label."""
        dm = self.mesh.dm
        if not dm.hasLabel(label_name):
            available = [dm.getLabelName(i) for i in range(dm.getNumLabels())]
            regions = getattr(self.mesh, "regions", None)
            if regions is not None:
                available += [r.name for r in regions]
            raise KeyError(
                f"the mesh has no label {label_name!r}. Available: "
                f"{sorted(set(available))}"
            )
        if label_value is None:
            regions = getattr(self.mesh, "regions", None)
            if regions is not None and label_name in regions.__members__:
                label_value = regions[label_name].value
            else:
                values = dm.getLabelIdIS(label_name).getIndices()
                if len(values) != 1:
                    raise ValueError(
                        f"label {label_name!r} carries values {list(values)}; "
                        "say which with materials[name] = (label, value)"
                    )
                label_value = int(values[0])

        c_start, c_end = dm.getHeightStratum(0)
        stratum = dm.getStratumIS(label_name, label_value)
        cells = np.asarray(stratum.getIndices()) if stratum is not None else np.zeros(0, int)
        cells = cells[(cells >= c_start) & (cells < c_end)] - c_start

        selected = np.zeros((ncells, nq), dtype=bool)
        selected[cells] = True
        return selected.reshape(-1)


def _condition_mask(condition, coords):
    """Evaluate a symbolic condition at ``coords`` -> boolean array."""
    import sympy.logic.boolalg as boolalg
    from sympy.core.relational import Relational

    n = coords.shape[0]

    if isinstance(condition, boolalg.BooleanTrue):
        return np.ones(n, dtype=bool)
    if isinstance(condition, boolalg.BooleanFalse):
        return np.zeros(n, dtype=bool)
    if isinstance(condition, sympy.And):
        return np.logical_and.reduce(
            [_condition_mask(a, coords) for a in condition.args])
    if isinstance(condition, sympy.Or):
        return np.logical_or.reduce(
            [_condition_mask(a, coords) for a in condition.args])
    if isinstance(condition, sympy.Not):
        return ~_condition_mask(condition.args[0], coords)

    if isinstance(condition, Relational):
        values = np.asarray(
            uw.function.evaluate(condition.lhs - condition.rhs, coords)
        ).reshape(-1)
        if isinstance(condition, sympy.StrictGreaterThan):
            return values > 0.0
        if isinstance(condition, sympy.GreaterThan):
            return values >= 0.0
        if isinstance(condition, sympy.StrictLessThan):
            return values < 0.0
        if isinstance(condition, sympy.LessThan):
            return values <= 0.0
        if isinstance(condition, sympy.Eq):
            return np.isclose(values, 0.0)
        if isinstance(condition, sympy.Ne):
            return ~np.isclose(values, 0.0)

    raise TypeError(
        f"cannot read {condition!r} as a region: expected a comparison of mesh "
        "coordinates, or an &/|/~ combination of them"
    )


# ---------------------------------------------------------------------------
# Convenience definitions. The values are SI; non-dimensionalise them, or set
# reference quantities on the model, before using them in a solve.
# ---------------------------------------------------------------------------

def create_standard_mantle_material(registry: MaterialRegistry):
    """A standard upper-mantle material (Turcotte & Schubert, 2014)."""
    return registry.add(
        "mantle",
        description="Standard upper mantle material",
        reference="Turcotte & Schubert (2014)",
        viscosity=1e21, density=3300,
        thermal_conductivity=3.0, thermal_diffusivity=1e-6,
        thermal_expansion=3e-5,
    )


def create_standard_crust_material(registry: MaterialRegistry):
    """A standard continental-crust material (Turcotte & Schubert, 2014)."""
    return registry.add(
        "crust",
        description="Standard continental crust material",
        reference="Turcotte & Schubert (2014)",
        viscosity=1e22, density=2700,
        thermal_conductivity=2.5, thermal_diffusivity=1e-6,
        thermal_expansion=3e-5,
    )


def create_high_viscosity_material(registry: MaterialRegistry,
                                   name: str = "high_visc",
                                   viscosity_contrast: float = 1000):
    """A stiff inclusion, for viscosity-contrast studies."""
    return registry.add(
        name,
        description=f"High viscosity material (contrast {viscosity_contrast}x)",
        reference="User defined",
        viscosity=1e21 * viscosity_contrast, density=3300,
        thermal_conductivity=3.0, thermal_diffusivity=1e-6,
    )
