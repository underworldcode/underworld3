r"""What Underworld3 can solve, discovered from the classes.

Every solver, constitutive model and history family describes itself at
the class level (``describe_class()``): the equation it declares, the terms
it is given, the conditions it accepts, its parameters, its documentation.
:func:`capabilities` gathers those into one record, so a notebook, a note
and the MCP server read the same catalogue, and none of it is written by
hand.

    uw.view(uw.capabilities())                     # every family, one line each
    uw.view(uw.capabilities("solvers", detail="full"), depth=2)
    uw.capabilities("constitutive_models")["children"][0]["children"]
"""

import inspect

from .describe import record

GROUPS = ("solvers", "constitutive_models", "histories")


def families():
    """Every solver, constitutive model and history family the package
    exports, by public name: ``{"solvers": {...}, "constitutive_models":
    {...}, "histories": {...}}``."""
    import underworld3 as uw
    from underworld3.cython.generic_solvers import SolverBaseClass

    out = {group: {} for group in GROUPS}
    for name, obj in vars(uw.systems).items():
        if inspect.isclass(obj) and issubclass(obj, SolverBaseClass) and not name.startswith("SNES_"):
            out["solvers"][name] = obj
    base = uw.constitutive_models.Constitutive_Model
    for name, obj in vars(uw.constitutive_models).items():
        if inspect.isclass(obj) and issubclass(obj, base) and obj is not base:
            out["constitutive_models"][name] = obj
    for name, obj in vars(uw.systems.ddt).items():
        if inspect.isclass(obj) and issubclass(obj, uw.systems.ddt._DDtBase) and not name.startswith("_"):
            out["histories"][name] = obj
    return out


def _summary_row(name, description):
    """A family reduced to what a catalogue line needs."""
    facts = dict(description.get("facts") or {})
    facts.pop("public name", None)
    if description.get("forms"):
        facts["equation"] = ", ".join(f"{k}: {v.get('description') or v.get('symbol')}"
                                      for k, v in description["forms"].items())
    if description.get("terms"):
        facts["given"] = [t["name"] for t in description["terms"]]
    if description.get("conditions"):
        facts["conditions"] = [c.get("mechanism") for c in description["conditions"]]
    return record(description.get("kind", "family"), name, description.get("summary", ""),
                  facts={"class": description.get("name"), **facts})


def capabilities(kind="all", detail="summary"):
    """The catalogue as a description record: a child per group, and a
    child per family under it. ``kind`` is ``"all"`` or one of
    ``"solvers"``, ``"constitutive_models"``, ``"histories"``. With
    ``detail="summary"`` each family is one line with its equation names,
    terms and conditions; with ``"full"`` each is its whole
    ``describe_class()`` record, documentation included."""
    found = families()
    if kind != "all" and kind not in found:
        raise ValueError(f"kind must be 'all' or one of {GROUPS}, not {kind!r}")
    if detail not in ("summary", "full"):
        raise ValueError("detail must be 'summary' or 'full'")
    groups = []
    for group in GROUPS:
        if kind not in ("all", group):
            continue
        members = []
        for name, cls in sorted(found[group].items()):
            description = cls.describe_class()
            members.append(description if detail == "full" else _summary_row(name, description))
        groups.append(record(group, None, f"{len(members)} {group.replace('_', ' ')}", children=members))
    total = sum(len(g["children"]) for g in groups)
    return record("capabilities", "underworld3",
                  f"{total} families: " + ", ".join(f"{len(g['children'])} {g['kind'].replace('_', ' ')}"
                                                   for g in groups),
                  children=groups)


def family(name):
    """One family's full class-level description, by public name or class
    name, or ``None``."""
    for members in families().values():
        cls = members.get(name)
        if cls is None:
            cls = next((c for c in members.values() if c.__name__ == name), None)
        if cls is not None:
            return cls.describe_class()
    return None
