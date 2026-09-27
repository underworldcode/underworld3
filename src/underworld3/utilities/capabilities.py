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


def guides_directory():
    """The checkout's ``docs/developer/guides``, found upward from the
    working directory or from ``UW_DOCS``; ``None`` outside a checkout."""
    import os
    candidates = []
    if os.environ.get("UW_DOCS"):
        candidates.append(os.path.join(os.environ["UW_DOCS"], "developer", "guides"))
    here = os.path.abspath(os.getcwd())
    while True:
        candidates.append(os.path.join(here, "docs", "developer", "guides"))
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def guides():
    """The capability guides in the checkout: ``{name: {name, description,
    families, kind, path}}`` read from each page's front matter. Empty
    outside a checkout."""
    import glob
    import os
    import yaml
    directory = guides_directory()
    out = {}
    if directory is None:
        return out
    for path in sorted(glob.glob(os.path.join(directory, "*.md"))):
        with open(path, encoding="utf-8") as handle:
            head = handle.read(4000)
        if not head.startswith("---"):
            continue
        parts = head.split("---", 2)
        if len(parts) < 3:
            continue
        try:
            meta = yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(meta, dict) or "families" not in meta:
            continue
        name = str(meta.get("name") or os.path.splitext(os.path.basename(path))[0])
        out[name] = {"name": name, "description": str(meta.get("description") or ""),
                     "families": [str(f) for f in (meta.get("families") or [])],
                     "kind": str(meta.get("kind") or "guide"), "path": path}
    return out


def guides_for(*names):
    """The names of the guides whose ``families`` include any of ``names``
    (a public name or a class name)."""
    wanted = {str(n) for n in names if n}
    return [g["name"] for g in guides().values() if wanted & set(g["families"])]


def guide_text(name):
    """The body of one guide, front matter removed, or ``None``."""
    g = guides().get(name)
    if g is None:
        return None
    with open(g["path"], encoding="utf-8") as handle:
        text = handle.read()
    parts = text.split("---", 2)
    return parts[2].lstrip("\n") if text.startswith("---") and len(parts) == 3 else text


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
    linked = guides_for(name, description.get("name"))
    if linked:
        facts["guides"] = linked
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
            d = cls.describe_class()
            linked = guides_for(name, cls.__name__)
            if linked:
                d.setdefault("facts", {})["guides"] = linked
            return d
    return None
