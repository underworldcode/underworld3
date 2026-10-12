"""Capability guides are documentation, and nothing else copies them.

Each curated guide is one page under docs/developer/guides with front
matter naming the families it applies to. The AI skills in .claude/skills
are symlinks to those pages, uw.capabilities() lists a guide beside its
family, and a family's class-level description names its guides. This
file enforces the single source: a copied skill, a guide without front
matter, or a family no class carries, fails here.
"""
import os
import pathlib

import pytest
import yaml

import underworld3 as uw
from underworld3.utilities.capabilities import families, guides, guide_text, guides_for

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

ROOT = pathlib.Path(__file__).resolve().parent.parent
GUIDES = ROOT / "docs" / "developer" / "guides"
SKILLS = ROOT / ".claude" / "skills"


def _front_matter(path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---"), f"{path.name}: no front matter"
    return yaml.safe_load(text.split("---", 2)[1]) or {}


def test_every_skill_is_a_symlink_into_the_guides():
    skills = sorted(p for p in SKILLS.glob("*/SKILL.md"))
    assert skills, "no skills found"
    for skill in skills:
        assert skill.is_symlink(), f"{skill} is a copy; make it a symlink into docs/developer/guides"
        target = (skill.parent / os.readlink(skill)).resolve()
        assert target.parent == GUIDES.resolve() and target.exists(), (skill, target)


def test_every_guide_names_real_families():
    known = {name for group in families().values() for name in group}
    known |= {cls.__name__ for group in families().values() for cls in group.values()}
    found = guides()
    assert {"transport-schemes", "boundary-condition-rulings", "nonlinear-solver"} <= set(found)
    for name, g in found.items():
        meta = _front_matter(pathlib.Path(g["path"]))
        assert meta.get("name") == name and meta.get("description"), name
        unknown = set(g["families"]) - known
        assert not unknown, f"guide {name} names families no class carries: {sorted(unknown)}"


def test_families_list_their_guides():
    assert "boundary-condition-rulings" in guides_for("Stokes")
    assert "transport-schemes" in guides_for("SemiLagrangian")
    stokes = uw.systems.Stokes.describe_class()
    assert "nonlinear-solver" in stokes["facts"]["guides"]
    cat = uw.capabilities("solvers")
    row = next(c for c in cat["children"][0]["children"] if c["name"] == "Stokes")
    assert "boundary-condition-rulings" in row["facts"]["guides"]
    text = guide_text("transport-schemes")
    assert text.startswith("# Transport schemes") and guide_text("nothing") is None
