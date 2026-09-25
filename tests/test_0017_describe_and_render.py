"""Every core object describes itself as data, and one renderer shows it.

``describe()`` returns a plain tree — kind, name, summary, and facts, terms,
forms, conditions and children as the object has them. ``uw.render`` turns
the tree into Markdown, text, LaTeX, YAML or JSON; ``view()`` is the render
in the form the session wants. The transcript serialises the same tree, so
a notebook, a note, a run record and a query cannot disagree about what an
object is. This file holds the contract: the shared keys are present on
every kind, every format renders, the serial formats round-trip, and the
solver's description keeps the keys the transcript records.
"""
import json

import pytest
import sympy
import yaml

import underworld3 as uw
from underworld3.utilities.describe import FORMATS, plain, render

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

SHARED = ("kind", "name", "summary")


@pytest.fixture(scope="module")
def objects():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 4, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    eta = uw.expression(r"\eta", 1.0, "viscosity")
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta * (1 + x)
    stokes.bodyforce = sympy.Matrix([0, -1])
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_essential_bc((1.0, 0.0), "Top")
    adv = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=v.sym)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0
    swarm = uw.swarm.Swarm(mesh)
    material = uw.swarm.SwarmVariable("M", swarm, 1, proxy_degree=1)
    swarm.populate(fill_param=2)
    model = uw.get_default_model()
    return {"mesh": mesh, "variable": v, "solver": stokes, "constitutive_model": stokes.constitutive_model,
            "history": adv.DuDt, "swarm": swarm, "swarm_variable": material, "model": model}


def test_every_kind_carries_the_shared_keys(objects):
    for kind, obj in objects.items():
        d = obj.describe()
        for key in SHARED:
            assert key in d, (kind, key)
        assert d["kind"] == kind, (kind, d["kind"])
        assert isinstance(d["summary"], str) and d["summary"], kind


def test_every_format_renders_every_kind(objects):
    for kind, obj in objects.items():
        d = obj.describe()
        for fmt in FORMATS:
            text = render(d, fmt)
            assert isinstance(text, str) and text.strip(), (kind, fmt)


def test_the_serial_formats_round_trip(objects):
    d = objects["solver"].describe()
    assert yaml.safe_load(render(d, "yaml")) == plain(d)
    assert json.loads(render(d, "json")) == plain(d)


def test_the_solver_description_keeps_the_transcript_keys(objects):
    d = objects["solver"].describe()
    for key in ("forms", "boundary_conditions", "terms", "terms_declared", "unknown", "dim"):
        assert key in d, key
    assert "F0" in d["forms"] and "F1" in d["forms"]
    # the constitutive model is a child, one level down, and the history of
    # an advection solver names its scheme
    kinds = {child["kind"] for child in d["children"]}
    assert "constitutive_model" in kinds
    history = objects["history"].describe()
    assert history["facts"]["scheme"] == type(objects["history"]).__name__
    assert history["facts"]["order"] == objects["history"].order


def test_markdown_carries_the_equation_and_text_carries_the_symbols(objects):
    d = objects["solver"].describe()
    md = render(d, "markdown")
    assert d["forms"]["F0"]["latex"] in md
    assert "Boundary conditions" in md and "essential on Top" in md
    txt = render(d, "text")
    assert "F0:" in txt and "$" not in txt.split("F0:")[1].split("\n")[0]
    tex = render(d, "latex")
    assert tex.startswith("\\section*{") and "\\[" in tex


def test_depth_prunes_children(objects):
    d = objects["model"].describe(depth=2)
    assert d["children"], "the model holds its meshes, swarms and solvers"
    shallow = render(d, "json", depth=0)
    assert "children" not in json.loads(shallow)


def test_view_prints_outside_a_notebook(objects, capsys):
    for kind, obj in objects.items():
        obj.view()
        out = capsys.readouterr().out
        assert kind.replace("_", " ") in out, kind
    objects["variable"].view(format="yaml")
    assert yaml.safe_load(capsys.readouterr().out)["kind"] == "variable"


def test_the_transcript_part_record_is_still_a_part(tmp_path, objects):
    # the live default model: the test harness resets it between tests, and
    # a solver registers its part with whichever model is current
    model = uw.get_default_model()
    model.transcript_file = str(tmp_path / "run.jsonl")
    stokes = objects["solver"]
    with model.step(0.0, label="describe"):
        stokes.solve()
    records = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    parts = [r for r in records if r.get("kind") == "part"]
    assert parts and "children" not in parts[0] and parts[0]["forms"]
