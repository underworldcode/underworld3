"""The transcript MCP server is a projection of the query object.

Every tool reads a transcript file and returns the query's answer as YAML.
The tests call the tool functions directly on a small recorded run, and
check that the server registers them with read-only annotations.
"""
import asyncio

import pytest
import yaml

import underworld3 as uw

pytest.importorskip("mcp")
from underworld3 import mcp as uwmcp  # noqa: E402

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture(scope="module")
def run_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mcp")
    uw.reset_default_model()
    model = uw.get_default_model()
    path = tmp / "transcripts" / "2026-09-26T00-00-00-run" / "transcript.jsonl"
    model.transcript_file = str(path)
    model.record_every = 1
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 4, qdegree=2)
    u = uw.discretisation.MeshVariable("u", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 1.0
    poisson.add_essential_bc(0.0, "Bottom")
    poisson.add_essential_bc(1.0, "Top")
    poisson.petsc_options.delValue("ksp_monitor")
    for _ in range(2):
        with model.step(0.1, label="march"):
            poisson.solve()
    with pytest.raises(RuntimeError):
        with model.step(0.1, label="bad"):
            poisson.solve()
            raise RuntimeError("rejected by the test")
    model.rewind(1, reason="rejected", observed=1.0, threshold=0.5)
    with model.step(0.05, label="retry"):
        poisson.solve()
    return path


def test_the_server_registers_read_only_tools():
    tools = asyncio.run(uwmcp.server.list_tools())
    names = {t.name for t in tools}
    assert {"uw_transcript_list", "uw_transcript_summary", "uw_transcript_problems",
            "uw_transcript_part", "uw_transcript_key", "uw_describe_render"} <= names
    assert all(t.annotations.read_only_hint for t in tools)
    assert all(t.description for t in tools)


def test_paths_resolve_to_the_latest_run(run_path):
    root = run_path.parent.parent            # the 'transcripts' directory
    assert uwmcp.resolve(str(root)) == str(run_path)
    assert uwmcp.resolve(str(run_path.parent)) == str(run_path)
    with pytest.raises(FileNotFoundError, match="uw_transcript_list"):
        uwmcp.resolve(str(root / "nowhere"))


def test_the_tools_answer_as_yaml(run_path):
    p = str(run_path)
    listed = yaml.safe_load(uwmcp.uw_transcript_list(str(run_path.parent.parent)))
    assert listed[0]["steps"] == 4 and listed[0]["abandoned"] == 1
    summary = yaml.safe_load(uwmcp.uw_transcript_summary(p))
    assert summary["kind"] == "transcript" and summary["facts"]["steps"] == 4
    assert summary["parts"][0]["solver"] == "SNES_Poisson"
    steps = yaml.safe_load(uwmcp.uw_transcript_steps(p, start=0, count=2))
    assert steps["total"] == 4 and [r["position"] for r in steps["rows"]] == [0, 1]
    patterns = yaml.safe_load(uwmcp.uw_transcript_patterns(p))
    assert [q["count"] for q in patterns] == [2, 1, 1]
    problems = yaml.safe_load(uwmcp.uw_transcript_problems(p))
    assert problems["abandoned"][0]["abandoned_by"]["type"] == "RuntimeError"
    assert problems["backtracks"][0]["reason"] == "rejected"
    assert problems["failed"] == [] and problems["capped"] == []
    events = yaml.safe_load(uwmcp.uw_transcript_events(p, kind="solve", outcome="ok"))
    assert events["total"] == 4
    step = yaml.safe_load(uwmcp.uw_transcript_step(p, index=2, attempt=0))
    assert step["row"]["completed"] is False
    assert "error" in uwmcp.uw_transcript_step(p, index=99)
    compare = yaml.safe_load(uwmcp.uw_transcript_compare(p, a=0, b=1))   # 1: the retry that stands
    assert compare["completed"] == [True, True]
    parts = yaml.safe_load(uwmcp.uw_transcript_parts(p))
    assert parts[0]["boundary_conditions"] == ["essential on Bottom", "essential on Top"]
    part = yaml.safe_load(uwmcp.uw_transcript_part(p, part=parts[0]["part"], detail="forms"))
    assert "F1" in part["forms"] and part["terms"]
    exact = yaml.safe_load(uwmcp.uw_transcript_part(p, part=parts[0]["label"], detail="exact"))
    assert exact["kind"] == "part"
    assert "error" in uwmcp.uw_transcript_part(p, part="nothing")
    assert "Poisson" in uwmcp.uw_transcript_key(p, format="text")
    rendered = uwmcp.uw_describe_render(uwmcp.uw_transcript_summary(p), format="markdown")
    assert rendered.startswith("## transcript")


def test_the_capabilities_catalogue_comes_from_the_classes():
    cat = yaml.safe_load(uwmcp.uw_capabilities())
    groups = {g["kind"]: g["children"] for g in cat["children"]}
    assert set(groups) == {"solvers", "constitutive_models", "histories"}
    stokes = next(r for r in groups["solvers"] if r["name"] == "Stokes")
    assert "F0" in stokes["facts"]["equation"] and "add_essential_bc" in stokes["facts"]["conditions"]
    assert any(r["name"] == "ViscoPlasticFlowModel" for r in groups["constitutive_models"])
    assert any(r["name"] == "SemiLagrangian" for r in groups["histories"])
    assert "error" in uwmcp.uw_capabilities(kind="nothing")
    full = uwmcp.uw_capability("Stokes")
    assert full.startswith("## solver family SNES_Stokes") and "Boundary conditions" in full
    assert yaml.safe_load(uwmcp.uw_capability("SemiLagrangian", format="yaml"))["kind"] == "history_family"
    assert "error" in uwmcp.uw_capability("Nothing")
