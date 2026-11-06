"""The PyVista conversions work without a notebook stack (#665).

``visualisation.initialise`` asks for the trame Jupyter backend, and every
conversion and render goes through it, although none of them needs a notebook.
pyvista's theme setter raises ImportError in two cases, and they want different
answers:

- no IPython: it refuses every backend, static included, so nothing can be set
  and nothing needs to be;
- IPython but no trame: it refuses trame, and the static backend stands in.

A backend asked for by name is another matter: if it cannot be had, that raises.
"""
import importlib.util
import subprocess
import sys

import pytest

import underworld3 as uw
import underworld3.visualisation as vis
from underworld3.visualisation.visualisation import initialise

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

pv = pytest.importorskip("pyvista")


@pytest.fixture
def trame_missing(monkeypatch):
    """The theme's backend setter raises ImportError for trame, as without it."""
    pytest.importorskip("IPython")  # without it initialise returns before trame matters
    theme_class = type(pv.global_theme)
    real = theme_class.jupyter_backend

    def setter(self, value):
        if value == "trame":
            raise ImportError("trame is not installed")
        real.fset(self, value)

    monkeypatch.setattr(theme_class, "jupyter_backend", property(real.fget, setter))


@pytest.fixture
def ipython_missing(monkeypatch):
    """pyvista's validator finds no IPython, which is how it checks for one."""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "IPython" else real(name, *a, **k))


def test_initialise_leaves_the_backend_alone_without_ipython(ipython_missing):
    before = pv.global_theme._jupyter_backend
    initialise(None)
    assert pv.global_theme._jupyter_backend == before


def test_mesh_to_pv_mesh_without_ipython(ipython_missing):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    start, end = mesh.dm.getHeightStratum(0)
    assert pvmesh.n_cells == end - start


@pytest.mark.parametrize("missing", ["trame_missing", "ipython_missing"])
def test_a_backend_asked_for_by_name_raises_if_it_cannot_be_had(missing, request):
    request.getfixturevalue(missing)
    with pytest.raises(ImportError):
        initialise("trame")


@pytest.mark.parametrize("backend", [None, "trame", "client", "server", "html"])
def test_a_remote_host_gets_the_server_proxy(backend, monkeypatch):
    """On Binder, every trame-family backend reaches its server through the proxy."""
    pytest.importorskip("IPython")
    pytest.importorskip("trame")
    monkeypatch.setenv("BINDER_LAUNCH_HOST", "https://mybinder.org")
    monkeypatch.delenv("JUPYTERHUB_SERVICE_PREFIX", raising=False)
    monkeypatch.setattr(pv.global_theme.trame, "server_proxy_enabled", False)
    initialise(backend)
    assert pv.global_theme.trame.server_proxy_enabled


def test_a_mesh_converts_with_the_notebook_stack_really_absent():
    """In a fresh process with IPython and trame not importable: the fixtures above fake
    how pyvista checks for them, so a new hard import on this path would pass them."""
    script = """
import importlib.abc, sys
HIDDEN = ("IPython", "ipykernel", "ipywidgets", "jupyter", "comm", "trame")
class Hide(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0].startswith(HIDDEN):
            raise ModuleNotFoundError(name)
        return None
sys.meta_path.insert(0, Hide())
import underworld3 as uw
import underworld3.visualisation as vis
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
pvmesh = vis.mesh_to_pv_mesh(mesh)
start, end = mesh.dm.getHeightStratum(0)
assert pvmesh.n_cells == end - start
print("CONVERTED")
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0 and "CONVERTED" in done.stdout, done.stdout + done.stderr


def test_the_notebook_default_is_trame_where_it_can_be_had():
    pytest.importorskip("IPython")
    pytest.importorskip("trame")
    initialise(None)
    assert pv.global_theme.jupyter_backend == "trame"


def test_asking_for_no_backend_without_ipython_sets_nothing(ipython_missing):
    before = pv.global_theme._jupyter_backend
    initialise("none")
    assert pv.global_theme._jupyter_backend == before


def test_initialise_falls_back_to_static_without_trame(trame_missing):
    initialise(None)
    assert pv.global_theme.jupyter_backend == "static"


def test_mesh_to_pv_mesh_without_trame(trame_missing):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    start, end = mesh.dm.getHeightStratum(0)
    assert pvmesh.n_cells == end - start
