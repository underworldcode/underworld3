"""WorkflowProducts — make-like product persistence for workflow pipelines.

Maps named products (meshes, variables, surfaces) to files on disk.
A YAML manifest tracks what exists, enabling selective rebuild and
parameter-study workflows where expensive products are reused.

Usage::

    products = WorkflowProducts(config)

    # Save after expensive steps
    products.save("adapted_mesh", mesh)
    products.save("fault_surfaces", surface_collection)

    # Later — reload without recomputing
    mesh = products.load("adapted_mesh")
    surfaces = products.load("fault_surfaces", mesh=mesh)

    # Check what's available
    products.list()
    products.status(h2ex_module)
"""

import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


# ── Type detection ──────────────────────────────────────────────────

def _detect_type(obj):
    """Return a string type tag for dispatch."""
    cls_name = type(obj).__name__
    mod = type(obj).__module__ or ""

    # Mesh types
    if cls_name in ("Mesh", "MeshClass") or "discretisation" in mod and "Mesh" in cls_name:
        if hasattr(obj, "write_checkpoint"):
            return "mesh"

    # MeshVariable
    if hasattr(obj, "load_from_h5_plex_vector") and hasattr(obj, "save"):
        return "mesh_variable"

    # SurfaceCollection
    if cls_name == "SurfaceCollection" and hasattr(obj, "surfaces"):
        return "surface_collection"

    # Surface
    if cls_name == "Surface" and hasattr(obj, "pv_mesh"):
        return "surface"

    # numpy array
    if isinstance(obj, np.ndarray):
        return "ndarray"

    # dict of surfaces
    if isinstance(obj, dict) and obj:
        first = next(iter(obj.values()))
        if type(first).__name__ == "Surface":
            return "surface_dict"

    return "unknown"


# ── Manifest helpers ────────────────────────────────────────────────

def _load_manifest(path):
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)


# ── Main class ──────────────────────────────────────────────────────

class WorkflowProducts:
    """Manages workflow product persistence (make-like).

    Products are named objects (meshes, variables, surfaces) saved to a
    products directory.  Each product has a type-aware save/load method.
    A YAML manifest tracks what exists.

    Parameters
    ----------
    config : WorkflowConfig, optional
        If provided, ``products_dir`` defaults to
        ``config.output_dir / "products"``.
    products_dir : str or Path, optional
        Explicit products directory. Overrides config.
    """

    def __init__(self, config=None, products_dir=None):
        if products_dir is not None:
            self._dir = Path(products_dir)
        elif config is not None:
            self._dir = Path(config.output_dir) / "products"
        else:
            self._dir = Path("products")

        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._dir / "manifest.yaml"

    @property
    def products_dir(self):
        return self._dir

    # ── Save ────────────────────────────────────────────────────────

    def save(self, name: str, obj, **metadata):
        """Save a product to disk.

        Type dispatch:

        - **Mesh** — HDF5 via ``write_timestep()`` (vertex-based, portable)
        - **MeshVariable** — HDF5 via ``write_timestep()`` with the mesh
        - **Surface** — VTK via ``.save()``
        - **SurfaceCollection** — directory of VTK files
        - **dict of Surfaces** — directory of VTK files
        - **ndarray** — ``.npz``

        Parameters
        ----------
        name : str
            Product name (used as filename stem and manifest key).
        obj : object
            The object to save.
        **metadata
            Extra key-value pairs stored in the manifest entry.
        """
        product_type = _detect_type(obj)
        files = []

        if product_type == "mesh":
            files = self._save_mesh(name, obj)

        elif product_type == "mesh_variable":
            files = self._save_mesh_variable(name, obj)
            metadata["var_name"] = obj.clean_name

        elif product_type == "surface":
            files = self._save_surface(name, obj)

        elif product_type == "surface_collection":
            files = self._save_surface_collection(name, obj)

        elif product_type == "surface_dict":
            files = self._save_surface_dict(name, obj)

        elif product_type == "ndarray":
            fname = f"{name}.npz"
            np.savez(self._dir / fname, data=obj)
            files = [fname]

        else:
            # Try YAML serialisation as fallback
            fname = f"{name}.yaml"
            with open(self._dir / fname, "w") as f:
                yaml.dump(obj, f, default_flow_style=False)
            files = [fname]
            product_type = "yaml"

        # Update manifest
        manifest = _load_manifest(self._manifest_path)
        manifest[name] = {
            "type": product_type,
            "files": files,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if metadata:
            manifest[name]["metadata"] = metadata
        _save_manifest(self._manifest_path, manifest)

    def _save_mesh(self, name, mesh):
        """Save a mesh via write_timestep (vertex-based, portable HDF5+XDMF)."""
        mesh.write_timestep(
            name, index=0, outputPath=str(self._dir),
            meshVars=[], swarmVars=[], meshUpdates=True,
        )
        # write_timestep creates <outputPath>/<name>.mesh.00000.h5
        return [f"{name}.mesh.00000.h5"]

    def _save_mesh_variable(self, name, var):
        """Save a MeshVariable via write_timestep.

        Writes the mesh geometry and the variable's vertex values to
        separate HDF5 files with XDMF metadata.  On reload,
        ``read_timestep`` uses kd-tree interpolation so the variable
        can even be loaded onto a different mesh.
        """
        mesh = var.mesh
        mesh.write_timestep(
            name, index=0, outputPath=str(self._dir),
            meshVars=[var], swarmVars=[], meshUpdates=True,
        )
        # Produces: <name>.mesh.00000.h5 and <name>.mesh.<clean_name>.00000.h5
        mesh_h5 = f"{name}.mesh.00000.h5"
        var_h5 = f"{name}.mesh.{var.clean_name}.00000.h5"
        return [mesh_h5, var_h5]

    def _save_surface(self, name, surface):
        """Save a single Surface to VTK."""
        fname = f"{name}.vtk"
        surface.save(str(self._dir / fname))
        return [fname]

    def _save_surface_collection(self, name, collection):
        """Save a SurfaceCollection as a directory of VTK files."""
        subdir = self._dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        files = []
        for surf_name, surface in collection.surfaces.items():
            fname = f"{name}/{surf_name}.vtk"
            surface.save(str(self._dir / fname))
            files.append(fname)
        return files

    def _save_surface_dict(self, name, surface_dict):
        """Save a dict of Surfaces as a directory of VTK files."""
        subdir = self._dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        files = []
        for surf_name, surface in surface_dict.items():
            fname = f"{name}/{surf_name}.vtk"
            surface.save(str(self._dir / fname))
            files.append(fname)
        return files

    # ── Load ────────────────────────────────────────────────────────

    def load(self, name: str, mesh=None, mesh_variable=None):
        """Load a product by name.

        Parameters
        ----------
        name : str
            Product name (must exist in manifest).
        mesh : Mesh, optional
            Required for loading Surfaces that need a mesh reference.
        mesh_variable : MeshVariable, optional
            For ``mesh_variable`` products: the target variable to load
            data into (via ``read_timestep``).  Must already exist on a
            mesh.

        Returns
        -------
        object
            The loaded product.

        Raises
        ------
        KeyError
            If the product is not in the manifest.
        FileNotFoundError
            If product files are missing from disk.
        """
        manifest = _load_manifest(self._manifest_path)
        if name not in manifest:
            available = ", ".join(manifest.keys()) or "(none)"
            raise KeyError(
                f"Product '{name}' not found. Available: {available}"
            )

        entry = manifest[name]
        product_type = entry["type"]
        files = entry["files"]

        if product_type == "mesh":
            return self._load_mesh(name, files)

        elif product_type == "mesh_variable":
            if mesh_variable is None:
                raise ValueError(
                    f"Loading mesh_variable '{name}' requires "
                    f"mesh_variable= argument (the target MeshVariable "
                    f"to load data into via read_timestep)"
                )
            return self._load_mesh_variable(name, entry, mesh_variable)

        elif product_type == "surface":
            return self._load_surface(name, files, mesh)

        elif product_type in ("surface_collection", "surface_dict"):
            return self._load_surface_collection(name, files, mesh)

        elif product_type == "ndarray":
            data = np.load(self._dir / files[0])
            return data["data"]

        elif product_type == "yaml":
            with open(self._dir / files[0]) as f:
                return yaml.safe_load(f)

        else:
            raise ValueError(f"Unknown product type '{product_type}' for '{name}'")

    def _load_mesh(self, name, files):
        """Load a mesh from the timestep HDF5 file."""
        import underworld3 as uw

        h5_path = str(self._dir / files[0])
        return uw.discretisation.Mesh(h5_path)

    def _load_mesh_variable(self, name, entry, mesh_variable):
        """Load a MeshVariable saved via write_timestep.

        Uses ``read_timestep`` which reads vertex-based HDF5 and
        interpolates via kd-tree onto the target variable's DOFs.
        Works even if the mesh has changed (kd-tree interpolation).
        """
        var_meta = entry.get("metadata", {})
        var_name = var_meta.get("var_name", mesh_variable.clean_name)

        mesh_variable.read_timestep(
            name, var_name, index=0, outputPath=str(self._dir),
        )
        return mesh_variable

    def _load_surface(self, name, files, mesh):
        """Load a Surface from VTK."""
        from underworld3.meshing import Surface

        vtk_path = str(self._dir / files[0])
        return Surface.from_vtk(vtk_path, mesh=mesh)

    def _load_surface_collection(self, name, files, mesh):
        """Load a SurfaceCollection from a directory of VTK files."""
        from underworld3.meshing import SurfaceCollection

        collection = SurfaceCollection()
        for f in files:
            vtk_path = str(self._dir / f)
            if Path(vtk_path).exists():
                collection.add_from_vtk(vtk_path, mesh=mesh)
        return collection

    # ── Query ───────────────────────────────────────────────────────

    def exists(self, name: str) -> bool:
        """Check if a product exists in the manifest and on disk."""
        manifest = _load_manifest(self._manifest_path)
        if name not in manifest:
            return False
        # Verify at least one file exists
        for f in manifest[name].get("files", []):
            if (self._dir / f).exists():
                return True
        return False

    def list(self):
        """Display available products.

        In Jupyter renders as an HTML table; falls back to plain text.
        """
        manifest = _load_manifest(self._manifest_path)

        if not manifest:
            try:
                from IPython.display import HTML, display
                display(HTML("<em>No products saved yet.</em>"))
            except ImportError:
                print("No products saved yet.")
            return

        rows = []
        for name, entry in manifest.items():
            on_disk = any(
                (self._dir / f).exists() for f in entry.get("files", [])
            )
            rows.append((
                name,
                entry.get("type", "?"),
                entry.get("saved_at", "?"),
                "yes" if on_disk else "MISSING",
            ))

        try:
            from IPython.display import HTML, display

            th = (
                '<th style="text-align:left; padding:4px 12px 4px 0; '
                'border-bottom:2px solid #ccc;">'
            )
            html = (
                f"<h4>Workflow Products</h4>"
                f'<p style="color:#888;">Directory: <code>{self._dir}</code></p>'
                f'<table style="border-collapse:collapse;">'
                f"<tr>{th}Product</th>{th}Type</th>{th}Saved</th>{th}On Disk</th></tr>"
            )
            for name, ptype, saved, on_disk in rows:
                colour = "#060" if on_disk == "yes" else "#c00"
                td = '<td style="padding:2px 12px 2px 0;'
                html += (
                    f"<tr>"
                    f'{td} font-family:monospace;">{name}</td>'
                    f'{td}">{ptype}</td>'
                    f'{td} color:#888;">{saved}</td>'
                    f'{td} color:{colour}; font-weight:bold;">{on_disk}</td>'
                    f"</tr>"
                )
            html += "</table>"
            display(HTML(html))

        except ImportError:
            name_w = max(len(r[0]) for r in rows)
            type_w = max(len(r[1]) for r in rows)
            print(f"Workflow Products  ({self._dir})")
            print(f"  {'Product':<{name_w}}  {'Type':<{type_w}}  {'Saved':<24}  On Disk")
            print(f"  {'-' * name_w}  {'-' * type_w}  {'-' * 24}  -------")
            for name, ptype, saved, on_disk in rows:
                print(f"  {name:<{name_w}}  {ptype:<{type_w}}  {saved:<24}  {on_disk}")

    def status(self, module=None):
        """Show which products exist vs need building.

        If *module* is provided (a workflow module with ``@workflow_step``
        decorated functions), cross-references product names from step
        metadata against the manifest.

        Parameters
        ----------
        module : module, optional
            A workflow module to scan for ``produces``/``requires`` metadata.
        """
        manifest = _load_manifest(self._manifest_path)

        # Gather all product names from module steps
        all_products = set()
        if module is not None:
            for _name, obj in inspect.getmembers(module, callable):
                if getattr(obj, "_is_workflow_step", False):
                    all_products.update(getattr(obj, "workflow_produces", []))
        else:
            all_products = set(manifest.keys())

        if not all_products:
            print("No products defined.")
            return

        rows = []
        for product in sorted(all_products):
            if product in manifest:
                on_disk = any(
                    (self._dir / f).exists()
                    for f in manifest[product].get("files", [])
                )
                status_str = "ready" if on_disk else "MISSING"
                saved = manifest[product].get("saved_at", "")
            else:
                status_str = "not built"
                saved = ""
            rows.append((product, status_str, saved))

        try:
            from IPython.display import HTML, display

            th = (
                '<th style="text-align:left; padding:4px 12px 4px 0; '
                'border-bottom:2px solid #ccc;">'
            )
            html = (
                f"<h4>Product Status</h4>"
                f'<table style="border-collapse:collapse;">'
                f"<tr>{th}Product</th>{th}Status</th>{th}Saved</th></tr>"
            )
            for product, status_str, saved in rows:
                if status_str == "ready":
                    colour, icon = "#060", "\u2713"
                elif status_str == "MISSING":
                    colour, icon = "#c00", "\u2717"
                else:
                    colour, icon = "#888", "\u2022"
                td = '<td style="padding:2px 12px 2px 0;'
                html += (
                    f"<tr>"
                    f'{td} font-family:monospace;">{product}</td>'
                    f'{td} color:{colour}; font-weight:bold;">{icon} {status_str}</td>'
                    f'{td} color:#888;">{saved}</td>'
                    f"</tr>"
                )
            html += "</table>"
            display(HTML(html))

        except ImportError:
            name_w = max(len(r[0]) for r in rows)
            print("Product Status:")
            for product, status_str, saved in rows:
                mark = "[x]" if status_str == "ready" else "[ ]"
                print(f"  {mark} {product:<{name_w}}  {status_str}  {saved}")

    def remove(self, name: str):
        """Remove a product from disk and the manifest.

        Parameters
        ----------
        name : str
            Product name to remove.
        """
        manifest = _load_manifest(self._manifest_path)
        if name not in manifest:
            return

        entry = manifest.pop(name)
        for f in entry.get("files", []):
            p = self._dir / f
            if p.exists():
                p.unlink()

        # Clean up empty subdirectories
        subdir = self._dir / name
        if subdir.is_dir() and not any(subdir.iterdir()):
            subdir.rmdir()

        _save_manifest(self._manifest_path, manifest)

    def clear(self):
        """Remove all products from disk and the manifest."""
        manifest = _load_manifest(self._manifest_path)
        for name in list(manifest.keys()):
            self.remove(name)
