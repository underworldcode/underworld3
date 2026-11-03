r"""True-sphere prescribed pressure lifting on fitted affine shell meshes.

For p_J=J(X) H(r-R), extraction of J e_r delta_sphere leaves -H grad(J)
and the exterior natural datum +J n_poly. Integrate complete topologically
Outer tetrahedra, then subtract the radial caps inside the true sphere.
Cap loads use exact packet routing and actual containing-cell P2 basis values.
The vectors below are FORCE right-hand sides, not positive residual loads.
"""

from contextlib import ExitStack
from typing import NamedTuple
import weakref

import numpy as np
import sympy as sp
from sympy.vector.scalar import BaseScalar
from sympy.core.add import _addsort
from sympy.core.mul import _mulsort
from sympy.core.function import Application
from sympy.core.relational import Relational
from sympy.functions.elementary.piecewise import ExprCondPair
from mpi4py import MPI
import underworld3 as uw

from underworld3.function.expressions import unwrap
from underworld3.utilities._fitted_regions import _label_fitted_shell_regions
from underworld3.utilities._physical_spherical_sheet import (
    _owned_spherical_facets, _p2_load_workspace, _add_p2_point_load, _route_spherical_packets,
)
from underworld3.utilities._spherical_gap import _spherical_facet_quadrature
from underworld3.utilities._spherical_stokes_gap import _coordinate_values
from underworld3.utilities._tetrahedral_gap import _tetrahedron_quadrature


class _SphericalPressureLift(NamedTuple):
    pressure_jump: object
    pressure_jump_integral: float
    physical_pressure: object
    regular_force_expression: object
    volume_force: object
    native_outer_boundary_force: object
    checks: dict
    owner_stokes: weakref.ReferenceType

    def destroy(self):
        """Release both caller-owned force vectors after copying or completing the solve."""
        self.volume_force.destroy()
        self.native_outer_boundary_force.destroy()


def _bind_spherical_coordinates(expression, dummies, coordinates):
    """Bind pure current-frame scalars without SymPy reconstruction-cache aliases.

    SymPy's private uncached constructors are required by UWCoordinate's shared
    BaseScalar hashes. The pinned-runtime multi-mesh regression covers these
    constructors; unsupported expression structures fail explicitly.
    """
    memo = {}

    def visit(node):
        if id(node) in memo:
            return memo[id(node)]
        if isinstance(node, BaseScalar):
            return coordinates[int(node.args[0])]
        if node in dummies:
            return coordinates[dummies.index(node)]
        if not node.args:
            return node
        args = tuple(visit(arg) for arg in node.args)
        if all(a is b for a, b in zip(args, node.args, strict=True)):
            return node
        if isinstance(node, (sp.Add, sp.Mul)):
            ordered = list(args)
            (_addsort if isinstance(node, sp.Add) else _mulsort)(ordered)
            result = node.func._from_args(tuple(ordered))
        elif isinstance(node, sp.Pow):
            result = sp.Pow.__new__.__wrapped__(sp.Pow, *args, evaluate=False)
        elif isinstance(node, sp.Function):
            result = Application.__new__.__wrapped__(node.func, *args, evaluate=False)
        elif isinstance(node, (sp.Piecewise, Relational)):
            result = node.func(*args, evaluate=False)
        elif isinstance(node, ExprCondPair):
            result = node.func(*args)
        elif isinstance(node, sp.Tuple):
            result = node.func(*args, sympify=False)
        else:
            raise TypeError(f"Unsupported spherical coordinate expression: {type(node).__name__}")
        memo[id(node)] = result
        return result

    return visit(expression)


def _spherical_pressure_lifting(
    stokes, inner_boundary: str, interface: str, interface_radius: float,
    outer_boundary: str, outer_radius: float, jump,
    volume_order: int, surface_order: int, radial_order: int,
) -> _SphericalPressureLift:
    """Construct true-radius pressure lifting and separated regular/natural loads.

    Reserved SphericalLiftInner/Outer labels are reconstructed by topology;
    existing mesh.regions and user material labels are preserved. The internal
    surface must be closed and star-shaped about the origin, with outward
    facets. Outer facet planes must lie beyond the internal sphere. No wall BC
    or native SNES callback is installed. Native force-vector insertion must
    respect essential rows; coupled walls use their own physical-sphere datum.
    """
    mesh, dm = stokes.mesh, stokes.mesh.dm
    comm = dm.comm.tompi4py()
    if (mesh.dim != 3 or mesh.degree != 1 or not mesh.isSimplex
            or stokes.u.degree != 2 or not stokes.u.continuous
            or stokes.p.degree != 1 or not stokes.p.continuous):
        raise ValueError("Spherical lifting requires affine tetrahedra and continuous P2/P1 fields.")
    if MPI.Comm.Compare(comm, uw.mpi.comm) not in (MPI.IDENT, MPI.CONGRUENT) or uw.get_default_model().has_units_active():
        raise ValueError("Spherical lifting requires a nondimensional world-communicator mesh.")
    if not np.isfinite([interface_radius, outer_radius]).all() or not 0 < interface_radius < outer_radius:
        raise ValueError("Require positive ordered internal and outer radii.")
    if any(not isinstance(q, (int, np.integer)) or isinstance(q, bool) or q < 2
           for q in (volume_order, surface_order, radial_order)):
        raise ValueError("Volume, surface and radial quadrature orders must be integers >=2.")
    jump = unwrap(sp.sympify(jump), keep_constants=False, return_self=False)
    # UWCoordinate and plain BaseScalar share hashes. SymPy reconstruction
    # caches can retain an earlier mesh's coordinate wrappers after unwrap.
    # This is a Cartesian coordinate function, so differentiate in independent
    # symbols, then map back to this mesh's frame. No mesh field is transferred.
    axes = tuple(sp.Dummy(f"spherical_lift_{axis}", real=True) for axis in "xyz")
    substitutions = {}
    for atom in jump.atoms(BaseScalar):
        if atom._system != mesh.N or int(atom.args[0]) not in range(3):
            raise ValueError("The pressure jump must use the mesh Cartesian reference frame.")
        substitutions[atom] = axes[int(atom.args[0])]
    canonical_jump = jump.xreplace(substitutions)
    if (canonical_jump.free_symbols - set(axes)
            or any(hasattr(type(atom), "meshvar") for atom in canonical_jump.atoms(sp.Function))
            or canonical_jump.has(sp.Piecewise, sp.Heaviside, sp.DiracDelta)):
        raise ValueError("Prescribe a smooth coordinate-only jump extension, not an unknown field or internal cut.")
    coordinates = mesh.N.base_scalars()
    bind = lambda expression: _bind_spherical_coordinates(expression, axes, coordinates)
    jump = bind(canonical_jump)
    gradient = sp.Matrix([bind(sp.diff(canonical_jump, coordinate)) for coordinate in axes])
    region_names = ("SphericalLiftInner", "SphericalLiftOuter")
    region_counts = _label_fitted_shell_regions(
        dm, (inner_boundary, mesh.boundaries[inner_boundary].value),
        (outer_boundary, mesh.boundaries[outer_boundary].value),
        (interface, mesh.boundaries[interface].value), (region_names[0], 101), (region_names[1], 102),
    )
    outer_cells = mesh.cells_labelled(region_names[1], 102)
    facets, triangles = _owned_spherical_facets(mesh, interface)
    upper_facets, upper_triangles = _owned_spherical_facets(mesh, outer_boundary)
    rule = _spherical_facet_quadrature(triangles, interface_radius, surface_order, radial_order)
    upper = _spherical_facet_quadrature(upper_triangles, outer_radius, surface_order, radial_order)
    cstart, _ = dm.getHeightStratum(0)

    # Determine actual fitted-side orientation from the adjacent cell label.
    # Positive radial normals plus closed topology make radial caps a tiling.
    invalid_geometry = False
    for face, triangle in zip(facets, triangles, strict=True):
        cell = int(dm.getSupport(face)[0])
        cross = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        toward_outer = np.dot(cross, dm.computeCellGeometryFVM(cell)[1]-triangle.mean(axis=0))
        toward_outer *= 1 if outer_cells[cell-cstart] else -1
        invalid_geometry |= toward_outer == 0 or np.sign(toward_outer)*np.dot(cross, triangle[0]) <= 0
    upper_cells, upper_normals = [], []
    for face, triangle in zip(upper_facets, upper_triangles, strict=True):
        cell = int(dm.getSupport(face)[0]); upper_cells.append(cell-cstart)
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        normal /= np.linalg.norm(normal)
        normal *= np.sign(normal@(triangle.mean(axis=0)-dm.computeCellGeometryFVM(cell)[1]))
        invalid_geometry |= normal@triangle[0] <= interface_radius
        upper_normals.append(normal)
    if comm.allreduce(invalid_geometry, op=MPI.LOR):
        raise ValueError("Internal radial caps do not form outward star-shaped patches, or outer planes cut the internal sphere.")
    area = comm.allreduce(float(rule["surface_weights"].sum()), op=MPI.SUM)
    # Positive sphere normals orient the triangle plane independently of vertex order.
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    plane_height = np.abs(np.einsum("ed,ed->e", cross, triangles[:, 0]))/np.linalg.norm(cross, axis=1)
    poly_internal = comm.allreduce(float(np.sum(rule["planar_weights"]*plane_height[:, None])/3), op=MPI.SUM)
    cap_volume = comm.allreduce(float(rule["gap_weights"].sum()), op=MPI.SUM)
    if not np.isclose(area, 4*np.pi*interface_radius**2, rtol=1e-10, atol=0) or not np.isclose(
            cap_volume+poly_internal, 4*np.pi*interface_radius**3/3, rtol=1e-10, atol=0):
        raise ValueError("Internal spherical patches fail closed-sphere area or cap-volume tiling checks.")
    cap_points = np.ascontiguousarray(rule["gap_points"].reshape(-1, 3))
    cap_weights = rule["gap_weights"].ravel()
    packets_before = comm.allreduce(len(cap_points), op=MPI.SUM)
    cap_points, cap_weights, cap_cells, moved = _route_spherical_packets(mesh, cap_points, cap_weights)
    if comm.allreduce(bool(np.any(~outer_cells[cap_cells])), op=MPI.LOR):
        raise ValueError("An internal radial cap enters the topologically Inner region.")
    packets_after = comm.allreduce(len(cap_points), op=MPI.SUM)
    if packets_after != packets_before or not np.isclose(comm.allreduce(float(cap_weights.sum()), op=MPI.SUM), cap_volume, rtol=1e-13, atol=0):
        raise ArithmeticError("Spherical lift routing changed cap counts or weights.")

    radius = bind(sp.sqrt(sum(coordinate**2 for coordinate in axes)))
    step = sp.Piecewise((sp.S.One, radius > sp.nsimplify(interface_radius, rational=True)), (sp.S.Zero, True))
    pressure_jump = bind(jump*step)
    with ExitStack() as cleanup:
        volume_force, vertices, ids = _p2_load_workspace(stokes.u)
        cleanup.callback(volume_force.destroy)
        natural = volume_force.duplicate(); natural.set(0); cleanup.callback(natural.destroy)
        owned = mesh._get_owned_cells_mask()
        selected = np.flatnonzero(owned & outer_cells)
        integral, minimum = 0.0, 1.0
        for start in range(0, len(selected), 32):
            cells = selected[start:start+32]
            points, weights = _tetrahedron_quadrature(vertices[cells], volume_order)
            force = -_coordinate_values(gradient, points, 3, mesh)
            minimum = min(minimum, _add_p2_point_load(volume_force, vertices, ids, cells, points, weights[..., None]*force))
            integral += float(np.sum(weights*_coordinate_values(jump, points, 1, mesh)[..., 0]))
        for start in range(0, len(cap_points), 512):
            sl = slice(start, start+512)
            points, weights, cells = cap_points[sl, None], cap_weights[sl, None], cap_cells[sl]
            force = _coordinate_values(gradient, points, 3, mesh)
            minimum = min(minimum, _add_p2_point_load(volume_force, vertices, ids, cells, points, weights[..., None]*force))
            integral -= float(np.sum(weights*_coordinate_values(jump, points, 1, mesh)[..., 0]))
        upper_cells, upper_normals = np.asarray(upper_cells, dtype=int), np.asarray(upper_normals).reshape(-1, 3)
        for start in range(0, len(upper_cells), 32):
            sl = slice(start, start+32)
            points, weights = upper["planar_points"][sl], upper["planar_weights"][sl]
            value = _coordinate_values(jump, points, 1, mesh)[..., 0]
            minimum = min(minimum, _add_p2_point_load(natural, vertices, ids, upper_cells[sl], points,
                                                     weights[..., None]*value[..., None]*upper_normals[sl, None]))
        minimum = comm.allreduce(minimum, op=MPI.MIN)
        if minimum < -1e-10:
            raise ValueError(f"Spherical lift P2 load extrapolated outside a cell: {minimum}")
        volume_force.assemble(); natural.assemble()
        integral = comm.allreduce(integral, op=MPI.SUM)
        outer_gap_integral = comm.allreduce(float(np.sum(upper["gap_weights"]*_coordinate_values(jump, upper["gap_points"].reshape(len(upper_cells), surface_order*surface_order*radial_order, 3), 1, mesh)[..., 0].reshape(upper["gap_weights"].shape))), op=MPI.SUM)
        outer_volume = comm.allreduce(float(np.einsum("eq,eqd,ed->", upper["planar_weights"], upper["planar_points"], upper_normals)/3), op=MPI.SUM)
        checks = dict(region_labels=dict(zip(region_names, (101, 102), strict=True)), **region_counts,
                      internal_area=area, internal_cap_volume=cap_volume, poly_outer_enclosed_volume=outer_volume,
                      packets_before=packets_before, packets_after=packets_after, migrated_packets=moved,
                      minimum_barycentric=minimum, physical_lift_integral=integral+outer_gap_integral,
                      volume_order=volume_order, surface_order=surface_order, radial_order=radial_order)
        result = _SphericalPressureLift(pressure_jump, integral, bind(stokes.p.sym[0]+pressure_jump),
                                        (-gradient*step).applyfunc(bind), volume_force, natural, checks, weakref.ref(stokes))
        cleanup.pop_all()
    return result
