"""Split-node faults in 3-D: the labelled patch becomes a discontinuity.

The 3-D counterparts of the ``test_0845`` checks, one dimension up: exact
chart deltas and the Euler-characteristic invariant (a ball with an interior
slit gains the patch's interior characteristic, chi 1 -> 2 for a disc-like
patch), conformity, coincident replica coordinates, volume conservation,
label integrity — and the load-bearing DOF-independence solve, where a
Dirichlet datum on the Plus side must NOT leak onto the coincident Minus
DOFs while the (unsplit) rim reads it from both sides.
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.fault_split import split_fault

PATCH = np.array([[0.5, 0.30, 0.30], [0.5, 0.70, 0.30],
                  [0.5, 0.70, 0.70], [0.5, 0.30, 0.70]])


@pytest.fixture(scope="module")
def box_with_patch():
    return uw.meshing.BoxInternalPatch(cellSize=0.15, patch_points=PATCH,
                                       patch_name="FltA")


@pytest.fixture(scope="module")
def split_box(box_with_patch):
    return split_fault(box_with_patch, "FltA")


def _counts(dm):
    return {key: b - a for key, (a, b) in dict(
        c=dm.getHeightStratum(0), f=dm.getHeightStratum(1),
        e=dm.getDepthStratum(1), v=dm.getDepthStratum(0)).items()}


def _patch_census(dm, name, value):
    """(faces, interior_edges, interior_verts) of the labelled patch."""
    fS, fE = dm.getHeightStratum(1)
    vS, vE = dm.getDepthStratum(0)
    faces = [int(p) for p in
             dm.getLabel(name).getStratumIS(int(value)).getIndices()
             if fS <= int(p) < fE]
    edge_use, rim_verts, all_verts = {}, set(), set()
    for f in faces:
        for e in dm.getCone(f):
            edge_use[int(e)] = edge_use.get(int(e), 0) + 1
        all_verts.update(int(p) for p in dm.getTransitiveClosure(f)[0]
                         if vS <= int(p) < vE)
    interior_edges = [e for e, n in edge_use.items() if n == 2]
    for e, n in edge_use.items():
        if n == 1:
            rim_verts.update(int(q) for q in dm.getCone(e))
    return faces, interior_edges, sorted(all_verts - rim_verts)


@pytest.mark.level_1
@pytest.mark.tier_b
def test_chart_deltas_and_euler(box_with_patch, split_box):
    dm0, dm1 = box_with_patch.dm, split_box.dm
    c0, c1 = _counts(dm0), _counts(dm1)
    value = int(box_with_patch.boundaries["FltA"].value)
    faces, interior_edges, interior_verts = _patch_census(dm0, "FltA", value)

    assert c1["c"] == c0["c"]
    assert c1["v"] - c0["v"] == len(interior_verts)
    assert c1["e"] - c0["e"] == len(interior_edges)
    assert c1["f"] - c0["f"] == len(faces)

    chi0 = c0["v"] - c0["e"] + c0["f"] - c0["c"]
    chi1 = c1["v"] - c1["e"] + c1["f"] - c1["c"]
    # the slit adds the patch interior's characteristic: V_i - E_i + F,
    # which is 1 for any disc-like patch, independent of resolution
    assert chi0 == 1 and chi1 == 2
    assert (chi1 - chi0) == (len(interior_verts) - len(interior_edges)
                             + len(faces)) == 1


@pytest.mark.level_1
@pytest.mark.tier_b
def test_conformity_and_sides(split_box):
    dm = split_box.dm
    fS, fE = dm.getHeightStratum(1)
    supports = np.array([dm.getSupportSize(f) for f in range(fS, fE)])
    assert supports.max() <= 2

    for side in ("FltAPlus", "FltAMinus"):
        value = int(split_box.boundaries[side].value)
        assert dm.getLabel(side).getStratumSize(value) > 0
        pts = dm.getLabel(side).getStratumIS(value).getIndices()
        assert all(dm.getSupportSize(int(p)) == 1 for p in pts)
    n_plus = dm.getLabel("FltAPlus").getStratumSize(
        int(split_box.boundaries["FltAPlus"].value))
    n_minus = dm.getLabel("FltAMinus").getStratumSize(
        int(split_box.boundaries["FltAMinus"].value))
    assert n_plus == n_minus


@pytest.mark.level_1
@pytest.mark.tier_b
def test_volume_and_coincidence(box_with_patch, split_box):
    def total_volume(dm):
        dm.getCoordinatesLocal()
        cS, cE = dm.getHeightStratum(0)
        volumes = [dm.computeCellGeometryFVM(c)[0] for c in range(cS, cE)]
        assert min(volumes) > 0
        return sum(volumes)

    v0, v1 = total_volume(box_with_patch.dm), total_volume(split_box.dm)
    assert abs(v1 - v0) < 1e-12 * v0

    vS0, _ = box_with_patch.dm.getDepthStratum(0)
    vS1, _ = split_box.dm.getDepthStratum(0)
    X0 = np.asarray(box_with_patch.dm.getCoordinatesLocal().array
                    ).reshape(-1, 3)
    X1 = np.asarray(split_box.dm.getCoordinatesLocal().array).reshape(-1, 3)
    pairs = split_box._fault_point_pairs["FltA"]
    n_vertex_pairs = 0
    for q_minus, q_plus in pairs.items():
        if split_box.dm.getPointDepth(q_minus) == 0:
            assert (X1[q_minus - vS1] == X1[q_plus - vS1]).all()
            n_vertex_pairs += 1
    assert n_vertex_pairs > 0


@pytest.mark.level_1
@pytest.mark.tier_b
def test_pairing_census(box_with_patch, split_box):
    value = int(box_with_patch.boundaries["FltA"].value)
    faces, interior_edges, interior_verts = _patch_census(
        box_with_patch.dm, "FltA", value)
    pairs = split_box._fault_point_pairs["FltA"]
    by_depth = {}
    for q_minus in pairs:
        d = split_box.dm.getPointDepth(q_minus)
        by_depth[d] = by_depth.get(d, 0) + 1
    # replica vertices + doubled interior edges (P2 midpoints) + doubled
    # faces; re-homed minus spokes are excluded (their originals dropped)
    assert by_depth == {0: len(interior_verts), 1: len(interior_edges),
                        2: len(faces)}


@pytest.mark.level_2
@pytest.mark.tier_b
def test_dof_independence_p2(split_box):
    u = uw.discretisation.MeshVariable("u_0848", split_box, 1, degree=2)
    poisson = uw.systems.Poisson(split_box, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    for wall in ("Bottom", "Top", "Right", "Left", "Front", "Back"):
        poisson.add_dirichlet_bc(0.0, wall)
    poisson.add_dirichlet_bc(1.0, "FltAPlus")
    poisson.solve()

    vals = np.asarray(u.data[:, 0])
    coords = u.coords
    sec = split_box.dm.getLocalSection()

    def dof_of(pt):
        if sec.getFieldDof(pt, u.field_id) == 0:
            return None
        return sec.getFieldOffset(pt, u.field_id)

    n_checked = 0
    for q_minus, q_plus in split_box._fault_point_pairs["FltA"].items():
        d_minus, d_plus = dof_of(q_minus), dof_of(q_plus)
        if d_minus is None or d_plus is None:
            continue
        assert np.allclose(coords[d_minus], coords[d_plus])
        assert abs(vals[d_plus] - 1.0) < 1e-9
        # a shared DOF would read exactly 1.0 on the minus side too
        assert 0.0 < vals[d_minus] < 1.0 - 1e-9
        n_checked += 1
    assert n_checked > 0

    # the rim is UNSPLIT: every datum-side value there reads from both
    # sides, so the whole-fault label's rim points sit at exactly 1
    assert vals.min() >= 0.0 and vals.max() <= 1.0 + 1e-12


@pytest.mark.level_1
@pytest.mark.tier_b
def test_refusals(box_with_patch):
    from underworld3.utilities.fault_split import split_along_label_3d

    # missing label
    with pytest.raises(ValueError, match="no faces carry label"):
        split_along_label_3d(box_with_patch.dm, "NoSuchFault", 1,
                             "P", 90, "M", 91)

    # double split: the child's patch faces are one-sided boundaries now
    child = split_fault(box_with_patch, "FltA")
    with pytest.raises(ValueError):
        split_fault(child, "FltA")


def _stokes_on(split_box, drive, tag):
    v = uw.discretisation.MeshVariable(f"v_{tag}", split_box, 3, degree=2)
    p = uw.discretisation.MeshVariable(f"p_{tag}", split_box, 1, degree=0,
                                       continuous=False)
    stokes = uw.systems.Stokes(split_box, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = [0.0, 0.0, 0.0]
    for wall in ("Bottom", "Top", "Right", "Left", "Front", "Back"):
        stokes.add_dirichlet_bc(drive, wall)
    stokes.tolerance = 1e-6
    return stokes


def _slip_and_leak(stokes):
    from underworld3.utilities import fault_contact

    coords, jumps, normals = fault_contact.fault_pair_jumps(
        stokes, "FltA", stokes._rotated_freeslip_info)
    leak = np.einsum("ij,ij->i", jumps, normals)
    slip = np.linalg.norm(jumps - leak[:, None] * normals, axis=1)
    return coords, slip, leak


@pytest.mark.level_2
@pytest.mark.tier_b
def test_viscous_family_3d(split_box):
    """The interface dashpot in 3-D: slip monotone in eta_f, weld limit."""
    x, y, z = split_box.X
    shear = (0.0, x - 0.5, 0.0)
    peaks = []
    for tag, eta_f in (("vf0", 0.0), ("vf1", 5.0), ("vf2", 500.0)):
        stokes = _stokes_on(split_box, shear, tag)
        stokes.add_fault_bc(eta_f, boundary="FltA")
        stokes.solve(verbose=False)
        _coords, slip, leak = _slip_and_leak(stokes)
        assert np.abs(leak).max() < 1e-10
        peaks.append(slip.max())
    assert peaks[0] > peaks[1] > peaks[2]
    assert peaks[2] < 0.05 * peaks[0]          # the weld removes the fault


@pytest.mark.level_2
@pytest.mark.tier_b
def test_coulomb_slide_stick_3d(split_box):
    """Coulomb with reaction-fed sigma_n in 3-D: the drive
    v = (-(x-c), (y-c) + gamma (x-c), 0) carries compression 2 eta onto
    the patch and resolved shear gamma eta; against strength mu sigma_n
    the fault slides at gamma = 2 and sticks (creep ~ V0) at 0.4."""
    from underworld3.utilities import fault_contact

    x, y, z = split_box.X
    results = {}
    for tag, gamma in (("cs2", 2.0), ("cs04", 0.4)):
        drive = (-(x - 0.5), (y - 0.5) + gamma * (x - 0.5), 0.0)
        stokes = _stokes_on(split_box, drive, tag)
        fault_contact.add_coulomb_fault_bc(stokes, 0.6, "FltA",
                                           sigma_n="reaction", V0=1e-3)
        fault_contact.solve_with_fault(stokes, picard=2)
        coords, slip, leak = _slip_and_leak(stokes)
        assert np.abs(leak).max() < 1e-10
        crds, sig = fault_contact.fault_normal_traction(
            stokes, "FltA", stokes._rotated_freeslip_info)
        inner = np.linalg.norm(crds[:, 1:] - 0.5, axis=1) < 0.12
        # the de-smeared reaction recovers the driven compression
        assert -3.5 < np.median(sig[inner]) < -1.2
        results[gamma] = slip.max()
    assert results[2.0] > 0.05                 # sliding
    assert results[0.4] < 5e-3                 # stuck at ~V0 creep


@pytest.mark.level_1
@pytest.mark.tier_b
def test_daylighting_patch_refused():
    # a patch reaching the wall is refused at mesh construction
    spanning = np.array([[0.5, -0.1, 0.3], [0.5, 1.1, 0.3],
                         [0.5, 1.1, 0.7], [0.5, -0.1, 0.7]])
    with pytest.raises(ValueError, match="strictly inside"):
        uw.meshing.BoxInternalPatch(cellSize=0.2, patch_points=spanning)


@pytest.mark.level_1
@pytest.mark.tier_b
def test_fault_surface_route():
    """The FaultSurface interface end to end: the object names the
    patch, its rim polygon is the conforming embed, it rides onto the
    split mesh, and normal="surface" compiles the constraint frame from
    the surface's OWN face normals (exactly +-y for this planar patch).
    """
    xs = np.linspace(0.30, 0.60, 5)
    zs = np.linspace(0.35, 0.65, 5)
    XX, ZZ = np.meshgrid(xs, zs)
    pts = np.column_stack([XX.ravel(), np.full(XX.size, 0.5), ZZ.ravel()])
    fs = uw.meshing.FaultSurface("Rupture", pts)
    fs.triangulate()

    rim = fs.rim_polygon()
    assert len(rim) == 16                       # the 5x5 grid's boundary
    assert np.allclose(rim[:, 1], 0.5)

    mesh = uw.meshing.BoxInternalPatch(cellSize=0.15, patch_points=fs)
    assert "Rupture" in [b.name for b in mesh.boundaries]
    child = split_fault(mesh, "Rupture")
    assert "Rupture" in child._fault_surfaces

    from underworld3.utilities.fault_contact import _compile_normal_spec
    fn = _compile_normal_spec("surface", child, "Rupture")
    n = fn(np.array([0.45, 0.5, 0.5]))
    assert abs(abs(n[1]) - 1.0) < 1e-12 and abs(n[0]) < 1e-12

    # a non-planar surface refuses the rim extraction, loudly
    warped = pts.copy()
    warped[:, 1] += 0.05 * np.sin(6 * warped[:, 0])
    fs2 = uw.meshing.FaultSurface("Warped", warped)
    fs2.triangulate()
    with pytest.raises(NotImplementedError, match="planar"):
        fs2.rim_polygon()
