"""Velocity multigrid iterations against rank count, two configurations:
  plain  — UnstructuredSimplexBox(refinement=2), standard Stokes path (native hierarchy)
  fault  — BoxInternalPatch(refinement=2) + split_fault, rotated path, hierarchy tail
"""
import sys, time
import numpy as np, sympy, underworld3 as uw
from mpi4py import MPI
from underworld3.utilities.fault_split import split_fault
mode = sys.argv[1]
rank0 = MPI.COMM_WORLD.rank == 0
Lx, Ly, H = 3.0, 2.0, 1.0
h, hf, f = 0.2, 0.1, 4
yf, xa, xb, zb = 1.0, 1.0, 2.0, 0.5; ztop = H - 2 * hf
t = time.perf_counter()
if mode in ("plain", "wall"):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0, 0), maxCoords=(Lx, Ly, H), cellSize=f * h, regular=False, qdegree=3, refinement=2)
else:
    patch = np.array([[xa, yf, zb], [xb, yf, zb], [xb, yf, ztop], [xa, yf, ztop]])
    fine = uw.meshing.BoxInternalPatch(cellSize=f * h, minCoords=(0, 0, 0), maxCoords=(Lx, Ly, H), patch_points=patch, patch_name="Fault",
                                       patch_cellSize=f * hf, grading_distance=8 * hf, refinement=2, qdegree=3)
    mesh = split_fault(fine, "Fault")
ncell = MPI.COMM_WORLD.allreduce(int(mesh.dm.getHeightStratum(0)[1] - mesh.dm.getHeightStratum(0)[0]), op=MPI.SUM)
v = uw.discretisation.MeshVariable("v", mesh, 3, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.bodyforce = sympy.Matrix([0, 0, 0])
stokes.tolerance = 1e-4
stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Left")
stokes.add_natural_bc(sympy.Matrix([[-0.01, -0.01, 0.0]]), "Right")
if mode != "wall":
    stokes.add_dirichlet_bc((None, None, 0.0), "Bottom")
if mode.startswith("fault"):
    stokes.add_fault_bc(0.0, "Fault")
if mode == "wall":
    # rotated free-slip on the base in place of the Dirichlet u_z = 0: the rotated
    # custom-P path with constrained (normal) rows but NO fault pairs
    stokes.add_rotated_freeslip_bc(0, "Bottom")
    from underworld3.utilities.custom_mg import set_custom_fmg
    set_custom_fmg(stokes, list(mesh._coarse_level_meshes()[:-1]), field_id=0)
if mode == "fault_xp":
    from underworld3.utilities.custom_mg import set_custom_fmg
    set_custom_fmg(stokes, list(mesh._custom_mg_coarse_meshes), field_id=0, cross_partition=True)
t = time.perf_counter()
stokes.solve(zero_init_guess=True)
dt_solve = time.perf_counter() - t
if mode.startswith("fault") or mode == "wall":
    info = stokes._rotated_freeslip_info
    rep = f"velocity_pc {info.get('velocity_pc')} vel its {info.get('vel_its_last')} pres its {info.get('pres_its_last')} converged {info.get('converged')}"
else:
    ksp = stokes.snes.getKSP(); pc = ksp.getPC()
    try:
        sub = pc.getFieldSplitSubKSP()
        rep = f"pc {pc.getType()} vel pc {sub[0].getPC().getType()} vel its {sub[0].getIterationNumber()} pres its {sub[1].getIterationNumber()} outer its {ksp.getIterationNumber()} reason {ksp.getConvergedReason()}"
    except Exception as e:
        rep = f"pc {pc.getType()} outer its {ksp.getIterationNumber()} ({e})"
if mode.startswith("fault") or mode == "wall":
    cache = getattr(stokes, "_rotated_linear_cache", None) or {}
    Pl = cache.get("custom_Pl") or []
    H = (getattr(stokes, "_custom_mg", None) or {}).get("hierarchy")
    raw = getattr(H, "transfers", None) or []
    from underworld3.utilities.custom_mg import _count_zero_columns_parallel
    comm = MPI.COMM_WORLD
    zc = []
    for l, Praw in enumerate(raw):
        zc.append(f"raw L{l} zero cols {_count_zero_columns_parallel(Praw, Praw.comm)}")
    if rank0:
        print("   zero columns in the RAW transfers (before rotation / constrained-row zeroing): " + "; ".join(zc), flush=True)
    # transfer quality: a Lagrange interpolation transfer maps the constant 1 to 1
    # (row weights sum to one); empty rows are fine DOFs with no coarse image
    from petsc4py import PETSc
    pu = []
    for l, Praw in enumerate(raw):
        xc = Praw.createVecRight(); xc.set(1.0)
        yf = Praw.createVecLeft(); Praw.mult(xc, yf)
        a = np.abs(yf.getArray() - 1.0)
        nbad = comm.allreduce(int((a > 1e-8).sum()), op=MPI.SUM)
        nzero = comm.allreduce(int((np.abs(yf.getArray()) < 1e-14).sum()), op=MPI.SUM)
        nrow = Praw.getSize()[0]
        per_rank = comm.gather(int((np.abs(yf.getArray()) < 1e-14).sum()), root=0)
        pu.append(f"L{l}: rows {nrow}, rows not summing to 1: {nbad}, empty rows: {nzero} (per rank {per_rank})")
    if rank0:
        print("   partition of unity of the RAW transfers: " + "; ".join(pu), flush=True)
    try:
        ctx = cache.get("ctx") or {}
        vpc = ctx["pc"].getFieldSplitSubKSP()[0].getPC()
        nl = vpc.getMGLevels()
        sm = [f"L{l}: {vpc.getMGSmoother(l).getType()}/{vpc.getMGSmoother(l).getPC().getType()}" for l in range(nl)]
        cs = vpc.getMGCoarseSolve()
        if rank0:
            print(f"   live velocity MG: {nl} levels; smoothers {sm}; coarse {cs.getType()}/{cs.getPC().getType()}", flush=True)
    except Exception as e:
        if rank0:
            print(f"   live velocity MG: not readable ({type(e).__name__}: {e})", flush=True)
    # does the transfer reproduce the coordinate field? y = P x_c vs x_f, in the
    # reduced numbering of each level (ordering + location test in one)
    try:
        from underworld3.utilities.custom_mg import _level_dof_layout, _coarse_dof_layout
        meshes = list(H.level_meshes)
        rep_x = []
        for l in range(1, len(meshes)):
            cm, fm = meshes[l - 1], meshes[l]
            P = raw[l - 1]
            lay_c = _coarse_dof_layout(stokes, cm, 0) if l - 1 < len(meshes) - 1 else _level_dof_layout(stokes.dm, 0)
            lay_f = _level_dof_layout(stokes.dm, 0) if l == len(meshes) - 1 else _coarse_dof_layout(stokes, fm, 0)
            xc_nodes = np.asarray(cm._get_coords_for_basis(2, True)); xf_nodes = np.asarray(fm._get_coords_for_basis(2, True))
            xc = P.createVecRight(); xf = P.createVecLeft()
            ncomp = 3
            for comp in range(1):                     # x-coordinate only
                for i in range(xc_nodes.shape[0]):
                    g = int(lay_c.l2g[i * ncomp + comp])
                    if lay_c.rstart <= g < lay_c.rend:
                        xc.setValue(g, float(xc_nodes[i, 0]))
                for i in range(xf_nodes.shape[0]):
                    g = int(lay_f.l2g[i * ncomp + comp])
                    if lay_f.rstart <= g < lay_f.rend:
                        xf.setValue(g, float(xf_nodes[i, 0]))
            xc.assemble(); xf.assemble()
            y = P.createVecLeft(); P.mult(xc, y)
            # compare only on the x-component rows (every 3rd reduced row is not guaranteed: use the rows we set)
            d = np.abs(y.getArray() - xf.getArray())
            mask = np.abs(xf.getArray()) > 0   # rows we set (x > 0 everywhere but the x=0 face, which is constrained)
            nbad = comm.allreduce(int((d[mask] > 1e-6).sum()), op=MPI.SUM)
            ntot = comm.allreduce(int(mask.sum()), op=MPI.SUM)
            dmax = comm.allreduce(float(d[mask].max()) if mask.any() else 0.0, op=MPI.MAX)
            rep_x.append(f"L{l-1}->L{l}: x-rows checked {ntot}, |P x_c - x_f| > 1e-6 on {nbad}, max {dmax:.3e}")
        if rank0:
            print("   coordinate reproduction: " + "; ".join(rep_x), flush=True)
    except Exception as e:
        import traceback
        if rank0:
            print(f"   coordinate reproduction: failed {type(e).__name__}: {e}", flush=True); traceback.print_exc()
    # the finest level: do the basis coordinates (custom_mg's source) match the
    # velocity variable's own DOF coordinates (section order), node by node?
    try:
        xb = np.asarray(mesh._get_coords_for_basis(2, True))
        xv = np.asarray(v.coords)
        same_shape = xb.shape == xv.shape
        dmax = float(np.abs(xb - xv).max()) if same_shape and xb.size else -1.0
        nmis = int((np.abs(xb - xv).max(axis=1) > 1e-10).sum()) if same_shape else -1
        rep2 = comm.gather((xb.shape, xv.shape, dmax, nmis), root=0)
        if rank0:
            print("   finest coords check (basis vs variable) per rank: " + "; ".join(f"{a} vs {b}: max diff {d:.2e}, nodes differing {n}" for a, b, d, n in rep2), flush=True)
        # the solver's dm vs the mesh dm: same chart?
        sd, md = stokes.dm.getChart(), mesh.dm.getChart()
        rep3 = comm.gather((sd, md, stokes.dm.handle == mesh.dm.handle), root=0)
        if rank0:
            print(f"   solver dm chart vs mesh dm chart per rank: {rep3}", flush=True)
    except Exception as e:
        if rank0:
            print(f"   finest coords check failed: {type(e).__name__}: {e}", flush=True)
    # the finest reduced layout itself: how many local DOFs carry -1 (constrained
    # OR an unresolved ghost), against the owned count and the ghost count
    try:
        from underworld3.utilities.custom_mg import _level_dof_layout
        lay = _level_dof_layout(stokes.dm, 0)
        n_minus = int((lay.l2g < 0).sum()); n_owned = lay.rend - lay.rstart; n_full = lay.n_full
        # genuinely constrained velocity DOFs on this rank: Left face (3 per node) + Bottom (1 per node), by coordinates
        xb = np.asarray(mesh._get_coords_for_basis(2, True))
        n_left = int((np.abs(xb[:, 0]) < 1e-9).sum()) * 3
        n_bot = int(((np.abs(xb[:, 2]) < 1e-9) & (np.abs(xb[:, 0]) > 1e-9)).sum())
        rep4 = comm.gather((n_full, n_owned, n_minus, n_left + n_bot), root=0)
        if rank0:
            print("   finest layout per rank (local DOFs, owned, l2g<0, expected constrained by geometry): " + str(rep4), flush=True)
    except Exception as e:
        if rank0:
            print(f"   finest layout check failed: {type(e).__name__}: {e}", flush=True)
    # are the SOLVER dm's P2 node coordinates in the same local order as the mesh's?
    try:
        from petsc4py import PETSc
        def p2_coords_of(dm):
            dmold = dm.getCoordinateDM(); dmold.createDS(); dmnew = dmold.clone()
            o = PETSc.Options(); o["probe_petscspace_degree"] = 2; o["probe_petscdualspace_lagrange_continuity"] = True; o["probe_petscdualspace_lagrange_node_endpoints"] = False
            fe = PETSc.FE().createDefault(mesh.dim, mesh.cdim, mesh.isSimplex, mesh.qdegree, "probe_", PETSc.COMM_SELF)
            dmnew.setField(0, fe); dmnew.createDS()
            mi, _ = dmold.createInterpolation(dmnew)
            g = dmnew.getGlobalVec(); lv = dmnew.getLocalVec()
            mi.mult(dm.getCoordinates(), g); dmnew.globalToLocal(g, lv)
            arr = lv.array.reshape(-1, mesh.cdim).copy()
            dmnew.restoreGlobalVec(g); dmnew.restoreLocalVec(lv); return arr
        xs = p2_coords_of(stokes.dm); xm = np.asarray(mesh._get_coords_for_basis(2, True))
        same = xs.shape == xm.shape
        nd = int((np.abs(xs - xm).max(axis=1) > 1e-10).sum()) if same else -1
        # also: is the mesh's node set a permutation of the solver's? (sort both)
        perm = bool(np.allclose(np.sort(xs, axis=0), np.sort(xm, axis=0))) if same else False
        rep5 = comm.gather((xs.shape[0], xm.shape[0], nd, perm), root=0)
        if rank0:
            print("   solver-dm P2 coords vs mesh P2 coords per rank (n_solver, n_mesh, nodes differing in order, same set): " + str(rep5), flush=True)
    except Exception as e:
        import traceback
        if rank0:
            print(f"   solver-dm coords check failed: {type(e).__name__}: {e}", flush=True); traceback.print_exc()
    lev = []
    for l, P in enumerate(Pl):
        m, n = P.getSize()
        rn = raw[l].getSize()[1] if l < len(raw) else -1
        lev.append(f"L{l}: live {n} (raw {rn}, dropped {rn - n if rn >= 0 else '?'}) -> {m}")
    if rank0:
        print("   levels (coarse DOFs live/raw -> fine rows): " + "; ".join(lev), flush=True)
if rank0:
    print(f"{mode} np {MPI.COMM_WORLD.size} cells {ncell} tail {len(getattr(mesh, '_custom_mg_coarse_meshes', None) or [])} hierarchy {len(mesh.dm_hierarchy)}: solve {dt_solve:.1f}s: {rep}", flush=True)
