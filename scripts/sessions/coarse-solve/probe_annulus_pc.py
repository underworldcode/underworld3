import sympy, underworld3 as uw
mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, refinement=3, qdegree=3)
v = uw.discretisation.MeshVariable("V", mesh, 2, degree=2); p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
s.constitutive_model = uw.constitutive_models.ViscousFlowModel; s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
x, y = mesh.X; s.bodyforce = sympy.Matrix([x, y]) * sympy.cos(4 * sympy.atan2(y, x))
s.add_rotated_freeslip_bc(0, "Lower"); s.add_rotated_freeslip_bc(0, "Upper"); s.petsc_use_pressure_nullspace = True
s.solve(zero_init_guess=True)
uw.pprint("PC", s._rotated_freeslip_info.get("velocity_pc"), "fallbacks", s.pc_fallbacks)
uw.pprint("mg levels", len(mesh._coarse_level_meshes()), "custom tail", getattr(mesh, "_custom_mg_coarse_meshes", None) is not None)
