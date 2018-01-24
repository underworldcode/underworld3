ex1: ex1.o
	-${CLINKER} -o $@ $< ${PETSC_LIB}
	${DSYMUTIL} $@
	${RM} -f $<

ex2: ex2.o
	-${CLINKER} -o $@ $< ${PETSC_LIB}
	${DSYMUTIL} $@
	${RM} -f $<

run: ex1
	./ex1 -f ./mesh.dat -dm_view hdf5:mesh.h5 -sol_vec_view hdf5:mesh.h5::append \
      -run_type analytic_simple \
      -pc_type fieldsplit -pc_fieldsplit_type schur \
        -fieldsplit_velocity_pc_type lu \
        -fieldsplit_pressure_pc_type jacobi \
      -snes_monitor -snes_converged_reason -ksp_error_if_not_converged 0 -ksp_converged_reason \
      -velocity_petscspace_order 1 -pressure_petscspace_order 0 \
      ${opts}
	${PETSC_DIR}/bin/petsc_gen_xdmf.py mesh.h5

run2:
	./ex2 -f ./mesh.dat -dm_view hdf5:mesh.h5 -sol_vec_view hdf5:mesh.h5::append ${opts}
	${PETSC_DIR}/bin/petsc_gen_xdmf.py mesh.h5

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
