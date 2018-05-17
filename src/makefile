ex1: ex1.o
	-${CLINKER} -o $@ $< ${PETSC_LIB}
	${DSYMUTIL} $@
	${RM} -f $<

ex2: ex2.o
	-${CLINKER} -o $@ $< ${PETSC_LIB}
	${DSYMUTIL} $@
	${RM} -f $<

ex3: ex3.o
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
	${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py mesh.h5

run3: ex3
	mpirun -np 2 ./ex3 -f ./mesh.11_6.dat -dm_view hdf5:mesh.h5 -initial_vec_view hdf5:mesh.h5::append -sol_vec_view hdf5:mesh.h5::append -dm_partition_view \
      -velocity_petscspace_order 1 -pressure_petscspace_order 0 \
      -run_type convection \
      -snes_monitor -snes_converged_reason -ksp_error_if_not_converged 0 -ksp_converged_reason \
      ${opts}
	${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py mesh.h5

#    -ksp_type fgmres\
	  -pc_type fieldsplit\
	      -pc_fieldsplit_detect_saddlepoint\
		      -pc_fieldsplit_type schur \
			        -pc_fieldsplit_schur_factorization_type full\
					      -fieldsplit_velocity_pc_type lu \
						        -fieldsplit_pressure_ksp_rtol 1e-9 \
		-pc_type fieldsplit -pc_fieldsplit_type schur \
        -fieldsplit_velocity_pc_type lu \
        -fieldsplit_pressure_pc_type jacobi \

run2: ex2
	./ex2 -f ./mesh.dat -dm_view hdf5:mesh.h5 -sol_vec_view hdf5:mesh.h5::append -temperature_petscspace_order 1 -pc_type gamg ${opts}
	${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py mesh.h5

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
