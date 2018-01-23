ex1: ex1.o
	-${CLINKER} -o $@ $< ${PETSC_LIB}
	${RM} -f $<

run:
	./ex1 -f ./mesh.dat -dm_view hdf5:mesh.h5 -sol_vec_view hdf5:mesh.h5::append
	${PETSC_DIR}/bin/petsc_gen_xdmf mesh.h5

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
