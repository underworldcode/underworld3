compile: ex4.c
	$(CC) -g -o ex4 $^ $(PETSC_CC_INCLUDES) $(PETSC_LIB)

#-snes_view \
#	-dm_plex_annular_lower 1.5,1.3089969389957472 \
#		-dm_plex_annular_upper 2.,1.8325957145940461 \
#		-snes_error_if_not_converged
#		-dm_plex_annular_lower 1.5,1.374447 \
#		-dm_plex_annular_upper 2.,1.767146  \
#		-vel_petscspace_degree 2 -pres_petscspace_degree 1 \
#	-simplex 1 \
#

run: ex4 
	mpirun -np 3 \
	./ex4  -dm_plex_annular_faces 60,32 $(opts)  \
		-dm_plex_annular_lower 1.5,1.3089969389957472 \
	  -dm_plex_annular_upper 2.,1.8325957145940461 \
		-dm_plex_separate_marker -dm_partition_view        \
		-vel_petscspace_degree 2 -pres_petscspace_degree 1 \
	  -snes_monitor  \
		-simplex 1 \
		-dm_plex_print_fem 0 -pc_type lu -dm_view hdf5:${PWD}/sol1.h5  -sol_vec_view hdf5:${PWD}/sol1.h5::append \
		-dm_aux_view hdf5:${PWD}/aux1.h5  -vec_aux_view hdf5:${PWD}/aux1.h5::append

include ${PETSC_DIR}/lib/petsc/conf/variables
