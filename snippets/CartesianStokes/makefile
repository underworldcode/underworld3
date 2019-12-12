H52XDMF=python3 ${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py

SRC = stokes.c
EXE = $(SRC:.c=) # string substitution out the '.c'

$(EXE): $(SRC)
	$(CC) -g $< -o $@ $(PETSC_CC_INCLUDES) $(PETSC_LIB)
	@echo ""
	@echo "Built now execute:"
	@echo "  make run1 opts='' "
	@echo ""
	@echo "See 'makefile' for other model runs "
	
# 2D stokes: simplicies, p2/p1 
run1: $(EXE) 
	mpirun -np 3 ./$^ -cells 15,20 $(opts)  \
		-dm_plex_separate_marker -dm_partition_view        \
		-vel_petscspace_degree 2 -pres_petscspace_degree 1 \
		-snes_monitor  \
		-simplex 1 \
		-dm_plex_print_fem 0 -pc_type lu -dm_view hdf5:sol1.h5  -sol_vec_view hdf5:sol1.h5::append \
		-dm_aux_view hdf5:aux1.h5  -vec_aux_view hdf5:aux1.h5::append
	${H52XDMF} sol1.h5 aux1.h5

# 2D stokes: q1/p0 
run2: $(EXE) 
	mpirun -np 3 ./$^ -cells 15,20 $(opts)  \
		-dm_plex_separate_marker -dm_partition_view        \
		-vel_petscspace_degree 1 -pres_petscspace_degree 0 \
		-snes_monitor  \
		-simplex 0 \
		-dm_plex_print_fem 0 -pc_type lu -dm_view hdf5:sol1.h5  -sol_vec_view hdf5:sol1.h5::append \
		-dm_aux_view hdf5:aux1.h5  -vec_aux_view hdf5:aux1.h5::append
	${H52XDMF} sol1.h5 aux1.h5


clean:
	-rm $(EXE) sol1.* aux1.*
 
# includes should go after targets
include ${PETSC_DIR}/lib/petsc/conf/variables
