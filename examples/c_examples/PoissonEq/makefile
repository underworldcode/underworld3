H52XDMF=python3 ${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py

SRC = poisson.c
EXE = $(SRC:.c=) # string substitution out the '.c'

$(EXE): $(SRC)
	$(CC) -g $< -o $@ $(PETSC_CC_INCLUDES) $(PETSC_LIB)
	@echo ""
	@echo "Built now execute:"
	@echo "  make run opts='' "

run: $(EXE) 
	-mpirun -np 4 \
	./$^ -elRes 10,10 -dm_plex_separate_marker -simplex 1 -temperature_petscspace_degree 1 -options_view \
	-sol_vec_view -dm_view hdf5:sol.h5 -initial_vec_view hdf5:sol.h5::append -sol_vec_view hdf5:sol.h5::append \
	-ana_vec_view hdf5:sol.h5::append -dm_partition_view \
	-local_vec_view \
	-snes_view $(opts)
	${H52XDMF} sol.h5

clean:
	rm $(EXE) sol.*
 
# includes should go after targets
include ${PETSC_DIR}/lib/petsc/conf/variables
