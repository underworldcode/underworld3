# The code in this file only runs successfully under 
# underworldcode/carbonite:petsc_knep@sha256:3557a63d3ced99e43953242198876388e7f3dc791e25682f5dca4089ac3208f8
#
# A test, using the file branchComp.sh, is made to 
# check for the REQUIRED petsc branch is available at $PETSC_DIR.
# See ./branchComp.sh for details


DOCKER_NAME = "underworldcode/carbonite:petsc_knep"
H52XMF      = ${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py

TEST := $(shell ./branchComp.sh; echo $$?)

compile: stokes.c
ifeq ($(TEST),1)
	@echo "*********************************"
	@echo "WRONG PETSC BRANCH FOR THIS MODEL"
	@echo "*********************************"
	@echo "Use the docker image called:"
	@echo "    $(DOCKER_NAME)"
else
	$(CC) -g -o carbonite $^ $(PETSC_CC_INCLUDES) $(PETSC_LIB)
endif

run_annulus: carbonite 
	./carbonite  $(opts)  \
		-annular 1 \
		-dm_plex_annular_faces 60,32 \
		-dm_plex_annular_lower 1.5,1.3089969389957472 \
	  -dm_plex_annular_upper 2.,1.8325957145940461 \
		-dm_plex_separate_marker -dm_partition_view        \
		-vel_petscspace_degree 2 -pres_petscspace_degree 1 \
	  -snes_monitor  \
		-simplex 1 \
		-dm_plex_print_fem 0 -pc_type lu -dm_view hdf5:${PWD}/sol1.h5  -sol_vec_view hdf5:${PWD}/sol1.h5::append \
		-dm_aux_view hdf5:${PWD}/aux1.h5  -vec_aux_view hdf5:${PWD}/aux1.h5::append
	${H52XMF} sol1.h5 aux1.h5
	

run_cart: carbonite 
	./carbonite  -cells 30,40 $(opts)  \
		-annular 0 \
		-dm_plex_separate_marker -dm_partition_view        \
		-vel_petscspace_degree 1 -pres_petscspace_degree 0 \
	  -snes_monitor  \
		-simplex 0 \
		-dm_plex_print_fem 0 -pc_type lu -dm_view hdf5:${PWD}/sol1.h5  -sol_vec_view hdf5:${PWD}/sol1.h5::append \
		-dm_aux_view hdf5:${PWD}/aux1.h5  -vec_aux_view hdf5:${PWD}/aux1.h5::append
	${H52XMF} sol1.h5 aux1.h5


clean:
	rm carbonite aux1.* sol1.*

include ${PETSC_DIR}/lib/petsc/conf/variables
