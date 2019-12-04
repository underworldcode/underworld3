#/Currently 
H52XDMF=python3 ${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py

compile: uw_swarm.c
	$(CC) -g $^ -o run $(PETSC_CC_INCLUDES) $(PETSC_LIB)
	python3 setup.py build_ext --inplace

	
###	valgrind --log-file=val.dat --leak-check=full --show-leak-kinds=all \
	gdb --args \

run: compile
	mpirun -np 3 \
	./run -elements 8,8 -ppcell 3 -simplex 0 -steps 5 \
	-vel_petscspace_degree 1 pres_petscspace_degree 0 \
	-dm_partition_view ${opts}

	${H52XDMF} mesh.h5


clean:
	rm -f run swarm-* mesh.*
	
# includes should go after targets
include ${PETSC_DIR}/lib/petsc/conf/variables
