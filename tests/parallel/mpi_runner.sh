PYTHON=python

echo "ptest 001 -np 1"
mpirun -np 1 $PYTHON ./ptest_001_start_stop.py
echo "ptest 001 -np 1 --discontinuous"
mpirun -np 1 $PYTHON ./ptest_001_start_stop.py  --discontinuous

echo "ptest 001 -np 4"
mpirun -np 4 $PYTHON ./ptest_001_start_stop.py
echo "ptest 001 -np 2 --discontinuous"
mpirun -np 4 $PYTHON ./ptest_001_start_stop.py  --discontinuous

echo "ptest 001a -np 1"
mpirun -np 1 $PYTHON ./ptest_001a_start_stop_petsc4py.py
echo "ptest 001a -np 4"
mpirun -np 4 $PYTHON ./ptest_001a_start_stop_petsc4py.py

echo "ptest 002 -np 1"
mpirun -np 1 $PYTHON ./ptest_002_projection.py
echo "ptest 002 -np 4"
mpirun -np 4 $PYTHON ./ptest_002_projection.py

#echo "ptest 003 -np 1"
#mpirun -np 1 $PYTHON ./ptest_003_swarm_projection.py
#echo "ptest 003 -np 4"
#mpirun -np 4 $PYTHON ./ptest_003_swarm_projection.py
