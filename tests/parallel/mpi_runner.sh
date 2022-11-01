echo "ptest 001 -np 1"
mpirun -np 1 python3 ./ptest_001_start_stop.py
echo "ptest 001 -np 1 --discontinuous"
mpirun -np 1 python3 ./ptest_001_start_stop.py  --discontinuous 

echo "ptest 001 -np 2"
mpirun -np 2 python3 ./ptest_001_start_stop.py
echo "ptest 001 -np 2 --discontinuous"
mpirun -np 2 python3 ./ptest_001_start_stop.py  --discontinuous 

echo "ptest 001a -np 1"
mpirun -np 1 python3 ./ptest_001a_start_stop_petsc4py.py
echo "ptest 001a -np 2"
mpirun -np 2 python3 ./ptest_001a_start_stop_petsc4py.py 

echo "ptest 002 -np 1"
mpirun -np 1 python3 ./ptest_002_projection.py
echo "ptest 002 -np 2"
mpirun -np 2 python3 ./ptest_002_projection.py 

echo "ptest 003 -np 1"
mpirun -np 1 python3 ./ptest_003_swarm_projection.py
echo "ptest 003 -np 2"
mpirun -np 2 python3 ./ptest_003_swarm_projection.py 

