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

echo "ptest 0007 snapshot in-memory -np 1"
mpirun -np 1 $PYTHON ./ptest_0007_snapshot_inmemory.py
echo "ptest 0007 snapshot in-memory -np 3 (uneven partition)"
mpirun -np 3 $PYTHON ./ptest_0007_snapshot_inmemory.py
echo "ptest 0007 snapshot in-memory -np 4"
mpirun -np 4 $PYTHON ./ptest_0007_snapshot_inmemory.py

echo "ptest 0010 snapshot on-disk -np 1"
mpirun -np 1 $PYTHON ./ptest_0010_snapshot_disk.py
echo "ptest 0010 snapshot on-disk -np 3 (uneven)"
mpirun -np 3 $PYTHON ./ptest_0010_snapshot_disk.py
echo "ptest 0010 snapshot on-disk -np 4"
mpirun -np 4 $PYTHON ./ptest_0010_snapshot_disk.py

echo "ptest 0004 checkpoint FMG hierarchy -np 2"
mpirun -np 2 $PYTHON ./ptest_0004_checkpoint_fmg_hierarchy.py
echo "ptest 0004 checkpoint FMG hierarchy -np 3 (uneven partition)"
mpirun -np 3 $PYTHON ./ptest_0004_checkpoint_fmg_hierarchy.py

# The wall-clock guard's expiry decision must be identical on every rank: a PETSc
# convergence test that disagrees across ranks deadlocks the next collective. This
# script skews the per-rank clocks deliberately, so it HANGS rather than fails if the
# reduction is ever dropped -- run it under a timeout.
echo "ptest 0203 wall-clock guard -np 2"
mpirun --timeout 300 -np 2 $PYTHON ./ptest_0203_wallclock_guard_parallel.py
echo "ptest 0203 wall-clock guard -np 4"
mpirun --timeout 300 -np 4 $PYTHON ./ptest_0203_wallclock_guard_parallel.py
