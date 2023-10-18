#!/usr/bin/bash
#
# Usage:
#  source pathSetup.sh 

#  Warn if petsc4py is NOT taken from the PETSC_DIR 
P4PY_DIR=`python -c 'import petsc4py; print(petsc4py.__file__)' | xargs dirname`
if [[ ${P4PY_DIR} != ${PETSC_DIR}* ]]; then
    echo "Warning: petsc4py isn't install under the petsc at ${PETSC_DIR}"
    echo "         To ensure your petsc4py and petsc align configure petsc with"
    echo "         --download-petsc4py=1"
    echo ""
fi

#  Put Underworld package is in the PYTHONPATH
export PYTHONPATH=`pwd`:${PYTHONPATH}

