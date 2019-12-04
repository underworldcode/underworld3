#!/bin/sh

RNAME=\""knepley/feature-plex-examples\"" # the required name of matt's branch with basis rotations
#echo ${RNAME}

# current branch name of petsc
#BNAME=`grep BRANCH ${PETSC_DIR}/include/petscconf.h | cut -d ' ' -f 3`

BNAME=`find $PETSC_DIR -name petscconf.h -exec grep BRANCH {} + | cut -d ' ' -f 3`
echo ${BNAME}

if [ ${BNAME} = ${RNAME} ]; then 
  exit 0
else
  exit 1
fi
