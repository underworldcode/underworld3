# This image supports underworld2 and active development in underworld3.

FROM underworldcode/underworld2:dev
MAINTAINER https://github.com/underworldcode/

USER root

# install dev tools + replace petsc_gen_xdmf.py with modified versions (for py3 support)
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
     valgrind   \
     tmux       \
     python2.7-minimal \
     cgdb           && \
    apt-get clean   && \
    rm -rf /var/lib/apt/lists/* && \
    wget -O $PETSC_DIR/lib/petsc/bin/petsc_gen_xdmf.py \
    https://bitbucket.org/!api/2.0/snippets/bucket_of_jules/Ajyyr8/e4c2abc99a10269984e16a98f33f6948019c4215/files/petsc_gen_xdmf.py

RUN pip3 install Cython
 
WORKDIR $NB_WORK
USER $NB_USER
