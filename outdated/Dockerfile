FROM julesg/underworld2:2.15.0b as runtime
LABEL maintainer="https://github.com/underworldcode/"

# install all build tool as root
USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends $(awk '{print $1'} /opt/installed.txt) \
    libglu1 \
    libxcursor-dev \
    libxft2 \
    libxinerama1 \
    libfltk1.3-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install python packages for uw3
RUN pip install sympy gmsh pyvista panel xxhash

USER $NB_USER
WORKDIR $NB_HOME
RUN git clone --branch development --depth 1 https://github.com/underworldcode/underworld3.git uw3
WORKDIR $NB_HOME/uw3
RUN python3 setup.py develop
ENV PYTHONPATH=$NB_HOME/uw3:$PYTHONPATH
