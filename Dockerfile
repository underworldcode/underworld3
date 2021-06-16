FROM underworldcode/carbonite:petsc_dev
COPY . /tmp/uw3
WORKDIR /tmp/uw3
RUN pip3 install -e .

# add a user
ENV NB_USER jovyan
RUN useradd -m -s /bin/bash -N $NB_USER 

# # jovyan user, finalise jupyter env
# USER $NB_USER