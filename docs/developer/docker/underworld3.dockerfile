### how to build docker image

# (1) run from the underworld3 top directory
# podman build . \
# --rm \
# -f ./docs/developer/docker/underworld3.dockerfile \
# --format docker \
# -t underworldcode/underworld3:0.9.9

### needs the --format tag to run on podman

FROM docker.io/mambaorg/micromamba:debian11-slim


USER $MAMBA_USER
ENV NB_HOME=/home/$MAMBA_USER

### create the env
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

# ### activate env and install
ARG MAMBA_DOCKERFILE_ACTIVATE=1


# ### install UW3
WORKDIR /tmp
COPY --chown=$MAMBA_USER:$MAMBA_USER . /tmp/underworld3
WORKDIR /tmp/underworld3

### install uw3 using pip
RUN pip install --no-build-isolation .

#### copy files across
RUN mkdir -p $NB_HOME/workspace

COPY --chown=$MAMBA_USER:$MAMBA_USER ./tests   $NB_HOME/Underworld/tests

EXPOSE 8888
WORKDIR $NB_HOME
USER $MAMBA_USER

# Declare a volume space
VOLUME $NB_HOME/workspace

CMD ["jupyter-lab", "--no-browser", "--ip='0.0.0.0'"]
