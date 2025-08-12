The `Containerfile` is used to build the container and can be used with docker or podman. 
At present only amd64 architecture is built, because vtk-osmesa isn't available for arm by default. This may change in future, or we could build vtk-osmesa from source.

The `launch-container.sh` is a script for launching the container using `podman` only. Podman is the preferred container runner because of `rootless` support.
Expect the script to take a minute or so to load as permissions are updated.

Useful links:
- Container stacks with podman - https://jupyter-docker-stacks.readthedocs.io/en/latest/using/running.html#using-the-podman-cli
- Micromamba images - https://micromamba-docker.readthedocs.io/en/latest/quick_start.html#quick-start
- Pyvista containers: 
    - https://github.com/pyvista/pyvista/tree/main/docker
    - https://dev.pyvista.org/getting-started/installation#running-on-mybinder

