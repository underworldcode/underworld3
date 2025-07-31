set -x 

# Without arguments will launch a jupyterlab server on 9999 on the host machine

podman run \
  --rm -it \
  -p 9999:8888 \
  -u root \
  -v $HOME:/uw_dir/host_ro:ro \
  -v uw_vol:/uw_dir/workspace:rw \
  docker.io:underworldcode/underworld3:development $@
