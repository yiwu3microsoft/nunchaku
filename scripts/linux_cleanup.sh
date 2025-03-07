#!/bin/bash
set -ex

#docker run --rm \
#    -v "$(pwd)":/nunchaku \
#    pytorch/manylinux-builder:cuda12.4 \
#    bash -c "cd /nunchaku && rm -r *"
docker run --rm -it \
    -v "$(pwd)":/nunchaku \
    pytorch/manylinux-builder:cuda12.4 \
    bash