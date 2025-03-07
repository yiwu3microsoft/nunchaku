#!/bin/bash
# Modified from https://github.com/sgl-project/sglang/blob/main/sgl-kernel/build.sh
set -ex
PYTHON_VERSION=$1
TORCH_VERSION=$2
CUDA_VERSION=$3
MAX_JOBS=${4:-} # optional
PYTHON_ROOT_PATH=/opt/python/cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}

docker run --rm \
    -v "$(pwd)":/nunchaku \
    pytorch/manylinux-builder:cuda${CUDA_VERSION} \
    bash -c "
    cd /nunchaku && \
    rm -rf build && \
    yum install -y devtoolset-11 && \
    source scl_source enable devtoolset-11 && \
    gcc --version && g++ --version && \
    ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir torch==${TORCH_VERSION} numpy --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.} && \
    ${PYTHON_ROOT_PATH}/bin/pip install build ninja wheel setuptools && \
    export NUNCHAKU_INSTALL_MODE=ALL && \
    export NUNCHAKU_BUILD_WHEELS=1 && \
    export MAX_JOBS=${MAX_JOBS} && \
    ${PYTHON_ROOT_PATH}/bin/python -m build --wheel --no-isolation
    "