FROM nvidia/cuda:10.0-cudnn7-devel

LABEL author="Daisuke Tanaka <duaipp@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    git \
    cmake \
    libblas3 \
    libblas-dev \
    curl \
    wget \
    unzip \
    # from nGraph
    build-essential \
    clang-3.9 \
    zlib1g \
    zlib1g-dev \
    libtinfo-dev \
    autoconf \
    automake \
    libtool \
    # from chainer-compiler
    ninja-build && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# nGraph build, referenced ngraph-onnx/BUILDING.md
# NOTE(disktnk): should use 'DPYTHON_EXECUTABLE' (currently not supported)
# NOTE(disktnk): failed with '-DNGRAPH_GPU_ENABLE=TRUE', stop using CUDA enabled
# NOTE(disktnk): cannot build python bind with multiprocess
RUN ln -s /usr/bin/python3 /usr/bin/python
ARG NGRAPH_VERSION="v0.19.0"
RUN git clone https://github.com/NervanaSystems/ngraph.git -b ${NGRAPH_VERSION} && \
    mkdir ngraph/build && cd ngraph/build && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$HOME/ngraph_dist \
        -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE \
        -DNGRAPH_USE_PREBUILT_LLVM=TRUE \
        -DNGRAPH_INTELGPU_ENABLE=TRUE \
        -DNGRAPH_GPU_ENABLE=FALSE \
        -DNGRAPH_UNIT_TEST_ENABLE=FALSE \
    && \
    make -j1 && \
    make install && \
    cd .. && rm -rf build
RUN cd ngraph/python && sed -e "s/^distutils.ccompiler.CCompiler.compile/# &/" setup.py > setup_.py
RUN cd ngraph/python && \
    git clone --recursive https://github.com/jagerman/pybind11.git && \
    export PYBIND_HEADERS_PATH=$PWD/pybind11 && \
    export NGRAPH_CPP_BUILD_PATH=$HOME/ngraph_dist && \
    export NGRAPH_ONNX_IMPORT_ENABLE=TRUE && \
    python3 -m pip install numpy && \
    python3 setup_.py bdist_wheel && \
    python3 -m pip install -U dist/*.whl && rm -rf build

ARG NGRAPH_ONNX_VERSION="v0.14.0"
RUN cd ngraph && \
    git clone https://github.com/NervanaSystems/ngraph-onnx.git -b ${NGRAPH_ONNX_VERSION} && \
    cd ngraph-onnx && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install -r requirements_test.txt && \
    python3 -m pip install -e .

# TVM Build, referenced Dockerfile.demo_gpu
# NOTE(disktnk): official installer does not copy header file
ARG TVM_VERSION="v0.5"
RUN wget https://raw.githubusercontent.com/dmlc/tvm/${TVM_VERSION}/docker/install/ubuntu_install_core.sh -P /install && \
    bash /install/ubuntu_install_core.sh && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main \
    >> /etc/apt/sources.list.d/llvm.list && \
    wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add - && \
    apt-get update && apt-get install -y --force-yes llvm-6.0 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
# based https://raw.githubusercontent.com/dmlc/tvm/${TVM_VERSION}/docker/install/install_tvm_gpu.sh
RUN cd /usr && \
    git clone https://github.com/dmlc/tvm --recursive && \
    cd /usr/tvm && \
    echo set\(USE_LLVM llvm-config-6.0\) >> config.cmake && \
    echo set\(USE_CUDA ON\) >> config.cmake && \
    echo set\(USE_CUDNN ON\) >> config.cmake && \
    echo set\(USE_RPC ON\) >> config.cmake && \
    echo set\(USE_SORT ON\) >> config.cmake && \
    echo set\(USE_GRAPH_RUNTIME ON\) >> config.cmake && \
    echo set\(USE_BLAS openblas\) >> config.cmake && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j10 && make install && \
    cd .. && rm -rf build
RUN ln -s /usr/tvm/include/tvm/*.h /usr/local/include/tvm && \
    ln -s /usr/tvm/include/relay /usr/local/include/tvm && \
    ln -s /usr/tvm/topi/include/topi /usr/local/include & \
    mkdir -p /usr/local/include/HalideIR &&  ln -s /usr/tvm/3rdparty/HalideIR/src/* /usr/local/include/HalideIR
RUN python3 -m pip install --user decorator attrs
ENV PYTHONPATH=/usr/tvm/python:/usr/tvm/topi/python:/usr/tvm/nnvm/python/:/usr/tvm/vta/python:${PYTHONPATH}
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
RUN mkdir -p /usr/tvm/3rdparty/dmlc-core/build && \
    cd /usr/tvm/3rdparty/dmlc-core/build && \
    cmake .. && make && make install
RUN mkdir -p /usr/tvm/3rdparty/dlpack/build && \
    cd /usr/tvm/3rdparty/dlpack/build && \
    cmake .. && make && make install
