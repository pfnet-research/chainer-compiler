FROM nvidia/cuda:10.0-cudnn7-devel AS ngraph
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    clang-3.9 \
    git \
    curl \
    zlib1g \
    zlib1g-dev \
    libtinfo-dev \
    unzip \
    autoconf \
    automake \
    libtool \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
# nGraph build, referenced ngraph-onnx/BUILDING.md
# NOTE(disktnk): should use 'DPYTHON_EXECUTABLE' (currently not supported)
# NOTE(disktnk): failed with '-DNGRAPH_GPU_ENABLE=TRUE', stop using CUDA enabled
# NOTE(disktnk): cannot build python bind with multiprocess
RUN ln -s /usr/bin/python3 /usr/bin/python
ARG NGRAPH_VERSION="0.19.0"
RUN git clone https://github.com/NervanaSystems/ngraph.git -b v${NGRAPH_VERSION} && \
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
    python3 -m pip install --user numpy && \
    python3 setup_.py bdist_wheel && \
    python3 -m pip install --user -U dist/*.whl && rm -rf build
ARG NGRAPH_ONNX_VERSION="0.14.0"
RUN cd ngraph && \
    git clone https://github.com/NervanaSystems/ngraph-onnx.git -b v${NGRAPH_ONNX_VERSION} && \
    cd ngraph-onnx && \
    python3 -m pip install --user -r requirements.txt && \
    python3 -m pip install --user -r requirements_test.txt && \
    python3 -m pip install --user -e .

FROM nvidia/cuda:10.0-cudnn7-devel AS tvm
# TVM Build, referenced Dockerfile.demo_gpu
# NOTE(disktnk): official installer does not copy header file
ARG TVM_VERSION="0.5"
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN wget https://raw.githubusercontent.com/dmlc/tvm/v${TVM_VERSION}/docker/install/ubuntu_install_core.sh -P /install && \
    bash /install/ubuntu_install_core.sh && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main \
    >> /etc/apt/sources.list.d/llvm.list && \
    wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add - && \
    apt-get update && apt-get install -y --force-yes llvm-6.0 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
# based https://raw.githubusercontent.com/dmlc/tvm/v${TVM_VERSION}/docker/install/install_tvm_gpu.sh
RUN cd /usr && \
    git clone https://github.com/dmlc/tvm --recursive -b v${TVM_VERSION} && \
    cd /usr/tvm && \
    echo set\(USE_LLVM llvm-config-6.0\) >> config.cmake && \
    echo set\(USE_CUDA ON\) >> config.cmake && \
    echo set\(USE_CUDNN ON\) >> config.cmake && \
    echo set\(USE_RPC ON\) >> config.cmake && \
    echo set\(USE_SORT ON\) >> config.cmake && \
    echo set\(USE_GRAPH_RUNTIME ON\) >> config.cmake && \
    echo set\(USE_BLAS openblas\) >> config.cmake && \
    echo set\(INSTALL_DEV ON\) >> config.cmake && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/tvm_dist && \
    make -j10 && make install && \
    cd .. && rm -rf build
RUN mkdir -p /usr/tvm/3rdparty/dmlc-core/build && \
    cd /usr/tvm/3rdparty/dmlc-core/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/dmlc_core_dist && \
    make && make install
RUN mkdir -p /usr/tvm/3rdparty/dlpack/build && \
    cd /usr/tvm/3rdparty/dlpack/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/dlpack_dist && \
    make && make install

FROM ubuntu:18.04 AS dldt
ARG DLDT_VERSION="2019_R1.1"
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    git \
    wget \
    sudo && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN mkdir neo && cd neo && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-gmmlib_18.4.1_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-igc-core_18.50.1270_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-igc-opencl_18.50.1270_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-opencl_19.04.12237_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-ocloc_19.04.12237_amd64.deb &&\
    sudo dpkg -i *.deb
RUN python3 -m pip install cython
RUN git clone https://github.com/opencv/dldt.git --recursive -b ${DLDT_VERSION} && \
    cd dldt/inference-engine && \
    bash install_dependencies.sh && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$HOME/dldt_dist \
        -DCMAKE_BUILD_TYPE=Release \
        -DGEMM=OPENBLAS \
        -DENABLE_PYTHON=ON \
    && \
    make -j4 && make install && \
    cd .. && rm -rf build
RUN cd dldt/model-optimizer && python3 -m pip install --user -r requirements_onnx.txt

FROM nvidia/cuda:10.0-cudnn7-devel AS ci-base
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
    libopenblas-dev \
    curl \
    wget \
    unzip \
    sudo \
    ninja-build \
    libopencv-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main \
    >> /etc/apt/sources.list.d/llvm.list && \
    wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add - && \
    apt-get update && apt-get install -y --force-yes llvm-6.0 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
COPY --from=ngraph /root/ngraph_dist /root/ngraph_dist
COPY --from=ngraph /root/.local /root/.local
COPY --from=tvm /root/tvm_dist /usr/local/
COPY --from=tvm /root/dmlc_core_dist /usr/local/
COPY --from=tvm /root/dlpack_dist /usr/local/
COPY --from=tvm /usr/tvm/include/tvm/runtime /usr/local/include/tvm/runtime
RUN mkdir neo && cd neo && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-gmmlib_18.4.1_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-igc-core_18.50.1270_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-igc-opencl_18.50.1270_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-opencl_19.04.12237_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-ocloc_19.04.12237_amd64.deb &&\
    sudo dpkg -i *.deb
COPY --from=dldt /root/dldt_dist /usr/local/
COPY --from=dldt /dldt/inference-engine/include /root/dldt/inference-engine/include
COPY --from=dldt /dldt/inference-engine/bin/intel64/Release/lib /root/dldt/inference-engine/bin/intel64/Release/lib
COPY --from=dldt /root/.local /root/.local
COPY --from=dldt /dldt/model-optimizer /root/dldt/model-optimizer
ENV PYTHONPATH=/root/dldt/model-optimizer:${PYTHONPATH}
RUN python3 -m pip install --user decorator attrs
ENV PYTHONPATH=/usr/tvm/python:/usr/tvm/topi/python:/usr/tvm/nnvm/python/:/usr/tvm/vta/python:${PYTHONPATH}
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/root/.local/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/root/ngraph_dist/lib:${LD_LIBRARY_PATH}

FROM ci-base AS dev-base
RUN git clone --recursive https://github.com/pfnet-research/chainer-compiler.git
RUN python3 -m pip install --user gast
RUN export CHAINER_VERSION=$(python3 -c "import imp;print(imp.load_source('_version','chainer-compiler/third_party/chainer/chainer/_version.py').__version__)") && \
    python3 -m pip install --user cupy-cuda100==$CHAINER_VERSION
RUN CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 MAKEFLAGS=-j8 \
    CHAINERX_NVCC_GENERATE_CODE=arch=compute_70,code=sm_70 \
    python3 -m pip install --user chainer-compiler/third_party/chainer[test]
RUN python3 -m pip install --user chainer-compiler/third_party/onnx-chainer[test-gpu]
RUN mkdir -p chainer-compiler/build && \
    cd chainer-compiler/build && \
    cmake .. \
        -DCHAINER_COMPILER_ENABLE_CUDA=ON \
        -DCHAINER_COMPILER_ENABLE_CUDNN=ON \
        -DCHAINER_COMPILER_ENABLE_OPENCV=ON \
        -DCHAINER_COMPILER_ENABLE_PYTHON=ON \
        -DCHAINER_COMPILER_NGRAPH_DIR=$HOME/ngraph_dist \
        -DCHAINER_COMPILER_DLDT_DIR=$HOME/dldt \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DCHAINER_COMPILER_ENABLE_TVM=ON \
        -DCHAINERX_BUILD_CUDA=ON \
        -DCHAINERX_BUILD_PYTHON=ON \
        -DCHAINER_COMPILER_PREBUILT_CHAINERX_DIR=$(pip3 show chainer | awk '/^Location: / {print $2}')/chainerx \
        && \
    make -j8

FROM dev-base
LABEL author="Daisuke Tanaka <duaipp@gmail.com>"
# Optional packages.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    less \
    lv \
    screen \
    vim \
    zsh \
    && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
