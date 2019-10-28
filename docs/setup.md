# Set up guide

Currently, Chainer compiler does not provide any install scripts. You need to try it at the top of the chainer-compiler repository or manually set up paths by yourself.

## Prerequisites

If you feel lucky, you can proceed to [building chainer compiler](#building-chainer-compiler) and come back to this section if your build fails.

### Setting up toolchains

```shell-session
$ apt-get install git curl wget build-essential cmake libopenblas-dev
```

### Setting up CUDA

Download and install the latest CUDA Toolkit from NVIDIA's website.

- https://developer.nvidia.com/cuda-downloads

#### Non-GPU environment

Chainer compiler can be built on the non-GPU environment because it just requires a few methods in CUDA Runtime API (`cuda_runtime.h`), and CMake's `FindCUDA` only requires NVCC.

There are two ways to build Chainer compiler without CUDA.

##### Specifying `CHAINER_COMPILER_ENABLE_CUDA`

You can enable CUDA by specifying `CHAINER_COMPILER_ENABLE_CUDA=ON`.

##### Using stub driver

NVIDIA provides [external package repositories](https://developer.download.nvidia.com/compute/cuda/repos/) for many Linux distributions, and you can introduce the specific package for non-GPU environment via the package manager instead of installing all components including the non-stub driver.

For example, you only need `cuda-cudart-dev-10-0` and `cuda-nvcc-10-0` on Ubuntu 18.04:

```shell-session
$ apt-get update
$ apt-get install --no-install-recommends gnupg2 curl ca-certificates
$ curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add
$ echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list
$ echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
$ apt-get update
$ apt-get install --no-install-recommends cuda-cudart-dev-10-0 cuda-nvcc-10-0
Reading package lists... Done
Building dependency tree
Reading state information... Done
The following additional packages will be installed:
  cuda-cudart-10-0 cuda-driver-dev-10-0 cuda-license-10-0
  cuda-misc-headers-10-0
The following NEW packages will be installed:
  cuda-cudart-10-0 cuda-cudart-dev-10-0 cuda-driver-dev-10-0 cuda-license-10-0
  cuda-misc-headers-10-0 cuda-nvcc-10-0
0 upgraded, 6 newly installed, 0 to remove and 10 not upgraded.
Need to get 21.2 MB of archives.
After this operation, 72.7 MB of additional disk space will be used.
Do you want to continue? [Y/n] y
...
```

The detailed instruction is described in the [official document](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation), and [`Dockerfile`s](https://gitlab.com/nvidia/cuda
) for [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/) image are also helpful (see `base` and `devel` images).

Ref. [Installing CUDA as a non-root user with no GPU - Stack Overflow](https://stackoverflow.com/questions/33842543/installing-cuda-as-a-non-root-user-with-no-gpu)

Note: `cuda-driver-dev-*` only installs a stub driver (`/usr/local/cuda-10.0/lib64/stubs/libcuda.so`) instead of the actual implementation.

### Setting up Protocol Buffers

Chainer compiler requires `libprotobuf.so` and `protoc` command at build time. You can install them by using the package manager or [building yourself](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md).

```shell-session
$ apt-get install libprotobuf-dev protobuf-compiler
```

### Setting up Python and the dependent Python packages

Chainer compiler requires Python and some libraries to build.

Check out the source code if you didn't:

```shell-session
$ git clone https://github.com/pfnet-research/chainer-compiler.git
$ cd chainer-compiler
```

And run:

```shell-session
$ apt-get install python3 python3-pip
$ git submodule update --init
$ ONNX_ML=1 pip3 install gast==0.3.2 numpy pytest packaging onnx
```

You need to install Chainer in a submodule directory (`third_party/chainer`).

```shell-session
$ CHAINER_BUILD_CHAINERX=1 pip3 install third_party/chainer   # install ChainerX without cuda
or
$ CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1\
  CUDNN_ROOT_DIR=/path/to/cudnn_root pip3 install third_party/chainer   # install ChainerX with cuda
```

You need to install `third_party/chainer` to run its Python interface which requires ABI compatibility.

## Building Chainer compiler from source

You may build Chainer compiler from source or install a python package via pip.
In this section, we explain how to build from source.

1. Check out Chainer compiler repository to your local environment:

```shell-session
$ git clone https://github.com/pfnet-research/chainer-compiler.git
```

2. Build Chainer compiler

```bash
$ mkdir -p build
$ cd build

$ cmake -DCHAINER_COMPILER_ENABLE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0 ..
or
$ cmake -DCHAINER_COMPILER_ENABLE_CUDA=OFF ..

$ make
```

### Setting up OpenCV (optional)

Install OpenCV via the package manager or by building the source code.

```shell-session
$ apt-get install libopencv-dev
```

You can enable `tools/train_imagenet` by adding `-DCHAINER_COMPILER_ENABLE_OPENCV=ON`.

### Other CMake variables

There is a few more CMake variables. Try

```shell-session
$ grep '^option' CMakeLists.txt
```

to see the list of supported options.

TODO(hamaji): Document some of them. Notably,

1. `CHAINER_COMPILER_ENABLE_CUDNN` is important for EspNet.
1. `CHAINER_COMPILER_ENABLE_PYTHON` is necessary for [Python interface](../python/chainer_compiler.py).

## Installing Chainer compiler via pip (optional)

You can install Chainer compiler as a python package.
In this case, you do not need to build Chainer compiler from source.

```bash
$ CUDA_PATH=/path/to/cuda CUDNN_ROOT_DIR=/path/to/cudnn_root pip3 install .
```

## Run tests

```shell-session
$ cd build
$ make test
$ cd ..
$ ./scripts/runtests.py
$ PYTHONPATH=. pytest tests  # If you set -DCHAINER_COMPILER_ENABLE_PYTHON=ON
```

Now you can proceed to [example usage](usage.md).
