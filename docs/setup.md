# Set up guide

Currently, Chainer compiler does not provide any install scripts. You need to try it at the top of chainer-compiler repository or manually set up paths by yourself.

## Prerequisites

### Setting up toolchains

```shell-session
$ apt-get install git curl wget build-essential cmake
```

### Setting up CUDA

Download and install the latest CUDA Toolkit from NVIDIA's website.

- https://developer.nvidia.com/cuda-downloads

#### Non-GPU environment

Oniku can be built on the non-GPU environment because it just requires a few methods in CUDA Runtime API (`cuda_runtime.h`), and CMake's `FindCUDA` only requires NVCC.

There are two ways to build Oniku without CUDA.

##### Specifying `CHAINER_COMPILER_BUILD_CUDA`

You can exclude CUDA dependency from Oniku by specifying `CHAINER_COMPILER_BUILD_CUDA=OFF`.

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

ONNX library used by Oniku requires `libprotobuf.so` and `protoc` command at build time. You can install them by using the package manager or [building yourself](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md).

```shell-session
$ apt-get install libprotobuf-dev protobuf-compiler
```

### Setting up Python and the dependent Python packages
Oniku requires Python and some libraries to build *(Should we use the same ONNX version used inside Oniku?)*.

```shell-session
$ apt-get install python3 python3-pip
$ pip3 install gast numpy chainer onnx==1.3.0 onnx_chainer
```

## Building Oniku

1. Check out Oniku repository to your local environment:

```shell-session
$ git clone https://github.com/pfnet/oniku.git
```

2. Run `setup.sh` (You may need to add `-DPYTHON_EXECUTABLE=python3` manually to `cmake` for ONNX):

```shell-session
$ cd oniku
$ ./setup.sh
```

3. Build Oniku

```bash
$ mkdir -p build
$ cd build

$ cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0 ..
or
$ cmake -DCHAINER_COMPILER_BUILD_CUDA=OFF ..

$ make
```

### Setting up OpenCV (optional)

Install OpenCV via the package manager or by building the source code.

```shell-session
$ apt-get install libopencv-dev
```

You can enable `tools/train_imagenet` by adding
`-DCHAINER_COMPILER_ENABLE_OPENCV=ON`.

### Other CMake variables

There is a few more CMake variables. Try

```shell-session
$ grep '^option' CMakeLists.txt
```

to see the list of supported options.

TODO(hamaji): Document some of them. Notably,

1. `CHAINER_COMPILER_ENABLE_CUDNN` is important for EspNet.
1. `CHAINER_COMPILER_ENABLE_NVTX` and `CHAINER_COMPILER_ENABLE_NVRTC` are important for tuning CUDA performance.
1. `CHAINER_COMPILER_ENABLE_PYTHON` is necessary for [Python interface](python/oniku.py).

## Usage

```shell-session
$ mkdir build
$ cd build
$ cmake ..
$ make
$ make test
$ cd ..
$ ./scripts/runtests.py
$ pytest python  # If you set -DCHAINER_COMPILER_ENABLE_PYTHON=ON
```

### Run ResNet50 with XCVM backend

```shell-session
$ ./setup.sh
$ ./build/tools/run_onnx --dump_xcvm --device cuda --test data/resnet50 --trace

```

VGG19 works, too:

```shell-session
$ wget https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz
$ tar -xvzf vgg19.tar.gz
$ ./build/tools/run_onnx --dump_xcvm --device cuda --test vgg19 --trace
```

You can run more models defined in [ONNX's tests](https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/real):

```shell-session
$ ./scripts/runtests.py onnx_real -g
```

### Generate a training graph from your Chainer model

First prepare a model which outputs a loss value as a single float. Here we use `ch2o/tests/model/Resnet_with_loss.py` as a sample.

You can generate an ONNX graph for inference by

```shell-session
$ PYTHONPATH=ch2o python3 ch2o/tests/model/Resnet_with_loss.py resnet50
```

Your ONNX file should be stored as `resnet50/model.onnx`. Then, you can generate a training graph by

```shell-session
$ ./build/tools/run_onnx --onnx resnet50/model.onnx --out_onnx resnet50/backprop.onnx --backprop --compile_only
```

There would be a bunch of ways to analyze `resnet50/backprop.onnx`. For example, if you want a text dump, run

```shell-session
$ ./build/tools/dump resnet50/backprop.onnx
```

Or investigate it programmatically by Python:

```shell-session
>>> import onnx
>>> model = onnx.ModelProto()
>>> model.ParseFromString(open('resnet50/backprop.onnx', 'rb').read())
>>> [o.name for o in model.graph.output]
```

You can also use visualizers for ONNX such as [netron](https://github.com/lutzroeder/netron).
