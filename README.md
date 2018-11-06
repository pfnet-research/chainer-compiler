# Oniku

## Doc

https://docs.google.com/document/d/1j07zkE71NxQjhd6DynpT7i5LTwr1rCbWHJbk286pG2k/edit#

## Usage

Setup prerequisites (e.g., ChainerX and protoc) first in

https://github.com/pfnet/oniku/wiki/Building-Oniku

```shell-session
$ ./setup.sh
$ mkdir build
$ cd build
$ cmake ..
$ make
$ cd ..
$ ./scripts/runtests.py
```

You can disable the CUDA support by specifying `-DONIKU_BUILD_CUDA=OFF` for `cmake`.

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
