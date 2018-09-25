# Oniku

## Doc

https://docs.google.com/document/d/1j07zkE71NxQjhd6DynpT7i5LTwr1rCbWHJbk286pG2k/edit#

## Usage

```shell-session
$ ./setup.sh
$ cmake .
$ make
$ ./scripts/runtests.py
```

### Run ResNet50 with XCVM backend

```shell-session
$ ./setup.sh
$ ./tools/run_onnx --dump_xcvm --device cuda --test data/resnet50 --trace

```

VGG19 works, too:

```shell-session
$ wget https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz
$ tar -xvzf vgg19.tar.gz
$ ./tools/run_onnx --dump_xcvm --device cuda --test vgg19 --trace
```

You can run more models defined in [ONNX's tests](https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/real):

```shell-session
$ ./scripts/runtests.py real_onnx -g
```
