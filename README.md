# Oniku

## Doc

https://docs.google.com/document/d/1j07zkE71NxQjhd6DynpT7i5LTwr1rCbWHJbk286pG2k/edit#

## Usage

```shell-session
$ ./setup.sh
$ cmake .
$ make
$ ./scripts/runtests.py
$ ./runtests.sh
```

### Run ResNet50 with XCVM backend

```shell-session
$ ./setup.sh
$ ./tools/run_onnx --dump_xcvm 1 --device cuda --test data/resnet50

```

VGG19 works, too:

```shell-session
$ wget https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz
$ tar -xvzf vgg19.tar.gz
$ ./tools/run_onnx --dump_xcvm 1 --device cuda --test vgg19
```

### Run ResNet50 with C++ backend (to be removed)

```shell-session
$ ./setup.sh
$ ONIKU_DEVICE=cuda ./runtests.sh data/resnet50
```

VGG19 works, too:

```shell-session
$ wget https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz
$ tar -xvzf vgg19.tar.gz
$ ONIKU_DEVICE=cuda ./runtests.sh vgg19
```
