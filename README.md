# Oniku

## Doc

https://docs.google.com/document/d/1j07zkE71NxQjhd6DynpT7i5LTwr1rCbWHJbk286pG2k/edit#

## Usage

```shell-session
$ ./setup.sh
$ cmake .
$ make
$ ./runtests.sh
```

### Run ResNet50

```shell-session
$ wget https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz
$ tar -xvzf resnet50.tar.gz
$ ONIKU_DEVICE=cuda ./runtests.sh resnet50
```

VGG19 works, too. Use the following ONNX model:

https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz
