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
$ ./setup.sh
$ ONIKU_DEVICE=cuda ./runtests.sh data/resnet50
```

VGG19 works, too:

```shell-session
$ wget https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz
$ tar -xvzf vgg19.tar.gz
$ ONIKU_DEVICE=cuda ./runtests.sh vgg19
```
