# Example usage

This document assumes you have successfully built the code by following [set up guide](setup.md).

## Run existing ONNX models

In this section, we run an existing ONNX model by the runtime of Chainer compiler.

To run ShuffleNet model in the repository, run

```shell-session
$ make -C build
$ ./build/tools/run_onnx --device cuda --test data/shufflenet --trace
```

The command above uses inputs and outputs in `data/shufflenet/test_data_set_?` to feed and verify the model.

VGG19 works, too:

```shell-session
$ wget https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz
$ tar -xvzf vgg19.tar.gz
$ ./build/tools/run_onnx --device cuda --test vgg19 --trace
```

You can run more models defined in [ONNX's tests](https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/real):

```shell-session
$ ./scripts/runtests.py onnx_real -g
```

## Generate a training graph from your Chainer model

First prepare a model which outputs a loss value as a single float. Here we use `ch2o/tests/model/Resnet_with_loss.py` as a sample.

You can generate an ONNX graph for inference by

```shell-session
$ PYTHONPATH=ch2o python3 ch2o/tests/model/Resnet_with_loss.py resnet50
```

Your ONNX file should be stored as `resnet50/model.onnx`. This ONNX file contains the computation graph for inference. Then, you can generate a training graph by

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

## Use chainer-compiler from Chainer

To use chainer-compiler from Chainer code, you first need to install Chainer from source code, for example:

```shell-session
$ CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 MAKEFLAGS=-j2 pip3 install --user third_party/chainer
```

You also need to rebuild chainer compiler with three flags:

```shell-session
$ cd build
$ cmake -DCHAINER_COMPILER_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCHAINERX_BUILD_PYTHON=ON ..
$ make
$ cd ..
```

Check if everything is fine by running pytest:

```shell-session
$ pytest python
```

You need to wrap your model by `chainer_compiler.compile`. E.g.,

```python
model = YourModel()
model = chainer_compiler.compile(model, dump_onnx=args.dump_onnx)
```

See examples directory for more details. You can run the MNIST example by

```shell-session
$ python3 examples/mnist/train_mnist.py --compile --dump_onnx -d cuda
```
