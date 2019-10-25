#!/bin/bash

set -eux

# MNIST from
# https://github.com/onnx/models/tree/master/vision/classification/mnist
rm -fr mnist mnist.tar.gz
wget https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz
tar -xvzf mnist.tar.gz
git add mnist

# ShuffleNet from
# https://github.com/onnx/models/tree/master/vision/classification/shufflenet
rm -fr shufflenet shufflenet.tar.gz
wget https://s3.amazonaws.com/download.onnx/models/opset_9/shufflenet.tar.gz
tar -xvzf shufflenet.tar.gz
git add shufflenet

# SqueezeNet form
# https://github.com/onnx/models/tree/master/vision/classification/squeezenet
rm -rf squeezenet1.1 squeezenet1.1.tar.gz
wget "https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.tar.gz"
tar -xvzf squeezenet1.1.tar.gz
git add squeezenet1.1

# MobileNet from
# https://github.com/onnx/models/tree/master/vision/classification/mobilenet
rm -rf mobilenetv2-1.0 mobilenetv2-1.0.tar.gz
wget "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz"
tar -xvzf mobilenetv2-1.0.tar.gz
git add mobilenetv2-1.0
