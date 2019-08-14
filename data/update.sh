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
