#!/bin/bash

set -eux

cat /proc/cpuinfo
cat /proc/meminfo

mkdir -p data
cd data
wget -q https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz && \
    tar -xzf resnet50.tar.gz
cd ..
