#!/usr/bin/python3

import os
import sys

import numpy as np
from onnx import onnx_pb


def np_array_to_onnx(name, a):
    t = onnx_pb.TensorProto(name=name, dims=a.shape)
    if a.dtype == np.float32:
        dtype = onnx_pb.TensorProto.FLOAT
        out = t.float_data
    elif a.dtype == np.uint8:
        dtype = onnx_pb.TensorProto.UINT8
        out = t.int32_data
    elif a.dtype == np.int8:
        dtype = onnx_pb.TensorProto.INT8
        out = t.int32_data
    elif a.dtype == np.uint16:
        dtype = onnx_pb.TensorProto.UINT16
        out = t.int32_data
    elif a.dtype == np.int16:
        dtype = onnx_pb.TensorProto.INT16
        out = t.int32_data
    elif a.dtype == np.int32:
        dtype = onnx_pb.TensorProto.INT32
        out = t.int32_data
    elif a.dtype == np.int64:
        dtype = onnx_pb.TensorProto.INT64
        out = t.int64_data
    elif a.dtype == np.str:
        dtype = onnx_pb.TensorProto.STRING
        out = t.string_data
    elif a.dtype == np.bool:
        dtype = onnx_pb.TensorProto.BOOL
        out = t.int32_data
    elif a.dtype == np.double:
        dtype = onnx_pb.TensorProto.DOUBLE
        out = t.double_data
    elif a.dtype == np.uint32:
        dtype = onnx_pb.TensorProto.UINT32
        out = t.uint32_data
    elif a.dtype == np.uint64:
        dtype = onnx_pb.TensorProto.UINT64
        out = t.uint32_data
    else:
        raise RuntimeError('Unsupported numpy dtype: %s' % a.dtype)
    t.data_type = dtype
    for v in a.flat:
        out.append(v)
    return t


def npz_to_onnx(npz_filename, onnx_prefix):
    data = np.load(npz_filename)
    for i, key in enumerate(data):
        t = np_array_to_onnx(key, data[key])
        with open('%s_%d.pb' % (onnx_prefix, i), 'wb') as f:
            f.write(t.SerializeToString())


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s <in.npz> <out.onnx>' % sys.argv[0])
        sys.exit(1)
    npz_to_onnx(sys.argv[1], sys.argv[2])
