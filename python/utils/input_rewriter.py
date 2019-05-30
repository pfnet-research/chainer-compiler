import glob
import os
import sys

import numpy as np
import onnx
from onnx import mapping, numpy_helper, shape_inference


class Type(object):
    def __init__(self, dtype=None, shape=None):
        self.dtype = dtype
        self.shape = shape if shape is None else tuple(shape)


def rewrite_onnx_tensor(xtensor, new_type):
    value = numpy_helper.to_array(xtensor)
    if new_type.shape is not None and value.shape != new_type.shape:
        sys.stderr.write('The shape of tensor `%s` was changed from '
                         '%s to %s and values were randomized\n' %
                         (xtensor.name, value.shape, new_type.shape))
        value = np.random.rand(*new_type.shape)
    if new_type.dtype is not None:
        value = value.astype(new_type.dtype)
    xtensor.CopyFrom(numpy_helper.from_array(value, xtensor.name))


def rewrite_onnx_tensor_type(xtensor_type, new_type):
    if new_type.dtype is not None:
        xtensor_type.elem_type = mapping.NP_TYPE_TO_TENSOR_TYPE[new_type.dtype]
    if new_type.shape is not None:
        xtensor_type.shape.Clear()
        for d in new_type.shape:
            xtensor_type.shape.dim.add().dim_value = d


def rewrite_onnx_model(xmodel, new_input_types):
    xgraph = xmodel.graph

    # Update parameter types.
    new_param_dtype = new_input_types[0].dtype
    if new_param_dtype is not None:
        old_param_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[
            xgraph.input[0].type.tensor_type.elem_type]
        initializers = {i.name: i for i in xgraph.initializer}
        for input in xgraph.input:
            if input.name not in initializers:
                continue
            param_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[
                input.type.tensor_type.elem_type]
            if param_dtype != old_param_dtype:
                sys.stderr.write(
                    'WARNING: This assumes all parameters have the same dtype '
                    'as the first input (%s) but the dtype of `%s` is %s\n' %
                    (old_param_dtype, input.name, param_dtype))
                continue

            new_type = Type(dtype=new_param_dtype)
            rewrite_onnx_tensor(initializers[input.name], new_type)
            rewrite_onnx_tensor_type(input.type.tensor_type, new_type)

    initializer_names = set(init.name for init in xgraph.initializer)
    inputs = [input for input in xgraph.input
              if input.name not in initializer_names]
    assert len(new_input_types) <= len(inputs)

    # Update input types.
    for input_type, input in zip(new_input_types, inputs):
        rewrite_onnx_tensor_type(input.type.tensor_type, input_type)

    for vi in xgraph.value_info:
        vi.type.Clear()
    for vi in xgraph.output:
        vi.type.Clear()

    return shape_inference.infer_shapes(xmodel)


def rewrite_onnx_file(model_filename, out_filename, new_input_types):
    xmodel = onnx.load(model_filename)
    xmodel = rewrite_onnx_model(xmodel, new_input_types)
    onnx.save(xmodel, out_filename)
    return xmodel


def rewrite_onnx_testdir(model_testdir, out_testdir, new_input_types):
    os.makedirs(out_testdir, exist_ok=True)
    xmodel = rewrite_onnx_file(os.path.join(model_testdir, 'model.onnx'),
                               os.path.join(out_testdir, 'model.onnx'),
                               new_input_types)

    name_to_type = {}
    for vi in (list(xmodel.graph.input) +
               list(xmodel.graph.value_info) +
               list(xmodel.graph.output)):
        dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[vi.type.tensor_type.elem_type]
        shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        name_to_type[vi.name] = Type(dtype=dtype, shape=shape)

    for test_set in glob.glob(os.path.join(model_testdir, 'test_data_set_*')):
        dest_dir = os.path.join(out_testdir, os.path.basename(test_set))
        os.makedirs(dest_dir, exist_ok=True)
        for tensor_proto in glob.glob(os.path.join(test_set, '*.pb')):
            xtensor = onnx.load_tensor(tensor_proto)
            if xtensor.name not in name_to_type:
                raise RuntimeError('Unknown tensor name: %s' % xtensor.name)
            rewrite_onnx_tensor(xtensor, name_to_type[xtensor.name])

            out_tensor_proto = os.path.join(dest_dir,
                                            os.path.basename(tensor_proto))
            onnx.save_tensor(xtensor, out_tensor_proto)
