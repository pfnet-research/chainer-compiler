import glob

import numpy as np
import onnx

import input_rewriter


def _get_inputs(xgraph):
    initializer_names = set(init.name for init in xgraph.initializer)
    inputs = [input for input in xgraph.input
              if input.name not in initializer_names]
    return inputs


def test_rewrite_onnx_file():
    input_rewriter.rewrite_onnx_file(
        'out/backprop_test_mnist_mlp/model.onnx',
        'out/backprop_test_mnist_mlp/model_bs3.onnx',
        [input_rewriter.Type(shape=(3, 784)),
         input_rewriter.Type(shape=(3, 10))])
    xmodel = onnx.load('out/backprop_test_mnist_mlp/model_bs3.onnx')
    xgraph = xmodel.graph

    def get_shape(vi):
        return tuple([d.dim_value for d in vi.type.tensor_type.shape.dim])

    inputs = _get_inputs(xgraph)

    assert 1 == inputs[0].type.tensor_type.elem_type
    assert 1 == inputs[1].type.tensor_type.elem_type
    assert (3, 784) == get_shape(inputs[0])
    assert (3, 10) == get_shape(inputs[1])
    assert 1 == xgraph.output[0].type.tensor_type.elem_type
    assert () == get_shape(xgraph.output[0])
    for init in xgraph.initializer:
        assert 1 == init.data_type


def test_rewrite_onnx_testdir():
    input_rewriter.rewrite_onnx_testdir(
        'out/backprop_test_mnist_mlp',
        'out/backprop_test_mnist_mlp_fp64',
        [input_rewriter.Type(dtype=np.float64),
         input_rewriter.Type(dtype=np.float64)])
    xmodel = onnx.load('out/backprop_test_mnist_mlp_fp64/model.onnx')
    xgraph = xmodel.graph

    assert 11 == xgraph.input[0].type.tensor_type.elem_type
    assert 11 == xgraph.input[1].type.tensor_type.elem_type
    assert 11 == xgraph.output[0].type.tensor_type.elem_type
    for init in xgraph.initializer:
        assert 11 == init.data_type

    for tensor_proto in glob.glob(
            'out/backprop_test_mnist_mlp_fp64/test_data_set_0/*.pb'):
        xtensor = onnx.load_tensor(tensor_proto)
        assert 11 == xtensor.data_type
