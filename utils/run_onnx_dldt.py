import glob
import logging as log
import os
import sys

import numpy as np
import onnx
import onnx.numpy_helper

import run_onnx_util


def load_test_data(data_dir, input_names, output_names):
    inout_values = []
    for kind, names in [('input', input_names), ('output', output_names)]:
        names = list(names)
        values = []
        for pb in sorted(glob.glob(os.path.join(data_dir, '%s_*.pb' % kind))):
            with open(pb, 'rb') as f:
                tensor = onnx.TensorProto()
                tensor.ParseFromString(f.read())
            if tensor.name in names:
                name = tensor.name
                names.remove(name)
            else:
                name = names.pop(0)
            values.append((name, onnx.numpy_helper.to_array(tensor)))
        inout_values.append(values)
    return tuple(inout_values)


def onnx_input_output_names(onnx_filename):
    onnx_model = onnx.load(onnx_filename)
    initializer_names = set()
    for initializer in onnx_model.graph.initializer:
        initializer_names.add(initializer.name)

    input_names = []
    for input in onnx_model.graph.input:
        if input.name not in initializer_names:
            input_names.append(input.name)

    output_names = []
    for output in onnx_model.graph.output:
        output_names.append(output.name)

    return input_names, output_names


def inference(args, model_xml, model_bin, inputs, outputs):
    from openvino.inference_engine import IENetwork
    from openvino.inference_engine import IEPlugin
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    log.info('Loading network files:\n\t{}\n\t{}'.format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == 'CPU':
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if
                                l not in supported_layers]
        if not_supported_layers:
            log.error('Folowing layers are not supported by the plugin for '
                      'specified device {}:\n {}'.format(
                          plugin.device, ', '.join(not_supported_layers)))
            log.error('Please try to specify cpu extensions library path in '
                      'sample\'s command line parameters using '
                      '--cpu-extension command line argument')
            sys.exis(1)

    assert len(net.inputs) == len(inputs)
    ie_inputs = {}
    for item in inputs:
        assert item[0] in set(net.inputs.keys())
        ie_inputs[item[0]] = item[1]

    log.info('Loading model to the plugin')
    exec_net = plugin.load(network=net)

    res = exec_net.infer(inputs=ie_inputs)

    assert len(res) == len(outputs)
    for name, output in outputs:
        assert name in res
        actual_output = res[name]
        np.testing.assert_allclose(output, actual_output, rtol=1e-3, atol=1e-4)
        log.info('{}: OK'.format(name))
    log.info('ALL OK')

    def compute():
        exec_net.infer(inputs=ie_inputs)

    return run_onnx_util.run_benchmark(compute, args.iterations)


def run(args):
    test_dir = os.path.abspath(args.test_dir)
    test_dir_name = test_dir.split(os.path.sep)[-1]

    onnx_filename = os.path.join(test_dir, 'model.onnx')
    input_names, output_names = onnx_input_output_names(onnx_filename)
    test_data_dir = os.path.join(test_dir, 'test_data_set_0')
    inputs, outputs = load_test_data(test_data_dir, input_names, output_names)

    mo_output_dir = os.path.join('out', 'dldt_{}.{}'.format(
        test_dir_name, args.data_type.lower()))
    mo_model_xml = os.path.join(mo_output_dir, 'model.xml')
    mo_model_bin = os.path.join(mo_output_dir, 'model.bin')

    # make optimized model
    not_found_mo = True
    if not os.path.exists(mo_output_dir):
        os.makedirs(mo_output_dir, exist_ok=True)
    else:
        if os.path.exists(mo_model_xml) and os.path.exists(mo_model_bin):
            not_found_mo = False
    if args.force_mo or not_found_mo:
        args.input_model = onnx_filename
        args.output_dir = mo_output_dir
        from mo.main import driver
        driver(args)
    else:
        log.basicConfig(
            format="[ %(levelname)s ] %(message)s", level=args.log_level,
            stream=sys.stdout)

    # compute inference engine
    inference(args, mo_model_xml, mo_model_bin, inputs, outputs)


def main():
    """Calculate ONNX model using DLDT

    1. model-optimizer(mo) part

    Add mo directory path to `$PYTHONPATH` to use mo module. Options are same
    with model-optimizer(mo), see "common" group parser. Picked up:

      - --input_model: get model path from 'test_dir', don't use.
      - --output_dir: set path automatically, don't use.
      - --log_level: default: ERROR, if set '-g', overwritten by 'DEBUG'
      - --data_type: choose  from 'FP16', 'FP32', 'half', 'float'
      - --disable_fusing, --disablegfusing, --move-to-process,
          --reverse_input_channels: mo's option for ONNX to DLDT Graph

    Output path is 'out/dldt_{test_dir}'. If model.{xml,bin} has alread
    existed, skip mo part unless set --force_mo option.

    2. inference-engine(ie) part

    Add ie python api path to `$PYTHONPATH` to use openvino module. Logic
    references ie_bridges example.

    """
    from mo.utils.cli_parser import get_onnx_cli_parser
    parser = get_onnx_cli_parser(parser=None)  # setup for mo
    parser.add_argument('test_dir')
    parser.add_argument('--debug', '-g', action='store_true')
    parser.add_argument('--force_mo', action='store_true')
    parser.add_argument('--iterations', '-I', type=int, default=1)
    # for inference-engine
    parser.add_argument(
        '--device', choices=['CPU', 'GPU', 'MYRIAD'], default='CPU')
    parser.add_argument(
        '--plugin_dir', help='Path to a plugin folder', default=None)
    parser.add_argument(
        '--cpu_extension', help='Required for CPU custom layers. '
        'MKLDNN (CPU)-targeted custom layers. Absolute path to a shared '
        'library with the kernels implementations', default=None
    )
    args = parser.parse_args()

    args.framework = 'onnx'
    if args.debug:
        args.log_level = 'DEBUG'

    run(args)


if __name__ == '__main__':
    main()
