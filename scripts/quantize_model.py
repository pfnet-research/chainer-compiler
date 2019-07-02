import argparse
import json
import onnx
import quantize


def main():
    parser = argparse.ArgumentParser(
        description='Quantize model with specified parameters')
    parser.add_argument('--no_per_channel', '-t',
                        action='store_true', default=False)
    parser.add_argument('--nbits', type=int, default=8)
    parser.add_argument('--quantization_mode', default='Integer',
                        choices=('Integer', 'QLinear'))
    parser.add_argument('--static', '-s', action='store_true', default=False)
    parser.add_argument('--asymmetric_input_types',
                        action='store_true', default=False)
    parser.add_argument('--input_quantization_params', default='')
    parser.add_argument('--output_quantization_params', default='')
    parser.add_argument('model')
    parser.add_argument('output')
    args = parser.parse_args()

    if args.no_per_channel:
        args.per_channel = False
    del args.no_per_channel

    if args.quantization_mode == 'QLinear':
        args.quantization_mode = quantize.QuantizationMode.QLinearOps
    else:
        args.quantization_mode = quantize.QuantizationMode.IntegerOps

    if len(args.input_quantization_params) != 0:
        args.input_quantization_params = json.loads(
            args.input_quantization_params)
    else:
        args.input_quantization_params = None

    if len(args.output_quantization_params) != 0:
        args.output_quantization_params = json.loads(
            args.output_quantization_params)
    else:
        args.output_quantization_params = None

    # Load the onnx model
    model_file = args.model
    model = onnx.load(model_file)
    del args.model

    output_file = args.output
    del args.output

    # Quantize
    print('Quantize config: {}'.format(vars(args)))
    quantized_model = quantize.quantize(model, **vars(args))

    print('Saving "{}" to "{}"'.format(model_file, output_file))

    # Save the quantized model
    onnx.save(quantized_model, output_file)


if __name__ == '__main__':
    main()
