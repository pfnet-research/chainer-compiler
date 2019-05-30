#!/usr/bin/python3
#
# Rewrite input types of an ONNX model.
#
# Usage:
#
# $ python/utils/rewrite_input.py from to '[{"shape":[2,3,224,224]}]'
# $ python/utils/rewrite_input.py from.onnx to.onnx '[{"dtype":"float16"}]'
#

import argparse
import json
import os
import sys

import numpy as np

import input_rewriter


def json_to_types(js_list):
    assert isinstance(js_list, list)
    types = []
    for js_type in js_list:
        dtype = None
        shape = None
        if 'dtype' in js_type:
            dtype = np.dtype(js_type['dtype'])
        if 'shape' in js_type:
            shape = js_type['shape']
        types.append(input_rewriter.Type(dtype=dtype, shape=shape))
    return types


def main():
    parser = argparse.ArgumentParser(description='Rewrite inputs of ONNX')
    parser.add_argument('input', type=str,
                        help='The input ONNX model file or test directory.')
    parser.add_argument('output', type=str,
                        help='The output ONNX model file or test directory.')
    parser.add_argument('types', type=str,
                        help='A JSON for new input types.')
    args = parser.parse_args()

    new_input_types = json_to_types(json.loads(args.types))

    if os.path.isdir(args.input):
        input_rewriter.rewrite_onnx_testdir(args.input, args.output,
                                            new_input_types)
    else:
        input_rewriter.rewrite_onnx_file(args.input, args.output,
                                         new_input_types)


if __name__ == '__main__':
    main()
