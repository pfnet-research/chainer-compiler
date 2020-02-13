#!/usr/bin/python3
#
# Expected workflow:
#
# $ # Do whatever you want for chainer_compiler/elichika/typing
# $ git commit -a -m 'A wonderful change'
# $ ./scripts/rebase_elichika_type_tests.py
# $ git diff  # Review the changes for test expectations
# $ git commit -a -m 'Test update for the great change'
#

import importlib
import os
import re
import six
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root))
sys.path.insert(0, os.path.join(root, 'tests'))

from chainer_compiler.elichika.testtools import type_inference_tools


def rebase_testcase(filename, model_name, gen_model_fn):
    model, forward_args = gen_model_fn()
    id2type, id2node = type_inference_tools.generate_type_inference_results(
        model, forward_args, is_debug=False)
    sio = six.StringIO()
    type_inference_tools.generate_assertion("id2type", id2type, id2node, sio)

    with open(filename) as f:
        code = f.read()

    begin_marker = '# === BEGIN ASSERTIONS for {} ==='.format(model_name)
    end_marker = '# === END ASSERTIONS for {} ==='.format(model_name)
    regexp = begin_marker + '.*?' + end_marker
    new_assertions = begin_marker + '\n' + sio.getvalue() + ' ' * 8 + end_marker
    code, num_replaced = re.subn(regexp, new_assertions, code,
                                 count=1, flags=re.DOTALL | re.MULTILINE)
    if not num_replaced:
        raise RuntimeError('No assertion markers for {}'.format(model_name))

    with open(filename, 'w') as f:
        f.write(code)


def main():
    unsupported_models = set(['ResNet50', 'Decoder', 'E2E', 'GoogLeNet'])

    for test_module_name in ['EspNet', 'Models']:
        filename = 'tests/elichika_typing/{}_test.py'.format(test_module_name)
        assert os.path.exists(filename)
        module = importlib.import_module(
            'elichika_typing.{}_test'.format(test_module_name))
        for symbol_name in dir(module):
            matched = re.match(r'gen_(\w+)_model', symbol_name)
            if not matched:
                continue
            model_name = matched.group(1)
            if model_name in unsupported_models:
                continue
            print('Rebase test for {}...'.format(model_name))
            rebase_testcase(filename, model_name, getattr(module, symbol_name))


if __name__ == '__main__':
    main()
