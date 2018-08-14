#!/usr/bin/python3

import os
import sys
import subprocess

import gen_backprop_tests


class TestCase(object):

    def __init__(self, dirname, name):
        self.dirname = dirname
        self.name = name

    def test_dir(self):
        return os.path.join(self.dirname, self.name)


NODE_TEST = 'onnx/onnx/backend/test/data/node'

TEST_CASES = [
    TestCase(NODE_TEST, 'test_relu'),

    TestCase(NODE_TEST, 'test_add'),
    TestCase(NODE_TEST, 'test_add_bcast'),
    TestCase(NODE_TEST, 'test_sub'),
    TestCase(NODE_TEST, 'test_sub_bcast'),
    TestCase(NODE_TEST, 'test_sub_example'),
    TestCase(NODE_TEST, 'test_mul'),
    TestCase(NODE_TEST, 'test_mul_bcast'),
    TestCase(NODE_TEST, 'test_mul_example'),
    TestCase(NODE_TEST, 'test_div'),
    TestCase(NODE_TEST, 'test_div_bcast'),
    TestCase(NODE_TEST, 'test_div_example'),
    TestCase(NODE_TEST, 'test_neg'),
    TestCase(NODE_TEST, 'test_neg_example'),

    # TODO(xchainer): Support non-2D dot.
    # terminate called after throwing an instance of 'xchainer::NotImplementedError'
    #   what():  dot does not support rhs operand with ndim > 2
    TestCase(NODE_TEST, 'test_matmul_2d'),

    TestCase(NODE_TEST, 'test_basic_conv_with_padding'),
    TestCase(NODE_TEST, 'test_basic_conv_without_padding'),
    TestCase(NODE_TEST, 'test_conv_with_strides_no_padding'),
    TestCase(NODE_TEST, 'test_conv_with_strides_padding'),
    TestCase(NODE_TEST, 'test_conv_with_strides_and_asymmetric_padding'),

    # TODO(hamaji): auto_pad is not supported.
    # TODO(hamaji): Support non-2D pools.
    TestCase(NODE_TEST, 'test_maxpool_2d_default'),
    TestCase(NODE_TEST, 'test_maxpool_2d_pads'),
    TestCase(NODE_TEST, 'test_maxpool_2d_precomputed_pads'),
    TestCase(NODE_TEST, 'test_maxpool_2d_precomputed_strides'),
    TestCase(NODE_TEST, 'test_maxpool_2d_strides'),
    TestCase(NODE_TEST, 'test_averagepool_2d_default'),
    TestCase(NODE_TEST, 'test_averagepool_2d_precomputed_pads'),
    TestCase(NODE_TEST, 'test_averagepool_2d_precomputed_pads_count_include_pad'),
    TestCase(NODE_TEST, 'test_averagepool_2d_precomputed_strides'),
    TestCase(NODE_TEST, 'test_averagepool_2d_strides'),
    TestCase(NODE_TEST, 'test_averagepool_2d_pads'),
    TestCase(NODE_TEST, 'test_averagepool_2d_pads_count_include_pad'),

    # TODO(xchainer): Support negative reshape.
    TestCase(NODE_TEST, 'test_reshape_extended_dims'),
    TestCase(NODE_TEST, 'test_reshape_one_dim'),
    TestCase(NODE_TEST, 'test_reshape_reduced_dims'),
    TestCase(NODE_TEST, 'test_reshape_reordered_dims'),
    # Broadcast from (3, 1) to (2, 1, 6).
    # TODO(xchainer): Do we really want to support this?
    # TestCase(NODE_TEST, 'test_expand_dim_changed'),
    TestCase(NODE_TEST, 'test_expand_dim_unchanged'),

    TestCase(NODE_TEST, 'test_gemm_nobroadcast'),
    TestCase(NODE_TEST, 'test_gemm_broadcast'),

    # TODO(hamaji): Investigate 3D softmax ops do not agree (though
    # xChainer agrees with Chainer).
    # TODO(hamaji): Relax equality check for "large_number" tests.
    TestCase(NODE_TEST, 'test_softmax_example'),
    TestCase(NODE_TEST, 'test_logsoftmax_example_1'),

    TestCase(NODE_TEST, 'test_sum_example'),
    TestCase(NODE_TEST, 'test_sum_one_input'),
    TestCase(NODE_TEST, 'test_sum_two_inputs'),

    TestCase(NODE_TEST, 'test_reduce_sum_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_sum_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_sum_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_mean_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_mean_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_mean_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_mean_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_mean_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_mean_keepdims_random'),

    TestCase(NODE_TEST, 'test_batchnorm_example'),
    TestCase(NODE_TEST, 'test_batchnorm_epsilon'),
]

gen_backprop_tests.replace_id(__builtins__)
for backprop_test in gen_backprop_tests.get_backprop_tests():
    dirname = 'out'
    name = 'backprop_test_' + backprop_test.name
    if not os.path.exists(os.path.join(dirname, name)):
        backprop_test.generate()
    assert os.path.exists(os.path.join(dirname, name))
    TEST_CASES.append(TestCase(dirname, name))


def main():
    if os.path.exists('Makefile'):
        subprocess.check_call(['make', '-j4'])
    elif os.path.exists('build.ninja'):
        subprocess.check_call('ninja')

    test_cnt = 0
    fail_cnt = 0
    for test_case in TEST_CASES:
        sys.stdout.write('%s... ' % test_case.name)
        args = ['tools/run_onnx', '--test', test_case.test_dir(), '--quiet', '1']
        if test_case.name.startswith('backprop_'):
            args.extend(['--backprop', '1'])
        try:
            test_cnt += 1
            subprocess.check_call(args)
            sys.stdout.write('OK\n')
        except subprocess.CalledProcessError:
            fail_cnt += 1
            sys.stdout.write('FAIL\n')
    if fail_cnt:
        print('%d/%d tests failed!' % (fail_cnt, test_cnt))
    else:
        print('ALL %d tests OK!' % test_cnt)


main()
