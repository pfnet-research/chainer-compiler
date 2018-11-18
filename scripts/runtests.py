#!/usr/bin/python3

import argparse
import copy
import glob
import multiprocessing
import os
import re
import sys
import subprocess

import ch2o_tests
import gen_backprop_tests_oc
import gen_backprop_tests_pc
import gen_extra_test
import onnx_real_tests
from test_case import TestCase


parser = argparse.ArgumentParser(description='Run tests for oniku')
parser.add_argument('test_filter', default=None, nargs='?',
                    help='A regular expression to filter tests')
parser.add_argument('--all', '-a', action='store_true',
                    help='Run all tests')
parser.add_argument('--build_dir', '-b', default=None,
                    help='The build directory')
parser.add_argument('--jobs', '-j', type=int,
                    default=multiprocessing.cpu_count(),
                    help='Number of parallel jobs')
parser.add_argument('--show_log', action='store_true',
                    help='Show logs')
parser.add_argument('--skip_build', action='store_true',
                    help='Skip the build before running tests')
parser.add_argument('--use_gpu', '-g', action='store_true',
                    help='Run heavy tests with GPU')
parser.add_argument('--use_gpu_all', '-G', action='store_true',
                    help='Run all tests with GPU')
parser.add_argument('--fuse', action='store_true', help='Enable fusion')
parser.add_argument('--verbose', action='store_true',
                    help='Run tests with --verbose flag')
args = parser.parse_args()


GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

ONNX_TEST_DATA = 'onnx/onnx/backend/test/data'
NODE_TEST = os.path.join(ONNX_TEST_DATA, 'node')

TEST_CASES = [
    TestCase(NODE_TEST, 'test_identity'),

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
    TestCase(NODE_TEST, 'test_pow'),
    TestCase(NODE_TEST, 'test_pow_bcast_array'),
    TestCase(NODE_TEST, 'test_pow_bcast_scalar'),
    TestCase(NODE_TEST, 'test_pow_example'),

    TestCase(NODE_TEST, 'test_neg'),
    TestCase(NODE_TEST, 'test_neg_example'),
    TestCase(NODE_TEST, 'test_reciprocal'),
    TestCase(NODE_TEST, 'test_reciprocal_example'),
    TestCase(NODE_TEST, 'test_exp'),
    TestCase(NODE_TEST, 'test_exp_example'),
    TestCase(NODE_TEST, 'test_log'),
    TestCase(NODE_TEST, 'test_log_example'),
    TestCase(NODE_TEST, 'test_sqrt'),
    TestCase(NODE_TEST, 'test_sqrt_example'),
    TestCase(NODE_TEST, 'test_tanh'),
    TestCase(NODE_TEST, 'test_tanh_example'),
    TestCase(NODE_TEST, 'test_abs'),
    TestCase(NODE_TEST, 'test_relu'),
    TestCase(NODE_TEST, 'test_elu'),
    TestCase(NODE_TEST, 'test_elu_default'),
    TestCase(NODE_TEST, 'test_elu_example'),
    TestCase(NODE_TEST, 'test_leakyrelu'),
    TestCase(NODE_TEST, 'test_leakyrelu_default'),
    TestCase(NODE_TEST, 'test_leakyrelu_example'),
    TestCase(NODE_TEST, 'test_selu'),
    TestCase(NODE_TEST, 'test_selu_default'),
    TestCase(NODE_TEST, 'test_selu_example'),
    TestCase(NODE_TEST, 'test_sigmoid'),
    TestCase(NODE_TEST, 'test_sigmoid_example'),
    TestCase(NODE_TEST, 'test_floor'),
    TestCase(NODE_TEST, 'test_floor_example'),
    TestCase(NODE_TEST, 'test_ceil'),
    TestCase(NODE_TEST, 'test_ceil_example'),

    TestCase(NODE_TEST, 'test_not_2d'),
    TestCase(NODE_TEST, 'test_not_3d'),
    TestCase(NODE_TEST, 'test_not_4d'),
    TestCase(NODE_TEST, 'test_equal'),
    TestCase(NODE_TEST, 'test_equal_bcast'),
    TestCase(NODE_TEST, 'test_greater'),
    TestCase(NODE_TEST, 'test_greater_bcast'),
    TestCase(NODE_TEST, 'test_less'),
    TestCase(NODE_TEST, 'test_less_bcast'),

    TestCase(NODE_TEST, 'test_constant'),
    TestCase(NODE_TEST, 'test_constantlike_ones_with_input'),
    TestCase(NODE_TEST, 'test_constantlike_threes_with_shape_and_dtype'),
    TestCase(NODE_TEST, 'test_constantlike_zeros_without_input_dtype'),
    TestCase(NODE_TEST, 'test_onehot_with_axis'),
    TestCase(NODE_TEST, 'test_onehot_without_axis'),

    # TODO(xchainer): Support float16?
    TestCase(NODE_TEST, 'test_cast_DOUBLE_to_FLOAT'),
    # TestCase(NODE_TEST, 'test_cast_DOUBLE_to_FLOAT16'),
    # TestCase(NODE_TEST, 'test_cast_FLOAT16_to_DOUBLE'),
    # TestCase(NODE_TEST, 'test_cast_FLOAT16_to_FLOAT'),
    TestCase(NODE_TEST, 'test_cast_FLOAT_to_DOUBLE'),
    # TestCase(NODE_TEST, 'test_cast_FLOAT_to_FLOAT16'),

    # TODO(xchainer): Support non-2D dot.
    # terminate called after throwing an instance of 'chainerx::NotImplementedError'
    #   what():  dot does not support rhs operand with ndim > 2
    TestCase(NODE_TEST, 'test_matmul_2d'),

    TestCase(NODE_TEST, 'test_basic_conv_with_padding'),
    TestCase(NODE_TEST, 'test_basic_conv_without_padding'),
    TestCase(NODE_TEST, 'test_conv_with_strides_no_padding'),
    TestCase(NODE_TEST, 'test_conv_with_strides_padding'),
    TestCase(NODE_TEST, 'test_conv_with_strides_and_asymmetric_padding'),
    # TODO(hamaji): Revisit parameters of ConvTranspose.
    TestCase(NODE_TEST, 'test_convtranspose'),
    # TestCase(NODE_TEST, 'test_convtranspose_1d'),
    # TestCase(NODE_TEST, 'test_convtranspose_3d'),
    TestCase(NODE_TEST, 'test_convtranspose_kernel_shape'),
    TestCase(NODE_TEST, 'test_convtranspose_output_shape'),
    # TODO(hamaji): output_pads is not handled yet.
    # TestCase(NODE_TEST, 'test_convtranspose_pad'),
    TestCase(NODE_TEST, 'test_convtranspose_pads'),
    TestCase(NODE_TEST, 'test_convtranspose_with_kernel'),

    TestCase(NODE_TEST, 'test_constant_pad'),
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
    TestCase(NODE_TEST, 'test_globalmaxpool'),
    TestCase(NODE_TEST, 'test_globalmaxpool_precomputed'),
    TestCase(NODE_TEST, 'test_globalaveragepool'),
    TestCase(NODE_TEST, 'test_globalaveragepool_precomputed'),

    TestCase(NODE_TEST, 'test_shape'),
    TestCase(NODE_TEST, 'test_shape_example'),
    TestCase(NODE_TEST, 'test_size'),
    TestCase(NODE_TEST, 'test_size_example'),

    TestCase(NODE_TEST, 'test_reshape_extended_dims'),
    TestCase(NODE_TEST, 'test_reshape_negative_dim'),
    TestCase(NODE_TEST, 'test_reshape_one_dim'),
    TestCase(NODE_TEST, 'test_reshape_reduced_dims'),
    TestCase(NODE_TEST, 'test_reshape_reordered_dims'),
    # Broadcast from (3, 1) to (2, 1, 6).
    # TODO(xchainer): Do we really want to support this?
    # TestCase(NODE_TEST, 'test_expand_dim_changed'),
    TestCase(NODE_TEST, 'test_expand_dim_unchanged'),
    TestCase(NODE_TEST, 'test_squeeze'),
    TestCase(NODE_TEST, 'test_unsqueeze'),
    TestCase(NODE_TEST, 'test_flatten_axis0'),
    TestCase(NODE_TEST, 'test_flatten_axis1'),
    TestCase(NODE_TEST, 'test_flatten_axis2'),
    TestCase(NODE_TEST, 'test_flatten_axis3'),
    TestCase(NODE_TEST, 'test_flatten_default_axis'),

    TestCase(NODE_TEST, 'test_slice'),
    TestCase(NODE_TEST, 'test_slice_default_axes'),
    TestCase(NODE_TEST, 'test_slice_end_out_of_bounds'),
    TestCase(NODE_TEST, 'test_slice_neg'),
    TestCase(NODE_TEST, 'test_slice_start_out_of_bounds'),
    TestCase(NODE_TEST, 'test_dynamic_slice'),
    TestCase(NODE_TEST, 'test_dynamic_slice_default_axes'),
    TestCase(NODE_TEST, 'test_dynamic_slice_end_out_of_bounds'),
    TestCase(NODE_TEST, 'test_dynamic_slice_neg'),
    TestCase(NODE_TEST, 'test_dynamic_slice_start_out_of_bounds'),
    TestCase(NODE_TEST, 'test_gather_0'),
    TestCase(NODE_TEST, 'test_gather_1'),
    TestCase(NODE_TEST, 'test_concat_1d_axis_0'),
    TestCase(NODE_TEST, 'test_concat_2d_axis_0'),
    TestCase(NODE_TEST, 'test_concat_2d_axis_1'),
    TestCase(NODE_TEST, 'test_concat_3d_axis_0'),
    TestCase(NODE_TEST, 'test_concat_3d_axis_1'),
    TestCase(NODE_TEST, 'test_concat_3d_axis_2'),
    TestCase(NODE_TEST, 'test_split_equal_parts_1d'),
    TestCase(NODE_TEST, 'test_split_equal_parts_2d'),
    TestCase(NODE_TEST, 'test_split_equal_parts_default_axis'),
    TestCase(NODE_TEST, 'test_split_variable_parts_1d'),
    TestCase(NODE_TEST, 'test_split_variable_parts_2d'),
    TestCase(NODE_TEST, 'test_split_variable_parts_default_axis'),

    TestCase(NODE_TEST, 'test_transpose_all_permutations_0'),
    TestCase(NODE_TEST, 'test_transpose_all_permutations_1'),
    TestCase(NODE_TEST, 'test_transpose_all_permutations_2'),
    TestCase(NODE_TEST, 'test_transpose_all_permutations_3'),
    TestCase(NODE_TEST, 'test_transpose_all_permutations_4'),
    TestCase(NODE_TEST, 'test_transpose_all_permutations_5'),
    TestCase(NODE_TEST, 'test_transpose_default'),

    TestCase(NODE_TEST, 'test_gemm_nobroadcast'),
    TestCase(NODE_TEST, 'test_gemm_broadcast'),

    TestCase(NODE_TEST, 'test_rnn_seq_length'),
    TestCase(NODE_TEST, 'test_simple_rnn_defaults'),
    TestCase(NODE_TEST, 'test_simple_rnn_with_initial_bias'),
    TestCase(NODE_TEST, 'test_gru_defaults'),
    # TODO(hamaji): Implement sequence_lens.
    # TestCase(NODE_TEST, 'test_gru_seq_length'),
    TestCase(NODE_TEST, 'test_gru_with_initial_bias'),
    TestCase(NODE_TEST, 'test_lstm_defaults'),
    TestCase(NODE_TEST, 'test_lstm_with_initial_bias'),
    TestCase(NODE_TEST, 'test_lstm_with_peepholes', rtol=5e-2),

    # TODO(hamaji): Investigate 3D softmax ops do not agree (though
    # xChainer agrees with Chainer).
    # TODO(hamaji): Relax equality check for "large_number" tests.
    TestCase(NODE_TEST, 'test_softmax_example'),
    TestCase(NODE_TEST, 'test_softmax_axis_2'),
    TestCase(NODE_TEST, 'test_logsoftmax_example_1'),
    TestCase(NODE_TEST, 'test_logsoftmax_axis_2'),
    TestCase(NODE_TEST, 'test_softplus'),
    TestCase(NODE_TEST, 'test_softplus_example'),
    TestCase(NODE_TEST, 'test_softsign'),
    TestCase(NODE_TEST, 'test_softsign_example'),

    TestCase(NODE_TEST, 'test_sum_example'),
    TestCase(NODE_TEST, 'test_sum_one_input'),
    TestCase(NODE_TEST, 'test_sum_two_inputs'),
    TestCase(NODE_TEST, 'test_mean_example'),
    TestCase(NODE_TEST, 'test_mean_one_input'),
    TestCase(NODE_TEST, 'test_mean_two_inputs'),
    TestCase(NODE_TEST, 'test_max_example'),
    TestCase(NODE_TEST, 'test_max_one_input'),
    TestCase(NODE_TEST, 'test_max_two_inputs'),
    TestCase(NODE_TEST, 'test_min_example'),
    TestCase(NODE_TEST, 'test_min_one_input'),
    TestCase(NODE_TEST, 'test_min_two_inputs'),
    TestCase(NODE_TEST, 'test_clip'),
    TestCase(NODE_TEST, 'test_clip_default_inbounds'),
    TestCase(NODE_TEST, 'test_clip_default_max'),
    TestCase(NODE_TEST, 'test_clip_default_min'),
    TestCase(NODE_TEST, 'test_clip_example'),
    TestCase(NODE_TEST, 'test_clip_inbounds'),
    TestCase(NODE_TEST, 'test_clip_outbounds'),
    TestCase(NODE_TEST, 'test_clip_splitbounds'),

    TestCase(NODE_TEST, 'test_argmax_default_axis_example'),
    TestCase(NODE_TEST, 'test_argmax_default_axis_random'),
    TestCase(NODE_TEST, 'test_argmax_keepdims_example'),
    TestCase(NODE_TEST, 'test_argmax_keepdims_random'),
    TestCase(NODE_TEST, 'test_argmax_no_keepdims_example'),
    TestCase(NODE_TEST, 'test_argmax_no_keepdims_random'),
    TestCase(NODE_TEST, 'test_argmin_default_axis_example'),
    TestCase(NODE_TEST, 'test_argmin_default_axis_random'),
    TestCase(NODE_TEST, 'test_argmin_keepdims_example'),
    TestCase(NODE_TEST, 'test_argmin_keepdims_random'),
    TestCase(NODE_TEST, 'test_argmin_no_keepdims_example'),
    TestCase(NODE_TEST, 'test_argmin_no_keepdims_random'),
    TestCase(NODE_TEST, 'test_hardmax_axis_0'),
    TestCase(NODE_TEST, 'test_hardmax_axis_1'),
    TestCase(NODE_TEST, 'test_hardmax_axis_2'),
    TestCase(NODE_TEST, 'test_hardmax_default_axis'),
    TestCase(NODE_TEST, 'test_hardmax_example'),
    TestCase(NODE_TEST, 'test_hardmax_one_hot'),

    TestCase(NODE_TEST, 'test_reduce_max_default_axes_keepdim_example'),
    TestCase(NODE_TEST, 'test_reduce_max_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_max_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_max_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_max_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_max_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_min_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_min_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_min_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_min_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_min_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_min_keepdims_random'),
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
    TestCase(NODE_TEST, 'test_reduce_l1_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_l1_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_l1_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_l1_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_l1_keep_dims_example'),
    TestCase(NODE_TEST, 'test_reduce_l1_keep_dims_random'),
    TestCase(NODE_TEST, 'test_reduce_l2_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_l2_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_l2_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_l2_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_l2_keep_dims_example'),
    TestCase(NODE_TEST, 'test_reduce_l2_keep_dims_random'),
    TestCase(NODE_TEST, 'test_reduce_log_sum'),
    TestCase(NODE_TEST, 'test_reduce_log_sum_asc_axes'),
    TestCase(NODE_TEST, 'test_reduce_log_sum_default'),
    TestCase(NODE_TEST, 'test_reduce_log_sum_desc_axes'),
    TestCase(NODE_TEST, 'test_reduce_log_sum_exp_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_log_sum_exp_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_log_sum_exp_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_log_sum_exp_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_log_sum_exp_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_log_sum_exp_keepdims_random'),

    TestCase(NODE_TEST, 'test_batchnorm_example'),
    TestCase(NODE_TEST, 'test_batchnorm_epsilon'),
    TestCase(NODE_TEST, 'test_lrn'),
    TestCase(NODE_TEST, 'test_lrn_default'),

    TestCase(NODE_TEST, 'test_dropout_default'),
    TestCase(NODE_TEST, 'test_dropout_random'),
]

TEST_CASES += [
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_AvgPool1d'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_AvgPool1d_stride'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_AvgPool2d'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_AvgPool2d_stride'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_AvgPool3d'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_AvgPool3d_stride'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_AvgPool3d_stride1_pad0_gpu_input'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_BatchNorm1d_3d_input_eval', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_BatchNorm2d_eval', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_BatchNorm2d_momentum_eval', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_BatchNorm3d_eval', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_BatchNorm3d_momentum_eval', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_ConstantPad2d'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_dilated', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_groups'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_pad1'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_pad1size1'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_pad2'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_pad2size1'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_stride'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_depthwise'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_depthwise_padded'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_depthwise_strided'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_depthwise_with_multiplier'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_dilated', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_groups'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_groups_thnn', rtol=2e-4),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_no_bias'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_padding'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_strided'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv3d'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv3d_dilated', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv3d_dilated_strided', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv3d_groups'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv3d_no_bias'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv3d_stride'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv3d_stride_padding'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_ConvTranspose2d', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_ConvTranspose2d_no_bias', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_ELU'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Embedding'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Embedding_sparse'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_GLU', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_GLU_dim'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_LeakyReLU'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_LeakyReLU_with_negval'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Linear', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Linear_no_bias'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_LogSoftmax'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_MaxPool1d'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_MaxPool1d_stride'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_MaxPool2d'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_MaxPool3d'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_MaxPool3d_stride'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_MaxPool3d_stride_padding'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_PReLU_1d', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_PReLU_1d_multiparam', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_PReLU_2d', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_PReLU_2d_multiparam', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_PReLU_3d', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_PReLU_3d_multiparam', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_PixelShuffle'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_PoissonNLLLLoss_no_reduce'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_ReLU'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_ReflectionPad2d', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_ReplicationPad2d', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_SELU'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Sigmoid'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Softmax'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Softmin'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Softplus'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Softsign', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Tanh'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_ZeroPad2d'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_log_softmax_dim3'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_log_softmax_lastdim'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_softmax_functional_dim3'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_softmax_lastdim'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_add_broadcast', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_add_size1_broadcast', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_add_size1_right_broadcast', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_add_size1_singleton_broadcast', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_addconstant', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_addmm', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_basic'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_chunk'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_clip'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_concat2'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_conv'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_convtranspose', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_exp'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_flatten'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_index'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_lstm'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_max'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_maxpool'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_min'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_mm', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_non_float_params'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_pad', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_params'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_permute2'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_pow', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_reduced_mean'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_reduced_mean_keepdim'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_reduced_sum'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_reduced_sum_keepdim'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_repeat', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_repeat_dim_overflow', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_rnn', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_rnn_single_layer', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_selu'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_sqrt', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_symbolic_override', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_symbolic_override_nested'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_view', fail=True),
]

TEST_PATHS = set()
for test_case in TEST_CASES:
    TEST_PATHS.add(test_case.test_dir)


if args.all:
    models = glob.glob(os.path.join(ONNX_TEST_DATA, '*/*/model.onnx'))
    for onnx in sorted(models):
        path = os.path.dirname(onnx)
        if path not in TEST_PATHS:
            case = TestCase(os.path.dirname(path), os.path.basename(path),
                            fail=True)
            TEST_CASES.append(case)

num_official_onnx_tests = len(TEST_CASES)

for backprop_test in gen_backprop_tests_oc.get_backprop_tests():
    dirname = 'out'
    name = 'backprop_test_oc_' + backprop_test.name
    assert os.path.exists(os.path.join(dirname, name))
    TEST_CASES.append(TestCase(dirname, name))

for backprop_test in gen_backprop_tests_pc.get_backprop_tests():
    dirname = 'out'
    name = 'backprop_test_pc_' + backprop_test.name
    assert os.path.exists(os.path.join(dirname, name))
    # TODO(hamaji): Do not skip shape inference.
    skip_shape_inference = name.endswith('_embed')
    TEST_CASES.append(TestCase(dirname, name, rtol=backprop_test.rtol,
                               fail=backprop_test.fail,
                               skip_shape_inference=skip_shape_inference))

for test in gen_extra_test.get_tests():
    dirname = 'out'
    assert os.path.exists(os.path.join(dirname, test.name))
    TEST_CASES.append(TestCase(dirname, test.name,
                               rtol=test.rtol, fail=test.fail,
                               skip_shape_inference=test.skip_shape_inference))

TEST_CASES.append(TestCase('out', 'backprop_test_mnist_mlp'))

TEST_CASES.append(TestCase('data', 'resnet50', want_gpu=True))

TEST_CASES.extend(ch2o_tests.get())

TEST_CASES.extend(onnx_real_tests.get())

for test_case in list(TEST_CASES):
    if not test_case.is_backprop:
        continue
    new_test = copy.copy(test_case)
    new_test.name = test_case.name + '_stack'
    new_test.always_retain_in_stack = True
    new_test.fail = new_test.fail or (
        # TODO(hamaji): Fix scheduling of backprop stack.
        'StatelessLSTM' in test_case.name or
        'Sum' in test_case.name or
        'EspNet' in test_case.name or
        'BroadcastTo' in test_case.name or
        'AttDot' in test_case.name or
        'pc_pad' in test_case.name or
        'pc_for' in test_case.name or
        'if_pd' in test_case.name or
        'separate' in test_case.name or
        'dynamic_slice' in test_case.name or
        test_case.name.endswith('softmax_cross_entropy') or
        'mlp' in test_case.name)
    TEST_CASES.append(new_test)

if args.test_filter is not None:
    reg = re.compile(args.test_filter)
    TEST_CASES = [case for case in TEST_CASES if reg.search(case.name)]

if not args.all:
    TEST_CASES = [case for case in TEST_CASES if not case.fail]


def _start_output(msg):
    if sys.stdout.isatty():
        sys.stdout.write('\r' + ' ' * 78 + '\r' + msg)
    else:
        sys.stdout.write(msg)


class TestRunner(object):
    def __init__(self, test_cases, show_log):
        self.test_cases = test_cases
        self.test_cnt = 0
        self.fail_cnt = 0
        self.show_log = show_log

    def run(self, num_parallel_jobs):
        tests = list(reversed(self.test_cases))
        procs = {}
        while tests or procs:
            if tests and len(procs) < num_parallel_jobs:
                test_case = tests.pop()
                if num_parallel_jobs == 1:
                    _start_output('%s... ' % test_case.name)
                proc = subprocess.Popen(test_case.args,
                                        stdout=subprocess.PIPE,
                                        stderr=test_case.log_writer())
                procs[proc.pid] = (test_case, proc)
                continue

            assert procs
            pid, status = os.wait()
            assert pid in procs
            test_case, proc = procs[pid]
            del procs[pid]

            if num_parallel_jobs != 1:
                _start_output('%s... ' % test_case.name)
            self.test_cnt += 1
            if status == 0:
                if test_case.fail:
                    sys.stdout.write('%sOK (unexpected)%s\n' % (YELLOW, RESET))
                else:
                    sys.stdout.write('%sOK%s' % (GREEN, RESET))
                    if not sys.stdout.isatty():
                        sys.stdout.write('\n')
            else:
                self.fail_cnt += 1
                filtered = [a for a in test_case.args if a != '--quiet']
                sys.stdout.write('%sFAIL%s: %s\n' %
                                 (RED, RESET, ' '.join(filtered)))
            if status != 0 or self.show_log:
                sys.stdout.buffer.write(test_case.log_read())
                if status != 0:
                    sys.stdout.write('%s$%s %s\n' %
                                     (RED, RESET, ' '.join(filtered)))

            sys.stdout.flush()
        _start_output('')
        sys.stdout.write('\n')


def main():
    if not args.skip_build:
        if os.path.exists('Makefile'):
            subprocess.check_call(['make', '-j4'])
        elif os.path.exists('build.ninja'):
            subprocess.check_call('ninja')

    if args.build_dir is None:
        if os.path.exists('build/CMakeCache.txt'):
            args.build_dir = 'build'
        elif os.path.exists('CMakeCache.txt'):
            args.build_dir = '.'
        else:
            args.build_dir = 'build'

    run_onnx = os.path.join(args.build_dir, 'tools/run_onnx')
    print('Testing %s \U0001f356' % run_onnx)

    test_cnt = 0
    fail_cnt = 0
    tests = []
    gpu_tests = []
    for test_case in TEST_CASES:
        test_case.args = [run_onnx, '--test', test_case.test_dir]
        is_gpu = False
        if test_case.rtol is not None:
            test_case.args += ['--rtol', str(test_case.rtol)]
        if test_case.skip_shape_inference:
            test_case.args.append('--skip_shape_inference')
        if test_case.always_retain_in_stack:
            test_case.args.append('--always_retain_in_stack')
        if test_case.is_backprop:
            test_case.args.append('--backprop')
        if args.verbose:
            test_case.args.append('--verbose')
        if test_case.want_gpu or args.use_gpu_all:
            if not args.use_gpu and not args.use_gpu_all:
                continue
            test_case.args.extend(['-d', 'cuda'])
            is_gpu = True

        if args.fuse:
            test_case.args.append('--fuse_operations')
            if is_gpu:
                test_case.args.append('--use_nvrtc')

        if is_gpu:
            gpu_tests.append(test_case)
        else:
            tests.append(test_case)

    for test in tests + gpu_tests:
        test.prepare()

    for tests, num_jobs in [(tests, args.jobs), (gpu_tests, 1)]:
        runner = TestRunner(tests, args.show_log)
        runner.run(num_jobs)
        test_cnt += runner.test_cnt
        fail_cnt += runner.fail_cnt

    if fail_cnt:
        print('%d/%d tests failed!' % (fail_cnt, test_cnt))
    else:
        print('ALL %d tests OK! (%d from ONNX)' %
              (test_cnt, num_official_onnx_tests))


main()
