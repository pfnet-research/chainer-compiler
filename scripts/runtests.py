#!/usr/bin/python3

import argparse
import glob
import os
import re
import sys
import subprocess

import gen_backprop_tests_oc
import gen_backprop_tests_pc
import gen_extra_node_test


parser = argparse.ArgumentParser(description='Run tests for oniku')
parser.add_argument('test_filter', default=None, nargs='?',
                    help='A regular expression to filter tests')
parser.add_argument('--all', '-a', action='store_true',
                    help='Run all tests')
parser.add_argument('--use_gpu', '-g', action='store_true',
                    help='Run heavy tests with GPU')
parser.add_argument('--use_gpu_all', '-G', action='store_true',
                    help='Run all tests with GPU')
cmdline = parser.parse_args()


TEST_PATHS = set()


class TestCase(object):

    def __init__(self, dirname, name, rtol=None, fail=False,
                 skip_shape_inference=False):
        self.dirname = dirname
        self.name = name
        self.rtol = rtol
        self.fail = fail
        self.skip_shape_inference = skip_shape_inference
        self.test_dir = os.path.join(self.dirname, self.name)
        TEST_PATHS.add(self.test_dir)


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
    TestCase(NODE_TEST, 'test_lrn', rtol=5e-3),
    TestCase(NODE_TEST, 'test_lrn_default', rtol=5e-3),

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
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_groups', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_pad1'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_pad1size1'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_pad2'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_pad2size1'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_stride'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_depthwise', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_depthwise_padded', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_depthwise_strided', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_depthwise_with_multiplier', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_dilated', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_groups', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_groups_thnn', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_no_bias'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_padding'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv2d_strided'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv3d'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv3d_dilated', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv3d_dilated_strided', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv3d_groups', fail=True),
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

if cmdline.all:
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
    TEST_CASES.append(TestCase(dirname, name, rtol=backprop_test.rtol,
                               skip_shape_inference=True))

for test in gen_extra_node_test.get_tests():
    dirname = 'out'
    assert os.path.exists(os.path.join(dirname, test.name))
    TEST_CASES.append(TestCase(dirname, test.name,
                               rtol=test.rtol, fail=test.fail))

TEST_CASES.append(TestCase('out', 'backprop_test_mnist_mlp'))

TEST_CASES.append(TestCase('data', 'resnet50'))

if cmdline.test_filter is not None:
    reg = re.compile(cmdline.test_filter)
    TEST_CASES = [case for case in TEST_CASES if reg.search(case.name)]

if not cmdline.all:
    TEST_CASES = [case for case in TEST_CASES if not case.fail]


def main():
    if os.path.exists('Makefile'):
        subprocess.check_call(['make', '-j4'])
    elif os.path.exists('build.ninja'):
        subprocess.check_call('ninja')

    test_cnt = 0
    fail_cnt = 0
    unexpected_pass = 0
    for test_case in TEST_CASES:
        args = ['tools/run_onnx', '--test', test_case.test_dir, '--quiet']
        if test_case.rtol is not None:
            args += ['--rtol', str(test_case.rtol)]
        if test_case.skip_shape_inference:
            args.append('--skip_shape_inference')
        if test_case.name.startswith('backprop_'):
            args.append('--backprop')
        if test_case.name == 'resnet50' or cmdline.use_gpu_all:
            if not cmdline.use_gpu and not cmdline.use_gpu_all:
                continue
            args.extend(['-d', 'cuda'])

        sys.stderr.write('%s... ' % test_case.name)
        try:
            test_cnt += 1
            subprocess.check_call(args)
            if test_case.fail:
                sys.stderr.write('OK (unexpected)\n')
            else:
                sys.stderr.write('OK\n')
        except subprocess.CalledProcessError:
            fail_cnt += 1
            sys.stderr.write('FAIL: %s\n' % ' '.join(args))
    if fail_cnt:
        print('%d/%d tests failed!' % (fail_cnt, test_cnt))
    else:
        print('ALL %d tests OK! (%d from ONNX)' %
              (test_cnt, num_official_onnx_tests))


main()
