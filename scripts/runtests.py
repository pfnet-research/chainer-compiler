#!/usr/bin/env python3

import argparse
import copy
import glob
import multiprocessing
import os
import re
import sys
import subprocess

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import ch2o_tests
import elichika_tests
import gen_backprop_tests_oc
import gen_backprop_tests_pc
import gen_chainercv_model_tests
import gen_extra_test
import gen_large_tests_oc
import onnx_chainer_tests
import onnx_real_tests
from test_case import TestCase


parser = argparse.ArgumentParser(description='Run tests for chainer_compiler')
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
parser.add_argument('--device', '-d', default=None,
                    help='ChainerX device to be used')
parser.add_argument('--use_gpu_all', '-G', action='store_true',
                    help='Run all tests with GPU')
parser.add_argument('--failed', action='store_true',
                    help='Run tests which failed last time')
parser.add_argument('--failure_log', default='out/failed_tests.log',
                    help='The file where names of failed tests are stored')
parser.add_argument('--fuse', action='store_true', help='Enable fusion')
parser.add_argument('--ngraph', action='store_true', help='Enable nGraph')
parser.add_argument('--snpe', action='store_true', help='Enable SNPE')
parser.add_argument('--computation_order', default=None,
                    help='Force setting --computation_order flag')
parser.add_argument('--cache', action='store_true', help='Enable model caching')
parser.add_argument('--verbose', action='store_true',
                    help='Run tests with --verbose flag')
parser.add_argument('--target_opsets',
                    help='Specify target opsets to run with comma separated string')
parser.add_argument('--only_opset_targetable', action='store_true',
                    help='Run test cases with opset_version is not None')
args = parser.parse_args()


GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

ONNX_TEST_DATA = 'third_party/onnx/onnx/backend/test/data'
NODE_TEST = os.path.join(ONNX_TEST_DATA, 'node')
SIMPLE_TEST = os.path.join(ONNX_TEST_DATA, 'simple')

# ChainerX does not support 1D conv/pool.
fail_1d_conv_pool = args.use_gpu_all

target_opsets = []
if args.target_opsets is not None:
    target_opsets = [int(o) for o in args.target_opsets.split(',')]
if len(target_opsets) > 0:
    print('Targeting opsets: {}'.format(target_opsets))

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
    TestCase(NODE_TEST, 'test_mod_broadcast'),
    TestCase(NODE_TEST, 'test_mod_int64_fmod'),
    TestCase(NODE_TEST, 'test_mod_mixed_sign_float16'),
    TestCase(NODE_TEST, 'test_mod_mixed_sign_float32'),
    TestCase(NODE_TEST, 'test_mod_mixed_sign_float64'),
    TestCase(NODE_TEST, 'test_mod_mixed_sign_int16'),
    TestCase(NODE_TEST, 'test_mod_mixed_sign_int32'),
    TestCase(NODE_TEST, 'test_mod_mixed_sign_int64'),
    TestCase(NODE_TEST, 'test_mod_mixed_sign_int8'),
    # TODO(hamaji): Unsupported dtypes.
    # TestCase(NODE_TEST, 'test_mod_uint16'),
    # TestCase(NODE_TEST, 'test_mod_uint32'),
    # TestCase(NODE_TEST, 'test_mod_uint64'),
    TestCase(NODE_TEST, 'test_mod_uint8'),
    TestCase(NODE_TEST, 'test_pow'),
    TestCase(NODE_TEST, 'test_pow_bcast_array'),
    TestCase(NODE_TEST, 'test_pow_bcast_scalar'),
    TestCase(NODE_TEST, 'test_pow_example'),

    TestCase(NODE_TEST, 'test_and2d'),
    TestCase(NODE_TEST, 'test_and3d'),
    TestCase(NODE_TEST, 'test_and4d'),
    TestCase(NODE_TEST, 'test_and_bcast3v1d'),
    TestCase(NODE_TEST, 'test_and_bcast3v2d'),
    TestCase(NODE_TEST, 'test_and_bcast4v2d'),
    TestCase(NODE_TEST, 'test_and_bcast4v3d'),
    TestCase(NODE_TEST, 'test_and_bcast4v4d'),
    TestCase(NODE_TEST, 'test_or2d'),
    TestCase(NODE_TEST, 'test_or4d'),
    TestCase(NODE_TEST, 'test_or_bcast3v1d'),
    TestCase(NODE_TEST, 'test_or3d'),
    TestCase(NODE_TEST, 'test_or_bcast4v2d'),
    TestCase(NODE_TEST, 'test_or_bcast3v2d'),
    TestCase(NODE_TEST, 'test_or_bcast4v3d'),
    TestCase(NODE_TEST, 'test_or_bcast4v4d'),
    TestCase(NODE_TEST, 'test_xor2d'),
    TestCase(NODE_TEST, 'test_xor3d'),
    TestCase(NODE_TEST, 'test_xor_bcast3v1d'),
    TestCase(NODE_TEST, 'test_xor4d'),
    TestCase(NODE_TEST, 'test_xor_bcast3v2d'),
    TestCase(NODE_TEST, 'test_xor_bcast4v2d'),
    TestCase(NODE_TEST, 'test_xor_bcast4v4d'),
    TestCase(NODE_TEST, 'test_xor_bcast4v3d'),

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
    TestCase(NODE_TEST, 'test_sin'),
    TestCase(NODE_TEST, 'test_sin_example'),
    TestCase(NODE_TEST, 'test_cos'),
    TestCase(NODE_TEST, 'test_cos_example'),
    TestCase(NODE_TEST, 'test_tan'),
    TestCase(NODE_TEST, 'test_tan_example'),
    TestCase(NODE_TEST, 'test_sinh'),
    TestCase(NODE_TEST, 'test_sinh_example'),
    TestCase(NODE_TEST, 'test_cosh'),
    TestCase(NODE_TEST, 'test_cosh_example'),
    TestCase(NODE_TEST, 'test_tanh'),
    TestCase(NODE_TEST, 'test_tanh_example'),
    TestCase(NODE_TEST, 'test_asin'),
    TestCase(NODE_TEST, 'test_asin_example'),
    TestCase(NODE_TEST, 'test_acos'),
    TestCase(NODE_TEST, 'test_acos_example'),
    TestCase(NODE_TEST, 'test_atan'),
    TestCase(NODE_TEST, 'test_atan_example'),
    TestCase(NODE_TEST, 'test_asinh'),
    TestCase(NODE_TEST, 'test_asinh_example'),
    TestCase(NODE_TEST, 'test_acosh'),
    TestCase(NODE_TEST, 'test_acosh_example'),
    # TODO(hamaji): Enable these tests once atanh is implemented.
    # TestCase(NODE_TEST, 'test_atanh'),
    # TestCase(NODE_TEST, 'test_atanh_example'),
    TestCase(NODE_TEST, 'test_erf'),
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
    TestCase(NODE_TEST, 'test_sign'),

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
    TestCase(NODE_TEST, 'test_constantofshape_float_ones'),
    TestCase(NODE_TEST, 'test_constantofshape_int_zeros'),
    TestCase(NODE_TEST, 'test_onehot_with_axis'),
    TestCase(NODE_TEST, 'test_onehot_with_negative_axis'),
    TestCase(NODE_TEST, 'test_onehot_without_axis'),
    TestCase(NODE_TEST, 'test_eyelike_populate_off_main_diagonal'),
    TestCase(NODE_TEST, 'test_eyelike_with_dtype'),
    TestCase(NODE_TEST, 'test_eyelike_without_dtype'),

    TestCase(NODE_TEST, 'test_cast_DOUBLE_to_FLOAT'),
    TestCase(NODE_TEST, 'test_cast_DOUBLE_to_FLOAT16'),
    TestCase(NODE_TEST, 'test_cast_FLOAT16_to_DOUBLE'),
    TestCase(NODE_TEST, 'test_cast_FLOAT16_to_FLOAT'),
    TestCase(NODE_TEST, 'test_cast_FLOAT_to_DOUBLE'),
    TestCase(NODE_TEST, 'test_cast_FLOAT_to_FLOAT16'),

    TestCase(NODE_TEST, 'test_matmul_2d'),
    TestCase(NODE_TEST, 'test_matmul_3d'),
    TestCase(NODE_TEST, 'test_matmul_4d'),

    TestCase(NODE_TEST, 'test_basic_conv_with_padding'),
    TestCase(NODE_TEST, 'test_basic_conv_without_padding'),
    TestCase(NODE_TEST, 'test_conv_with_strides_no_padding'),
    TestCase(NODE_TEST, 'test_conv_with_strides_padding'),
    TestCase(NODE_TEST, 'test_conv_with_strides_and_asymmetric_padding'),
    TestCase(NODE_TEST, 'test_convtranspose'),
    TestCase(NODE_TEST, 'test_convtranspose_1d', fail=fail_1d_conv_pool),
    TestCase(NODE_TEST, 'test_convtranspose_3d'),
    TestCase(NODE_TEST, 'test_convtranspose_kernel_shape'),
    TestCase(NODE_TEST, 'test_convtranspose_output_shape'),
    # TODO(hamaji): output_pads is not handled yet.
    # TestCase(NODE_TEST, 'test_convtranspose_pad'),
    TestCase(NODE_TEST, 'test_convtranspose_pads'),
    TestCase(NODE_TEST, 'test_convtranspose_with_kernel'),

    TestCase(NODE_TEST, 'test_constant_pad'),
    # TODO(hamaji): auto_pad is not supported.
    TestCase(NODE_TEST, 'test_maxpool_1d_default', fail=fail_1d_conv_pool),
    TestCase(NODE_TEST, 'test_maxpool_2d_ceil'),
    TestCase(NODE_TEST, 'test_maxpool_2d_default'),
    TestCase(NODE_TEST, 'test_maxpool_2d_pads'),
    TestCase(NODE_TEST, 'test_maxpool_2d_precomputed_pads'),
    TestCase(NODE_TEST, 'test_maxpool_2d_precomputed_same_upper'),
    TestCase(NODE_TEST, 'test_maxpool_2d_precomputed_strides'),
    TestCase(NODE_TEST, 'test_maxpool_2d_strides'),
    TestCase(NODE_TEST, 'test_maxpool_3d_default'),
    TestCase(NODE_TEST, 'test_averagepool_1d_default', fail=fail_1d_conv_pool),
    TestCase(NODE_TEST, 'test_averagepool_2d_default'),
    TestCase(NODE_TEST, 'test_averagepool_2d_precomputed_pads'),
    TestCase(NODE_TEST, 'test_averagepool_2d_precomputed_pads_count_include_pad'),
    TestCase(NODE_TEST, 'test_averagepool_2d_precomputed_strides'),
    TestCase(NODE_TEST, 'test_averagepool_2d_strides'),
    TestCase(NODE_TEST, 'test_averagepool_2d_pads'),
    TestCase(NODE_TEST, 'test_averagepool_2d_pads_count_include_pad'),
    TestCase(NODE_TEST, 'test_averagepool_3d_default'),
    TestCase(NODE_TEST, 'test_globalmaxpool'),
    TestCase(NODE_TEST, 'test_globalmaxpool_precomputed'),
    TestCase(NODE_TEST, 'test_globalaveragepool'),
    TestCase(NODE_TEST, 'test_globalaveragepool_precomputed'),
    TestCase(NODE_TEST, 'test_upsample_nearest'),
    # TODO(take-cheeze): Other Resize-11 tests
    TestCase(NODE_TEST, 'test_resize_upsample_scales_nearest'),
    # The second ROI values mismatch. Let the test pass with
    # ridiculously large tolerance.
    TestCase(NODE_TEST, 'test_roialign', rtol=0.5, atol=0.5),

    TestCase(NODE_TEST, 'test_shape'),
    TestCase(NODE_TEST, 'test_shape_example'),
    TestCase(NODE_TEST, 'test_size'),
    TestCase(NODE_TEST, 'test_size_example'),

    TestCase(NODE_TEST, 'test_reshape_extended_dims'),
    TestCase(NODE_TEST, 'test_reshape_negative_dim'),
    TestCase(NODE_TEST, 'test_reshape_negative_extended_dims'),
    TestCase(NODE_TEST, 'test_reshape_one_dim'),
    TestCase(NODE_TEST, 'test_reshape_reduced_dims'),
    TestCase(NODE_TEST, 'test_reshape_reordered_all_dims'),
    TestCase(NODE_TEST, 'test_reshape_reordered_last_dims'),
    # TODO(hamaji): Support zero dims in reshape op.
    TestCase(NODE_TEST, 'test_reshape_zero_dim', fail=True),
    TestCase(NODE_TEST, 'test_reshape_zero_and_negative_dim', fail=True),

    TestCase(NODE_TEST, 'test_expand_dim_changed'),
    TestCase(NODE_TEST, 'test_expand_dim_unchanged'),
    TestCase(NODE_TEST, 'test_squeeze'),
    TestCase(NODE_TEST, 'test_squeeze_negative_axes'),
    TestCase(NODE_TEST, 'test_unsqueeze_axis_0'),
    TestCase(NODE_TEST, 'test_unsqueeze_axis_1'),
    TestCase(NODE_TEST, 'test_unsqueeze_axis_2'),
    TestCase(NODE_TEST, 'test_unsqueeze_axis_3'),
    TestCase(NODE_TEST, 'test_unsqueeze_negative_axes'),
    TestCase(NODE_TEST, 'test_unsqueeze_two_axes'),
    TestCase(NODE_TEST, 'test_unsqueeze_three_axes'),
    TestCase(NODE_TEST, 'test_unsqueeze_unsorted_axes'),
    TestCase(NODE_TEST, 'test_flatten_axis0'),
    TestCase(NODE_TEST, 'test_flatten_axis1'),
    TestCase(NODE_TEST, 'test_flatten_axis2'),
    TestCase(NODE_TEST, 'test_flatten_axis3'),
    TestCase(NODE_TEST, 'test_flatten_default_axis'),
    TestCase(NODE_TEST, 'test_flatten_negative_axis1'),
    TestCase(NODE_TEST, 'test_flatten_negative_axis2'),
    TestCase(NODE_TEST, 'test_flatten_negative_axis3'),
    TestCase(NODE_TEST, 'test_flatten_negative_axis4'),

    TestCase(NODE_TEST, 'test_slice'),
    TestCase(NODE_TEST, 'test_slice_default_axes'),
    TestCase(NODE_TEST, 'test_slice_default_steps'),
    TestCase(NODE_TEST, 'test_slice_end_out_of_bounds'),
    TestCase(NODE_TEST, 'test_slice_neg'),
    TestCase(NODE_TEST, 'test_slice_neg_steps'),
    TestCase(NODE_TEST, 'test_slice_negative_axes'),
    TestCase(NODE_TEST, 'test_slice_start_out_of_bounds'),
    TestCase(NODE_TEST, 'test_gather_0'),
    TestCase(NODE_TEST, 'test_gather_1'),
    TestCase(NODE_TEST, 'test_gather_elements_0'),
    TestCase(NODE_TEST, 'test_gather_elements_1'),
    TestCase(NODE_TEST, 'test_gather_elements_negative_indices'),
    TestCase(NODE_TEST, 'test_gather_negative_indices'),
    TestCase(NODE_TEST, 'test_gathernd_example_int32'),
    TestCase(NODE_TEST, 'test_gathernd_example_float32'),
    TestCase(NODE_TEST, 'test_scatter_with_axis'),
    TestCase(NODE_TEST, 'test_scatter_without_axis'),
    TestCase(NODE_TEST, 'test_scatter_elements_with_axis'),
    TestCase(NODE_TEST, 'test_scatter_elements_with_negative_indices'),
    TestCase(NODE_TEST, 'test_scatter_elements_without_axis'),
    TestCase(NODE_TEST, 'test_scatternd'),
    TestCase(NODE_TEST, 'test_concat_1d_axis_0'),
    TestCase(NODE_TEST, 'test_concat_1d_axis_negative_1'),
    TestCase(NODE_TEST, 'test_concat_2d_axis_0'),
    TestCase(NODE_TEST, 'test_concat_2d_axis_1'),
    TestCase(NODE_TEST, 'test_concat_2d_axis_negative_1'),
    TestCase(NODE_TEST, 'test_concat_2d_axis_negative_2'),
    TestCase(NODE_TEST, 'test_concat_3d_axis_0'),
    TestCase(NODE_TEST, 'test_concat_3d_axis_1'),
    TestCase(NODE_TEST, 'test_concat_3d_axis_2'),
    TestCase(NODE_TEST, 'test_concat_3d_axis_negative_1'),
    TestCase(NODE_TEST, 'test_concat_3d_axis_negative_2'),
    TestCase(NODE_TEST, 'test_concat_3d_axis_negative_3'),
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

    TestCase(NODE_TEST, 'test_depthtospace_crd_mode'),
    TestCase(NODE_TEST, 'test_depthtospace_crd_mode_example'),
    TestCase(NODE_TEST, 'test_depthtospace_dcr_mode'),
    TestCase(NODE_TEST, 'test_depthtospace_example'),

    TestCase(NODE_TEST, 'test_gemm_all_attributes'),
    TestCase(NODE_TEST, 'test_gemm_alpha'),
    TestCase(NODE_TEST, 'test_gemm_beta'),
    TestCase(NODE_TEST, 'test_gemm_default_matrix_bias'),
    TestCase(NODE_TEST, 'test_gemm_default_no_bias'),
    TestCase(NODE_TEST, 'test_gemm_default_scalar_bias'),
    TestCase(NODE_TEST, 'test_gemm_default_single_elem_vector_bias'),
    TestCase(NODE_TEST, 'test_gemm_default_vector_bias'),
    TestCase(NODE_TEST, 'test_gemm_default_zero_bias'),
    TestCase(NODE_TEST, 'test_gemm_transposeA'),
    TestCase(NODE_TEST, 'test_gemm_transposeB'),

    TestCase(NODE_TEST, 'test_rnn_seq_length'),
    TestCase(NODE_TEST, 'test_simple_rnn_defaults'),
    TestCase(NODE_TEST, 'test_simple_rnn_with_initial_bias'),
    TestCase(NODE_TEST, 'test_gru_defaults'),
    TestCase(NODE_TEST, 'test_gru_seq_length'),
    TestCase(NODE_TEST, 'test_gru_with_initial_bias'),
    TestCase(NODE_TEST, 'test_lstm_defaults'),
    TestCase(NODE_TEST, 'test_lstm_with_initial_bias'),
    TestCase(NODE_TEST, 'test_lstm_with_peepholes', rtol=5e-2),

    TestCase(NODE_TEST, 'test_softmax_axis_0'),
    TestCase(NODE_TEST, 'test_softmax_axis_1'),
    TestCase(NODE_TEST, 'test_softmax_axis_2'),
    TestCase(NODE_TEST, 'test_softmax_default_axis'),
    TestCase(NODE_TEST, 'test_softmax_example'),
    TestCase(NODE_TEST, 'test_softmax_large_number'),
    TestCase(NODE_TEST, 'test_softmax_negative_axis'),
    TestCase(NODE_TEST, 'test_logsoftmax_axis_0'),
    TestCase(NODE_TEST, 'test_logsoftmax_axis_1'),
    TestCase(NODE_TEST, 'test_logsoftmax_axis_2'),
    TestCase(NODE_TEST, 'test_logsoftmax_default_axis'),
    TestCase(NODE_TEST, 'test_logsoftmax_example_1'),
    TestCase(NODE_TEST, 'test_logsoftmax_large_number', rtol=5e-3),
    TestCase(NODE_TEST, 'test_logsoftmax_negative_axis'),
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
    TestCase(NODE_TEST, 'test_cumsum_1d'),
    TestCase(NODE_TEST, 'test_cumsum_2d_axis_0'),
    TestCase(NODE_TEST, 'test_cumsum_2d_axis_1'),
    TestCase(NODE_TEST, 'test_cumsum_2d_negative_axis'),

    TestCase(NODE_TEST, 'test_argmax_default_axis_example'),
    TestCase(NODE_TEST, 'test_argmax_default_axis_random'),
    TestCase(NODE_TEST, 'test_argmax_keepdims_example'),
    TestCase(NODE_TEST, 'test_argmax_keepdims_random'),
    TestCase(NODE_TEST, 'test_argmax_negative_axis_keepdims_example'),
    TestCase(NODE_TEST, 'test_argmax_negative_axis_keepdims_random'),
    TestCase(NODE_TEST, 'test_argmax_no_keepdims_example'),
    TestCase(NODE_TEST, 'test_argmax_no_keepdims_random'),
    TestCase(NODE_TEST, 'test_argmin_default_axis_example'),
    TestCase(NODE_TEST, 'test_argmin_default_axis_random'),
    TestCase(NODE_TEST, 'test_argmin_keepdims_example'),
    TestCase(NODE_TEST, 'test_argmin_keepdims_random'),
    TestCase(NODE_TEST, 'test_argmin_negative_axis_keepdims_example'),
    TestCase(NODE_TEST, 'test_argmin_negative_axis_keepdims_random'),
    TestCase(NODE_TEST, 'test_argmin_no_keepdims_example'),
    TestCase(NODE_TEST, 'test_argmin_no_keepdims_random'),
    TestCase(NODE_TEST, 'test_hardmax_axis_0'),
    TestCase(NODE_TEST, 'test_hardmax_axis_1'),
    TestCase(NODE_TEST, 'test_hardmax_axis_2'),
    TestCase(NODE_TEST, 'test_hardmax_default_axis'),
    TestCase(NODE_TEST, 'test_hardmax_example'),
    TestCase(NODE_TEST, 'test_hardmax_negative_axis'),
    TestCase(NODE_TEST, 'test_hardmax_one_hot'),

    TestCase(NODE_TEST, 'test_reduce_l1_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_l1_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_l1_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_l1_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_l1_keep_dims_example'),
    TestCase(NODE_TEST, 'test_reduce_l1_keep_dims_random'),
    TestCase(NODE_TEST, 'test_reduce_l1_negative_axes_keep_dims_example'),
    TestCase(NODE_TEST, 'test_reduce_l1_negative_axes_keep_dims_random'),
    TestCase(NODE_TEST, 'test_reduce_l2_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_l2_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_l2_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_l2_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_l2_keep_dims_example'),
    TestCase(NODE_TEST, 'test_reduce_l2_keep_dims_random'),
    TestCase(NODE_TEST, 'test_reduce_l2_negative_axes_keep_dims_example'),
    TestCase(NODE_TEST, 'test_reduce_l2_negative_axes_keep_dims_random'),
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
    TestCase(NODE_TEST, 'test_reduce_log_sum_exp_negative_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_log_sum_exp_negative_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_log_sum_negative_axes'),
    TestCase(NODE_TEST, 'test_reduce_max_default_axes_keepdim_example'),
    TestCase(NODE_TEST, 'test_reduce_max_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_max_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_max_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_max_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_max_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_max_negative_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_max_negative_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_mean_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_mean_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_mean_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_mean_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_mean_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_mean_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_mean_negative_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_mean_negative_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_min_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_min_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_min_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_min_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_min_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_min_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_min_negative_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_min_negative_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_prod_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_prod_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_sum_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_sum_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_sum_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_sum_negative_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_negative_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_default_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_default_axes_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_do_not_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_do_not_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_keepdims_random'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_negative_axes_keepdims_example'),
    TestCase(NODE_TEST, 'test_reduce_sum_square_negative_axes_keepdims_random'),

    TestCase(NODE_TEST, 'test_batchnorm_example'),
    TestCase(NODE_TEST, 'test_batchnorm_epsilon'),
    TestCase(NODE_TEST, 'test_lrn'),
    TestCase(NODE_TEST, 'test_lrn_default'),

    TestCase(NODE_TEST, 'test_dropout_default'),
    TestCase(NODE_TEST, 'test_dropout_random'),

    TestCase(NODE_TEST, 'test_isnan'),
    TestCase(NODE_TEST, 'test_isinf'),
    TestCase(NODE_TEST, 'test_isinf_negative'),
    TestCase(NODE_TEST, 'test_isinf_positive'),

    TestCase(NODE_TEST, 'test_where_example'),
    TestCase(NODE_TEST, 'test_where_long_example'),
    TestCase(NODE_TEST, 'test_nonzero_example'),
    TestCase(NODE_TEST, 'test_nonmaxsuppression_suppress_by_IOU'),
    TestCase(NODE_TEST, 'test_nonmaxsuppression_center_point_box_format'),
    TestCase(NODE_TEST, 'test_nonmaxsuppression_flipped_coordinates'),
    TestCase(NODE_TEST, 'test_nonmaxsuppression_identical_boxes'),
    TestCase(NODE_TEST, 'test_nonmaxsuppression_single_box'),
    TestCase(NODE_TEST, 'test_nonmaxsuppression_suppress_by_IOU_and_scores'),
    TestCase(NODE_TEST, 'test_nonmaxsuppression_two_batches'),
    TestCase(NODE_TEST, 'test_nonmaxsuppression_limit_output_size'),
    TestCase(NODE_TEST, 'test_nonmaxsuppression_two_classes'),
    TestCase(NODE_TEST, 'test_top_k'),
    TestCase(NODE_TEST, 'test_top_k_negative_axis'),
    TestCase(NODE_TEST, 'test_top_k_smallest'),
    TestCase(NODE_TEST, 'test_quantizelinear'),
    TestCase(NODE_TEST, 'test_dequantizelinear'),
    TestCase(NODE_TEST, 'test_qlinearmatmul_2D'),
    TestCase(NODE_TEST, 'test_qlinearmatmul_3D'),
    TestCase(NODE_TEST, 'test_qlinearconv'),
    TestCase(NODE_TEST, 'test_convinteger_with_padding'),
    TestCase(NODE_TEST, 'test_basic_convinteger'),
    TestCase(NODE_TEST, 'test_matmulinteger'),
    TestCase(NODE_TEST, 'test_round'),
    TestCase(NODE_TEST, 'test_bitshift_left_uint8'),
    TestCase(NODE_TEST, 'test_bitshift_right_uint8'),
    # TODO(take-cheeze): Support larger unsigned int types
    # TestCase(NODE_TEST, 'test_bitshift_left_uint64'),
    # TestCase(NODE_TEST, 'test_bitshift_left_uint32'),
    # TestCase(NODE_TEST, 'test_bitshift_left_uint16'),
    # TestCase(NODE_TEST, 'test_bitshift_right_uint64'),
    # TestCase(NODE_TEST, 'test_bitshift_right_uint32'),
    # TestCase(NODE_TEST, 'test_bitshift_right_uint16'),

    TestCase(NODE_TEST, 'test_scan9_sum'),

    TestCase(NODE_TEST, 'test_dynamicquantizelinear'),
    TestCase(NODE_TEST, 'test_dynamicquantizelinear_max_adjusted'),
    TestCase(NODE_TEST, 'test_dynamicquantizelinear_min_adjusted'),
    TestCase(NODE_TEST, 'test_dynamicquantizelinear_expanded'),
    TestCase(NODE_TEST, 'test_dynamicquantizelinear_max_adjusted_expanded'),
    TestCase(NODE_TEST, 'test_dynamicquantizelinear_min_adjusted_expanded'),
    TestCase(NODE_TEST, 'test_mvn'),
    TestCase(NODE_TEST, 'test_mvn_expanded'),

    TestCase(SIMPLE_TEST, 'test_sign_model'),
    TestCase(SIMPLE_TEST, 'test_single_relu_model'),
    # TODO(hamaji): Come up with a good way to handle ONNX's shape
    # inference for sequence types.
    TestCase(SIMPLE_TEST, 'test_sequence_model1', skip_shape_inference=True),
    TestCase(SIMPLE_TEST, 'test_sequence_model2', skip_shape_inference=True),
    TestCase(SIMPLE_TEST, 'test_sequence_model3', skip_shape_inference=True),
    TestCase(SIMPLE_TEST, 'test_sequence_model4', skip_shape_inference=True),
    TestCase(SIMPLE_TEST, 'test_sequence_model5', skip_shape_inference=True),
    TestCase(SIMPLE_TEST, 'test_sequence_model6', skip_shape_inference=True),
    TestCase(SIMPLE_TEST, 'test_sequence_model7', skip_shape_inference=True),

    TestCase(SIMPLE_TEST, 'test_expand_shape_model1'),
    TestCase(SIMPLE_TEST, 'test_expand_shape_model2'),
    TestCase(SIMPLE_TEST, 'test_expand_shape_model3'),
    TestCase(SIMPLE_TEST, 'test_expand_shape_model4'),
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
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d', fail=fail_1d_conv_pool),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_dilated', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_groups', fail=fail_1d_conv_pool),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_pad1', fail=fail_1d_conv_pool),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_pad1size1', fail=fail_1d_conv_pool),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_pad2', fail=fail_1d_conv_pool),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_pad2size1', fail=fail_1d_conv_pool),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Conv1d_stride', fail=fail_1d_conv_pool),
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
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_GLU'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_GLU_dim'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_LeakyReLU'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_LeakyReLU_with_negval'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Linear', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_Linear_no_bias'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_LogSoftmax'),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_MaxPool1d', fail=fail_1d_conv_pool),
    TestCase(ONNX_TEST_DATA, 'pytorch-converted/test_MaxPool1d_stride', fail=fail_1d_conv_pool),
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
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_max'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_maxpool'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_min'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_mm', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_non_float_params'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_pad', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_params'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_permute2'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_pow', equal_nan=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_reduced_mean'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_reduced_mean_keepdim'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_reduced_sum'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_reduced_sum_keepdim'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_repeat', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_repeat_dim_overflow', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_selu'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_sqrt', equal_nan=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_symbolic_override', fail=True),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_symbolic_override_nested'),
    TestCase(ONNX_TEST_DATA, 'pytorch-operator/test_operator_view', fail=True),
]

TEST_PATHS = set()
for test_case in TEST_CASES:
    TEST_PATHS.add(test_case.test_dir)

opsets_blacklist = {
    8: set([
        # Fixed in https://github.com/onnx/onnx/pull/1437
        'test_convtranspose_output_shape',
    ]),
}

if args.all:
    def extend_all_onnx_test_data(opset_version=None):
        tests_dir = ONNX_TEST_DATA
        if opset_version is not None:
            tests_dir = tests_dir.replace(
                'third_party/onnx/',
                'third_party/onnx-{}/'.format(opset_version))
        models = glob.glob(os.path.join(tests_dir, '*/*/model.onnx'))
        for onnx in sorted(models):
            path = os.path.dirname(onnx)
            if path not in TEST_PATHS:
                case = TestCase(os.path.dirname(path), os.path.basename(path),
                                fail=True)
                TEST_CASES.append(case)

    extend_all_onnx_test_data()
    for opset in target_opsets:
        extend_all_onnx_test_data(opset)
elif len(target_opsets) > 0:
    print('Finding opset variants: {}'.format(target_opsets))
    new_tcs = []
    skip_tcs = {}
    for opset in target_opsets:
        for tc in TEST_CASES:
            if opset in opsets_blacklist and tc.name in opsets_blacklist[opset]:
                continue

            var_test_dir = tc.test_dir.replace(
                'third_party/onnx/',
                'third_party/onnx-{}/'.format(opset))
            if os.path.isdir(var_test_dir) is False:
                if tc.name not in skip_tcs:
                    skip_tcs[tc.name] = []
                skip_tcs[tc.name].append(opset)
                continue
            new_tcs.append(TestCase(
                name=tc.name, test_dir=var_test_dir,
                rtol=tc.rtol, atol=tc.atol, equal_nan=tc.equal_nan,
                fail=tc.fail, opset_version=opset))

    for tc_name, opsets in skip_tcs.items():
        print('Skipping {} for opsets: {}'.format(tc_name, opsets))
    TEST_CASES.extend(new_tcs)

num_official_onnx_tests = len(TEST_CASES)

for backprop_test in gen_backprop_tests_oc.get_backprop_tests():
    assert os.path.exists(backprop_test.test_dir)
    TEST_CASES.append(backprop_test)

for backprop_test in gen_backprop_tests_pc.get_backprop_tests():
    assert os.path.exists(backprop_test.test_dir)
    TEST_CASES.append(backprop_test)

for test in gen_extra_test.get_tests():
    assert os.path.exists(test.test_dir), test.test_dir
    TEST_CASES.append(test)

for name, _, _, kwargs in gen_large_tests_oc.get_large_tests():
    dirname = 'out'
    TEST_CASES.append(TestCase(dirname, name, want_gpu=True, **kwargs))

TEST_CASES.append(TestCase('out', 'backprop_test_mnist_mlp'))

TEST_CASES.append(TestCase('data', 'shufflenet', want_gpu=True))
TEST_CASES.append(TestCase('data', 'mnist'))

TEST_CASES.extend(ch2o_tests.get())

TEST_CASES.extend(elichika_tests.get())

TEST_CASES.extend(onnx_chainer_tests.get(target_opsets))

TEST_CASES.extend(onnx_real_tests.get())

TEST_CASES.extend(gen_chainercv_model_tests.get_tests())

new_tests = []
for test in TEST_CASES:
    if not test.is_backprop:
        continue

    # TODO(mkusumoto): remove this "if" after fixing issue
    if not test.name.startswith('large_oc'):
        new_test = copy.copy(test)
        new_test.name = test.name + '_two_phase'
        new_test.is_backprop_two_phase = True
        new_tests.append(new_test)

    # TODO(hamaji): Temporarily disabled due to shape inference change in ONNX.
    if test.name.startswith('backprop_test_oc_split_2'):
        continue

    # TODO(hamaji): Unexpected shape will appear due to broadcast.
    if test.name.startswith('backprop_test_oc_pow_const'):
        continue

    if test.fixed_batch_norm:
        continue

    # computation_order is supported in limited test cases
    if test.name.startswith('backprop_test_oc'):
        for two_phase in [False, True]:
            new_test = copy.copy(test)
            new_test.name = test.name + '_computation_order'
            new_test.computation_order = 'dummy'
            new_test.is_backprop_two_phase = two_phase
            new_tests.append(new_test)

    # add more tests for computation_order using CustomPolicy
    if test.name.startswith('backprop_test_oc_tanh2') or\
       test.name.startswith('backprop_test_oc_mul2') or\
       test.name.startswith('backprop_test_oc_max_pool2'):
        order_strings = [
            'CF0,CF1,BF1,BF0',
            'CF0,CF1,FFo0,CF1,BF1,BF0',
            'CF0,CF1,FFo0,FFt0,CF0,CF1,BF1,BF0',
            'CF0,CF1,FFt0,FFo0,CF0,CF1,BF1,BF0',
            'CF0,CF1,FFt0,CF0,BF1,BF0',
            'CF0,CF1,FFt0,CF0,FFt0,CF0,BF1,BF0',
            'CF0,CF1,FFt0,CF0,FFo0,CF1,BF1,BF0',
            'CF0,CF1,FFt0,CF0,FFo0,CF1,FFt0,CF0,BF1,BF0',
            'CF0,CF1,BF1,FFt0,CF0,BF0',
        ]
        if not test.name.startswith('backprop_test_oc_max_pool2'):
            order_strings.append(
                'CF0,CF1,BF1,FFt0,CF0,FFo0,FFt0,CF0,CF1,BF0'
            )
        for order_string in order_strings:
            for two_phase in [False, True]:
                new_test = copy.copy(test)
                new_test.name = test.name + '_custom_computation_order_'\
                    + order_string
                new_test.computation_order = 'custom_' + order_string
                new_test.is_backprop_two_phase = two_phase
                new_tests.append(new_test)

    if test.name.startswith('backprop_test_oc_branched_conv'):
        order_string = 'CF0,CF1,CF2,CF3,BF3,FFt2,BF1,CF2,BF2,BF0'
        for two_phase in [False, True]:
            new_test = copy.copy(test)
            new_test.name = test.name + '_custom_computation_order_'\
                + order_string
            new_test.computation_order = 'custom_' + order_string
            new_test.is_backprop_two_phase = two_phase
            new_tests.append(new_test)

    # run gpu test for the test cases of onnx_chainer
    # NOTE: We don't add tests for float16 case because they fail with --fuse
    # option. We may resolve this in future.
    if test.name.startswith('backprop_test_oc') and\
       not test.name.endswith('float16'):
        new_test = copy.copy(test)
        new_test.want_gpu = True
        new_tests.append(new_test)

for test in new_tests:
    TEST_CASES.append(test)


if args.ngraph:
    # TODO(hamaji): Triage these failures.
    ngraph_blacklist = [
        'extra_test_loop_scan_out',
        'extra_backprop_test_need_stack_loop',
        'ch2o_node_Linear_backprop',
        'ch2o_node_Linear_backprop_diversed',
        'backprop_test_oc_mul_same_float32_two_phase',
        'backprop_test_oc_mul_same_float64_two_phase',
        'backprop_test_oc_sigmoid_float64_two_phase',
        'extra_backprop_test_need_stack_loop_two_phase',
        'test_gemm_default_no_bias',
    ]
    for test in TEST_CASES:
        if test.name in ngraph_blacklist:
            test.fail = True
        if '_float16' in test.name:
            # TODO(hamaji): Skip float16 tests since nGraph
            # automatically promote float16 to float32.
            test.fail = True
        if test.name.endswith('_sigmoid_float64'):
            # TODO(hamaji): nGraph seems not to support fp64 sigmoid.
            test.fail = True
        if re.search(r'grouped_conv_.*float64', test.name):
            test.fail = True

if args.failed:
    if not os.path.exists(args.failure_log):
        raise RuntimeError('No failure log in %s' % args.failure_log)
    failed_test_names = set()
    with open(args.failure_log, 'rb') as f:
        for line in f:
            if line.startswith(b'=== '):
                matched = re.match(r'=== (\S+) ===', line.decode())
                if matched:
                    failed_test_names.add(matched.group(1))
    TEST_CASES = [case for case in TEST_CASES
                  if case.name in failed_test_names]

if args.test_filter is not None:
    reg = re.compile(args.test_filter)
    TEST_CASES = [case for case in TEST_CASES if reg.search(case.name)]

if not args.all:
    TEST_CASES = [case for case in TEST_CASES if not case.fail]


def _start_output(msg):
    if sys.stdout.isatty():
        if len(msg) > 75:
            msg = msg[:36] + '...' + msg[-36:]
        sys.stdout.write('\r' + ' ' * 78 + '\r' + msg)
    else:
        sys.stdout.write(msg)


class TestRunner(object):
    def __init__(self, test_cases, show_log):
        self.test_cases = test_cases
        self.tested = []
        self.failed = []
        self.show_log = show_log

    def run(self, num_parallel_jobs):
        tests = list(reversed(self.test_cases))
        procs = {}
        while tests or procs:
            if tests and len(procs) < num_parallel_jobs:
                test_case = tests.pop()
                if num_parallel_jobs == 1:
                    _start_output('%s... ' % test_case.name)
                log_file = open(test_case.log_filename, 'wb')
                proc = subprocess.Popen(test_case.args,
                                        stdout=subprocess.PIPE,
                                        stderr=log_file)
                procs[proc.pid] = (test_case, proc, log_file)
                continue

            assert procs
            pid, status = os.wait()
            assert pid in procs
            test_case, proc, log_file = procs[pid]
            del procs[pid]
            log_file.close()

            if num_parallel_jobs != 1:
                _start_output('%s... ' % test_case.name)
            self.tested.append(test_case)
            if status == 0:
                if test_case.fail:
                    sys.stdout.write('%sOK (unexpected)%s\n' % (YELLOW, RESET))
                else:
                    sys.stdout.write('%sOK%s' % (GREEN, RESET))
                    if not sys.stdout.isatty():
                        sys.stdout.write('\n')
            else:
                self.failed.append(test_case)
                sys.stdout.write('%sFAIL%s: %s\n' %
                                 (RED, RESET, test_case.repro_cmdline()))
            if status != 0 or self.show_log:
                sys.stdout.buffer.write(test_case.log_read())
                if status != 0:
                    sys.stdout.write('%s$%s %s\n' %
                                     (RED, RESET, test_case.repro_cmdline()))

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
    run_onnx_menoh = os.path.join(args.build_dir, 'menoh/run_onnx_menoh')

    tested = []
    failed = []
    tests = []
    gpu_tests = []
    for test_case in TEST_CASES:
        runner = run_onnx_menoh
        if (test_case.is_backprop or
            test_case.is_backprop_two_phase or
            test_case.equal_nan or
            test_case.skip_shape_inference or
            test_case.skip_runtime_type_check or
            test_case.want_gpu or
            test_case.computation_order or
            not test_case.test_dir.startswith(NODE_TEST)):
            runner = run_onnx

        if len(target_opsets) != 0:
            if args.only_opset_targetable and test_case.opset_version is None:
                continue
            if not (test_case.opset_version in target_opsets):
                continue

        test_case.runner = run_onnx
        test_case.args = [runner, '--test', test_case.test_dir]
        test_case.args.append('--compiler_log')
        is_gpu = False
        if test_case.rtol is not None:
            test_case.args += ['--rtol', str(test_case.rtol)]
        if test_case.atol is not None:
            test_case.args += ['--atol', str(test_case.atol)]
        if test_case.equal_nan:
            test_case.args += ['--equal_nan']
        if test_case.skip_shape_inference:
            test_case.args.append('--skip_inference')
        if test_case.skip_runtime_type_check:
            test_case.args.append('--skip_runtime_type_check')
        if test_case.fixed_batch_norm:
            test_case.args.append('--fixed_batch_norm')
        if test_case.is_backprop_two_phase:
            test_case.args.append('--backprop_two_phase')
        elif test_case.is_backprop:
            test_case.args.append('--backprop')

        if test_case.computation_order:
            test_case.args.append(
                '--computation_order=' + test_case.computation_order)
        elif args.computation_order:
            test_case.args.append(
                '--computation_order=' + args.computation_order)

        if test_case.backend is not None:
            test_case.args.append('--backend')
            test_case.args.append(test_case.backend)
        if args.verbose:
            test_case.args.append('--verbose')
        device = args.device
        if test_case.want_gpu or args.use_gpu_all:
            if not args.use_gpu and not args.use_gpu_all:
                continue
            if device is None:
                device = 'cuda'
            is_gpu = True
        if device is not None:
            test_case.args.extend(['-d', device])

        if args.fuse:
            test_case.args.append('--fuse_operations')
            if is_gpu:
                test_case.args.append('--use_nvrtc')
        if args.ngraph:
            test_case.args.append('--fuse_operations')
            test_case.args.append('--use_ngraph')

        if args.snpe:
            test_case.args.append('--use_snpe')

        if args.cache:
            test_case.args.append('--use_cached_model')

        if is_gpu:
            gpu_tests.append(test_case)
        else:
            tests.append(test_case)

    print('Testing %d tests with %s and %s' %
          (len(tests + gpu_tests), run_onnx, run_onnx_menoh))

    for test in tests + gpu_tests:
        test.prepare()

    for tests, num_jobs in [(tests, args.jobs), (gpu_tests, 1)]:
        runner = TestRunner(tests, args.show_log)
        runner.run(num_jobs)
        tested += runner.tested
        failed += runner.failed

    if failed:
        with open(args.failure_log, 'wb') as f:
            for test in failed:
                f.write(('=== %s ===\n' % test.name).encode())
                f.write(('$ %s\n' % test.repro_cmdline()).encode())
                f.write(test.log_read())
                f.write('\n'.encode())
        print('%d/%d tests failed! (see %s)' %
              (len(failed), len(tested), args.failure_log))
        sys.exit(1)
    else:
        print('ALL %d tests OK! (%d from ONNX)' %
              (len(tested), num_official_onnx_tests))


main()
