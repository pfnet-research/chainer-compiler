import contextlib

import topi
import tvm


@contextlib.contextmanager
def autotvm_context(target, autotvm_log):
    with target:
        if autotvm_log:
            with tvm.autotvm.apply_history_best(autotvm_log):
                yield
        else:
            tophub_context = tvm.autotvm.tophub.context(target)
            with tophub_context:
                yield


@tvm.api.register_func('oniku.tvm.conv2d')
def _conv2d(target, autotvm_log, inputs, pad_h, pad_w, stride_h, stride_w):
    with autotvm_context(target, autotvm_log):
        return topi.nn.conv2d(inputs[0],
                              inputs[1],
                              [stride_h, stride_w],
                              [pad_h, pad_w],
                              [1, 1],
                              out_dtype=inputs[0].dtype)


@tvm.api.register_func('oniku.tvm.conv2d_transpose')
def _conv2d_transpose(target, autotvm_log,
                      inputs, pad_h, pad_w, stride_h, stride_w):
    with autotvm_context(target, autotvm_log):
        return topi.nn.conv2d_transpose_nchw(inputs[0],
                                             inputs[1],
                                             [stride_h, stride_w],
                                             [pad_h, pad_w],
                                             out_dtype=inputs[0].dtype)


@tvm.api.register_func('oniku.tvm.schedule_conv2d')
def _schedule_conv2d(target, autotvm_log, outputs):
    with autotvm_context(target, autotvm_log):
        schedule = topi.generic.schedule_conv2d_nchw(outputs)
        return schedule


@tvm.api.register_func('oniku.tvm.schedule_conv2d_transpose')
def _schedule_conv2d_transpose(target, autotvm_log, outputs):
    with autotvm_context(target, autotvm_log):
        schedule = topi.generic.schedule_conv2d_transpose_nchw(outputs)
        return schedule



def init():
    pass
