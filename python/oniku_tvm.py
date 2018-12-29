import topi
import tvm


@tvm.api.register_func('oniku.tvm.conv2d')
def _conv2d(target, inputs, pad_h, pad_w, stride_h, stride_w):
    with target:
        tophub_context = tvm.autotvm.tophub.context(target)
        with tophub_context:
            return topi.nn.conv2d(inputs[0],
                                  inputs[1],
                                  [stride_h, stride_w],
                                  [pad_h, pad_w],
                                  [1, 1],
                                  out_dtype=inputs[0].dtype)


@tvm.api.register_func('oniku.tvm.schedule_conv2d')
def _schedule_conv2d(target, outputs):
    with target:
        tophub_context = tvm.autotvm.tophub.context(target)
        with tophub_context:
            schedule = topi.generic.schedule_conv2d_nchw(outputs)
            return schedule



def init():
    pass
