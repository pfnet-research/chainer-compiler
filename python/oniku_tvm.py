import topi
import tvm


@tvm.api.register_func('oniku.tvm.conv2d')
def _conv2d(target, inputs, pad_h, pad_w, stride_h, stride_w):
    with target:
        return topi.nn.conv2d(inputs[0],
                              inputs[1],
                              [stride_h, stride_w],
                              [pad_h, pad_w],
                              [1, 1])


@tvm.api.register_func('oniku.tvm.schedule')
def _schedule(target, outputs):
    # TODO(hamaji): Implement this.
    pass


def init():
    pass
