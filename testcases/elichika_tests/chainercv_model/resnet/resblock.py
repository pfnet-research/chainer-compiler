import chainer
import chainer.functions as F

from testcases.elichika_tests.chainercv_model.utils import Conv2DBNActiv
from testcases.elichika_tests.chainercv_model.utils import PickableSequentialChain
from testcases.elichika_tests.chainercv_model.utils import SEBlock
from chainer_compiler.elichika.parser import flags


class ResBlock(PickableSequentialChain):

    """A building block for ResNets.

    in --> Bottleneck with residual_conv --> Bottleneck * (n_layer - 1) --> out

    Args:
        n_layer (int): The number of layers used in the building block.
        in_channels (int): The number of channels of the input array.
        mid_channels (int): The number of channels of intermediate arrays.
        out_channels (int): The number of channels of the output array.
        stride (int or tuple of ints): Stride of filter application.
        dilate (int or tuple of ints): Dilation factor of filter applications.
            :obj:`dilate=d` and :obj:`dilate=(d, d)` are equivalent.
        groups (int): The number of groups to use grouped convolution in the
            second layer of each bottleneck. The default is one, where
            grouped convolution is not used.
        initialW (callable): Initial weight value used in
            the convolutional layers.
        bn_kwargs (dict): Keyword arguments passed to initialize
            :class:`chainer.links.BatchNormalization`.
        stride_first (bool): This determines the behavior of the
            bottleneck with a shortcut. If :obj:`True`, apply strided
            convolution with the first convolution layer.
            Otherwise, apply strided convolution with the
            second convolution layer.
        add_seblock (bool): If :obj:`True`, apply a squeeze-and-excitation
            block to each residual block.

    """

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, dilate=1, groups=1, initialW=None,
                 bn_kwargs={}, stride_first=False, add_seblock=False):
        super(ResBlock, self).__init__()
        # Dilate option is applied to all bottlenecks.
        with self.init_scope():
            self.a = Bottleneck(
                in_channels, mid_channels, out_channels, stride, dilate,
                groups, initialW, bn_kwargs=bn_kwargs, residual_conv=True,
                stride_first=stride_first, add_seblock=add_seblock)
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = Bottleneck(
                    out_channels, mid_channels, out_channels, stride=1,
                    dilate=dilate, initialW=initialW, bn_kwargs=bn_kwargs,
                    residual_conv=False, add_seblock=add_seblock,
                    groups=groups)
                setattr(self, name, bottleneck)
            self._pick = (name,)
            self._return_tuple = False


class Bottleneck(chainer.Chain):

    """A bottleneck layer.

    Args:
        in_channels (int): The number of channels of the input array.
        mid_channels (int): The number of channels of intermediate arrays.
        out_channels (int): The number of channels of the output array.
        stride (int or tuple of ints): Stride of filter application.
        dilate (int or tuple of ints): Dilation factor of filter applications.
            :obj:`dilate=d` and :obj:`dilate=(d, d)` are equivalent.
        groups (int): The number of groups to use grouped convolution in the
            second layer. The default is one, where grouped convolution is
            not used.
        initialW (callable): Initial weight value used in
            the convolutional layers.
        bn_kwargs (dict): Keyword arguments passed to initialize
            :class:`chainer.links.BatchNormalization`.
        residual_conv (bool): If :obj:`True`, apply a 1x1 convolution
            to the residual.
        stride_first (bool): If :obj:`True`, apply strided convolution
            with the first convolution layer. Otherwise, apply
            strided convolution with the second convolution layer.
        add_seblock (bool): If :obj:`True`, apply a squeeze-and-excitation
            block to each residual block.

    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=1, dilate=1, groups=1, initialW=None, bn_kwargs={},
                 residual_conv=False, stride_first=False, add_seblock=False):
        if stride_first:
            first_stride = stride
            second_stride = 1
        else:
            first_stride = 1
            second_stride = stride
        super(Bottleneck, self).__init__()
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(in_channels, mid_channels,
                                       1, first_stride, 0,
                                       nobias=True, initialW=initialW,
                                       bn_kwargs=bn_kwargs)
            # pad = dilate
            self.conv2 = Conv2DBNActiv(mid_channels, mid_channels,
                                       3, second_stride, dilate, dilate,
                                       groups, nobias=True, initialW=initialW,
                                       bn_kwargs=bn_kwargs)
            self.conv3 = Conv2DBNActiv(mid_channels, out_channels, 1, 1, 0,
                                       nobias=True, initialW=initialW,
                                       activ=None, bn_kwargs=bn_kwargs)
            self._pick = ('conv3',)
            if add_seblock:
                self.se = SEBlock(out_channels)
                self._pick = ('se',)
            if residual_conv:
                self.residual_conv = Conv2DBNActiv(
                    in_channels, out_channels, 1, stride, 0,
                    nobias=True, initialW=initialW,
                    activ=None, bn_kwargs=bn_kwargs)
                self._pick = ('residual_conf',)
            self._return_tuple = False


    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        if hasattr(self, 'se'):
            h = self.se(h)

        if hasattr(self, 'residual_conv'):
            residual = self.residual_conv(x)
        else:
            residual = x
        h += residual
        h = F.relu(h)
        return h


from chainer_compiler.elichika import testtools
from chainer import initializers
import numpy as np

def main():
    np.random.seed(314)

    model = ResBlock(3, None, 64, 256, 1, initialW=initializers.HeNormal(scale=1., fan_option='fan_out'), stride_first=False)

    v = np.random.rand(2, 64, 56, 56).astype(np.float32)

    testtools.generate_testcase(model, [v])

if __name__ == '__main__':
    main()