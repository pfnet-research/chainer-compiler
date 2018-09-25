import collections
import onnx

from typing import List, Mapping

from chainer2onnx.utils import Env, totensor


def _is_float_value(v):
    # The latter is for numpy-like things.
    return isinstance(v, float) or int(v) != v


class Value(object):
    """An object which holds either an ONNX value or a Python object."""

    def __init__(self, value):
        if isinstance(value, Value):
            value = value.value
        self.value = value
        self.is_py = not isinstance(self.value, onnx.ValueInfoProto)

    def get_attribute(self, key: str) -> 'Value':
        if not self.is_py:
            raise TypeError('Unsupported attribute %s for an ONNX value' % key)
        return Value(getattr(self.value, key))

    def call(self, args: List['Value'],
             kwargs: Mapping[str, 'Value'], env: Env):
        assert self.is_py
        return Value(self.value.call(args, kwargs, env))

    def value_info(self) -> onnx.ValueInfoProto:
        print(self.value)
        assert not self.is_py
        return self.value

    def is_none(self) -> bool:
        return self.is_py and self.value is None

    def to_tensor(self, env: Env) -> onnx.ValueInfoProto:
        if self.is_py:
            # TODO(hamaji): Rewrite `totensor` to convert a Python
            # list to a tensor.
            self.value = totensor(self.value, env)
            self.is_py = False

        if not self.value.type.tensor_type:
            raise TypeError('Expected a tensor: %s' % self.value)

        return self.value

    def to_float(self) -> float:
        if not self.is_py:
            raise TypeError('Expected a float scalar: %s' % self.value)
        return float(self.value)

    def to_int(self) -> int:
        if not self.is_py or _is_float_value(self.value):
            raise TypeError('Expected an int scalar: %s' % self.value)
        return int(self.value)

    def to_bool(self) -> bool:
        if not self.is_py or not isinstance(self.value, bool):
            raise TypeError('Expected a bool scalar: %s' % self.value)
        return bool(self.value)

    def to_int_list(self) -> List[int]:
        if not self.is_py or not isinstance(self.value, collections.Iterable):
            raise TypeError('Expected an int list: %s' % self.value)
        ints = list(self.value)
        if ints and _is_float_value(ints[0]):
            raise TypeError('Expected an int list: %s' % self.value)
        return ints
