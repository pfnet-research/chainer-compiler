import collections
import onnx

from typing import List, Mapping

from ch2o.utils import Env, totensor, new_tensor


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
        if not self.is_py:
            assert self.is_tensor() or self.is_sequence()
            assert not (self.is_tensor() and self.is_sequence())

    def __str__(self):
        if self.is_py:
            return 'Value(%s)' % str(self.value)
        else:
            return 'Value(%s)' % self.value.name

    def get_attribute(self, key: str) -> 'Value':
        if not self.is_py:
            raise TypeError('Unsupported attribute %s for an ONNX value' % key)
        return Value(getattr(self.value, key))

    def is_none(self) -> bool:
        return self.is_py and self.value is None

    def is_tensor(self) -> bool:
        return not self.is_py and self.value.type.HasField('tensor_type')

    def is_sequence(self) -> bool:
        return not self.is_py and self.value.type.HasField('sequence_type')

    def to_value_info(self, env: Env) -> onnx.ValueInfoProto:
        if self.is_py:
            # TODO(hamaji): Rewrite `totensor` to convert a Python
            # list to a tensor.
            self.value = totensor(self.value, env)
            self.is_py = False

        if not self.value.type.tensor_type:
            raise TypeError('Expected a tensor: %s' % self.value)

        return self.value

    def to_tensor(self, env: Env, dtype: type = None) -> onnx.ValueInfoProto:
        if self.is_py:
            # TODO(hamaji): Rewrite `totensor` to convert a Python
            # list to a tensor.
            self.value = totensor(self.value, env, dtype=dtype)
            self.is_py = False
        elif self.is_sequence():
            self.value = env.calc('OnikuxSequenceStack',
                                  inputs=[self.value.name])
            self.is_py = False

        assert self.is_tensor()
        return self.value

    def to_sequence(self, env: Env) -> onnx.ValueInfoProto:
        if self.is_py:
            if not isinstance(self.value, collections.Iterable):
                raise TypeError('Expected a sequence: %s' % self.value)
            res = env.calc_seq(
                "OnikuxSequenceCreate",
                inputs=[],
            )
            for v in self.value:
                v = v.to_tensor(env)
                res = env.calc_seq(
                    "OnikuxSequenceAppend",
                    inputs=[res.name, v.name],
                )
            self.value = res
            self.is_py = False
        elif self.is_tensor():
            self.value = env.calc_seq(
                'OnikuxSequenceSplit',
                inputs=[self.value.name]
            )

        assert self.is_sequence()
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
