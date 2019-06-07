import inspect

from chainer_compiler.ch2o.value import Value


class Callable(object):

    def __init__(self, fn):
        self.fn = fn
        self.sig = inspect.signature(self.fn)

    def call(self, args, kwargs, env):
        bound = self.sig.bind(*args, **kwargs)
        bound.apply_defaults()
        args = [Value(a) for a in bound.args]
        kwargs = {k: Value(a) for k, a in bound.kwargs.items()}
        return self.call_impl(env, *args, **kwargs)

    def call_impl(self, env, *args):
        raise NotImplementedError('call_impl must be implemented')
