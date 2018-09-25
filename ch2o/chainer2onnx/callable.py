import inspect


class Callable(object):

    def __init__(self, fn):
        self.fn = fn
        self.sig = inspect.signature(self.fn)

    def call(self, args, kwargs, env):
        bound = self.sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return self.call_impl(env, *bound.args, **bound.kwargs)

    def call_impl(self, env, *args):
        raise NotImplementedError('call_impl must be implemented')
