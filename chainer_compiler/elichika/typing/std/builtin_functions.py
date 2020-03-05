from   chainer_compiler.elichika.typing.types  import *

__all__ = [ 'builtin_func_ty' ]


def ty_len(ty_args, ty_kwargs):
    x_type, = ty_args
    if isinstance(x_type, TySequence):
        return TyInt(x_type.size())
    if isinstance(x_type, TyTensor):
        return TyInt(x_type.shape[0].value)
    if isinstance(x_type, TyUserDefinedClass):
        assert hasattr(x_type.instance, '__len__')
        return TyInt(len(x_type.instance))
    assert False


builtin_func_ty = {
        len   : ty_len,
        }
