from chainer_compiler.elichika.parser.utils import DummyFlag


def eval_as_written_target():
    return DummyFlag()

def ignore_branch():
    return DummyFlag()

def for_unroll(unroll=True):
    return DummyFlag()