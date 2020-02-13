from copy import deepcopy
import unittest

from chainer_compiler.elichika.typing.types import *


def is_unifiable(ty1, ty2):
    try:
        unify(ty1, ty2)
        return True
    except UnifyError as e:
        return False


class TestUnify(unittest.TestCase):
    def test_not_unifiable(self):
        x = TyVar()
        ty1 = TyArrow([x], x)
        ty2 = TyArrow([TyString()], TyInt())
        self.assertFalse(is_unifiable(ty1, ty2))

    def test_deepcopy(self):
        x = TyVar()
        ty = TyArrow([x], x)
        ty1 = deepcopy(ty)
        ty2 = TyArrow([TyString()], TyInt())
        self.assertFalse(is_unifiable(ty1, ty2))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
