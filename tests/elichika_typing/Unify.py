import unittest

import chainer_compiler.elichika.parser.types as T


def is_unifiable(ty1, ty2):
    try:
        T.unify(ty1, ty2)
        return True
    except T.UnifyError as e:
        return False


class TestUnify(unittest.TestCase):
    def test_1(self):
        x = T.TyVar()
        ty1 = T.TyArrow([x], x)
        ty2 = T.TyArrow([T.TyString()], T.TyInt())
        self.assertFalse(is_unifiable(ty1, ty2))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
