import gast

class TypeChecker(gast.NodeTransformer):
    def __init__(self):
        super().__init__()
        pass

    def generic_visit(self, node):
        return super().generic_visit(node)
