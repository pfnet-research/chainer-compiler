def the_preprocessor(ast):
    ast = simplify_unary_literal(ast)
    # ast = simplify_other(ast)
    return ast
