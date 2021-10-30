""" Functions to parse the string conversion expressions retrieved from the ITOS
database.

Author:

    C.M. Gosmeyer, June 2018

References:

    https://stackoverflow.com/questions/2371436/evaluating-a-mathematical-expression-in-a-string

"""

import ast
import numpy as np
import operator as op

# supported operators
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}

def eval_expr(expr, parameter):
    """
    >>> eval_expr('2^6')
    4
    >>> eval_expr('2**6')
    64
    >>> eval_expr('1 + 2*3**(4^5) / (6 + -7)')
    -5.0

    Parameters
    ----------
    expr : string
        The expression.
    parameter : float
        The value to be evaluated with expression.

    Source
    ------
    https://stackoverflow.com/questions/2371436/evaluating-a-mathematical-expression-in-a-string    
    """
    expr = prepare_expr(expr, parameter)
    return eval_(ast.parse(expr, mode='eval').body)

def eval_(node):
    """

    Source
    ------
    https://stackoverflow.com/questions/2371436/evaluating-a-mathematical-expression-in-a-string
    """
    if isinstance(node, ast.Num): # <number>
        return node.n
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_(node.operand))
    #elif isinstance(node, ast.Set):
    #    return node.n
    else:
        raise TypeError(node)

def prepare_expr(expr, parameter):
    """ Prepare the expression string.

    Parameters
    ----------
    expr : string
        The expression.
    parameter : float
        The value to be evaluated with expression.
    """
    # First take out of tuple.
    if type(expr) == tuple:
        expr = expr[0]
    # Strip of whitespace
    expr = expr.strip(" ")
    # Do necessary replacements to Python can interpret expression.
    expr = expr.replace('^', '**')
    expr = expr.replace('{'+'0}', 'x')
    expr = expr.replace('{'+'1}', 'x')  ## does this need be another variable???
    expr = expr.replace('x', str(np.float64(parameter)))
    expr = expr.replace('X', str(np.float64(parameter)))
    expr = expr.replace('32768', str(np.float64(32768.0)))

    return expr


def build_poly_expr(query_tuple):
    """ Shove the polynomial coefficients returned from the Polynomial
    table into a string so can be compatible with functions used
    for evaluating the returns from the Expression table.

    ** depricated? **
    """
    print("query_tuple: ", query_tuple)
    expression = '0 + '
    factors = np.arange(7)

    for coeff, factor in zip(query_tuple, factors):
        if coeff != None:
            expression += '(' + str(np.float64(coeff)) + '*x^{}) + '.format(factor)

    # Remove trailing '+'
    expression = expression[:-3]
    
    # Return as a tuple.
    return (expression,)

