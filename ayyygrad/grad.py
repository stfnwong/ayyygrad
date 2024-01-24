# Stuff for doing autograd

from typing import Any, Callable, Tuple, Union
from itertools import count
from collections import defaultdict
import numpy as np

from ayyygrad.trace import Box, Node, trace


# Topological sort for nodes


# Replace ith value of X with V
def subval(X, i, V) -> Tuple[Any]:
    x = list(X)
    x[i] = V
    return tuple(x)


def backward(graph: Callable, end_node: Node):
    pass


# Construct Vector-Jacobian Products
def make_vjp(func: Callable, X: Any) -> Tuple[Callable, Union[Box, Any]]:
    """
    Trace the computation to construct a graph and return a function that implements
    the backward pass
    """

    start_node = Node.new_root()
    end_value, end_node = trace(start_node, func, X)

    def vjp(g: Any):
        return backward(g, end_node)

    return (vjp, end_value)



def grad(func: Callable, argnum:int=0) -> Callable:
    """
    Construct a function f'(x) from a function f(x). Both f(x) and f'(x) are assumed to
    be single argument functions

    func (Callable): A single argument function
    argnum (int): Argument to differentiate with respect to.

    """

    def grad_func(*args, **kwargs):
        # replace args[argnum] with X
        unary_func = lambda x: func(subval(args, argnum, x), **kwargs)
        vjp, ans = make_vjp(unary_func, args[argnum])
        return vjp(np.ones_like(ans))

    return grad_func



# Keep track of all the VJPs here
base_vjps = defaultdict(dict)

def defvjp(func: Callable[..., ...], *vjps, **kwargs):
    """
    Register Vector-Jacobian Products functions.

    If we have a function f(x, y, ...) = ans, we want to register a VJP for each
    argument of f. Eg:

    vjp_x(g, ans, x, y, ...) = g df/dx
    vjp_y(g, ans, x, y, ...) = g df/dy


    Args:
        func    : Function to define Vector-Jacobian product for.
        *vjps   : Collection of Vector-Jacobian products, on for each argument of f().
        **kwargs: Any other keyword args. This function only takes 'argnums' from
                  this collection.
    """

    # We basically just stick these all in a dict with global scope.
    argnums = kwargs.get("argnums", count())
    for argnum, vjp in zip(argnums, vjps):
        base_vjps[func][argnum] = vjp
