# A graph tracer.
# Based on the material here (https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf)

from typing import Any, Callable, Dict, Generator, List, Optional, Self, Tuple, Union
from contextlib import contextmanager



class Node:
    """
    One node in a computational graph
    """

    def __init__(
        self,
        value: Any,
        func: Callable,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
        parent_argnums: List[int],
        parents: List[Self]
    ):
        """
        parent_argnums (List[int]): Positional indicies of boxed values

        parents: List of parent Nodes paired with each element of parent_argnums
        """

        self.parents: List[Self] = parents
        self.data = (func, value, args, kwargs, parent_argnums)

    def init_root(self):
        self.parents = []
        self.data = (lambda x: x, None, (), {}, [])

    @classmethod
    def new_root(cls, *args, **kwargs) -> Self:
        root = cls.__new__(cls)
        root.init_root()
        return root


class Box:
    """
    Boxes a single value in a computation graph
    """

    # Type -> Subclass mapping, where type may be a Box and subclass is any class
    # that takes the sam arguments for __init__
    type_mapping = dict()
    types = set()

    def __init__(self, value: Any, trace_id: int, node: Node):
        self.value = value
        self.trace_id = trace_id
        self.node = node

    def __bool__(self) -> bool:
        return bool(self.value)

    # NOTE: some legacy thing from Python 2?
    __nonzero__ = __bool__


    def __str__(self) -> str:
        return f"Ayygrad {self.__name__}, value {self.value}"

    @classmethod
    def register(cls, value_type):
        Box.types.add(cls)
        Box.type_mapping[value_type] = cls

        # The Box implementation for a Box-type is itself. This stops the inner Box's
        # computation graph from interacting with the outer Box's computation graph.
        Box.type_mapping[cls] = cls


# TODO: add TypeVar here?
box_type_mappings = Box.type_mapping
box_types = Box.types

isbox = lambda x: type(x) in box_types   # turns out to be faster than isinstance


def new_box(value: Any, trace_id: int, node: Node):
    """
    Box an unboxed value
    """

    try:
        box_type_mappings[type(value)](value, trace_id, node)
    except KeyError:
        raise TypeError(f"Can't differentiate with respect to {type(value)}")



class TraceStack:
    """
    Track the number of times trace() has been called.
    """

    def __init__(self):
        self.top = -1

    @contextmanager
    def new_trace(self) -> Generator[int, None, None]:
        self.top += 1
        yield self.top
        self.top -= 1



trace_stack = TraceStack()


def trace(start_node: Node, func: Callable, X: Any) -> Tuple[Union[Box, Any], Optional[Node]]:
    with trace_stack.new_trace() as trace_id:
        start_box = new_box(X, trace_id, start_node)

        # Apply the function to the boxed value
        end_box = func(start_box)

        if isbox(end_box) and (end_box, trace_id) == (start_box, trace_id):
            # Extract final value 
            return end_box.value, end_box.node
        else:
            return end_box, None

