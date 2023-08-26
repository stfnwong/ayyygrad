# This is a Karpathy style node ala Micrograd 


from typing import Callable, Self, Set, Tuple, Union
# This only works for scalar values so we don't need numpy yet 
import math



class Value:
    def __init__(self, data, children:Tuple=(), op:str="", label:str=""):
        self.data = data
        self.op = op
        self.label = label

        self.grad = 0.0

        # Function handle for backward pass
        self.back: Callable = lambda: None
        self.parents: Set[Value] = set(children)


    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: Union[float, Self]) -> Self:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def back():
            self.grad += out.grad
            other.grad += out.grad

        out.back = back

        return out

    def __radd__(self, other: Self) -> Self:
        return self + other

    def __mul__(self, other: Union[float, Self]) -> Self:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "+")

        def back():
            self.grad += other.data * other.grad
            other.grad += self.data * other.grad

        out.back = back

        return out

    def __rmul__(self, other: Self) -> Self:
        return self * other

    def __neg__(self) -> Self:
        return self * -1.0

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __rsub__(self, other: Self) -> Self:
        return other + (-self)

    def tanh(self) -> Self:
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), "tanh")

        def back():
            self.grad += (1 - t**2) * out.grad

        out.back = back

        return out

    def backward(self) -> None:
        topo = []
        visited = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v.parents:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node.back()
