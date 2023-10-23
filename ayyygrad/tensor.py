# A Tensor

from typing import Optional, Self, Union
from collections import namedtuple
import numpy as np
import copy

#from ayyygrad import ops     # TODO: later





# TODO: should we accept a list as a possible init?
class Tensor:
    def __init__(
        self,
        data: Union[float, np.ndarray],
        *parents: Self,
        requires_grad: Optional[bool]=None
    ):
        if type(data) is float:
            self.data = np.array([data])
        else:
            self.data = data

        self.grad: Optional[Self]=None

    # Operator overloads 
    def __mul__(self, X: Union[Tensor, float]) -> Self:
        # TODO: in tinygrad we call apply here 
        # eg: ops.Mul.apply(*self.broadcasted_(X)) 
        # full path is something like:  mul.apply() if X is Tensor else self.mul(1/X)
        pass

    # Arithmetic ops
    def mul(self, other: Self) -> Self:
        pass

    def copy(self) -> Self:
        return copy.copy(self)

    def backward(self) -> None:
        pass



class Function:
    def __init__(self, *tensors: Tensor):
        self.parents = tensors
        self.saved_tensors = []

    def save(self, *tensors: Tensor) -> None:
        self.saved_tensors.extend(tensors)

    def forward(self, *args, **kwargs):
        raise NotImplemented(f"forward() not implemented for {type(self)}")

    def backward(self, *args, **kwargs):
        raise NotImplemented(f"backward() not implemented for {type(self)}")








Children = namedtuple("Children", ["a", "b", "op"])


class OldTensor:
    def __init__(
        self,
        value: Optional[Union[float, np.ndarray]]=None,
        children: Optional[Children]=None,
        label:Optional[str]=None
    ):
        self.value: Optional[Union[float, np.ndarray]] = value
        self.children = children
        self.label = label

    def __repr__(self) -> str:
        return f"T:{self.value}"

    def __add__(self, other: Self) -> Self:
        c = Children(self, other, np.add)
        t = OldTensor(children=c)

        return t.forward()

    def __sub__(self, other: Self) -> Self:
        c = Children(self, other, np.subtract)
        t = OldTensor(children=c)

        return t.forward()

    def __mul__(self, other: Self) -> Self:
        c = Children(self, other, np.multiply)
        t = OldTensor(children=c)

        return t.forward()

    def __truediv__(self, other: Self) -> Self:
        c = Children(self, other, np.divide)
        t = OldTensor(children=c)

        return t.forward()

    def __neg__(self) -> Self:
        c = Children(OldTensor(np.zeros_like(self.value)), self, np.subtract)
        t = OldTensor(children=c)

        return t.forward()

    def exp(self) -> Self:
        c = Children(self, None, np.exp)
        t = OldTensor(children=c)

        return t.forward()

    def forward(self) -> Self:
        # Check if this is a leaf node
        if self.children is None:
            return self

        a = OldTensor()
        b = OldTensor()

        # Compute forward pass of children in tree
        if self.children.a is not None:
            a = self.children.a.forward()
        if self.children.b is not None:
            b = self.children.b.forward()

        # If a has a value we may be able to compute the value for this node
        if a.value is not None:
            if self.children.b is None:
                self.value = self.children.op(a.value)
            elif b.value is not None:
                self.value = self.children.op(a.value, b.value)

        return self

    def grad(self, derive_to: Self) -> Self:
        # Derivative of a OldTensor with itself is 1
        if self is derive_to:
            return OldTensor(1)

        # Derivative of a scalar with a OldTensor is 0
        if self.children is None:
            return OldTensor(0)

        # Recursively find derivatives for child nodes
        if self.children.op is np.add:    # (a + b)' = a' + b'
            t = self.children.a.grad(derive_to) + self.children.b.grad(derive_to)
        elif self.children.op is np.subtract:
            t = self.children.a.grad(derive_to) - self.children.b.grad(derive_to)
        elif self.children.op is np.multiply:    # (ab)' = a'b + ab'
            t = self.children.a.grad(derive_to) * self.children.b + self.children.a * self.children.b.grad(derive_to)
        elif self.children.op is np.divide:   # (ab)' = (a'b - ab') / b^2
            t = (
                self.children.a.grad(derive_to) * self.children.b -
                self.children.a * self.children.b.grad(derive_to)
            ) / (self.children.b * self.children.b)
        elif self.children.op is np.exp:        # exp(a)' = a' exp(a)
            t = self.children.a.grad(derive_to) * self.children.a.exp()
        else:
            raise NotImplementedError(f"Op [{self.children.op}] not implemented")

        return t
