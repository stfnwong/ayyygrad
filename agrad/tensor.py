# A Tensor

from typing import Any, Optional, Self
from collections import namedtuple
import numpy as np


Children = namedtuple("Children", ["a", "b", "op"])



class Tensor:
    def __init__(
        self,
        value: Optional[float]=None,
        children: Optional[Children]=None,
    ):
        self.value: Optional[float] = value
        self.children = children

    def __repr__(self) -> str:
        return f"T:{self.value}"

    def __add__(self, other: Self) -> Self:
        c = Children(self, other, np.add)
        t = Tensor(children=c)

        return t.forward()

    def __sub__(self, other: Self) -> Self:
        c = Children(self, other, np.subtract)
        t = Tensor(children=c)

        return t.forward()

    def __mul__(self, other: Self) -> Self:
        c = Children(self, other, np.multiply)
        t = Tensor(children=c)

        return t.forward()

    def __truediv__(self, other: Self) -> Self:
        c = Children(self, other, np.divide)
        t = Tensor(children=c)

        return t.forward()

    def __neg__(self) -> Self:
        c = Children(Tensor(np.zeros_like(self.value)), self, np.subtract)
        t = Tensor(children=c)

        return t.forward()

    def exp(self) -> Self:
        c = Children(self, None, np.exp)
        t = Tensor(children=c)

        return t.forward()

    def forward(self) -> Self:
        # Check if this is a leaf node
        if self.children is None:
            return self

        a = Tensor()
        b = Tensor()

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
        # Derivative of a Tensor with itself is 1
        if self is derive_to:
            return Tensor(1)

        # Derivative of a scalar with a Tensor is 0
        if self.children is None:
            return Tensor(0)

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

