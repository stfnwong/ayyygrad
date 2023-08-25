# A Tensor

from typing import Any, Optional, Self
from collections import namedtuple
import numpy as np


Children = namedtuple("Children", ["a", "b", "op"])



class Tensor:
    def __init__(
        self,
        value: Optional[Any]=None,
        children: Optional[Children]=None,
    ):
        self.value = value
        self.children = children

    def __repr__(self) -> str:
        return f"T:{self.value}"

    def __add__(self, other: Self) -> Self:
        c = Children(self, other, np.add)
        t = Tensor(children=c)

        return t.forward()

    def __mul__(self, other: Self) -> Self:
        c = Children(self, other, np.multiply)
        t = Tensor(children=c)

        return t.forward()

    def forward(self) -> Self:
        # Check if this is a leaf node
        if self.children is None:
            return self

        # Compute forward pass of children in tree
        a = self.children.a.forward()
        b = self.children.b.forward()

        # If values are set then compute the real value of this Tensor
        if a.value is not None and b.value is not None:
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
        elif self.children.op is np.multiply:    # (ab)' = a'b + ab'
            t = self.children.a.grad(derive_to) * self.children.b + self.children.a * self.children.b.grad(derive_to)
        else:
            raise NotImplementedError(f"Op [{self.children.op}] not implemented")

        return t



