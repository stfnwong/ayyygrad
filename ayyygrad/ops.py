# Functions that can go in a computation graph


from typing import Tuple
import numpy as np

# TODO: where should Function go?
from ayyygrad.tensor import Function,Tensor






class Mul(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.save(x, y)
        return x * y

    def backward(self, dz: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = self.saved_tensors
        dx = x * dz
        dy = y * dz

        return (dx, dy)

class Exp(Function):
    def forward(self, x: Tensor) -> Tensor:
        self.save(np.exp(x))
        return np.exp(x)


class ReLU(Function):
    def forward(self, x: Tensor) -> Tensor:
        self.save(x)
        return np.maximum(0, x)

    def backward(self, dz: Tensor) -> Tensor:
        x, = self.saved_tensors
        g = dz.copy()
        g[x < 0] = 0

