# Functions that can go in a computation graph


from typing import Tuple
from ayyydz.tensor import Tensor



class Function:
    def forward(self, *args, **kwargs):
        raise NotImplemented(f"forward() not implemented for {type(self)}")

    def backward(self, *args, **kwargs):
        raise NotImplemented(f"backward() not implemented for {type(self)}")




class Add(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x + y

    def backward(self, dz: Tensor) -> Tensor:
        return dz


class Sub(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x - y

    def backward(self, dz: Tensor) -> Tensor:
        return dz


class Mul(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.x = x
        self.y = y
        return x * y

    def backward(self, dz: Tensor) -> Tuple[Tensor, Tensor]:
        dx = self.x * dz
        dy = self.y * dz

        return (dx, dy)


class Div(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.x = x
        self.y = y
        return x / y

    def backward(self, dz: Tensor) -> Tensor:
        pass


