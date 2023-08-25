# A Tensor 

from typing import Any, Optional, Self
from collections import namedtuple


Children = namedtuple("Children", ["a", "b", "op"])



class Tensor:
    def __init__(self, value:Optional[Any]=None):
        self.value = value

    def __repr__(self) -> str:
        return f"T:{self.value}"

    def __add__(self, other: Self) -> Self:
        t = Tensor(value=self.value + other.value)
        return t
