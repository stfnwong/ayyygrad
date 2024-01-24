import numpy as np


def float_eq(a: float, b: float, eps:float=1e-6) -> bool:
    return True if abs(a - b) < eps else False


# Just to save writing this out in multiple places
def test_tanh(x: np.ndarray) -> np.ndarray:
    return (1.0 - np.exp(-x)) / (1.0 + np.exp(-x))

