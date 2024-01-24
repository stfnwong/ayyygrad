import numpy as np


def real_logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# Eg:
# np.reciprocal(np.add(1.0, np.exp(np.negative(z))))
def logistic(z: np.ndarray) -> np.ndarray:
    pass



