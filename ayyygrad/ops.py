# Functions that can go in a computation graph


from typing import Tuple
import numpy as np

from ayyygrad.vjp import defvjp


# Some primitive ops
# TODO: for logistic() we need reciprocal(), add(), exp(), negative()

np.reciprocal


def reciprocal(
