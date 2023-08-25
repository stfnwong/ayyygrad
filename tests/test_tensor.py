from agrad.tensor import Tensor


def test_tensor_add():
    a = Tensor(3)
    b = Tensor(4)

    c = a + b
    assert c.value == 7
