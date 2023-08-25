from agrad.tensor import Tensor


def test_tensor_add():
    a = Tensor(3)
    b = Tensor(4)

    c = a + b
    c.forward()
    assert c.value == 7



def test_tensor_add_children():
    a = Tensor(3)
    b = Tensor(4)

    z1 = a + b
    z2 = z1 * b

    z2.forward()     # need to call forward() to compute values in leaf nodes
    assert z2.value == 28  # (3 + 4) * 4
