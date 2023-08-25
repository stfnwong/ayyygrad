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



def test_tensor_grad():
    a = Tensor(None)
    b = Tensor(None)

    z1 = a + b
    z2  = z1 * b

    # Before we compute the gradient these should be empty
    assert a.value == None
    assert b.value == None
    assert z2.value == None

    # Now we adjust the leaf node values and call forward()
    a.value = 3
    b.value = 5
    z2.forward()

    assert z2.value == 40

    g = z2.grad(b)
    assert g.value == 13

