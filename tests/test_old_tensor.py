from ayyygrad.tensor import OldTensor
from ayyygrad.util import float_eq


# ======== OldTensor - remove this eventually ======== #
def test_tensor_add():
    a = OldTensor(3)
    b = OldTensor(4)

    c = a + b
    c.forward()
    assert c.value == 7


def test_tensor_add_children():
    a = OldTensor(3)
    b = OldTensor(4)

    z1 = a + b
    z2 = z1 * b

    z2.forward()     # need to call forward() to compute values in leaf nodes
    assert z2.value == 28  # (3 + 4) * 4



def test_tensor_grad():
    a = OldTensor(None)
    b = OldTensor(None)

    z1 = a + b
    z2 = z1 * b

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


def test_tensor_larger_comp_graph():
    a = OldTensor(3)
    b = OldTensor(5)

    # Find the graph of the expression 
    # z = (12 - (x * exp(y))) / (45 + x * y * exp(-x))

    z = (OldTensor(12) - (a * b.exp())) / (OldTensor(45) + a * b * (-a).exp())
    za = z.grad(a)
    zb = z.grad(b)

    assert float_eq(za.value, -3.34729777301069)
    assert float_eq(zb.value, -9.70176956641438)

    # Check against a symbolic implementation
    import sympy as sym

    xs = sym.Symbol("xs")
    ys = sym.Symbol("ys")
    zs = (12 - (xs * sym.exp(ys))) / (45 + ((xs * ys) * sym.exp(-xs)))

    d = zs.diff(ys)
    assert float_eq(zs.diff(xs).evalf(subs={xs: 3, ys: 5}), za.value)
    assert float_eq(zs.diff(ys).evalf(subs={xs: 3, ys: 5}), zb.value)
