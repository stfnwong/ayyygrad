

def float_eq(a: float, b: float, eps:float=1e-6) -> bool:
    return True if abs(a - b) < eps else False

