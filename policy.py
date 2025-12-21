import math
def main(obs: list[float]) -> list[float]:
    x, y, z = obs[:3]
    s = sum(obs) + 1e-3
    m = math.tanh(s / 3.0)
    w = math.tanh(x * y + y * z + z * x)
    a = math.tanh(x * y * z)
    b = math.tanh((x + y + z) / (abs(x * y + y * z + z * x) + 1e-3))
    p = math.tanh(math.cos(m * w))
    q = math.tanh(math.sin(x) + math.cos(y))
    u = math.tanh(10 * (x - 0.5)) * math.tanh(5 * (y - 0.0))
    v = math.tanh(10 * (x + 0.5)) * math.tanh(-5 * (y - 0.1))
    return [math.tanh(w * m * a + u), math.tanh(m * b + v), p, q]
