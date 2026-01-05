def main(obs: list[float]) -> list[float]:
    import math
    if len(obs) < 4:
        obs = obs + [0.0] * (4 - len(obs))
    a, b, c, d = obs[:4]
    f1 = math.tanh(10*a + 8*c + 8.5*b*d + 7.9*a*b + 7.8*c*d)
    f2 = math.tanh(8.5*b - 7.9*c + 8.8*d + 8.3*a*c + 7.9*c*d + 7.6*a*d)
    f3 = math.tanh(7.9*a*c + 7.9*b*d + 7.8*a - 7.2*b + 8.4*c + 7.8*a*b + 8.1*c*d)
    f4 = math.tanh(8.1*c*d - 7.2*a*b - 8.9*d + 7.5*b - 8*a*c)
    out0 = math.tanh(f1 + 7.4*f2 + 8.5*f3 + 3.2*f4 + 8.5*a*d - 2.5*b*c)
    out1 = math.tanh(8*f1 - 8.2*f2 + 8.8*f3 - 3.1*f4 + 2.5*d)
    out2 = math.tanh(8.3*f1 * f3 - 8*f2 * f4 + 2.5*c - 2.5*a*d)
    out3 = math.tanh(f1 - f2 + f3 - f4)
    return [out0, out1, out2, out3]
