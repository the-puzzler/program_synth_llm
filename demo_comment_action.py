from __future__ import annotations

from bipedal_fitness_loop import _code_signature_and_numbers, _miniscule_param_change_warning


def main() -> None:
    baseline = """
import math

def main(obs: list[float]) -> list[float]:
    a = obs[0]
    b = obs[1]
    return [math.tanh(0.10*a + 0.20*b), 0.0, 0.0, 0.0]
""".strip()

    identical = """
import math

def main(obs: list[float]) -> list[float]:
    a = obs[0]
    b = obs[1]
    return [math.tanh(0.10*a +0.20*b), 0.0, 0.0, 0.0]
""".strip()

    tiny_tweak = """
import math

def main(obs: list[float]) -> list[float]:
    a = obs[0]
    b = obs[1]
    return [math.tanh(0.101*a + 0.199*b), 0.0, 0.0, 0.0]
""".strip()

    bigger_change = """
import math

def main(obs: list[float]) -> list[float]:
    a = obs[0]
    b = obs[1]
    c = obs[2] if len(obs) > 2 else 0.0
    return [math.tanh(0.8*a - 0.3*b + 0.2*c), math.sin(a), 0.0, -math.cos(b)]
""".strip()

    seen: list[tuple[str, list[float], int, int]] = []
    sig0, nums0 = _code_signature_and_numbers(baseline)
    seen.append((sig0, nums0, 1, 0))

    for label, code in [
        ("identical", identical),
        ("tiny_tweak", tiny_tweak),
        ("bigger_change", bigger_change),
    ]:
        sig, nums = _code_signature_and_numbers(code)
        warning, meta = _miniscule_param_change_warning(sig, nums, seen=seen)
        print(f"\n== {label} ==")
        print(f"warning: {warning}")
        print(f"meta: {meta}")


if __name__ == "__main__":
    main()
