from __future__ import annotations

import argparse
import ast
import json
import math
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_code_env import clean_generated_code, validate_sandboxed_code


_AGENT_RUNNER_PY = r"""
import importlib.util
import json
import sys


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("user_code", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clip(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x != x:  # NaN
        return 0.0
    return max(-1.0, min(1.0, x))


def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    user_path = payload["user_path"]
    env_id = payload.get("env_id", "Walker2d-v5")
    seeds = payload.get("seeds", [0])
    max_steps = int(payload.get("max_steps", 1000))

    try:
        import gymnasium as gym
    except Exception as e:
        print(json.dumps({"ok": False, "error": "missing_gymnasium", "message": str(e)}))
        return 2

    def _make_env():
        # Override the default TimeLimit.
        try:
            return gym.make(env_id, max_episode_steps=max_steps)
        except TypeError:
            env = gym.make(env_id)
            try:
                from gymnasium.wrappers import TimeLimit

                return TimeLimit(env, max_episode_steps=max_steps)
            except Exception:
                return env

    try:
        mod = _load_module(user_path)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "load_failed", "message": str(e)}))
        return 3

    if not hasattr(mod, "main"):
        print(json.dumps({"ok": False, "error": "missing_main"}))
        return 4

    episodes = []
    step_errors = 0

    for s in seeds:
        env = _make_env()
        obs, _info = env.reset(seed=int(s))
        total = 0.0
        terminated = False
        truncated = False
        steps = 0
        start_x = None
        end_x = None

        try:
            # MuJoCo: x position is usually qpos[0]
            start_x = float(env.unwrapped.data.qpos[0])
        except Exception:
            start_x = None

        for _t in range(max_steps):
            try:
                action = mod.main(list(obs))
                if not isinstance(action, (list, tuple)) or len(action) != 6:
                    raise ValueError("action must be a list/tuple of length 6")
                a = [_clip(action[i]) for i in range(6)]
            except Exception:
                step_errors += 1
                a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            obs, reward, terminated, truncated, _info = env.step(a)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break

        try:
            end_x = float(env.unwrapped.data.qpos[0])
        except Exception:
            end_x = None

        env.close()
        hit_max_steps = bool(steps >= max_steps and not terminated)
        distance = None
        avg_speed = None
        if start_x is not None and end_x is not None:
            distance = float(end_x - start_x)
            if steps > 0:
                avg_speed = float(distance / steps)
        episodes.append(
            {
                "seed": int(s),
                "return": total,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "hit_max_steps": hit_max_steps,
                "distance": distance,
                "avg_speed": avg_speed,
            }
        )

    returns = [e["return"] for e in episodes]
    avg_return = sum(returns) / max(1, len(returns))
    distances = [e["distance"] for e in episodes if e.get("distance") is not None]
    avg_distance = (sum(distances) / max(1, len(distances))) if distances else None
    speeds = [e["avg_speed"] for e in episodes if e.get("avg_speed") is not None]
    avg_speed = (sum(speeds) / max(1, len(speeds))) if speeds else None
    print(
        json.dumps(
            {
                "ok": True,
                "avg_return": avg_return,
                "avg_distance": avg_distance,
                "avg_speed": avg_speed,
                "episodes": episodes,
                "step_errors": step_errors,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _python_for_repo() -> str:
    venv = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else "python3"


def evaluate_policy(
    code: str,
    *,
    env_id: str,
    seeds: list[int],
    max_steps: int,
    timeout_s: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="walker2d_weightopt_") as td:
        td_path = Path(td)
        user_path = td_path / "policy.py"
        runner_path = td_path / "runner.py"
        user_path.write_text(clean_generated_code(code), encoding="utf-8")
        runner_path.write_text(_AGENT_RUNNER_PY, encoding="utf-8")

        proc = subprocess.run(
            [_python_for_repo(), str(runner_path)],
            input=json.dumps({"user_path": str(user_path), "env_id": env_id, "seeds": seeds, "max_steps": max_steps}),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if not stdout:
            return {"ok": False, "error": "empty_stdout", "stderr": stderr, "returncode": proc.returncode}
        try:
            out = json.loads(stdout)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "error": "non_json_stdout",
                "stdout": stdout,
                "stderr": stderr,
                "returncode": proc.returncode,
            }
        out["returncode"] = proc.returncode
        if stderr:
            out["stderr"] = stderr
        return out


def _validate_main_exists(code: str) -> None:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            if len(node.args.args) != 1:
                raise ValueError("`main` must take exactly 1 positional arg: `obs`.")
            return
    raise ValueError("Code must define a `main(obs)` function.")


class _FloatExtractor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.values: list[float] = []

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
            if isinstance(node.operand.value, float):
                v = float(node.operand.value)
                if isinstance(node.op, ast.USub):
                    v = -v
                self.values.append(v)
                return
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, float):
            self.values.append(float(node.value))


class _FloatApplier(ast.NodeTransformer):
    def __init__(self, values: list[float]) -> None:
        self._it = iter(values)

    def _next(self) -> float:
        try:
            v = next(self._it)
        except StopIteration:
            raise ValueError("Internal error: weight vector length mismatch.") from None
        if not math.isfinite(v):
            v = 0.0
        return float(v)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        if isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
            if isinstance(node.operand.value, float):
                return ast.copy_location(ast.Constant(value=self._next()), node)
        return self.generic_visit(node)  # type: ignore[return-value]

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, float):
            return ast.copy_location(ast.Constant(value=self._next()), node)
        return node


def _extract_floats(code: str) -> list[float]:
    tree = ast.parse(code)
    ex = _FloatExtractor()
    ex.visit(tree)
    return ex.values


def _apply_floats(code: str, floats: list[float]) -> str:
    tree = ast.parse(code)
    new_tree = _FloatApplier(floats).visit(tree)
    ast.fix_missing_locations(new_tree)
    out = ast.unparse(new_tree).strip() + "\n"
    return clean_generated_code(out)


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return _jsonable(dump())
        except Exception:
            pass
    asdict = getattr(obj, "__dict__", None)
    if isinstance(asdict, dict):
        try:
            return _jsonable(asdict)
        except Exception:
            pass
    return repr(obj)


def _score_from_result(result: dict[str, Any]) -> tuple[float, float]:
    dist = result.get("avg_distance")
    spd = result.get("avg_speed")
    try:
        score_a = float(dist) if dist is not None else float("-inf")
    except Exception:
        score_a = float("-inf")
    try:
        score_b = float(spd) if spd is not None else float("-inf")
    except Exception:
        score_b = float("-inf")
    return score_a, score_b


def _perturb(
    base: list[float],
    *,
    rng: random.Random,
    rel_sigma: float,
    abs_sigma: float,
    clip: float,
) -> list[float]:
    out: list[float] = []
    rel = float(rel_sigma)
    abs_s = float(abs_sigma)
    c = float(clip)
    if not math.isfinite(rel) or rel < 0:
        rel = 0.0
    if not math.isfinite(abs_s) or abs_s < 0:
        abs_s = 0.0
    if not math.isfinite(c) or c <= 0:
        c = 10.0
    for v in base:
        try:
            v = float(v)
        except Exception:
            v = 0.0
        if not math.isfinite(v):
            v = 0.0
        dv = 0.0
        if abs_s > 0:
            dv += rng.gauss(0.0, abs_s)
        if rel > 0:
            dv += v * rng.gauss(0.0, rel)
        nv = v + dv
        if math.isfinite(nv):
            nv = max(-c, min(c, nv))
        else:
            nv = 0.0
        out.append(nv)
    return out


@dataclass
class Trial:
    floats: list[float]
    score_a: float
    score_b: float
    result: dict[str, Any]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "policy_path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a policy file (e.g. runs/.../iter_079.py).",
    )
    p.add_argument(
        "--policy-path",
        "--policy_path",
        dest="policy_path_flag",
        type=Path,
        default=None,
        help="Same as positional `policy_path` (flag form for convenience).",
    )
    p.add_argument("--out-path", type=Path, default=None, help="Where to write best policy code.")
    p.add_argument("--log-path", type=Path, default=None, help="Where to write JSONL trial logs.")
    p.add_argument("--env-id", default="Walker2d-v5")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--population", type=int, default=50)
    p.add_argument("--rel-sigma", type=float, default=0.10, help="Relative Gaussian stddev for float perturbations.")
    p.add_argument("--abs-sigma", type=float, default=0.02, help="Absolute Gaussian stddev for float perturbations.")
    p.add_argument("--clip", type=float, default=10.0, help="Clip each float parameter to [-clip, +clip].")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the optimizer.")
    p.add_argument("--concurrency", type=int, default=1, help="Parallel eval workers.")
    args = p.parse_args()

    if args.policy_path is not None and args.policy_path_flag is not None:
        raise SystemExit("Provide either positional `policy_path` or `--policy-path`, not both.")
    policy_path_raw = args.policy_path_flag or args.policy_path
    if policy_path_raw is None:
        raise SystemExit("Missing required argument: policy_path (positional) or `--policy-path`.")

    policy_path = policy_path_raw.expanduser().resolve()
    if not policy_path.exists() or not policy_path.is_file():
        raise SystemExit(f"Missing policy file: {policy_path}")

    code0 = clean_generated_code(policy_path.read_text(encoding="utf-8"))
    validate_sandboxed_code(
        code0,
        allowed_import_roots={"math", "random", "itertools", "functools", "statistics"},
    )
    _validate_main_exists(code0)

    floats0 = _extract_floats(code0)
    if not floats0:
        raise SystemExit("No float literals found to optimize.")

    out_path = (args.out_path or policy_path.with_name(policy_path.stem + "_optimized.py")).resolve()
    log_path = (args.log_path or policy_path.with_name(policy_path.stem + "_weight_opt.jsonl")).resolve()

    rng = random.Random(int(args.seed))
    best_floats = floats0
    best_trial: Trial | None = None

    def _eval_one(candidate_floats: list[float]) -> Trial:
        code = _apply_floats(code0, candidate_floats)
        validate_sandboxed_code(
            code,
            allowed_import_roots={"math", "random", "itertools", "functools", "statistics"},
        )
        _validate_main_exists(code)
        result = evaluate_policy(
            code,
            env_id=str(args.env_id),
            seeds=list(args.seeds),
            max_steps=int(args.max_steps),
            timeout_s=float(args.timeout),
        )
        if not result.get("ok"):
            return Trial(floats=candidate_floats, score_a=float("-inf"), score_b=float("-inf"), result=result)
        score_a, score_b = _score_from_result(result)
        return Trial(floats=candidate_floats, score_a=score_a, score_b=score_b, result=result)

    for it in range(1, int(args.iterations) + 1):
        pop = int(args.population)
        if pop < 1:
            raise SystemExit("`--population` must be >= 1")

        candidates: list[list[float]] = [best_floats]
        for _ in range(pop - 1):
            candidates.append(
                _perturb(
                    best_floats,
                    rng=rng,
                    rel_sigma=float(args.rel_sigma),
                    abs_sigma=float(args.abs_sigma),
                    clip=float(args.clip),
                )
            )

        with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as ex:
            trials = list(ex.map(_eval_one, candidates))

        it_best = max(trials, key=lambda t: (t.score_a, t.score_b))
        improved = best_trial is None or (it_best.score_a, it_best.score_b) > (best_trial.score_a, best_trial.score_b)
        if improved:
            best_trial = it_best
            best_floats = it_best.floats

        rec = {
            "iteration": it,
            "population": pop,
            "improved": bool(improved),
            "best_score_a": best_trial.score_a if best_trial else float("-inf"),
            "best_score_b": best_trial.score_b if best_trial else float("-inf"),
            "iter_best_score_a": it_best.score_a,
            "iter_best_score_b": it_best.score_b,
            "rel_sigma": float(args.rel_sigma),
            "abs_sigma": float(args.abs_sigma),
            "clip": float(args.clip),
            "best_result": _jsonable(best_trial.result if best_trial else None),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec, indent=2))

    if best_trial is None:
        raise SystemExit("No successful evaluations.")

    out_code = _apply_floats(code0, best_trial.floats)
    out_path.write_text(out_code, encoding="utf-8")
    print(json.dumps({"ok": True, "out_path": str(out_path), "log_path": str(log_path)}, indent=2))


if __name__ == "__main__":
    main()
