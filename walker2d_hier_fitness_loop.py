from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ai_code_env import clean_generated_code, extract_python_code, validate_sandboxed_code
from call_ai_utils import call_ai


@dataclass
class Attempt:
    iteration: int
    score_a: float
    score_b: float
    code: str
    comment: str | None = None


ReducerName = Literal["reduce_1", "reduce_2", "reduce_3"]

REDUCER_SPECS: dict[ReducerName, dict[str, Any]] = {
    "reduce_1": {
        "input_dim": 8,
        # qpos[1:] in Walker2d is usually length 8; in obs it is indices 0..7
        "obs_indices": list(range(0, 8)),
        "label": "positions (qpos[1:])",
    },
    "reduce_2": {
        "input_dim": 3,
        "obs_indices": [8, 9, 10],
        "label": "torso velocities (first 3 of qvel)",
    },
    "reduce_3": {
        "input_dim": 6,
        "obs_indices": list(range(11, 17)),
        "label": "leg velocities (remaining 6 of qvel)",
    },
}

FEATURES_DIM = 6  # 3 reducers * 2 floats each
ACTION_DIM = 6


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"walker2d_hier_{ts}"


def _load_run_meta(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "meta.json"
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _load_attempt_history(run_dir: Path, *, key: str) -> list[Attempt]:
    attempts_path = run_dir / "attempts.jsonl"
    if not attempts_path.exists():
        raise FileNotFoundError(f"Missing attempts file: {attempts_path}")

    history: list[Attempt] = []
    key_to_component = {
        "code_reduce_1": "reduce_1",
        "code_reduce_2": "reduce_2",
        "code_reduce_3": "reduce_3",
        "code_main": "main",
    }
    expected_component = key_to_component.get(key)
    for line in attempts_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        # In freeze-schedule mode, only treat iterations where this component was active
        # as part of its evolutionary history. This keeps prompts from being flooded
        # with repeated "frozen" copies.
        if expected_component is not None:
            active = rec.get("active_component")
            if isinstance(active, str) and active not in ("all", expected_component):
                # When reducers are evolved together, accept those steps too.
                if not (active == "reducers" and expected_component.startswith("reduce_")):
                    continue
        it = rec.get("iteration")
        if not isinstance(it, int):
            continue
        code = rec.get(key)
        if not isinstance(code, str) or not code.strip():
            # Fallback: try to read the composed program (if present) and extract the relevant function.
            code_path = rec.get("code_path")
            if isinstance(code_path, str) and code_path.endswith(".py"):
                code_file = run_dir / code_path
            else:
                code_file = run_dir / f"iter_{it:03d}.py"
            try:
                full = code_file.read_text(encoding="utf-8")
            except Exception:
                continue
            code = full

        score_a = rec.get("score_a")
        score_b = rec.get("score_b")
        try:
            sa = float(score_a) if score_a is not None else float("-inf")
        except Exception:
            sa = float("-inf")
        try:
            sb = float(score_b) if score_b is not None else float("-inf")
        except Exception:
            sb = float("-inf")

        comment = rec.get("comment")
        comment = str(comment) if isinstance(comment, str) else None
        history.append(Attempt(iteration=int(it), score_a=sa, score_b=sb, code=str(code), comment=comment))

    history.sort(key=lambda a: a.iteration)
    return history


def _load_seen_programs(run_dir: Path) -> list[tuple[str, list[float], int, int]]:
    path = run_dir / "candidates.jsonl"
    if not path.exists():
        return []
    out: list[tuple[str, list[float], int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        code = rec.get("code")
        it = rec.get("iteration")
        idx = rec.get("candidate_index")
        if not isinstance(code, str) or not isinstance(it, int) or not isinstance(idx, int):
            continue
        try:
            sig, nums = _code_signature_and_numbers(code)
            out.append((sig, nums, int(it), int(idx)))
        except Exception:
            continue
    return out


def _load_last_iteration(run_dir: Path) -> int | None:
    path = run_dir / "attempts.jsonl"
    if not path.exists():
        return None
    try:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return None
    for ln in reversed(lines):
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("iteration"), int):
            return int(rec["iteration"])
    return None


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


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


_SCORE_CACHE: dict[str, tuple[float, float]] = {}


def _score_component_code(
    *,
    component: str,
    component_code: str,
    elites: dict[str, str],
    env_id: str,
    seeds: list[int],
    max_steps: int,
    timeout_s: float,
) -> tuple[float, float]:
    r1 = elites["reduce_1"]
    r2 = elites["reduce_2"]
    r3 = elites["reduce_3"]
    m = elites["main"]
    if component == "reduce_1":
        r1 = component_code
    elif component == "reduce_2":
        r2 = component_code
    elif component == "reduce_3":
        r3 = component_code
    elif component == "main":
        m = component_code
    else:
        raise ValueError(f"Unknown component: {component}")

    program = _compose_program(reduce_1=r1, reduce_2=r2, reduce_3=r3, main=m)
    key = "|".join(
        [
            "v1",
            component,
            env_id,
            str(max_steps),
            ",".join(str(int(s)) for s in seeds),
            _sha1(program),
        ]
    )
    cached = _SCORE_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        result = evaluate_policy(program, env_id=env_id, seeds=seeds, max_steps=max_steps, timeout_s=timeout_s)
        if not result.get("ok"):
            scores = (float("-inf"), float("-inf"))
        else:
            avg_distance = result.get("avg_distance")
            avg_speed = result.get("avg_speed")
            score_a = float(avg_distance) if avg_distance is not None else float("-inf")
            score_b = float(avg_speed) if avg_speed is not None else float("-inf")
            scores = (score_a, score_b)
    except Exception:
        scores = (float("-inf"), float("-inf"))
    _SCORE_CACHE[key] = scores
    return scores


def _format_history_best_last_random_rescored(
    history: list[Attempt],
    *,
    seed: int,
    component: str,
    elites: dict[str, str],
    env_id: str,
    seeds: list[int],
    max_steps: int,
    timeout_s: float,
) -> str:
    if not history:
        return "(none)\n"
    rng = random.Random(int(seed))
    best = max(history, key=lambda a: (a.score_a, a.score_b))
    last_n = 3
    rand_n = 3
    last = history[-last_n:]
    random_picks = rng.sample(history, k=min(rand_n, len(history)))

    attempts: list[tuple[str, Attempt]] = [("BEST", best)]
    for i, a in enumerate(last, start=1):
        attempts.append((f"LAST_{i}", a))
    for i, a in enumerate(random_picks, start=1):
        attempts.append((f"RAND_{i}", a))

    out: list[str] = []
    seen: set[tuple[int, float, float]] = set()
    seen_sigs: set[str] = set()
    kept = 0
    for label, a in attempts:
        key = (a.iteration, a.score_a, a.score_b)
        if key in seen:
            continue
        seen.add(key)
        try:
            sig, _nums = _code_signature_and_numbers(a.code)
        except Exception:
            sig = ""
        if sig:
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
        else:
            text_key = f"code:{hash(a.code)}"
            if text_key in seen_sigs:
                continue
            seen_sigs.add(text_key)

        score_a, score_b = _score_component_code(
            component=component,
            component_code=a.code,
            elites=elites,
            env_id=env_id,
            seeds=seeds,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )

        snippet = "\n".join(a.code.strip().splitlines()[:12])
        header = f"{label}: score_a={score_a:.6f} score_b={score_b:.6f}"
        if a.comment:
            out.append(f"{header}\n{a.comment}\n{snippet}\n")
        else:
            out.append(f"{header}\n{snippet}\n")
        kept += 1
        if kept >= 1 + last_n + rand_n:
            break
    return "\n".join(out)


def _validate_hier_functions_exist(code: str) -> None:
    tree = ast.parse(code)
    found: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            found[node.name] = len(node.args.args)
    for name in ("reduce_1", "reduce_2", "reduce_3", "main"):
        if name not in found:
            raise ValueError(f"Generated code must define `{name}(...)`.")
        if found[name] != 1:
            raise ValueError(f"`{name}` must take exactly 1 positional argument.")


class _NumericNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (int, float)):
            return ast.copy_location(ast.Constant(value=0), node)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
                return ast.copy_location(ast.Constant(value=0), node)
        return node


class _NumericExtractor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.values: list[float] = []

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
            if isinstance(node.operand.value, (int, float)):
                v = float(node.operand.value)
                if isinstance(node.op, ast.USub):
                    v = -v
                self.values.append(v)
                return
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, (int, float)):
            self.values.append(float(node.value))


def _code_signature_and_numbers(code: str) -> tuple[str, list[float]]:
    tree = ast.parse(code)
    extractor = _NumericExtractor()
    extractor.visit(tree)
    normalized = _NumericNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    sig = ast.dump(normalized, include_attributes=False)
    return sig, extractor.values


def _miniscule_param_change_warning(
    sig: str,
    nums: list[float],
    *,
    seen: list[tuple[str, list[float], int, int]],
) -> tuple[str | None, dict[str, Any] | None]:
    if not nums:
        return None, None

    max_rel_thresh = 0.05
    mean_rel_thresh = 0.02
    max_abs_thresh = 0.2
    mean_abs_thresh = 0.05

    best: tuple[float, float, int, int, float, float] | None = None
    for prev_sig, prev_nums, prev_step, prev_idx in seen:
        if prev_sig != sig:
            continue
        if len(prev_nums) != len(nums) or not prev_nums:
            continue
        rels: list[float] = []
        abss: list[float] = []
        for a, b in zip(prev_nums, nums):
            d = abs(a - b)
            abss.append(d)
            denom = max(1e-6, abs(a), abs(b))
            rels.append(d / denom)
        max_rel = max(rels)
        mean_rel = sum(rels) / len(rels)
        max_abs = max(abss)
        mean_abs = sum(abss) / len(abss)

        if max_abs <= 1e-12:
            return (
                "WARNING: candidate repeats a previously seen program (or only tiny numeric changes).",
                {"type": "identical", "similar_step": prev_step, "similar_candidate_index": prev_idx},
            )

        if (
            max_rel <= max_rel_thresh
            and mean_rel <= mean_rel_thresh
            and max_abs <= max_abs_thresh
            and mean_abs <= mean_abs_thresh
        ):
            if best is None or (max_rel, mean_rel, max_abs, mean_abs) < (
                best[0],
                best[1],
                best[4],
                best[5],
            ):
                best = (max_rel, mean_rel, prev_step, prev_idx, max_abs, mean_abs)

    if best is None:
        return None, None

    _max_rel, _mean_rel, prev_step, prev_idx, _max_abs, _mean_abs = best
    warning = "WARNING: only miniscule parameter change detected; make a bigger structural change."
    meta = {
        "type": "minuscule_params",
        "similar_step": prev_step,
        "similar_candidate_index": prev_idx,
        "max_rel": _max_rel,
        "mean_rel": _mean_rel,
        "max_abs": _max_abs,
        "mean_abs": _mean_abs,
    }
    return warning, meta


def _format_history_best_random_last(history: list[Attempt], *, seed: int) -> str:
    if not history:
        return "(none)\n"
    rng = random.Random(int(seed))
    best = max(history, key=lambda a: (a.score_a, a.score_b))
    last_n = 3
    rand_n = 3
    last = history[-last_n:]
    random_picks = rng.sample(history, k=min(rand_n, len(history)))

    attempts: list[tuple[str, Attempt]] = [("BEST", best)]
    for i, a in enumerate(last, start=1):
        attempts.append((f"LAST_{i}", a))
    for i, a in enumerate(random_picks, start=1):
        attempts.append((f"RAND_{i}", a))

    out: list[str] = []
    seen: set[tuple[int, float, float]] = set()
    seen_sigs: set[str] = set()
    kept = 0
    for label, a in attempts:
        key = (a.iteration, a.score_a, a.score_b)
        if key in seen:
            continue
        seen.add(key)
        try:
            sig, _nums = _code_signature_and_numbers(a.code)
        except Exception:
            sig = ""
        if sig:
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
        else:
            text_key = f"code:{hash(a.code)}"
            if text_key in seen_sigs:
                continue
            seen_sigs.add(text_key)
        snippet = "\n".join(a.code.strip().splitlines()[:12])
        header = f"{label}: score_a={a.score_a:.6f} score_b={a.score_b:.6f}"
        if a.comment:
            out.append(f"{header}\n{a.comment}\n{snippet}\n")
        else:
            out.append(f"{header}\n{snippet}\n")
        kept += 1
        if kept >= 1 + last_n + rand_n:
            break
    return "\n".join(out)


def build_reducer_prompt(
    *,
    name: ReducerName,
    history: list[Attempt],
    seed: int,
    elites: dict[str, str],
    env_id: str,
    seeds: list[int],
    max_steps: int,
    timeout_s: float,
) -> str:
    spec = REDUCER_SPECS[name]
    input_dim = int(spec["input_dim"])
    return (
        "Output exactly ONE fenced Python code block and nothing else. Do not write comments.\n"
        "Define exactly this function signature:\n"
        f"def {name}(x: list[float]) -> list[float]:\n"
        "\n"
        "Contract:\n"
        f"- `x` is a list of {input_dim} floats.\n"
        "- Return a list of 2 floats.\n"
        "- The evaluator will clip floats to [-1, 1].\n"
        "- Allowed imports: `math`, `random`, `itertools`, `functools`, `statistics`.\n"
        "- Do not import anything else.\n"
        "- Do not read/write files, do not use network, do not print.\n"
        "\n"
        "Goal: maximize two scalar scores: `score_a` and `score_b`.\n"
        "\n"
        "Previous attempts to build from (scores + previous code):\n"
        f"{_format_history_best_last_random_rescored(history, seed=seed, component=name, elites=elites, env_id=env_id, seeds=seeds, max_steps=max_steps, timeout_s=timeout_s)}\n"
    )


def build_main_prompt(
    *,
    history: list[Attempt],
    seed: int,
    elites: dict[str, str],
    env_id: str,
    seeds: list[int],
    max_steps: int,
    timeout_s: float,
) -> str:
    return (
        "Output exactly ONE fenced Python code block and nothing else. Do not write comments.\n"
        "Define exactly this function signature:\n"
        "def main(features: list[float]) -> list[float]:\n"
        "\n"
        "Contract:\n"
        f"- `features` is a list of {FEATURES_DIM} floats.\n"
        f"- Return a list of {ACTION_DIM} floats.\n"
        "- The evaluator will clip floats to [-1, 1].\n"
        "- Allowed imports: `math`, `random`, `itertools`, `functools`, `statistics`.\n"
        "- Do not import anything else.\n"
        "- Do not read/write files, do not use network, do not print.\n"
        "\n"
        "Goal: maximize two scalar scores: `score_a` and `score_b`.\n"
        "Higher is better for both.\n"
        "\n"
        "Previous attempts to build from (scores + previous code):\n"
        f"{_format_history_best_last_random_rescored(history, seed=seed, component='main', elites=elites, env_id=env_id, seeds=seeds, max_steps=max_steps, timeout_s=timeout_s)}\n"
    )


def _compose_program(*, reduce_1: str, reduce_2: str, reduce_3: str, main: str) -> str:
    parts = [
        clean_generated_code(reduce_1),
        clean_generated_code(reduce_2),
        clean_generated_code(reduce_3),
        clean_generated_code(main),
    ]
    return "\n\n".join(p.strip() for p in parts if p.strip()) + "\n"


_AGENT_RUNNER_PY = r"""
import importlib.util
import json
import sys


IDX1 = list(range(0, 8))
IDX2 = [8, 9, 10]
IDX3 = list(range(11, 17))


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


def _as_list2(v):
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError("reducer must return a list/tuple of length 2")
    return [_clip(v[0]), _clip(v[1])]


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

    for name in ("reduce_1", "reduce_2", "reduce_3", "main"):
        if not hasattr(mod, name):
            print(json.dumps({"ok": False, "error": "missing_fn", "fn": name}))
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
            start_x = float(env.unwrapped.data.qpos[0])
        except Exception:
            start_x = None

        for _t in range(max_steps):
            try:
                o = list(obs)
                x1 = [o[i] for i in IDX1]
                x2 = [o[i] for i in IDX2]
                x3 = [o[i] for i in IDX3]
                f1 = _as_list2(mod.reduce_1(x1))
                f2 = _as_list2(mod.reduce_2(x2))
                f3 = _as_list2(mod.reduce_3(x3))
                features = f1 + f2 + f3
                action = mod.main(features)
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


def evaluate_policy(
    code: str,
    *,
    env_id: str = "Walker2d-v5",
    seeds: list[int],
    max_steps: int = 1000,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="walker2d_hier_eval_") as td:
        td_path = Path(td)
        user_path = td_path / "policy.py"
        runner_path = td_path / "runner.py"
        user_path.write_text(clean_generated_code(code), encoding="utf-8")
        runner_path.write_text(_AGENT_RUNNER_PY, encoding="utf-8")

        py = str(Path(__file__).resolve().parent / ".venv" / "bin" / "python")
        if not Path(py).exists():
            py = "python3"

        try:
            proc = subprocess.run(
                [py, str(runner_path)],
                input=json.dumps({"user_path": str(user_path), "env_id": env_id, "seeds": seeds, "max_steps": max_steps}),
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if not stdout:
            return {"ok": False, "error": "empty_output", "stderr": stderr}
        try:
            obj = json.loads(stdout.splitlines()[-1])
        except Exception:
            return {"ok": False, "error": "bad_json", "stdout": stdout, "stderr": stderr}
        if not isinstance(obj, dict):
            return {"ok": False, "error": "bad_payload", "payload": obj, "stderr": stderr}
        if stderr and not obj.get("stderr"):
            obj["stderr"] = stderr
        return obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None, help="Resume an existing run dir (e.g. runs/walker2d_hier_20251221T211922Z).")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--concurrent", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--env-id", default="Walker2d-v5")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    default_mode = "coevolve"
    p.add_argument("--mode", choices=("coevolve", "freeze"), default=default_mode)
    p.add_argument("--freeze-k", type=int, default=5, help="Iterations per frozen block in freeze mode.")
    p.add_argument(
        "--freeze-scheme",
        choices=("single", "reducers_together"),
        default="single",
        help="Freeze schedule scheme: evolve one component at a time, or evolve all reducers together.",
    )
    p.add_argument(
        "--baseline-prob",
        type=float,
        default=0.10,
        help="In freeze mode, include the elite (no-change) baseline candidate with this probability.",
    )
    args = p.parse_args()

    run_dir = Path(args.resume) if args.resume is not None else (Path(args.run_dir) if args.run_dir is not None else _default_run_dir())
    run_dir.mkdir(parents=True, exist_ok=True)

    run_meta = _load_run_meta(run_dir) or {}

    def _arg_or_meta(key: str, default: Any) -> Any:
        if key in run_meta:
            return run_meta.get(key)
        return default

    env_id_val = _arg_or_meta("env_id", str(args.env_id))
    env_id = str(env_id_val) if isinstance(env_id_val, str) else str(args.env_id)

    iterations_val = _arg_or_meta("iterations", int(args.iterations))
    iterations = int(iterations_val) if isinstance(iterations_val, int) else int(args.iterations)

    # Resume should keep the run's configured mode unless explicitly overridden.
    if args.resume is not None:
        meta_mode = run_meta.get("mode")
        if isinstance(meta_mode, str) and args.mode == default_mode:
            args.mode = meta_mode
        meta_fk = run_meta.get("freeze_k")
        if isinstance(meta_fk, int) and int(args.freeze_k) == 5:
            args.freeze_k = int(meta_fk)
        meta_fs = run_meta.get("freeze_scheme")
        if isinstance(meta_fs, str) and args.freeze_scheme == "single":
            args.freeze_scheme = meta_fs
        meta_bp = run_meta.get("baseline_prob")
        if isinstance(meta_bp, (int, float)) and float(args.baseline_prob) == 0.10:
            args.baseline_prob = float(meta_bp)

    seeds_val = _arg_or_meta("seeds", list(args.seeds))
    if isinstance(seeds_val, list) and all(isinstance(x, int) for x in seeds_val):
        seeds = list(seeds_val)
    else:
        seeds = list(args.seeds)

    max_steps_val = _arg_or_meta("max_steps", int(args.max_steps))
    max_steps = int(max_steps_val) if isinstance(max_steps_val, int) else int(args.max_steps)

    timeout_val = float(args.timeout) if args.timeout is not None else float(run_meta.get("timeout_s", 120.0))
    timeout_s = float(run_meta.get("timeout_s", timeout_val)) if args.timeout is None else timeout_val

    attempts_path = run_dir / "attempts.jsonl"
    candidates_path = run_dir / "candidates.jsonl"
    meta_path = run_dir / "meta.json"

    if args.resume is not None and attempts_path.exists():
        history_r1 = _load_attempt_history(run_dir, key="code_reduce_1")
        history_r2 = _load_attempt_history(run_dir, key="code_reduce_2")
        history_r3 = _load_attempt_history(run_dir, key="code_reduce_3")
        history_main = _load_attempt_history(run_dir, key="code_main")
        last_it = _load_last_iteration(run_dir)
        start_step = (last_it + 1) if isinstance(last_it, int) else 0
        seen_programs = _load_seen_programs(run_dir)
    else:
        history_r1, history_r2, history_r3, history_main = [], [], [], []
        start_step = 0
        seen_programs = []

    if args.resume is None and not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {
                    "experiment": "walker2d_hier_blackbox",
                    "env_id": env_id,
                    "iterations": iterations,
                    "seeds": seeds,
                    "max_steps": max_steps,
                    "timeout_s": timeout_s,
                    "concurrent": args.concurrent,
                    "temperature": args.temperature,
                    "mode": args.mode,
                    "freeze_k": int(args.freeze_k),
                    "freeze_scheme": args.freeze_scheme,
                    "baseline_prob": float(args.baseline_prob),
                    "candidates_path": candidates_path.name,
                    "spec": {
                        "reduce_1": {"in": 8, "out": 2, "obs_indices": REDUCER_SPECS["reduce_1"]["obs_indices"]},
                        "reduce_2": {"in": 3, "out": 2, "obs_indices": REDUCER_SPECS["reduce_2"]["obs_indices"]},
                        "reduce_3": {"in": 6, "out": 2, "obs_indices": REDUCER_SPECS["reduce_3"]["obs_indices"]},
                        "main": {"in": FEATURES_DIM, "out": ACTION_DIM},
                    },
                    "scores": ["score_a", "score_b"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _elite_codes() -> dict[str, str]:
        return {
            "reduce_1": history_r1[-1].code if history_r1 else "def reduce_1(x: list[float]) -> list[float]:\n    return [0.0, 0.0]\n",
            "reduce_2": history_r2[-1].code if history_r2 else "def reduce_2(x: list[float]) -> list[float]:\n    return [0.0, 0.0]\n",
            "reduce_3": history_r3[-1].code if history_r3 else "def reduce_3(x: list[float]) -> list[float]:\n    return [0.0, 0.0]\n",
            "main": history_main[-1].code if history_main else "def main(features: list[float]) -> list[float]:\n    return [0.0]*6\n",
        }

    def _one_prompt(pr: str):
        return call_ai(pr, concurrent_calls=1, temperature=args.temperature)[0]

    def _active_component(step: int) -> str:
        if args.mode != "freeze":
            return "all"
        k = max(1, int(args.freeze_k))
        if args.freeze_scheme == "reducers_together":
            order = ["main", "reducers"]
        else:
            order = ["main", "reduce_1", "reduce_2", "reduce_3"]
        return order[(step // k) % len(order)]

    for step in range(start_step, start_step + iterations):
        active = _active_component(step)
        elites = _elite_codes()

        by_candidate: list[dict[str, Any]] = [{"prompts": {}, "responses": {}} for _ in range(args.concurrent)]

        if active == "all":
            seed_base = step * 10_000
            prompt_r1 = build_reducer_prompt(
                name="reduce_1",
                history=history_r1,
                seed=seed_base + 1,
                elites=elites,
                env_id=env_id,
                seeds=seeds,
                max_steps=max_steps,
                timeout_s=timeout_s,
            )
            prompt_r2 = build_reducer_prompt(
                name="reduce_2",
                history=history_r2,
                seed=seed_base + 2,
                elites=elites,
                env_id=env_id,
                seeds=seeds,
                max_steps=max_steps,
                timeout_s=timeout_s,
            )
            prompt_r3 = build_reducer_prompt(
                name="reduce_3",
                history=history_r3,
                seed=seed_base + 3,
                elites=elites,
                env_id=env_id,
                seeds=seeds,
                max_steps=max_steps,
                timeout_s=timeout_s,
            )
            prompt_m = build_main_prompt(
                history=history_main,
                seed=seed_base + 4,
                elites=elites,
                env_id=env_id,
                seeds=seeds,
                max_steps=max_steps,
                timeout_s=timeout_s,
            )
            jobs: list[tuple[str, int, str]] = []
            for i in range(args.concurrent):
                jobs.append(("reduce_1", i, prompt_r1))
                jobs.append(("reduce_2", i, prompt_r2))
                jobs.append(("reduce_3", i, prompt_r3))
                jobs.append(("main", i, prompt_m))

            max_workers = max(1, min(32, len(jobs)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                responses = [f.result() for f in [executor.submit(_one_prompt, pr) for _kind, _cand, pr in jobs]]

            for (kind, cand_idx, prompt), resp in zip(jobs, responses, strict=False):
                by_candidate[cand_idx]["prompts"][kind] = prompt
                by_candidate[cand_idx]["responses"][kind] = _jsonable(resp.raw)

                content = resp.choices[0].message.content
                code = clean_generated_code(extract_python_code(content))
                by_candidate[cand_idx][f"code_{kind}"] = code
        else:
            # Freeze schedule: only generate candidates for the active component;
            # all other components stay fixed at their current elite implementation.
            jobs = []
            seed_base = step * 10_000
            prompt_r1 = None
            prompt_r2 = None
            prompt_r3 = None
            prompt_r = None
            prompt_m = None
            if active == "reducers":
                prompt_r1 = build_reducer_prompt(
                    name="reduce_1",
                    history=history_r1,
                    seed=seed_base + 1,
                    elites=elites,
                    env_id=env_id,
                    seeds=seeds,
                    max_steps=max_steps,
                    timeout_s=timeout_s,
                )
                prompt_r2 = build_reducer_prompt(
                    name="reduce_2",
                    history=history_r2,
                    seed=seed_base + 2,
                    elites=elites,
                    env_id=env_id,
                    seeds=seeds,
                    max_steps=max_steps,
                    timeout_s=timeout_s,
                )
                prompt_r3 = build_reducer_prompt(
                    name="reduce_3",
                    history=history_r3,
                    seed=seed_base + 3,
                    elites=elites,
                    env_id=env_id,
                    seeds=seeds,
                    max_steps=max_steps,
                    timeout_s=timeout_s,
                )
            elif active.startswith("reduce_"):
                prompt_r = build_reducer_prompt(
                    name=active,
                    history={"reduce_1": history_r1, "reduce_2": history_r2, "reduce_3": history_r3}[active],
                    seed=seed_base + 1,
                    elites=elites,
                    env_id=env_id,
                    seeds=seeds,
                    max_steps=max_steps,
                    timeout_s=timeout_s,
                )
            elif active == "main":
                prompt_m = build_main_prompt(
                    history=history_main,
                    seed=seed_base + 1,
                    elites=elites,
                    env_id=env_id,
                    seeds=seeds,
                    max_steps=max_steps,
                    timeout_s=timeout_s,
                )
            for i in range(args.concurrent):
                if active == "reducers":
                    assert prompt_r1 is not None and prompt_r2 is not None and prompt_r3 is not None
                    jobs.append(("reduce_1", i, prompt_r1))
                    jobs.append(("reduce_2", i, prompt_r2))
                    jobs.append(("reduce_3", i, prompt_r3))
                elif active.startswith("reduce_"):
                    assert prompt_r is not None
                    jobs.append((active, i, prompt_r))
                elif active == "main":
                    assert prompt_m is not None
                    jobs.append(("main", i, prompt_m))
                else:
                    raise RuntimeError(f"Unknown active component: {active}")

            with ThreadPoolExecutor(max_workers=max(1, min(32, len(jobs)))) as executor:
                responses = [f.result() for f in [executor.submit(_one_prompt, pr) for _kind, _cand, pr in jobs]]

            for (kind, cand_idx, prompt), resp in zip(jobs, responses, strict=False):
                by_candidate[cand_idx]["prompts"][kind] = prompt
                by_candidate[cand_idx]["responses"][kind] = _jsonable(resp.raw)
                content = resp.choices[0].message.content
                code = clean_generated_code(extract_python_code(content))
                by_candidate[cand_idx][f"code_{kind}"] = code

        best_code: str | None = None
        best_idx: int | None = None
        best_detail: dict[str, Any] | None = None
        best_score_a = float("-inf")
        best_score_b = float("-inf")

        # Optional "baseline" evaluation: include the current elites as a candidate,
        # so the active component can't regress if all generated samples are worse.
        baseline_bundle: dict[str, Any] | None = None
        if active != "all":
            prob = float(args.baseline_prob)
            if math.isfinite(prob) and prob > 0:
                rng = random.Random(step)
                if rng.random() < min(1.0, max(0.0, prob)):
                    baseline_bundle = {
                        "prompts": {},
                        "responses": {},
                        "code_reduce_1": elites["reduce_1"],
                        "code_reduce_2": elites["reduce_2"],
                        "code_reduce_3": elites["reduce_3"],
                        "code_main": elites["main"],
                    }

        def _candidate_components(bundle: dict[str, Any]) -> tuple[str, str, str, str]:
            if active == "all":
                r1 = str(bundle.get("code_reduce_1") or "")
                r2 = str(bundle.get("code_reduce_2") or "")
                r3 = str(bundle.get("code_reduce_3") or "")
                m = str(bundle.get("code_main") or "")
                return r1, r2, r3, m

            # Freeze: take non-active components from elites.
            r1 = elites["reduce_1"]
            r2 = elites["reduce_2"]
            r3 = elites["reduce_3"]
            m = elites["main"]
            if active == "reduce_1":
                r1 = str(bundle.get("code_reduce_1") or "")
            elif active == "reduce_2":
                r2 = str(bundle.get("code_reduce_2") or "")
            elif active == "reduce_3":
                r3 = str(bundle.get("code_reduce_3") or "")
            elif active == "reducers":
                r1 = str(bundle.get("code_reduce_1") or "")
                r2 = str(bundle.get("code_reduce_2") or "")
                r3 = str(bundle.get("code_reduce_3") or "")
            elif active == "main":
                m = str(bundle.get("code_main") or "")
            else:
                raise RuntimeError(f"Unknown active component: {active}")
            return r1, r2, r3, m

        def _iter_bundles() -> list[tuple[int | None, dict[str, Any]]]:
            out: list[tuple[int | None, dict[str, Any]]] = []
            if baseline_bundle is not None:
                out.append((None, baseline_bundle))
            out.extend((i, b) for i, b in enumerate(by_candidate))
            return out

        for idx, bundle in _iter_bundles():
            reduce_1_code, reduce_2_code, reduce_3_code, main_code = _candidate_components(bundle)
            code = _compose_program(reduce_1=reduce_1_code, reduce_2=reduce_2_code, reduce_3=reduce_3_code, main=main_code)

            error: str | None = None
            step_errors: int | None = None
            episodes: list[dict[str, Any]] | None = None
            avg_return: float | None = None
            avg_distance: float | None = None
            avg_speed: float | None = None
            comment: str | None = None
            similarity: dict[str, Any] | None = None

            score_a = float("-inf")
            score_b = float("-inf")

            try:
                validate_sandboxed_code(
                    code,
                    allowed_import_roots={"math", "random", "itertools", "functools", "statistics"},
                )
                _validate_hier_functions_exist(code)
                sig, nums = _code_signature_and_numbers(code)
                comment, similarity = _miniscule_param_change_warning(sig, nums, seen=seen_programs)
                result = evaluate_policy(
                    code,
                    env_id=env_id,
                    seeds=seeds,
                    max_steps=max_steps,
                    timeout_s=timeout_s,
                )
                if not result.get("ok"):
                    error = str(result.get("error") or "eval_failed")
                else:
                    avg_return = float(result.get("avg_return"))
                    avg_distance = result.get("avg_distance")
                    avg_distance = float(avg_distance) if avg_distance is not None else None
                    avg_speed = result.get("avg_speed")
                    avg_speed = float(avg_speed) if avg_speed is not None else None
                    score_a = avg_distance if avg_distance is not None else float("-inf")
                    score_b = avg_speed if avg_speed is not None else float("-inf")
                    step_errors = int(result.get("step_errors", 0))
                    episodes = list(result.get("episodes", []))
            except Exception as e:
                error = str(e)

            cand_record = {
                "iteration": step,
                "candidate_index": idx if idx is not None else -1,
                "active_component": active,
                "score_a": score_a,
                "score_b": score_b,
                "avg_return": avg_return,
                "avg_distance": avg_distance,
                "avg_speed": avg_speed,
                "episodes": episodes,
                "step_errors": step_errors,
                "error": error,
                "comment": comment,
                "similarity": similarity,
                "prompts": bundle.get("prompts"),
                "responses": bundle.get("responses"),
                "code_reduce_1": reduce_1_code,
                "code_reduce_2": reduce_2_code,
                "code_reduce_3": reduce_3_code,
                "code_main": main_code,
                "code": code,
            }
            with candidates_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(cand_record) + "\n")

            if (score_a, score_b) > (best_score_a, best_score_b):
                best_score_a, best_score_b = score_a, score_b
                best_code = code
                best_idx = idx
                best_detail = {
                    "score_a": score_a,
                    "score_b": score_b,
                    "avg_return": avg_return,
                    "avg_distance": avg_distance,
                    "avg_speed": avg_speed,
                    "episodes": episodes,
                    "step_errors": step_errors,
                    "error": error,
                    "comment": comment,
                    "similarity": similarity,
                    "code_reduce_1": reduce_1_code,
                    "code_reduce_2": reduce_2_code,
                    "code_reduce_3": reduce_3_code,
                    "code_main": main_code,
                }

            try:
                sig, nums = _code_signature_and_numbers(code)
                seen_programs.append((sig, nums, step, idx))
            except Exception:
                pass

        if best_code is None:
            best_code = (
                "def reduce_1(x: list[float]) -> list[float]:\n    return [0.0, 0.0]\n\n"
                "def reduce_2(x: list[float]) -> list[float]:\n    return [0.0, 0.0]\n\n"
                "def reduce_3(x: list[float]) -> list[float]:\n    return [0.0, 0.0]\n\n"
                "def main(features: list[float]) -> list[float]:\n    return [0.0]*6\n"
            )
            best_idx = None
            best_detail = {"error": "no_candidates"}

        # History updates.
        detail = best_detail or {}
        if active == "all":
            # Co-evolution update: each component gets the same final reward.
            history_r1.append(
                Attempt(
                    iteration=step,
                    score_a=best_score_a if math.isfinite(best_score_a) else -1e9,
                    score_b=best_score_b if math.isfinite(best_score_b) else -1e9,
                    code=str(detail.get("code_reduce_1") or ""),
                    comment=str(detail.get("comment")) if isinstance(detail.get("comment"), str) else None,
                )
            )
            history_r2.append(
                Attempt(
                    iteration=step,
                    score_a=best_score_a if math.isfinite(best_score_a) else -1e9,
                    score_b=best_score_b if math.isfinite(best_score_b) else -1e9,
                    code=str(detail.get("code_reduce_2") or ""),
                    comment=str(detail.get("comment")) if isinstance(detail.get("comment"), str) else None,
                )
            )
            history_r3.append(
                Attempt(
                    iteration=step,
                    score_a=best_score_a if math.isfinite(best_score_a) else -1e9,
                    score_b=best_score_b if math.isfinite(best_score_b) else -1e9,
                    code=str(detail.get("code_reduce_3") or ""),
                    comment=str(detail.get("comment")) if isinstance(detail.get("comment"), str) else None,
                )
            )
            history_main.append(
                Attempt(
                    iteration=step,
                    score_a=best_score_a if math.isfinite(best_score_a) else -1e9,
                    score_b=best_score_b if math.isfinite(best_score_b) else -1e9,
                    code=str(detail.get("code_main") or ""),
                    comment=str(detail.get("comment")) if isinstance(detail.get("comment"), str) else None,
                )
            )
        else:
            # Freeze schedule: only update the active component's evolutionary history.
            if active == "reduce_1":
                history_r1.append(
                    Attempt(
                        iteration=step,
                        score_a=best_score_a if math.isfinite(best_score_a) else -1e9,
                        score_b=best_score_b if math.isfinite(best_score_b) else -1e9,
                        code=str(detail.get("code_reduce_1") or ""),
                        comment=str(detail.get("comment")) if isinstance(detail.get("comment"), str) else None,
                    )
                )
            elif active == "reduce_2":
                history_r2.append(
                    Attempt(
                        iteration=step,
                        score_a=best_score_a if math.isfinite(best_score_a) else -1e9,
                        score_b=best_score_b if math.isfinite(best_score_b) else -1e9,
                        code=str(detail.get("code_reduce_2") or ""),
                        comment=str(detail.get("comment")) if isinstance(detail.get("comment"), str) else None,
                    )
                )
            elif active == "reduce_3":
                history_r3.append(
                    Attempt(
                        iteration=step,
                        score_a=best_score_a if math.isfinite(best_score_a) else -1e9,
                        score_b=best_score_b if math.isfinite(best_score_b) else -1e9,
                        code=str(detail.get("code_reduce_3") or ""),
                        comment=str(detail.get("comment")) if isinstance(detail.get("comment"), str) else None,
                    )
                )
            elif active == "reducers":
                history_r1.append(
                    Attempt(
                        iteration=step,
                        score_a=best_score_a if math.isfinite(best_score_a) else -1e9,
                        score_b=best_score_b if math.isfinite(best_score_b) else -1e9,
                        code=str(detail.get("code_reduce_1") or ""),
                        comment=str(detail.get("comment")) if isinstance(detail.get("comment"), str) else None,
                    )
                )
                history_r2.append(
                    Attempt(
                        iteration=step,
                        score_a=best_score_a if math.isfinite(best_score_a) else -1e9,
                        score_b=best_score_b if math.isfinite(best_score_b) else -1e9,
                        code=str(detail.get("code_reduce_2") or ""),
                        comment=str(detail.get("comment")) if isinstance(detail.get("comment"), str) else None,
                    )
                )
                history_r3.append(
                    Attempt(
                        iteration=step,
                        score_a=best_score_a if math.isfinite(best_score_a) else -1e9,
                        score_b=best_score_b if math.isfinite(best_score_b) else -1e9,
                        code=str(detail.get("code_reduce_3") or ""),
                        comment=str(detail.get("comment")) if isinstance(detail.get("comment"), str) else None,
                    )
                )
            elif active == "main":
                history_main.append(
                    Attempt(
                        iteration=step,
                        score_a=best_score_a if math.isfinite(best_score_a) else -1e9,
                        score_b=best_score_b if math.isfinite(best_score_b) else -1e9,
                        code=str(detail.get("code_main") or ""),
                        comment=str(detail.get("comment")) if isinstance(detail.get("comment"), str) else None,
                    )
                )
            else:
                raise RuntimeError(f"Unknown active component: {active}")

        record = {
            "iteration": step,
            "best_candidate_index": best_idx if best_idx is not None else -1,
            "active_component": active,
            "score_a": best_score_a,
            "score_b": best_score_b,
            "code_path": f"iter_{step:03d}.py",
            "concurrent": args.concurrent,
            **detail,
        }
        print(json.dumps(record, indent=2))
        (run_dir / f"iter_{step:03d}.py").write_text(best_code, encoding="utf-8")
        with attempts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
