from __future__ import annotations

import ast
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from program_synth.ai_code_env import clean_generated_code, validate_sandboxed_code
from program_synth.utils import repo_python_from_file


_EVAL_RUNNER = r"""
import importlib.util
import json
import math
import sys


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("candidate_policy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _finite(x: float) -> bool:
    return math.isfinite(float(x))


def _clip01(x: float) -> float:
    x = float(x)
    if not _finite(x):
        return 0.0
    return max(-1.0, min(1.0, x))


def _obs_to_features(obs, prev_achieved, t: int, max_steps: int):
    import numpy as np

    achieved = np.asarray(obs.get("achieved_goal", [0.0, 0.0, 0.0]), dtype=float).ravel()
    desired = np.asarray(obs.get("desired_goal", [0.0, 0.0, 0.0]), dtype=float).ravel()
    if achieved.shape[0] < 3:
        achieved = np.pad(achieved, (0, 3 - achieved.shape[0]))
    if desired.shape[0] < 3:
        desired = np.pad(desired, (0, 3 - desired.shape[0]))
    achieved = achieved[:3]
    desired = desired[:3]

    if prev_achieved is None:
        vel = np.zeros(3, dtype=float)
    else:
        vel = achieved - prev_achieved
    goal_rel = desired - achieved
    tau = float(t / max(1, max_steps - 1))

    feats = [
        float(achieved[0]),
        float(achieved[1]),
        float(achieved[2]),
        float(vel[0]),
        float(vel[1]),
        float(vel[2]),
        float(goal_rel[0]),
        float(goal_rel[1]),
        float(goal_rel[2]),
        float(tau),
    ]
    return feats, achieved


def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    policy_path = payload["policy_path"]
    env_id = payload.get("env_id", "FetchReach-v4")
    seed = int(payload.get("seed", 0))
    max_steps = int(payload.get("max_steps", 80))

    w_v = float(payload.get("w_v", 1.0))
    w_e = float(payload.get("w_e", 0.10))
    w_u = float(payload.get("w_u", 0.05))
    w_err = float(payload.get("w_err", 0.25))

    try:
        import gymnasium as gym
        from gymnasium.spaces import Box
        import numpy as np
        import gymnasium_robotics  # type: ignore

        gym.register_envs(gymnasium_robotics)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "missing_deps", "message": str(e)}))
        return 2

    try:
        env = gym.make(env_id)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "make_env_failed", "message": str(e)}))
        return 3

    try:
        mod = _load_module(policy_path)
    except Exception as e:
        env.close()
        print(json.dumps({"ok": False, "error": "load_failed", "message": str(e)}))
        return 4

    if not hasattr(mod, "main"):
        env.close()
        print(json.dumps({"ok": False, "error": "missing_main"}))
        return 5

    action_space = env.action_space
    if not isinstance(action_space, Box):
        env.close()
        print(json.dumps({"ok": False, "error": "unsupported_action_space", "space": repr(action_space)}))
        return 6
    if int(np.prod(action_space.shape)) < 3:
        env.close()
        print(json.dumps({"ok": False, "error": "action_space_too_small", "space": repr(action_space)}))
        return 7

    obs, info = env.reset(seed=seed)
    prev_achieved = None
    prev_action_xyz = None
    action_energy_sum = 0.0
    action_delta_sum = 0.0
    total_reward = 0.0
    step_errors = 0
    non_finite_outputs = 0
    terminated = False
    truncated = False
    steps_ran = 0
    terminal_speed = 0.0

    final_achieved = np.array([0.0, 0.0, 0.0], dtype=float)
    final_desired = np.array([0.0, 0.0, 0.0], dtype=float)

    for t in range(max_steps):
        features, achieved = _obs_to_features(obs, prev_achieved, t, max_steps)
        terminal_speed = float(np.linalg.norm(achieved - prev_achieved)) if prev_achieved is not None else 0.0
        prev_achieved = achieved

        try:
            out = mod.main(features)
            if not isinstance(out, (list, tuple)) or len(out) != 3:
                raise ValueError("policy output must be list[float] of length 3")
            ax, ay, az = float(out[0]), float(out[1]), float(out[2])
            if not (_finite(ax) and _finite(ay) and _finite(az)):
                non_finite_outputs += 1
                raise ValueError("non-finite output")
            a_xyz = np.array([_clip01(ax), _clip01(ay), _clip01(az)], dtype=float)
        except Exception:
            step_errors += 1
            a_xyz = np.zeros(3, dtype=float)

        full_dim = int(np.prod(action_space.shape))
        full_action = np.zeros(full_dim, dtype=float)
        full_action[0:3] = a_xyz
        action = full_action.reshape(action_space.shape)
        action = np.clip(action, action_space.low, action_space.high)

        action_energy_sum += float(np.dot(a_xyz, a_xyz))
        if prev_action_xyz is not None:
            d = a_xyz - prev_action_xyz
            action_delta_sum += float(np.dot(d, d))
        prev_action_xyz = a_xyz

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps_ran += 1

        try:
            final_achieved = np.asarray(obs.get("achieved_goal", final_achieved), dtype=float).ravel()[:3]
            final_desired = np.asarray(obs.get("desired_goal", final_desired), dtype=float).ravel()[:3]
        except Exception:
            pass

        if terminated or truncated:
            break

    env.close()

    mean_action_sq = action_energy_sum / max(1, steps_ran)
    mean_action_delta_sq = action_delta_sum / max(1, max(1, steps_ran - 1))
    reg_cost = (
        w_v * (terminal_speed ** 2)
        + w_e * mean_action_sq
        + w_u * mean_action_delta_sq
        + w_err * float(step_errors)
    )
    quality = -float(reg_cost)

    out = {
        "ok": True,
        "quality": quality,
        "descriptor_xyz": [float(final_achieved[0]), float(final_achieved[1]), float(final_achieved[2])],
        "terminal_speed": float(terminal_speed),
        "mean_action_sq": float(mean_action_sq),
        "mean_action_delta_sq": float(mean_action_delta_sq),
        "reg_cost": float(reg_cost),
        "step_errors": int(step_errors),
        "non_finite_outputs": int(non_finite_outputs),
        "steps_ran": int(steps_ran),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "env_reward_sum": float(total_reward),
        "goal_distance_final": float(
            math.sqrt(
                float((final_desired[0] - final_achieved[0]) ** 2)
                + float((final_desired[1] - final_achieved[1]) ** 2)
                + float((final_desired[2] - final_achieved[2]) ** 2)
            )
        ),
        "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else None,
        "obs_spec": {
            "shape": 10,
            "layout": [
                "achieved_x",
                "achieved_y",
                "achieved_z",
                "delta_achieved_x",
                "delta_achieved_y",
                "delta_achieved_z",
                "goal_rel_x",
                "goal_rel_y",
                "goal_rel_z",
                "tau",
            ],
        },
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def validate_policy_code(code: str) -> None:
    validate_sandboxed_code(
        code,
        allowed_import_roots={"math", "random", "itertools", "functools", "statistics"},
    )
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            if len(node.args.args) != 1:
                raise ValueError("`main` must take exactly one positional arg: obs.")
            return
    raise ValueError("Generated code must define `main(obs)`.")


def evaluate_candidate_code(
    code: str,
    *,
    env_id: str = "FetchReach-v4",
    seed: int = 0,
    max_steps: int = 80,
    timeout_s: float = 60.0,
    w_v: float = 1.0,
    w_e: float = 0.10,
    w_u: float = 0.05,
    w_err: float = 0.25,
) -> dict[str, Any]:
    validate_policy_code(code)

    with tempfile.TemporaryDirectory(prefix="qd_arm_eval_") as td:
        td_path = Path(td)
        policy_path = td_path / "policy.py"
        runner_path = td_path / "runner.py"
        policy_path.write_text(clean_generated_code(code), encoding="utf-8")
        runner_path.write_text(_EVAL_RUNNER, encoding="utf-8")

        proc = subprocess.run(
            [repo_python_from_file(__file__), str(runner_path)],
            input=json.dumps(
                {
                    "policy_path": str(policy_path),
                    "env_id": env_id,
                    "seed": int(seed),
                    "max_steps": int(max_steps),
                    "w_v": float(w_v),
                    "w_e": float(w_e),
                    "w_u": float(w_u),
                    "w_err": float(w_err),
                }
            ),
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
    if not isinstance(out, dict):
        return {
            "ok": False,
            "error": "runner_output_not_object",
            "stdout": stdout,
            "stderr": stderr,
            "returncode": proc.returncode,
        }
    out["returncode"] = proc.returncode
    if stderr:
        out["stderr"] = stderr
    if not out.get("ok"):
        return out

    desc = out.get("descriptor_xyz")
    if not isinstance(desc, list) or len(desc) != 3:
        return {"ok": False, "error": "bad_descriptor", "runner": out}
    if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in desc):
        return {"ok": False, "error": "non_finite_descriptor", "runner": out}
    q = out.get("quality")
    if not isinstance(q, (int, float)) or not math.isfinite(float(q)):
        return {"ok": False, "error": "non_finite_quality", "runner": out}

    return out

