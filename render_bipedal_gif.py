from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_code_env import clean_generated_code, extract_python_code, validate_sandboxed_code


_RUNNER_PY = r"""
import importlib.util
import json
import sys


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("policy", path)
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
    policy_path = payload["policy_path"]
    out_path = payload["out_path"]
    env_id = payload.get("env_id", "BipedalWalker-v3")
    seed = int(payload.get("seed", 0))
    max_steps = int(payload.get("max_steps", 1600))
    every = int(payload.get("every", 2))
    fps = int(payload.get("fps", 30))

    try:
        import gymnasium as gym
    except Exception as e:
        print(json.dumps({"ok": False, "error": "missing_gymnasium", "message": str(e)}))
        return 2

    def _make_env():
        # Override the default TimeLimit (BipedalWalker is commonly capped at 1600 steps).
        try:
            return gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_steps)
        except TypeError:
            env = gym.make(env_id, render_mode="rgb_array")
            try:
                from gymnasium.wrappers import TimeLimit

                return TimeLimit(env, max_episode_steps=max_steps)
            except Exception:
                return env

    try:
        import imageio.v2 as imageio
    except Exception as e:
        print(json.dumps({"ok": False, "error": "missing_imageio", "message": str(e)}))
        return 3

    try:
        mod = _load_module(policy_path)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "load_failed", "message": str(e)}))
        return 4

    if not hasattr(mod, "main"):
        print(json.dumps({"ok": False, "error": "missing_main"}))
        return 5

    env = _make_env()
    obs, _info = env.reset(seed=seed)

    frames = []
    total = 0.0
    step_errors = 0
    steps = 0
    terminated = False
    truncated = False
    start_x = None
    end_x = None

    try:
        start_x = float(env.unwrapped.hull.position[0])
    except Exception:
        start_x = None

    for t in range(max_steps):
        try:
            action = mod.main(list(obs))
            if not isinstance(action, (list, tuple)) or len(action) != 4:
                raise ValueError("action must be a list/tuple of length 4")
            a = [_clip(action[0]), _clip(action[1]), _clip(action[2]), _clip(action[3])]
        except Exception:
            step_errors += 1
            a = [0.0, 0.0, 0.0, 0.0]

        obs, reward, terminated, truncated, _info = env.step(a)
        total += float(reward)
        steps += 1

        if (t % max(1, every)) == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if terminated or truncated:
            break

    try:
        end_x = float(env.unwrapped.hull.position[0])
    except Exception:
        end_x = None

    env.close()

    if not frames:
        print(json.dumps({"ok": False, "error": "no_frames", "return": total, "step_errors": step_errors}))
        return 6

    distance = None
    avg_speed = None
    if start_x is not None and end_x is not None:
        distance = float(end_x - start_x)
        if steps > 0:
            avg_speed = float(distance / steps)

    imageio.mimsave(out_path, frames, fps=fps)
    print(
        json.dumps(
            {
                "ok": True,
                "out_path": out_path,
                "frames": len(frames),
                "return": total,
                "step_errors": step_errors,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "hit_max_steps": bool(steps >= max_steps and not terminated),
                "distance": distance,
                "avg_speed": avg_speed,
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


def _default_out_path(run_dir: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return run_dir / f"manual_bipedal_{ts}.gif"


def _load_code(args: argparse.Namespace) -> str:
    if args.code_path is not None:
        return Path(args.code_path).read_text(encoding="utf-8")
    return (Path("/dev/stdin").read_text(encoding="utf-8") or "").strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--code-path", "--code_path", type=Path, default=None, help="Path to a .py containing `main(obs)`.")
    p.add_argument("--out", type=Path, default=None, help="Output GIF path.")
    p.add_argument("--env-id", default="BipedalWalker-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--every", type=int, default=2, help="Record every Nth step.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()

    raw = _load_code(args)
    if not raw:
        raise SystemExit("No code provided. Pass `--code-path`, or pipe code into stdin.")
    code = clean_generated_code(extract_python_code(raw))
    validate_sandboxed_code(code, allowed_import_roots={"math"})

    out_path = args.out
    if out_path is None:
        out_dir = Path("runs") / "manual_gifs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = _default_out_path(out_dir)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="bipedal_manual_gif_") as td:
        td_path = Path(td)
        policy_path = td_path / "policy.py"
        runner_path = td_path / "runner.py"
        policy_path.write_text(code, encoding="utf-8")
        runner_path.write_text(_RUNNER_PY, encoding="utf-8")

        proc = subprocess.run(
            [_python_for_repo(), str(runner_path)],
            input=json.dumps(
                {
                    "policy_path": str(policy_path),
                    "out_path": str(out_path),
                    "env_id": args.env_id,
                    "seed": args.seed,
                    "max_steps": args.max_steps,
                    "every": args.every,
                    "fps": args.fps,
                }
            ),
            text=True,
            capture_output=True,
            timeout=float(args.timeout),
        )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if not stdout:
            raise SystemExit(f"Runner produced no output (rc={proc.returncode}). stderr:\n{stderr}")
        try:
            payload: dict[str, Any] = json.loads(stdout)
        except json.JSONDecodeError:
            raise SystemExit(f"Runner produced non-JSON output (rc={proc.returncode}). stdout:\n{stdout}\nstderr:\n{stderr}")

        payload["returncode"] = proc.returncode
        if stderr:
            payload["stderr"] = stderr
        print(json.dumps(payload, indent=2))

        if not payload.get("ok"):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
