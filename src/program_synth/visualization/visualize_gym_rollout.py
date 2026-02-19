from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from program_synth.ai_code_env import clean_generated_code, validate_sandboxed_code


_GYM_VIDEO_RUNNER = r"""
import importlib.util
import json
import sys


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("policy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    policy_path = payload["policy_path"]
    out_path = payload["out_path"]
    env_id = payload.get("env_id")
    seed = int(payload.get("seed", 0))
    max_steps = int(payload.get("max_steps", 1000))
    every = int(payload.get("every", 1))
    fps = int(payload.get("fps", 30))

    try:
        import gymnasium as gym
        from gymnasium.spaces import Box, Discrete
        import imageio.v2 as imageio
        import numpy as np
    except Exception as e:
        print(json.dumps({"ok": False, "error": "missing_deps", "message": str(e)}))
        return 2

    def _make_env():
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
        mod = _load_module(policy_path)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "load_failed", "message": str(e)}))
        return 3

    if not hasattr(mod, "main"):
        print(json.dumps({"ok": False, "error": "missing_main"}))
        return 4

    env = _make_env()
    obs, _info = env.reset(seed=seed)

    act_space = env.action_space

    frames = []
    total = 0.0
    step_errors = 0
    steps = 0
    terminated = False
    truncated = False

    for t in range(max_steps):
        try:
            obs_vec = np.asarray(obs, dtype=float).ravel().tolist()
            action = mod.main(obs_vec)
            if isinstance(act_space, Discrete):
                a_int = int(action)
                if a_int < 0:
                    a_int = 0
                if a_int >= act_space.n:
                    a_int = act_space.n - 1
                a = a_int
            elif isinstance(act_space, Box):
                arr = np.array(action, dtype=float).reshape(act_space.shape)
                a = np.clip(arr, act_space.low, act_space.high)
            else:
                raise ValueError("Unsupported action space type.")
        except Exception:
            step_errors += 1
            a = act_space.sample()

        obs, reward, terminated, truncated, _info = env.step(a)
        total += float(reward)
        steps += 1

        if (t % max(1, every)) == 0:
            frame = env.render()
            if frame is not None:
                frame = np.asarray(frame)
                if getattr(frame, "ndim", 0) == 3 and frame.shape[-1] >= 3:
                    frame = frame[..., :3]
                if frame.dtype != np.uint8:
                    m = float(np.nanmax(frame))
                    if m <= 1.5:
                        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
                    else:
                        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
                frames.append(frame)

        if terminated or truncated:
            break

    env.close()

    if not frames:
        print(json.dumps({"ok": False, "error": "no_frames", "return": total, "step_errors": step_errors}))
        return 5

    imageio.mimsave(out_path, frames, fps=fps, codec="libx264")
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
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _python_for_repo() -> str:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    venv_python = repo_root / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else "python3"


def _infer_env_id(run_dir: Path) -> str:
    meta = run_dir / "meta.json"
    if not meta.exists():
        raise SystemExit(f"meta.json not found under {run_dir}")
    try:
        obj = json.loads(meta.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Could not parse {meta}: {e}")
    if not isinstance(obj, dict):
        raise SystemExit(f"{meta} did not contain a JSON object")
    env_id = obj.get("env_id")
    if not isinstance(env_id, str):
        raise SystemExit(f"{meta} does not contain a valid 'env_id'")
    return env_id


def _latest_iter_file(run_dir: Path) -> Path:
    iters = sorted(run_dir.glob("iter_*.py"))
    if not iters:
        raise SystemExit(f"No iter_XXX.py files found under {run_dir}")
    return iters[-1]


def _run_one(
    code: str,
    *,
    out_video: Path,
    env_id: str,
    seed: int,
    max_steps: int,
    every: int,
    fps: int,
    timeout_s: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gym_video_") as td:
        td_path = Path(td)
        policy_path = td_path / "policy.py"
        runner_path = td_path / "runner.py"
        policy_path.write_text(clean_generated_code(code), encoding="utf-8")
        runner_path.write_text(_GYM_VIDEO_RUNNER, encoding="utf-8")

        proc = subprocess.run(
            [_python_for_repo(), str(runner_path)],
            input=json.dumps(
                {
                    "policy_path": str(policy_path),
                    "out_path": str(out_video),
                    "env_id": env_id,
                    "seed": seed,
                    "max_steps": max_steps,
                    "every": every,
                    "fps": fps,
                }
            ),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )

    if proc.returncode != 0:
        try:
            data = json.loads(proc.stdout.strip() or "{}")
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {
            "ok": False,
            "error": "runner_failed",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    try:
        data = json.loads(proc.stdout.strip() or "{}")
        if not isinstance(data, dict):
            raise ValueError("runner output was not a JSON object")
        return data
    except Exception as e:
        return {
            "ok": False,
            "error": f"bad_runner_json: {e}",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Render MP4 rollout(s) for a Gym-based run.\n"
            "- By default, renders videos for all iter_XXX.py files in the run-dir.\n"
            "- You can use --iter to restrict to a single iteration."
        )
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a run directory (e.g. runs/gym_islands_LunarLander_v3_...).",
    )
    ap.add_argument(
        "--iter",
        type=str,
        default=None,
        help="Iteration to visualize (e.g. '003' or 'iter_003.py'); "
        "when omitted, renders all iter_XXX.py files in the run-dir.",
    )
    ap.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Gym env ID override; defaults to env_id from run-dir/meta.json.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the rendered episode.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max steps per episode (defaults to value in meta.json if present, else 1000).",
    )
    ap.add_argument(
        "--every",
        type=int,
        default=1,
        help="Record every Nth frame (1 = every frame).",
    )
    ap.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video frames per second.",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Subprocess timeout in seconds.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output MP4 path; defaults to RUN_DIR/videos/iter_XXX.mp4.",
    )
    args = ap.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"run-dir is not a directory: {run_dir}")

    env_id = args.env_id or _infer_env_id(run_dir)

    # Determine which iter_XXX.py files to render.
    if args.iter is None:
        code_paths = sorted(run_dir.glob("iter_*.py"))
        if not code_paths:
            raise SystemExit(f"No iter_XXX.py files found under {run_dir}")
    else:
        name = args.iter
        if not name.endswith(".py"):
            try:
                # Allow "3" or "003".
                it = int(name)
                name = f"iter_{it:03d}.py"
            except Exception:
                # Assume literal filename under run_dir.
                pass
        code_path = run_dir / name
        if not code_path.is_file():
            raise SystemExit(f"Code file not found: {code_path}")
        code_paths = [code_path]

    meta = run_dir / "meta.json"
    default_max_steps = 1000
    if meta.exists():
        try:
            meta_obj = json.loads(meta.read_text(encoding="utf-8"))
            if isinstance(meta_obj, dict):
                ms = meta_obj.get("max_steps")
                if isinstance(ms, int) and ms > 0:
                    default_max_steps = ms
        except Exception:
            pass

    max_steps = int(args.max_steps) if args.max_steps is not None else default_max_steps

    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Render one video per selected iter file.
    for code_path in code_paths:
        code = code_path.read_text(encoding="utf-8")
        try:
            validate_sandboxed_code(
                code,
                allowed_import_roots={"math", "random", "itertools", "functools", "statistics"},
            )
        except Exception as e:
            print(f"Skipping {code_path.name}: sandbox validation failed: {e}")
            continue

        out_path = args.out or (videos_dir / f"{code_path.stem}.mp4")

        # If no explicit --out is provided and the video already exists, skip
        # rendering to avoid unnecessary work when batch-generating all videos.
        if args.out is None and out_path.exists():
            print(f"{code_path.name}: video already exists at {out_path}, skipping.")
            continue

        result = _run_one(
            code,
            out_video=out_path,
            env_id=env_id,
            seed=int(args.seed),
            max_steps=max_steps,
            every=max(1, int(args.every)),
            fps=int(args.fps),
            timeout_s=float(args.timeout),
        )

        print(f"{code_path.name}:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
