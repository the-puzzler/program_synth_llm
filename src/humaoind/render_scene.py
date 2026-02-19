from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_out_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("runs") / "humaoind"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"humanoid_scene_{ts}.mp4"


def _jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return repr(x)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a short Humanoid scene rollout to verify MuJoCo+rendering setup."
    )
    ap.add_argument("--env-id", default="Humanoid-v5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None, help="Output MP4 path.")
    args = ap.parse_args()

    try:
        import gymnasium as gym
    except Exception as e:
        raise SystemExit(f"Missing gymnasium install. Error: {e}")

    try:
        import imageio.v2 as imageio
    except Exception as e:
        raise SystemExit(f"Missing imageio install for video writing. Error: {e}")

    out_path = (args.out or _default_out_path()).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        env = gym.make(args.env_id, render_mode="rgb_array")
    except Exception as e:
        raise SystemExit(
            "Failed to create env. For Humanoid, make sure MuJoCo deps are installed "
            "(e.g. `uv add \"gymnasium[mujoco]\"`). "
            f"Error: {e}"
        )

    obs, info = env.reset(seed=int(args.seed))
    frames: list[Any] = []
    total_reward = 0.0
    steps_ran = 0
    terminated = False
    truncated = False

    for _ in range(max(1, int(args.steps))):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        total_reward += float(reward)
        steps_ran += 1
        if terminated or truncated:
            break

    env.close()

    if not frames:
        raise SystemExit("No frames captured from env render.")

    imageio.mimsave(out_path, frames, fps=max(1, int(args.fps)), codec="libx264")
    print(
        json.dumps(
            _jsonable(
                {
                    "ok": True,
                    "env_id": args.env_id,
                    "out_path": str(out_path),
                    "seed": int(args.seed),
                    "steps_requested": int(args.steps),
                    "steps_ran": int(steps_ran),
                    "frames": len(frames),
                    "total_reward": float(total_reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "obs_shape": list(getattr(obs, "shape", [])) if hasattr(obs, "shape") else None,
                    "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else None,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

