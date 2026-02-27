from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return repr(x)


def _default_out_path(env_id: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_env = env_id.replace("/", "_").replace("-", "_")
    return Path("runs") / f"qd_arm_smoke_{safe_env}_{ts}.mp4"


def main() -> None:
    env_id = "FetchReach-v4"
    seed = 0
    steps = 300
    every = 1
    fps = 30
    policy = "random"  # "random" | "zero"

    try:
        import gymnasium as gym
        from gymnasium.spaces import Box, Discrete
        import imageio.v2 as imageio
        import numpy as np
    except Exception as e:
        raise SystemExit(
            "Missing dependencies. Install with:\n"
            '  uv add "gymnasium[mujoco]" imageio[pyav] numpy\n'
            f"Error: {e}"
        )

    # Register extra robotics envs (e.g., FrankaKitchen) when available.
    try:
        import gymnasium_robotics  # type: ignore

        gym.register_envs(gymnasium_robotics)
    except Exception:
        pass

    out_path = _default_out_path(env_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _make_env():
        try:
            return gym.make(env_id, render_mode="rgb_array", max_episode_steps=int(steps))
        except TypeError:
            env = gym.make(env_id, render_mode="rgb_array")
            try:
                from gymnasium.wrappers import TimeLimit

                return TimeLimit(env, max_episode_steps=int(steps))
            except Exception:
                return env

    try:
        env = _make_env()
    except Exception as e:
        raise SystemExit(
            f"Could not create env {env_id!r}. "
            "If this is a MuJoCo env, ensure MuJoCo deps are installed.\n"
            f"Error: {e}"
        )

    obs, info = env.reset(seed=int(seed))
    action_space = env.action_space
    frames: list[Any] = []
    total_reward = 0.0
    terminated = False
    truncated = False
    steps_ran = 0

    for t in range(int(steps)):
        if policy == "zero":
            if isinstance(action_space, Box):
                action = np.zeros(action_space.shape, dtype=float)
            elif isinstance(action_space, Discrete):
                action = 0
            else:
                action = action_space.sample()
        else:
            action = action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps_ran += 1

        if (t % int(every)) == 0:
            frame = env.render()
            if frame is not None:
                arr = np.asarray(frame)
                if getattr(arr, "ndim", 0) == 3 and arr.shape[-1] >= 3:
                    arr = arr[..., :3]
                if arr.dtype != np.uint8:
                    m = float(np.nanmax(arr))
                    if m <= 1.5:
                        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
                    else:
                        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                frames.append(arr)

        if terminated or truncated:
            break

    env.close()

    if not frames:
        raise SystemExit("No frames captured; rendering failed.")

    imageio.mimsave(out_path, frames, fps=int(fps), codec="libx264")

    out = {
        "ok": True,
        "env_id": env_id,
        "seed": int(seed),
        "steps_ran": int(steps_ran),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "total_reward": float(total_reward),
        "frames": len(frames),
        "out_path": str(out_path),
        "policy": policy,
        "obs_type": type(obs).__name__,
        "obs_shape": list(getattr(obs, "shape", [])) if getattr(obs, "shape", None) is not None else None,
        "action_space": repr(action_space),
        "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else None,
    }
    print(json.dumps(_jsonable(out), indent=2))


if __name__ == "__main__":
    main()
