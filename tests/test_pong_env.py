from __future__ import annotations

import argparse
import json
from typing import Any


def _jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return repr(x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="ALE/Pong-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--frameskip", type=int, default=1)
    p.add_argument("--full-action-space", action="store_true", help="Use full ALE action space (larger).")
    args = p.parse_args()

    try:
        import gymnasium as gym
    except Exception as e:
        raise SystemExit(f"Missing gymnasium. Install deps first (e.g. `uv sync`). Error: {e}")

    # Gymnasium doesn't always auto-register ALE env ids; ensure they're registered when ale_py is available.
    try:
        from gymnasium.envs.registration import registry

        if args.env_id not in registry:
            import ale_py.registration as _ale_reg

            _ale_reg.register_v5_envs()
    except Exception:
        pass

    try:
        env = gym.make(
            args.env_id,
            obs_type="rgb",
            frameskip=int(args.frameskip),
            repeat_action_probability=0.0,
            full_action_space=bool(args.full_action_space),
        )
    except TypeError:
        env = gym.make(args.env_id)
    except Exception as e:
        raise SystemExit(
            "Failed to create Atari env. You likely need ALE + ROMs:\n"
            "- `uv add \"gymnasium[atari,accept-rom-license]\"`\n"
            "- Ensure ROMs are installed/accepted for Gymnasium ALE\n"
            f"\nError: {e}"
        )

    obs, info = env.reset(seed=int(args.seed))
    action_space = getattr(env, "action_space", None)

    total = 0.0
    terminated = False
    truncated = False
    reward_events = 0
    nonzero_rewards: list[float] = []

    for t in range(int(args.steps)):
        # Prefer sampling from the env's action space if available.
        try:
            action = int(action_space.sample()) if action_space is not None else 0
        except Exception:
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)
        r = float(reward)
        total += r
        if r != 0.0:
            reward_events += 1
            nonzero_rewards.append(r)
        if terminated or truncated:
            break

    env.close()

    shape = None
    try:
        shape = list(getattr(obs, "shape", None)) if getattr(obs, "shape", None) is not None else None
    except Exception:
        shape = None

    out = {
        "ok": True,
        "env_id": args.env_id,
        "seed": int(args.seed),
        "steps_ran": int(t + 1),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "total_reward": float(total),
        "reward_events": int(reward_events),
        "nonzero_rewards": nonzero_rewards[:10],
        "obs_type": type(obs).__name__,
        "obs_shape": shape,
        "action_space": repr(action_space),
        "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else None,
    }
    print(json.dumps(_jsonable(out), indent=2))


if __name__ == "__main__":
    main()
