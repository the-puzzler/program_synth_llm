from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ai_code_env import clean_generated_code, validate_sandboxed_code

try:
    import numpy as np  # type: ignore
    import imageio.v2 as imageio  # type: ignore

    _HAVE_IMAGEIO = True
except Exception:
    np = None  # type: ignore
    imageio = None  # type: ignore
    _HAVE_IMAGEIO = False

try:
    from PIL import Image, ImageDraw, ImageFont, ImageSequence  # type: ignore

    _HAVE_PIL = True
except Exception:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore
    ImageSequence = None  # type: ignore
    _HAVE_PIL = False


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
    label = payload.get("label")
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
        from PIL import Image, ImageDraw, ImageFont
        _have_pil = True
    except Exception:
        _have_pil = False

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
                if _have_pil and label:
                    img = Image.fromarray(frame[:, :, :3])
                    draw = ImageDraw.Draw(img)
                    try:
                        font = ImageFont.load_default()
                    except Exception:
                        font = None
                    x, y = 6, 6
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        draw.text((x + dx, y + dy), label, fill=(0, 0, 0), font=font)
                    draw.text((x, y), label, fill=(255, 255, 255), font=font)
                    frame = __import__("numpy").asarray(img)
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
                # `truncated` is how Gymnasium reports TimeLimit; treat that as "hit max steps".
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


def _latest_run_dir(runs_root: Path = Path("runs")) -> Path:
    candidates = sorted(p for p in runs_root.glob("bipedal_*") if p.is_dir())
    if not candidates:
        raise SystemExit(f"No bipedal runs found under {runs_root}")
    return candidates[-1]


def _iter_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("iter_*.py"))


def _run_one(
    code: str,
    *,
    out_gif: Path,
    env_id: str,
    seed: int,
    max_steps: int,
    every: int,
    fps: int,
    timeout_s: float,
    label: str | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="bipedal_gif_") as td:
        td_path = Path(td)
        policy_path = td_path / "policy.py"
        runner_path = td_path / "runner.py"
        policy_path.write_text(clean_generated_code(code), encoding="utf-8")
        runner_path.write_text(_RUNNER_PY, encoding="utf-8")

        proc = subprocess.run(
            [_python_for_repo(), str(runner_path)],
            input=json.dumps(
                {
                    "policy_path": str(policy_path),
                    "out_path": str(out_gif),
                    "label": label,
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
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if not stdout:
            return {"ok": False, "error": "empty_stdout", "stderr": stderr, "returncode": proc.returncode}
        try:
            out = json.loads(stdout)
        except json.JSONDecodeError:
            out = {"ok": False, "error": "non_json_stdout", "stdout": stdout}
        out["returncode"] = proc.returncode
        if stderr:
            out["stderr"] = stderr
        return out


def _make_collage_gif(
    gif_paths: list[Path],
    *,
    out_path: Path,
    cols: int,
    fps: int,
    limit: int | None = None,
) -> Path:
    raise RuntimeError("Grid collage mode removed; use sequence mode instead.")


def _iter_label_from_path(p: Path) -> str:
    # iter_012.gif -> iter 012
    try:
        stem = p.stem
        if stem.startswith("iter_"):
            return f"iter {int(stem.split('_', 1)[1]):03d}"
    except Exception:
        pass
    return p.stem


def _overlay_label(frame: Any, label: str) -> Any:
    if not _HAVE_PIL:
        return frame
    assert Image is not None and ImageDraw is not None and ImageFont is not None

    img = Image.fromarray(frame[:, :, :3])
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    x, y = 6, 6
    # Simple outline for readability.
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        draw.text((x + dx, y + dy), label, fill=(0, 0, 0), font=font)
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)  # type: ignore[no-any-return]


def _format_metrics_label(base: str, meta: dict[str, Any] | None) -> str:
    if not meta:
        return base
    dist = meta.get("distance")
    spd = meta.get("avg_speed")
    parts = [base]
    if dist is not None:
        try:
            parts.append(f"dist={float(dist):.3f}")
        except Exception:
            pass
    if spd is not None:
        try:
            parts.append(f"spd={float(spd):.4f}")
        except Exception:
            pass
    return "  ".join(parts)

def _load_attempt_metrics(run_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Reads run-level `attempts.jsonl` and maps iteration stem (e.g. "iter_004") to metrics.
    Uses whatever fields are present; currently prefers `avg_distance`/`avg_speed` if available.
    """
    path = run_dir / "attempts.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        code_path = obj.get("code_path")
        if isinstance(code_path, str) and code_path.endswith(".py"):
            stem = Path(code_path).stem
        else:
            it = obj.get("iteration")
            if isinstance(it, int):
                stem = f"iter_{it:03d}"
            else:
                continue
        out[stem] = {
            "distance": obj.get("avg_distance"),
            "avg_speed": obj.get("avg_speed"),
            "return": obj.get("avg_return"),
        }
    return out


def _make_sequence_gif(
    gif_paths: list[Path],
    *,
    out_path: Path,
    fps: int,
    speed: float = 2.0,
    limit: int | None = None,
    meta_by_stem: dict[str, dict[str, Any]] | None = None,
) -> Path:
    if not _HAVE_IMAGEIO:
        raise RuntimeError("Sequence requires `imageio` and `numpy`.")
    assert imageio is not None and np is not None

    gifs = [p for p in gif_paths if p.exists()]
    if limit is not None and limit > 0:
        gifs = gifs[-limit:]
    if not gifs:
        raise RuntimeError("No GIFs available to sequence.")

    def _first_frame_size(path: Path) -> tuple[int, int] | None:
        # Prefer Pillow for reading GIFs (avoids imageio backend/plugin issues).
        if _HAVE_PIL:
            assert Image is not None
            try:
                with Image.open(path) as im:
                    return (int(im.height), int(im.width))
            except Exception:
                return None
        try:
            reader = imageio.get_reader(path)
            try:
                first = reader.get_data(0)
                return (int(first.shape[0]), int(first.shape[1]))
            finally:
                reader.close()
        except Exception:
            return None

    def _iter_frames(path: Path):
        if _HAVE_PIL:
            assert Image is not None and ImageSequence is not None
            with Image.open(path) as im:
                for fr in ImageSequence.Iterator(im):
                    yield np.asarray(fr.convert("RGB"))
            return
        reader = imageio.get_reader(path)
        try:
            for fr in reader:
                yield fr
        finally:
            reader.close()

    # Stream processing to avoid imageio's memtest threshold (and huge RAM usage).
    # 1) Determine a common crop size by reading only the first frame of each GIF.
    min_h: int | None = None
    min_w: int | None = None
    readable_gifs: list[Path] = []
    for p in gifs:
        size = _first_frame_size(p)
        if size is None:
            continue
        h, w = size
        min_h = h if min_h is None else min(min_h, h)
        min_w = w if min_w is None else min(min_w, w)
        readable_gifs.append(p)
    if min_h is None or min_w is None:
        raise RuntimeError("Could not read any frames from the selected GIFs.")

    def crop(frame: Any) -> Any:
        h, w = frame.shape[0], frame.shape[1]
        top = max(0, (h - min_h) // 2)
        left = max(0, (w - min_w) // 2)
        return frame[top : top + min_h, left : left + min_w, :3]

    # Many GIF viewers clamp very small frame delays, so increasing FPS can paradoxically
    # look slower. Instead, speed up by dropping frames and keeping FPS unchanged.
    if not math.isfinite(speed) or speed <= 0:
        speed = 1.0
    stride = max(1, int(round(speed))) if speed > 1.0 else 1
    repeat = max(1, int(round(1.0 / speed))) if speed < 1.0 else 1

    writer = imageio.get_writer(out_path, mode="I", fps=max(1, int(fps)))
    try:
        for p in readable_gifs:
            base = _iter_label_from_path(p)
            meta = meta_by_stem.get(p.stem) if meta_by_stem else None
            label = _format_metrics_label(base, meta)
            try:
                for i, f in enumerate(_iter_frames(p)):
                    if stride > 1 and (i % stride) != 0:
                        continue
                    frame = crop(f)
                    if _HAVE_PIL:
                        frame = _overlay_label(frame, label)
                    for _ in range(repeat):
                        writer.append_data(frame)
            except Exception:
                # Skip unreadable/corrupt gifs instead of failing the whole collage.
                continue
    finally:
        writer.close()
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path, nargs="?", default=None)
    p.add_argument("--env-id", default="BipedalWalker-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--every", type=int, default=2, help="Record every Nth step")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--collage", action="store_true", help="Also write one sequential GIF (2x speed).")
    p.add_argument(
        "--collage-limit",
        type=int,
        default=0,
        help="Max GIFs to include (latest N); set 0 to include all.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-render per-iteration GIFs even if they already exist.",
    )
    args = p.parse_args()

    run_dir = (args.run_dir or _latest_run_dir()).resolve()
    iters = _iter_files(run_dir)
    if not iters:
        raise SystemExit(f"No `iter_*.py` files found in {run_dir}")

    out_dir = run_dir / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []
    meta_by_stem = _load_attempt_metrics(run_dir)
    for it_path in iters:
        code = it_path.read_text(encoding="utf-8")
        try:
            validate_sandboxed_code(
                code,
                allowed_import_roots={"math", "random", "itertools", "functools", "statistics"},
            )
        except Exception as e:
            print(json.dumps({"iter": it_path.name, "ok": False, "error": f"sandbox: {e}"}))
            continue

        out_gif = out_dir / f"{it_path.stem}.gif"
        if out_gif.exists() and not args.force:
            rendered.append(out_gif)
            print(
                json.dumps(
                    {"iter": it_path.name, "ok": True, "skipped": True, "out_path": str(out_gif)},
                    indent=2,
                )
            )
            continue

        label = _format_metrics_label(_iter_label_from_path(out_gif), meta_by_stem.get(out_gif.stem))
        result = _run_one(
            code,
            out_gif=out_gif,
            env_id=args.env_id,
            seed=args.seed,
            max_steps=args.max_steps,
            every=args.every,
            fps=args.fps,
            timeout_s=args.timeout,
            label=label,
        )
        if result.get("ok") and out_gif.exists():
            rendered.append(out_gif)
        print(json.dumps({"iter": it_path.name, **result}, indent=2))

    if args.collage:
        limit = None if args.collage_limit == 0 else args.collage_limit
        try:
            out = _make_sequence_gif(
                rendered,
                out_path=out_dir / "collage.gif",
                fps=args.fps,
                speed=2.0,
                limit=limit,
                meta_by_stem=meta_by_stem,
            )
            print(json.dumps({"collage_ok": True, "path": str(out)}, indent=2))
        except Exception as e:
            print(json.dumps({"collage_ok": False, "error": str(e)}, indent=2))


if __name__ == "__main__":
    main()
