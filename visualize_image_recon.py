from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ai_code_env import clean_generated_code


def _latest_run_dir(runs_root: Path = Path("runs")) -> Path:
    candidates = sorted(p for p in runs_root.glob("image_*") if p.is_dir())
    if not candidates:
        raise SystemExit(f"No image runs found under {runs_root}")
    return candidates[-1]


def _iter_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("iter_*.py"))


def _load_attempt_scores(run_dir: Path) -> dict[int, float]:
    path = run_dir / "attempts.jsonl"
    if not path.exists():
        return {}
    out: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        it = obj.get("iteration")
        score = obj.get("score")
        if isinstance(it, int) and isinstance(score, (int, float)):
            out[int(it)] = float(score)
    return out


def _load_attempt_score_pairs(run_dir: Path) -> dict[int, tuple[float, float]]:
    path = run_dir / "attempts.jsonl"
    if not path.exists():
        return {}
    out: dict[int, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        it = obj.get("iteration")
        a = obj.get("score_a")
        b = obj.get("score_b")
        if isinstance(it, int) and isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[int(it)] = (float(a), float(b))
    return out


def _load_module_from_code(path: Path) -> Any:
    # Load module from file path.
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render_prediction(
    mod: Any,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        fy = 0.0 if height == 1 else y / (height - 1)
        for x in range(width):
            fx = 0.0 if width == 1 else x / (width - 1)
            try:
                out = mod.main(float(fx), float(fy))
                if not isinstance(out, (list, tuple)) or len(out) != 3:
                    raise TypeError("bad output")
                r, g, b = float(out[0]), float(out[1]), float(out[2])
                if not (math.isfinite(r) and math.isfinite(g) and math.isfinite(b)):
                    raise ValueError("non-finite")
                img[y, x, 0] = np.clip(r, 0.0, 1.0)
                img[y, x, 1] = np.clip(g, 0.0, 1.0)
                img[y, x, 2] = np.clip(b, 0.0, 1.0)
            except Exception:
                img[y, x, :] = 0.0
    return img


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path, nargs="?", default=None)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    args = p.parse_args()

    run_dir = (args.run_dir or _latest_run_dir()).resolve()
    files = _iter_files(run_dir)
    if not files:
        raise SystemExit(f"No iter_*.py files found in {run_dir}")

    scores = _load_attempt_scores(run_dir)
    score_pairs = _load_attempt_score_pairs(run_dir)

    n = len(files)
    ncols = max(1, int(math.ceil(math.sqrt(n))))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)

    for ax in axes.ravel():
        ax.set_axis_off()

    for i, path in enumerate(files):
        ax = axes[i // ncols][i % ncols]
        ax.set_axis_on()

        # Ensure we don't have stray leading 'py' tags in saved files.
        code = clean_generated_code(path.read_text(encoding="utf-8"))
        tmp_path = run_dir / (path.stem + ".__clean__.py")
        tmp_path.write_text(code, encoding="utf-8")
        try:
            mod = _load_module_from_code(tmp_path)
            img = _render_prediction(mod, width=args.width, height=args.height)
            ax.imshow(img, origin="upper")
        except Exception as e:
            ax.text(0.5, 0.5, f"error:\n{e}", ha="center", va="center", transform=ax.transAxes)
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

        it = None
        try:
            it = int(path.stem.split("_", 1)[1])
        except Exception:
            pass
        if it is not None and it in scores:
            ax.set_title(f"iter {it:03d} (score={scores[it]:.4f})")
        elif it is not None and it in score_pairs:
            a, b = score_pairs[it]
            ax.set_title(f"iter {it:03d} (a={a:.3f}, b={b:.3f})")
        else:
            ax.set_title(path.stem)

    fig.suptitle(run_dir.name, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path = run_dir / "collage.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
