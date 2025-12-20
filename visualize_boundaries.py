from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ai_code_env import FunctionSpec, run_code_main_batch, validate_generated_code, validate_sandboxed_code

try:
    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.colors import Normalize  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore

    _HAVE_MATPLOTLIB = True
except Exception:
    np = None  # type: ignore
    plt = None  # type: ignore
    Normalize = None  # type: ignore
    Line2D = None  # type: ignore
    _HAVE_MATPLOTLIB = False


def _latest_run_dir(runs_root: Path = Path("runs")) -> Path:
    if not runs_root.exists():
        raise SystemExit(f"No runs directory found at {runs_root}")
    candidates = sorted(p for p in runs_root.glob("xor_*") if p.is_dir())
    if not candidates:
        raise SystemExit(f"No xor runs found under {runs_root}")
    return candidates[-1]


def _iter_code_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("iter_*.py"))


def _load_dataset_points(run_dir: Path) -> list[tuple[float, float, int]]:
    dataset_path = run_dir / "dataset.jsonl"
    if not dataset_path.exists():
        return []
    points: list[tuple[float, float, int]] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        points.append((float(obj["x0"]), float(obj["x1"]), int(obj["y"])))
    return points


def _load_attempt_rewards(run_dir: Path) -> dict[int, float]:
    path = run_dir / "attempts.jsonl"
    if not path.exists():
        return {}
    out: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        it = int(obj.get("iteration"))
        out[it] = float(obj.get("reward", 0.0))
    return out


def _iter_number_from_name(name: str) -> int | None:
    # iter_003.py -> 3
    try:
        stem = Path(name).stem
        _, num = stem.split("_", 1)
        return int(num)
    except Exception:
        return None


def _pred_to_float01(pred: Any) -> float:
    # Legacy helper (kept for compatibility).
    try:
        v = int(pred)
    except Exception:
        return float("nan")
    if v in (0, 1):
        return float(v)
    return float("nan")


def _pred_to_float(pred: Any) -> float:
    # Visualize raw model output as a continuous field (more "honest" even if spec is int).
    try:
        return float(pred)
    except Exception:
        return float("nan")


def _pred_to_class01(pred: Any) -> float:
    # Match the reward function: int-cast and accept only {0,1}; otherwise treat as invalid.
    try:
        v = int(pred)
    except Exception:
        return float("nan")
    if v in (0, 1):
        return float(v)
    return float("nan")


def visualize_run(
    run_dir: Path,
    *,
    grid: int = 200,
    x_min: float = -1.0,
    x_max: float = 1.0,
    y_min: float = -1.0,
    y_max: float = 1.0,
    timeout_s: float = 60.0,
    raw: bool = False,
) -> Path:
    if not _HAVE_MATPLOTLIB:
        raise SystemExit(
            "matplotlib is required for the combined subplot visualization. "
            "Install it (e.g. `uv sync`) and re-run."
        )
    assert np is not None and plt is not None

    run_dir = run_dir.resolve()
    code_files = _iter_code_files(run_dir)
    if not code_files:
        raise SystemExit(f"No `iter_*.py` files found in {run_dir}")

    spec = FunctionSpec(input_types=("float", "float"), output_types=("int",))
    dataset_points = _load_dataset_points(run_dir)
    rewards = _load_attempt_rewards(run_dir)

    xx_lin = np.linspace(x_min, x_max, grid)
    yy_lin = np.linspace(y_min, y_max, grid)
    xx, yy = np.meshgrid(xx_lin, yy_lin)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    grid_inputs = grid_points.astype(float).tolist()

    n = len(code_files)
    ncols = max(1, int(math.ceil(math.sqrt(n))))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    # First pass: evaluate all iterations so we can pick a consistent color scale.
    evaluated: list[tuple[Path, str, Any]] = []
    all_values: list[float] = []

    for code_path in code_files:
        it = _iter_number_from_name(code_path.name)
        title = code_path.name
        if it is not None and it in rewards:
            title = f"iter {it:03d} (reward={rewards[it]:.3f})"

        code = code_path.read_text(encoding="utf-8")
        try:
            validate_sandboxed_code(code)
            validate_generated_code(code, spec)
            run_result = run_code_main_batch(code, batch_inputs=grid_inputs, timeout_s=timeout_s)
            if not run_result.get("ok"):
                raise RuntimeError(str(run_result.get("error") or "run_failed"))
            preds = run_result.get("results", [])
            if raw:
                z = np.asarray([_pred_to_float(p) for p in preds], dtype=float).reshape((grid, grid))
            else:
                z = np.asarray([_pred_to_class01(p) for p in preds], dtype=float).reshape((grid, grid))
            finite = z[np.isfinite(z)]
            if finite.size:
                all_values.extend(finite.tolist())
            evaluated.append((code_path, title, z))
        except Exception as e:
            evaluated.append((code_path, title, e))

    if raw:
        if all_values:
            vmin = float(np.nanpercentile(np.asarray(all_values, dtype=float), 5))
            vmax = float(np.nanpercentile(np.asarray(all_values, dtype=float), 95))
            if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
                vmin, vmax = float(np.nanmin(all_values)), float(np.nanmax(all_values))
        else:
            vmin, vmax = 0.0, 1.0
        if vmin == vmax:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        # Discrete classes; keep a fixed scale.
        norm = Normalize(vmin=0.0, vmax=1.0)

    for ax in axes.ravel():
        ax.set_axis_off()

    mappable = None

    for idx, (code_path, title, result) in enumerate(evaluated):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_axis_on()

        if isinstance(result, Exception):
            ax.text(0.5, 0.5, f"error:\n{result}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_aspect("equal", "box")
            continue

        z = result
        # Heatmap of model outputs across the input space.
        cmap = plt.get_cmap("bwr").copy()
        cmap.set_bad(color=(0.9, 0.9, 0.9, 1.0))
        im = ax.imshow(
            z,
            origin="lower",
            extent=(x_min, x_max, y_min, y_max),
            cmap=cmap,
            norm=norm,
            alpha=0.35,
            interpolation="bilinear",
        )
        mappable = im

        if dataset_points:
            xs0 = np.asarray([p[0] for p in dataset_points], dtype=float)
            xs1 = np.asarray([p[1] for p in dataset_points], dtype=float)
            ys = np.asarray([p[2] for p in dataset_points], dtype=int)
            ax.scatter(xs0[ys == 0], xs1[ys == 0], c="blue", edgecolors="k", s=26)
            ax.scatter(xs0[ys == 1], xs1[ys == 1], c="red", edgecolors="k", s=26)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title)
        ax.set_aspect("equal", "box")

    # Layout: reserve a right gutter for the colorbar and a top band for title/legend.
    fig.suptitle(run_dir.name, y=0.995)
    fig.tight_layout(rect=(0.0, 0.03, 0.88, 0.93))

    if mappable is not None:
        cbar = fig.colorbar(
            mappable,
            ax=axes.ravel().tolist(),
            shrink=0.65,
            pad=0.02,
        )
        cbar.set_label("model raw output" if raw else "predicted class (0/1; gray=invalid)")

    # Put the legend in the top band (outside axes) so it never covers subplots or the colorbar.
    if dataset_points:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markeredgecolor="k",
                markersize=8,
                linestyle="None",
                label="true y=0",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markeredgecolor="k",
                markersize=8,
                linestyle="None",
                label="true y=1",
            ),
        ]
        # Center the legend over the axes region (left 88% of the figure).
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.44, 0.965),
            ncol=2,
            frameon=True,
        )
    out_path = run_dir / "all_boundaries.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Path like runs/xor_YYYYMMDDTHHMMSSZ (omit to use latest)",
    )
    p.add_argument("--grid", type=int, default=200)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument(
        "--raw",
        action="store_true",
        help="Visualize raw numeric outputs instead of reward-parsed class labels (0/1).",
    )
    args = p.parse_args()

    if args.grid < 25 or args.grid > 600:
        raise SystemExit("`--grid` must be an integer in 25..600")

    run_dir = args.run_dir or _latest_run_dir()
    out = visualize_run(run_dir, grid=args.grid, timeout_s=args.timeout, raw=args.raw)
    print(out)


if __name__ == "__main__":
    main()
