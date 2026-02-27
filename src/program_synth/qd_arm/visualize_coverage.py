from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_run_dir(runs_root: Path = Path("runs")) -> Path:
    candidates = sorted(p for p in runs_root.glob("qd_arm_xyz_*") if p.is_dir())
    if not candidates:
        raise SystemExit(f"No qd_arm runs found under {runs_root}")
    return candidates[-1]


def _load_archive_config(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Missing meta.json in {run_dir}")
    meta = _load_json(meta_path)
    if not isinstance(meta, dict):
        raise SystemExit(f"Invalid meta.json format in {run_dir}")
    archive = meta.get("archive")
    if not isinstance(archive, dict):
        raise SystemExit(f"meta.json missing archive config in {run_dir}")
    bins = archive.get("bins_xyz")
    min_xyz = archive.get("min_xyz")
    max_xyz = archive.get("max_xyz")
    if (
        not isinstance(bins, list)
        or len(bins) != 3
        or not isinstance(min_xyz, list)
        or len(min_xyz) != 3
        or not isinstance(max_xyz, list)
        or len(max_xyz) != 3
    ):
        raise SystemExit(f"Invalid archive config in {meta_path}")
    return {
        "bins_xyz": [int(x) for x in bins],
        "min_xyz": [float(x) for x in min_xyz],
        "max_xyz": [float(x) for x in max_xyz],
    }


def _load_final_elites(run_dir: Path) -> list[dict[str, Any]]:
    events_path = run_dir / "archive_events.jsonl"
    if not events_path.exists():
        raise SystemExit(f"Missing archive_events.jsonl in {run_dir}")
    by_cell: dict[tuple[int, int, int], dict[str, Any]] = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        cell = rec.get("cell")
        desc = rec.get("descriptor_xyz")
        q = rec.get("quality")
        if (
            not isinstance(cell, list)
            or len(cell) != 3
            or not isinstance(desc, list)
            or len(desc) != 3
            or not isinstance(q, (int, float))
        ):
            continue
        cell_key = (int(cell[0]), int(cell[1]), int(cell[2]))
        by_cell[cell_key] = {
            "cell": cell_key,
            "descriptor_xyz": (float(desc[0]), float(desc[1]), float(desc[2])),
            "quality": float(q),
        }
    return list(by_cell.values())


def _cell_center(cell: tuple[int, int, int], bins_xyz: list[int], min_xyz: list[float], max_xyz: list[float]) -> tuple[float, float, float]:
    out: list[float] = []
    for i in range(3):
        bins = bins_xyz[i]
        lo = min_xyz[i]
        hi = max_xyz[i]
        w = (hi - lo) / bins
        out.append(lo + (cell[i] + 0.5) * w)
    return (out[0], out[1], out[2])


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize 3D MAP-Elites coverage for qd_arm runs.")
    ap.add_argument("--run-dir", type=Path, default=None, help="Run directory (defaults to latest runs/qd_arm_xyz_*).")
    ap.add_argument("--out", type=Path, default=None, help="Output image path (default: <run-dir>/coverage_xyz.png).")
    ap.add_argument(
        "--mode",
        choices=("descriptor", "cell_center"),
        default="descriptor",
        help="Plot elite descriptor points or occupied-cell centers.",
    )
    ap.add_argument("--title", type=str, default=None, help="Optional custom figure title.")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve() if args.run_dir is not None else _latest_run_dir().resolve()
    cfg = _load_archive_config(run_dir)
    elites = _load_final_elites(run_dir)

    if not elites:
        raise SystemExit(f"No archive insertion events found in {run_dir}")

    try:
        import matplotlib.pyplot as plt  # type: ignore
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import numpy as np  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing plotting deps (matplotlib/numpy): {e}")

    bins_xyz = cfg["bins_xyz"]
    min_xyz = cfg["min_xyz"]
    max_xyz = cfg["max_xyz"]

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    qs: list[float] = []
    for e in elites:
        q = float(e["quality"])
        cell = e["cell"]
        if args.mode == "cell_center":
            x, y, z = _cell_center(cell, bins_xyz, min_xyz, max_xyz)
        else:
            x, y, z = e["descriptor_xyz"]
        xs.append(float(x))
        ys.append(float(y))
        zs.append(float(z))
        qs.append(q)

    q_arr = np.asarray(qs, dtype=float)
    vmin = float(np.nanmin(q_arr))
    vmax = float(np.nanmax(q_arr))
    if vmin == vmax:
        vmax = vmin + 1e-6

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, 1])

    sc = ax3d.scatter(xs, ys, zs, c=qs, cmap="viridis", s=32, alpha=0.9, vmin=vmin, vmax=vmax)
    cax = fig.add_axes([0.92, 0.18, 0.014, 0.66])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Elite quality")

    ax3d.set_xlim(min_xyz[0], max_xyz[0])
    ax3d.set_ylim(min_xyz[1], max_xyz[1])
    ax3d.set_zlim(min_xyz[2], max_xyz[2])
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("3D Coverage")

    occupied = len(elites)
    total = int(bins_xyz[0] * bins_xyz[1] * bins_xyz[2])
    coverage = occupied / max(1, total)
    mode_label = "descriptor points" if args.mode == "descriptor" else "occupied cell centers"
    bx, by, bz = bins_xyz
    # Top-down projection: x-y.
    ax_top.scatter(xs, ys, c=qs, cmap="viridis", s=22, alpha=0.9, vmin=vmin, vmax=vmax)
    ax_top.set_xlim(min_xyz[0], max_xyz[0])
    ax_top.set_ylim(min_xyz[1], max_xyz[1])
    ax_top.set_xlabel("x")
    ax_top.set_ylabel("y")
    ax_top.set_title(f"Top View (x-y), grid {bx}x{by}")
    ax_top.set_aspect("auto")

    # Side projection: x-z.
    ax_side.scatter(xs, zs, c=qs, cmap="viridis", s=22, alpha=0.9, vmin=vmin, vmax=vmax)
    ax_side.set_xlim(min_xyz[0], max_xyz[0])
    ax_side.set_ylim(min_xyz[2], max_xyz[2])
    ax_side.set_xlabel("x")
    ax_side.set_ylabel("z")
    ax_side.set_title(f"Side View (x-z), grid {bx}x{bz}")
    ax_side.set_aspect("auto")

    # Draw archive grid lines in the 2D views.
    x_edges = np.linspace(min_xyz[0], max_xyz[0], bx + 1)
    y_edges = np.linspace(min_xyz[1], max_xyz[1], by + 1)
    z_edges = np.linspace(min_xyz[2], max_xyz[2], bz + 1)

    for x in x_edges:
        ax_top.axvline(float(x), color="#b0b0b0", lw=0.5, alpha=0.5, zorder=0)
        ax_side.axvline(float(x), color="#b0b0b0", lw=0.5, alpha=0.5, zorder=0)
    for y in y_edges:
        ax_top.axhline(float(y), color="#b0b0b0", lw=0.5, alpha=0.5, zorder=0)
    for z in z_edges:
        ax_side.axhline(float(z), color="#b0b0b0", lw=0.5, alpha=0.5, zorder=0)

    # Edge ticks show full bin extent, so scale is visible even for sparse points.
    ax_top.set_xticks(x_edges)
    ax_top.set_yticks(y_edges)
    ax_side.set_xticks(x_edges)
    ax_side.set_yticks(z_edges)
    for ax in (ax_top, ax_side):
        ax.tick_params(axis="both", labelsize=7)

    ax3d.view_init(elev=22, azim=-52)
    title = args.title or f"QD Arm Coverage ({mode_label}) | {occupied}/{total} cells ({coverage:.2%})"
    fig.suptitle(title)

    out_path = args.out.resolve() if args.out is not None else (run_dir / "coverage_xyz.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 0.9, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(
        json.dumps(
            {
                "ok": True,
                "run_dir": str(run_dir),
                "out_path": str(out_path),
                "occupied_cells": occupied,
                "total_cells": total,
                "coverage": coverage,
                "mode": args.mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
