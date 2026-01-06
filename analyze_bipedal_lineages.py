#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _load_regular_history(run_dir: Path, max_score: float = 15.0) -> List[Tuple[int, float]]:
    """
    Load running-best score_a over iterations from a regular bipedal run.
    """
    attempts_path = run_dir / "attempts.jsonl"
    if not attempts_path.exists():
        raise FileNotFoundError(f"Missing attempts.jsonl in {run_dir}")

    best = float("-inf")
    points: List[Tuple[int, float]] = []
    with attempts_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            it = obj.get("iteration")
            score = obj.get("score_a")
            if not isinstance(it, int):
                continue
            try:
                val = float(score)
            except Exception:
                continue
            if val > max_score:
                # Ignore obviously bad / outlier scores entirely.
                continue
            if val > best:
                best = val
            points.append((it, best))
    points.sort(key=lambda t: t[0])
    return points


def _load_islands_lineages(
    run_dir: Path,
) -> Tuple[Dict[int, List[Tuple[int, float]]], List[Tuple[int, float]]]:
    """
    For an islands run:
      - per-lineage running-best score_a over iterations (from candidates.jsonl)
      - global canonical running-best score_a (from attempts.jsonl)
    """
    candidates_path = run_dir / "candidates.jsonl"
    attempts_path = run_dir / "attempts.jsonl"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing candidates.jsonl in {run_dir}")
    if not attempts_path.exists():
        raise FileNotFoundError(f"Missing attempts.jsonl in {run_dir}")

    # Per-lineage, per-iteration best for that iteration.
    per_lineage_iter: Dict[int, Dict[int, float]] = {}
    with candidates_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            it = obj.get("iteration")
            lid = obj.get("lineage_id")
            sc = obj.get("score_a")
            if not isinstance(it, int) or not isinstance(lid, int):
                continue
            try:
                val = float(sc)
            except Exception:
                continue
            if val > 15.0:
                # Ignore obviously bad / outlier scores entirely.
                continue
            d = per_lineage_iter.setdefault(lid, {})
            prev = d.get(it, float("-inf"))
            if val > prev:
                d[it] = val

    per_lineage_running: Dict[int, List[Tuple[int, float]]] = {}
    for lid, iter_scores in per_lineage_iter.items():
        best = float("-inf")
        pts: List[Tuple[int, float]] = []
        for it in sorted(iter_scores.keys()):
            val = float(iter_scores[it])
            if val > best:
                best = val
            pts.append((it, best))
        per_lineage_running[lid] = pts

    # Global canonical running-best from attempts.jsonl.
    global_curve = _load_regular_history(run_dir, max_score=15.0)
    return per_lineage_running, global_curve


def _latest_runs(prefix: str, limit: int = 10, exclude_prefix: str | None = None) -> List[Path]:
    runs_root = Path("runs")
    candidates: List[Path] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith(prefix):
            continue
        if exclude_prefix is not None and name.startswith(exclude_prefix):
            continue
        candidates.append(p)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Plot running-best distance (score_a) for regular vs islands bipedal runs.\n"
            "- By default, uses the last 10 regular and last 10 islands runs.\n"
            "- You can also pass a single pair of runs explicitly."
        )
    )
    ap.add_argument(
        "--regular-run",
        type=Path,
        required=False,
        help="Path to a regular bipedal run directory (e.g. runs/bipedal_YYYYMMDDTHHMMSSZ).",
    )
    ap.add_argument(
        "--islands-run",
        type=Path,
        required=False,
        help="Path to an islands bipedal run directory (e.g. runs/bipedal_islands_YYYYMMDDTHHMMSSZ).",
    )
    ap.add_argument(
        "--use-last-experiment",
        action="store_true",
        help="If set, ignore --regular-run/--islands-run and use the last 10 runs of each type.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("bipedal_lineages_comparison.png"),
        help="Output PNG path for the plot.",
    )
    args = ap.parse_args()

    if args.use_last_experiment:
        # Regular runs: plain bipedal_*, explicitly excluding bipedal_islands_*.
        regular_runs = _latest_runs(
            "bipedal_", limit=10, exclude_prefix="bipedal_islands_"
        )
        islands_runs = _latest_runs("bipedal_islands_", limit=10)
        if not regular_runs or not islands_runs:
            raise SystemExit(
                "Could not find enough runs under runs/ for regular or islands."
            )
    else:
        if args.regular_run is None or args.islands_run is None:
            ap.error(
                "Provide both --regular-run and --islands-run, or use --use-last-experiment."
            )
        regular_runs = [args.regular_run]
        islands_runs = [args.islands_run]

    # Keep names with curves so we can debug exact overlaps.
    regular_curves: List[Tuple[str, List[Tuple[int, float]]]] = []
    for run in regular_runs:
        curve = _load_regular_history(run, max_score=15.0)
        regular_curves.append((f"regular:{run.name}", curve))

    islands_lineages_all: List[Tuple[str, List[Tuple[int, float]]]] = []
    islands_global_curves: List[Tuple[str, List[Tuple[int, float]]]] = []
    for run in islands_runs:
        per_lineage, global_curve = _load_islands_lineages(run)
        islands_global_curves.append((f"islands_canonical:{run.name}", global_curve))
        for lid, pts in per_lineage.items():
            islands_lineages_all.append(
                (f"islands_lineage:{run.name}:lid={lid}", pts)
            )

    # Debug: check for exactly identical trajectories.
    def _curve_key(curve: List[Tuple[int, float]]) -> Tuple[Tuple[int, float], ...]:
        return tuple((int(x), float(y)) for x, y in curve)

    all_named_curves: List[Tuple[str, List[Tuple[int, float]]]] = (
        regular_curves + islands_global_curves + islands_lineages_all
    )
    seen: Dict[Tuple[Tuple[int, float], ...], List[str]] = {}
    for name, pts in all_named_curves:
        if not pts:
            continue
        key = _curve_key(pts)
        seen.setdefault(key, []).append(name)

    overlaps = [names for names in seen.values() if len(names) > 1]
    if overlaps:
        print("WARNING: found exactly overlapping trajectories:")
        for group in overlaps:
            print("  - " + ", ".join(group))
    else:
        print("No exactly overlapping trajectories found.")

    # Prepare summary statistics for box plot: best score per lineage/run.
    regular_best: List[float] = []
    for _, curve in regular_curves:
        if not curve:
            continue
        _, ys = zip(*curve)
        regular_best.append(max(ys))

    islands_best: List[float] = []
    for _, pts in islands_lineages_all:
        if not pts:
            continue
        _, ys = zip(*pts)
        islands_best.append(max(ys))

    # Two subplots: left = trajectories, right = histogram of best scores.
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        sharey=False,
        gridspec_kw={"width_ratios": [2.5, 1.0]},
    )

    # Use explicit colors to avoid any style confusion.
    islands_color = "#FF7F0E"  # soft orange
    regular_color = "#1F77B4"  # soft blue

    # Plot islands lineages (from islands runs) as thin orange lines.
    first_islands = True
    for _, pts in islands_lineages_all:
        if not pts:
            continue
        xs, ys = zip(*pts)
        label = "SimpleIslandShinka" if first_islands else None
        ax_left.plot(xs, ys, color=islands_color, linewidth=0.9, alpha=0.25, label=label)
        first_islands = False

    # Plot regular "lineages" (each regular run's trajectory) as slightly bolder blue lines.
    first_regular = True
    for _, curve in regular_curves:
        if not curve:
            continue
        xs_r, ys_r = zip(*curve)
        label = "SimpleShinka" if first_regular else None
        ax_left.plot(
            xs_r,
            ys_r,
            color=regular_color,
            linewidth=1.1,
            alpha=0.3,
            label=label,
        )
        first_regular = False

    ax_left.set_xlabel("Iteration")
    ax_left.set_ylabel("Running best distance")
    ax_left.grid(False)
    ax_left.legend()

    # Histogram: distribution of best scores for each method.
    bins = 15
    ax_right.hist(
        regular_best,
        bins=bins,
        color=regular_color,
        alpha=0.5,
        label="SimpleShinka",
    )
    ax_right.hist(
        islands_best,
        bins=bins,
        color=islands_color,
        alpha=0.5,
        label="SimpleIslandShinka",
    )
    ax_right.set_xlabel("Best distance")
    ax_right.set_ylabel("Count")
    ax_right.grid(False)
    ax_right.legend()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
