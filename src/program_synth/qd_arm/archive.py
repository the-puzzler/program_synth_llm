from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


Vec3 = tuple[float, float, float]
Cell3 = tuple[int, int, int]


@dataclass(frozen=True)
class ArchiveConfig:
    bins_xyz: tuple[int, int, int]
    min_xyz: Vec3
    max_xyz: Vec3

    def validate(self) -> None:
        bx, by, bz = self.bins_xyz
        if bx < 1 or by < 1 or bz < 1:
            raise ValueError("All bin counts must be >= 1.")
        for lo, hi in zip(self.min_xyz, self.max_xyz):
            if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                raise ValueError("Archive bounds must be finite and satisfy max > min for all axes.")


@dataclass
class Elite:
    cell: Cell3
    descriptor_xyz: Vec3
    quality: float
    code: str
    iteration: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class InsertResult:
    inserted: bool
    reason: str
    replaced_quality: float | None
    cell: Cell3


class MapElitesArchive3D:
    def __init__(self, config: ArchiveConfig) -> None:
        config.validate()
        self.config = config
        self._cells: dict[Cell3, Elite] = {}

    @property
    def cells(self) -> dict[Cell3, Elite]:
        return self._cells

    def __len__(self) -> int:
        return len(self._cells)

    def total_cells(self) -> int:
        bx, by, bz = self.config.bins_xyz
        return bx * by * bz

    def coverage(self) -> float:
        return float(len(self._cells) / max(1, self.total_cells()))

    def qd_score(self) -> float:
        return float(sum(e.quality for e in self._cells.values()))

    def best_elite(self) -> Elite | None:
        if not self._cells:
            return None
        return max(self._cells.values(), key=lambda e: e.quality)

    def descriptor_to_cell(self, descriptor_xyz: Vec3) -> Cell3:
        ix = self._bin_one(descriptor_xyz[0], 0)
        iy = self._bin_one(descriptor_xyz[1], 1)
        iz = self._bin_one(descriptor_xyz[2], 2)
        return (ix, iy, iz)

    def _bin_one(self, value: float, axis: int) -> int:
        bins = self.config.bins_xyz[axis]
        lo = self.config.min_xyz[axis]
        hi = self.config.max_xyz[axis]
        if not math.isfinite(value):
            raise ValueError(f"Descriptor axis {axis} is not finite.")
        ratio = (value - lo) / (hi - lo)
        ratio = min(1.0, max(0.0, ratio))
        idx = int(math.floor(ratio * bins))
        if idx >= bins:
            idx = bins - 1
        return idx

    def insert(
        self,
        *,
        descriptor_xyz: Vec3,
        quality: float,
        code: str,
        iteration: int,
        metadata: dict[str, Any] | None = None,
    ) -> InsertResult:
        if not math.isfinite(quality):
            raise ValueError("Quality must be finite.")

        cell = self.descriptor_to_cell(descriptor_xyz)
        current = self._cells.get(cell)
        if current is None:
            self._cells[cell] = Elite(
                cell=cell,
                descriptor_xyz=tuple(float(x) for x in descriptor_xyz),
                quality=float(quality),
                code=str(code),
                iteration=int(iteration),
                metadata=dict(metadata or {}),
            )
            return InsertResult(inserted=True, reason="new_cell", replaced_quality=None, cell=cell)

        if quality > current.quality:
            replaced_quality = current.quality
            self._cells[cell] = Elite(
                cell=cell,
                descriptor_xyz=tuple(float(x) for x in descriptor_xyz),
                quality=float(quality),
                code=str(code),
                iteration=int(iteration),
                metadata=dict(metadata or {}),
            )
            return InsertResult(inserted=True, reason="improved_cell", replaced_quality=replaced_quality, cell=cell)

        return InsertResult(inserted=False, reason="not_better_than_cell_elite", replaced_quality=current.quality, cell=cell)

    def random_elite(self, rng_seed: int) -> Elite | None:
        if not self._cells:
            return None
        import random

        rng = random.Random(rng_seed)
        return rng.choice(list(self._cells.values()))

    def random_two_elites(self, rng_seed: int) -> tuple[Elite, Elite] | None:
        if len(self._cells) < 2:
            return None
        import random

        rng = random.Random(rng_seed)
        picks = rng.sample(list(self._cells.values()), 2)
        return picks[0], picks[1]

    def to_summary(self) -> dict[str, Any]:
        best = self.best_elite()
        return {
            "occupied_cells": len(self._cells),
            "total_cells": self.total_cells(),
            "coverage": self.coverage(),
            "qd_score": self.qd_score(),
            "best_quality": (best.quality if best is not None else None),
            "config": {
                "bins_xyz": list(self.config.bins_xyz),
                "min_xyz": list(self.config.min_xyz),
                "max_xyz": list(self.config.max_xyz),
            },
        }


def append_archive_event_jsonl(path: Path, *, iteration: int, elite: Elite, reason: str) -> None:
    rec = {
        "iteration": int(iteration),
        "event": "inserted",
        "reason": str(reason),
        "cell": list(elite.cell),
        "descriptor_xyz": [float(x) for x in elite.descriptor_xyz],
        "quality": float(elite.quality),
        "elite_iteration": int(elite.iteration),
        "metadata": elite.metadata,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def append_archive_snapshot_json(path: Path, archive: MapElitesArchive3D) -> None:
    path.write_text(json.dumps(archive.to_summary(), indent=2) + "\n", encoding="utf-8")

