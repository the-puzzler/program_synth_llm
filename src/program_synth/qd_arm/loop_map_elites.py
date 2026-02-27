from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import random
from pathlib import Path
from typing import Any

from program_synth.ai_code_env import clean_generated_code, extract_python_code
from program_synth.call_ai_utils import call_ai
from program_synth.qd_arm.archive import (
    ArchiveConfig,
    MapElitesArchive3D,
    append_archive_event_jsonl,
    append_archive_snapshot_json,
)
from program_synth.qd_arm.evaluate import evaluate_candidate_code
from program_synth.utils import jsonable


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"qd_arm_xyz_{ts}"


def _obs_contract_text() -> str:
    return (
        "Observation vector format (len=10):\n"
        "- obs[0:3]   = achieved_goal_xyz\n"
        "- obs[3:6]   = delta_achieved_goal_xyz (current minus previous step)\n"
        "- obs[6:9]   = goal_rel_xyz = desired_goal_xyz - achieved_goal_xyz\n"
        "- obs[9]     = tau in [0,1] (normalized primitive time)\n"
        "Output format:\n"
        "- Return list of exactly 3 floats in [-1,1]: [ax, ay, az]\n"
    )


def _build_mutate_prompt(parent_code: str | None) -> str:
    parent = parent_code.strip() if isinstance(parent_code, str) else ""
    parent_block = (
        f"Parent controller to mutate:\n```python\n{parent}\n```\n"
        if parent
        else "No parent controller provided. Create a new controller from scratch.\n"
    )
    return (
        "Output exactly ONE fenced Python code block and nothing else.\n"
        "Do not write comments.\n"
        "Define exactly:\n"
        "def main(obs: list[float]) -> list[float]:\n"
        f"{_obs_contract_text()}\n"
        "Hard constraints:\n"
        "- Do not read/write files, do not use network, do not print.\n"
        "- Imports: math only.\n"
        "- Must always return a list with length 3.\n"
        "\n"
        "Task:\n"
        "- Produce a behaviorally different controller (not a near copy).\n"
        "- Make at least ONE substantial structural change from the parent:\n"
        "  (a) add/delete an intermediate feature computation block, OR\n"
        "  (b) add/delete an explicit phase split using tau (2+ regimes), OR\n"
        "  (c) add/delete one loop/comprehension over axes, OR\n"
        "  (d) add/delete one nonlinear gating term that mixes at least two obs groups.\n"
        "- Avoid trivial coefficient-only edits; change computation structure.\n"
        "- Keep output contract identical and robust to any finite float inputs.\n"
        "\n"
        f"{parent_block}"
    )


def _build_recombine_prompt(code_a: str, code_b: str) -> str:
    return (
        "Output exactly ONE fenced Python code block and nothing else.\n"
        "Do not write comments.\n"
        "Define exactly:\n"
        "def main(obs: list[float]) -> list[float]:\n"
        f"{_obs_contract_text()}\n"
        "Hard constraints:\n"
        "- Do not read/write files, do not use network, do not print.\n"
        "- Imports: math only.\n"
        "- Must always return a list with length 3.\n"
        "\n"
        "Task:\n"
        "- Recombine useful ideas from both parents into one coherent controller.\n"
        "- Do not output either parent unchanged.\n"
        "- Include at least two distinct motifs, one from each parent, in different code regions.\n"
        "- Use at least one add/delete structural edit, not only coefficient tweaks.\n"
        "\n"
        "Parent A:\n"
        f"```python\n{code_a.strip()}\n```\n\n"
        "Parent B:\n"
        f"```python\n{code_b.strip()}\n```\n"
    )


def _seed_programs() -> list[str]:
    return [
        """
def main(obs: list[float]) -> list[float]:
    gx, gy, gz = obs[6], obs[7], obs[8]
    return [gx, gy, gz]
""".strip(),
        """
def main(obs: list[float]) -> list[float]:
    gx, gy, gz = obs[6], obs[7], obs[8]
    tau = obs[9]
    s = 0.25 + 0.75 * tau
    return [s * gx, s * gy, s * gz]
""".strip(),
        """
def main(obs: list[float]) -> list[float]:
    ax = 0.8 * obs[6] - 0.4 * obs[3]
    ay = 0.8 * obs[7] - 0.4 * obs[4]
    az = 0.8 * obs[8] - 0.4 * obs[5]
    return [ax, ay, az]
""".strip(),
        """
def main(obs: list[float]) -> list[float]:
    import math
    gx, gy, gz = obs[6], obs[7], obs[8]
    tau = obs[9]
    k = 0.9 + 0.3 * math.sin(6.28318 * tau)
    return [k * gx, k * gy, k * gz]
""".strip(),
        """
def main(obs: list[float]) -> list[float]:
    import math
    gx, gy, gz = obs[6], obs[7], obs[8]
    vx, vy, vz = obs[3], obs[4], obs[5]
    r = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-6
    scale = min(1.0, 1.2 / r)
    return [scale * (gx - 0.2 * vx), scale * (gy - 0.2 * vy), scale * (gz - 0.2 * vz)]
""".strip(),
        """
def main(obs: list[float]) -> list[float]:
    gx, gy, gz = obs[6], obs[7], obs[8]
    vx, vy, vz = obs[3], obs[4], obs[5]
    tau = obs[9]
    if tau < 0.35:
        k = 1.2
        d = 0.05
    elif tau < 0.75:
        k = 0.9
        d = 0.20
    else:
        k = 0.5
        d = 0.35
    return [k * gx - d * vx, k * gy - d * vy, k * gz - d * vz]
""".strip(),
        """
def main(obs: list[float]) -> list[float]:
    import math
    gx, gy, gz = obs[6], obs[7], obs[8]
    vx, vy, vz = obs[3], obs[4], obs[5]
    tau = obs[9]
    kx = 0.8 + 0.4 * math.sin(9.0 * tau + 0.0)
    ky = 0.8 + 0.4 * math.sin(9.0 * tau + 2.1)
    kz = 0.8 + 0.4 * math.sin(9.0 * tau + 4.2)
    return [kx * gx - 0.15 * vx, ky * gy - 0.15 * vy, kz * gz - 0.15 * vz]
""".strip(),
        """
def main(obs: list[float]) -> list[float]:
    import statistics
    gx, gy, gz = obs[6], obs[7], obs[8]
    vx, vy, vz = obs[3], obs[4], obs[5]
    mags = [abs(gx), abs(gy), abs(gz)]
    s = statistics.fmean(mags) + 1e-6
    k = 0.6 + 0.8 * min(1.0, s / 0.2)
    return [k * gx - 0.25 * vx, k * gy - 0.25 * vy, k * gz - 0.25 * vz]
""".strip(),
        """
def main(obs: list[float]) -> list[float]:
    import math
    gx, gy, gz = obs[6], obs[7], obs[8]
    tau = obs[9]
    e = math.exp(-3.0 * tau)
    ax = (1.1 * gx) * e + 0.35 * gx
    ay = (1.1 * gy) * e + 0.35 * gy
    az = (1.1 * gz) * e + 0.35 * gz
    return [ax, ay, az]
""".strip(),
        """
def main(obs: list[float]) -> list[float]:
    import math
    gx, gy, gz = obs[6], obs[7], obs[8]
    vx, vy, vz = obs[3], obs[4], obs[5]
    tau = obs[9]
    r2 = gx * gx + gy * gy + gz * gz
    k = 1.4 / (1.0 + 6.0 * r2)
    d = 0.10 + 0.25 * (tau * tau)
    bx = math.tanh(k * gx - d * vx)
    by = math.tanh(k * gy - d * vy)
    bz = math.tanh(k * gz - d * vz)
    return [bx, by, bz]
""".strip(),
    ]


def _safe_parse_code(text: str) -> str:
    return clean_generated_code(extract_python_code(text or ""))


def _maybe_get_attr_or_key(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_usage_tokens(raw: Any) -> dict[str, int | float | None]:
    usage = _maybe_get_attr_or_key(raw, "usage")
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "provider_cost_usd": None}

    prompt = _maybe_get_attr_or_key(usage, "prompt_tokens")
    if prompt is None:
        prompt = _maybe_get_attr_or_key(usage, "input_tokens")

    completion = _maybe_get_attr_or_key(usage, "completion_tokens")
    if completion is None:
        completion = _maybe_get_attr_or_key(usage, "output_tokens")

    total = _maybe_get_attr_or_key(usage, "total_tokens")
    if total is None and isinstance(prompt, (int, float)) and isinstance(completion, (int, float)):
        total = int(prompt) + int(completion)

    def _as_int(x: Any) -> int | None:
        if isinstance(x, (int, float)):
            return int(x)
        return None

    def _as_float(x: Any) -> float | None:
        if isinstance(x, (int, float)):
            return float(x)
        return None

    # OpenRouter may provide exact cost in `usage.cost`.
    provider_cost = _maybe_get_attr_or_key(usage, "cost")
    if provider_cost is None:
        provider_cost = _maybe_get_attr_or_key(usage, "estimated_cost")

    return {
        "prompt_tokens": _as_int(prompt),
        "completion_tokens": _as_int(completion),
        "total_tokens": _as_int(total),
        "provider_cost_usd": _as_float(provider_cost),
    }


def _estimate_cost_usd(
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> float | None:
    if prompt_tokens is None and completion_tokens is None:
        return None
    p = float(prompt_tokens or 0)
    c = float(completion_tokens or 0)
    return (p / 1_000_000.0) * float(input_cost_per_1m) + (c / 1_000_000.0) * float(output_cost_per_1m)


def _softmax_weights(values: list[float], tau: float) -> list[float]:
    import math

    if not values:
        return []
    t = max(1e-6, float(tau))
    m = max(values)
    exps = [math.exp((v - m) / t) for v in values]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(values)] * len(values)
    return [e / s for e in exps]


def _weighted_choice(items: list[Any], probs: list[float], rng: random.Random) -> Any:
    if not items:
        raise ValueError("items must be non-empty")
    if len(items) != len(probs):
        raise ValueError("items/probs length mismatch")
    total = sum(max(0.0, float(p)) for p in probs)
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    acc = 0.0
    for item, p in zip(items, probs):
        acc += max(0.0, float(p))
        if r <= acc:
            return item
    return items[-1]


def _sample_parent_cell_weighted(
    archive: MapElitesArchive3D,
    parent_scores: dict[tuple[int, int, int], float],
    *,
    rng: random.Random,
    tau_parent: float,
    eps_parent: float,
) -> tuple[int, int, int] | None:
    if len(archive.cells) == 0:
        return None
    cells = list(archive.cells.keys())
    scores = [float(parent_scores.get(c, 0.0)) for c in cells]
    soft = _softmax_weights(scores, tau=tau_parent)
    n = len(cells)
    eps = min(1.0, max(0.0, float(eps_parent)))
    mixed = [(1.0 - eps) * p + eps * (1.0 / n) for p in soft]
    return _weighted_choice(cells, mixed, rng)


def _sample_two_parent_cells_weighted(
    archive: MapElitesArchive3D,
    parent_scores: dict[tuple[int, int, int], float],
    *,
    rng: random.Random,
    tau_parent: float,
    eps_parent: float,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    if len(archive.cells) < 2:
        return None
    c1 = _sample_parent_cell_weighted(
        archive, parent_scores, rng=rng, tau_parent=tau_parent, eps_parent=eps_parent
    )
    if c1 is None:
        return None
    remaining = [c for c in archive.cells.keys() if c != c1]
    if not remaining:
        return None
    scores = [float(parent_scores.get(c, 0.0)) for c in remaining]
    soft = _softmax_weights(scores, tau=tau_parent)
    n = len(remaining)
    eps = min(1.0, max(0.0, float(eps_parent)))
    mixed = [(1.0 - eps) * p + eps * (1.0 / n) for p in soft]
    c2 = _weighted_choice(remaining, mixed, rng)
    return c1, c2


def _rename_main_function(code: str, new_name: str) -> str:
    try:
        tree = ast.parse(code)

        class _Renamer(ast.NodeTransformer):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                if node.name == "main":
                    node.name = new_name
                return self.generic_visit(node)

        tree = _Renamer().visit(tree)  # type: ignore[assignment]
        ast.fix_missing_locations(tree)
        out = ast.unparse(tree).strip() + "\n"
        if f"def {new_name}(" in out:
            return out
    except Exception:
        pass
    return code.replace("def main(", f"def {new_name}(", 1)


def _write_iteration_bundle(run_dir: Path, step: int, inserted_batch: list[dict[str, Any]]) -> None:
    if not inserted_batch:
        return
    lines: list[str] = []
    lines.append(f"# Inserted controllers for iteration {step}")
    lines.append(f"# Count: {len(inserted_batch)}")
    lines.append("")
    exported: list[str] = []
    for i, rec in enumerate(inserted_batch, start=1):
        fn_name = f"main_candidate_{i:03d}"
        exported.append(fn_name)
        mode = str(rec.get("mode") or "")
        cell = rec.get("insert_cell")
        quality = rec.get("quality")
        code = str(rec.get("code") or "")
        renamed = _rename_main_function(code, fn_name).strip()
        lines.append(
            f"# candidate_index={rec.get('candidate_index')} mode={mode} cell={cell} quality={quality}"
        )
        lines.append(renamed)
        lines.append("")
    lines.append(f"__all__ = {exported!r}")
    out_path = run_dir / f"iter_{step:04d}.py"
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _parse_cell_key(key: str) -> tuple[int, int, int] | None:
    parts = [p.strip() for p in str(key).split(",")]
    if len(parts) != 3:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return None


def _load_checkpoint_state(
    run_dir: Path,
    *,
    config: ArchiveConfig,
    default_seed_programs: list[str],
) -> dict[str, Any]:
    attempts_path = run_dir / "attempts.jsonl"
    if not attempts_path.exists():
        raise SystemExit(f"Checkpoint missing attempts.jsonl: {attempts_path}")

    archive = MapElitesArchive3D(config)
    parent_scores: dict[tuple[int, int, int], float] = {}
    max_iter = 0
    cumulative_prompt_tokens = 0
    cumulative_completion_tokens = 0
    cumulative_provider_cost_usd = 0.0
    cumulative_token_estimated_cost_usd = 0.0
    cumulative_cost_usd = 0.0

    for line in attempts_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue

        it = rec.get("iteration")
        if isinstance(it, int):
            max_iter = max(max_iter, it)

        # Rebuild cumulative usage/cost counters from historical attempts.
        usage = rec.get("llm_usage")
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            if isinstance(pt, int) and pt >= 0:
                cumulative_prompt_tokens += pt
            if isinstance(ct, int) and ct >= 0:
                cumulative_completion_tokens += ct
        p_cost = rec.get("llm_provider_cost_usd")
        if isinstance(p_cost, (int, float)):
            cumulative_provider_cost_usd += float(p_cost)
        est_cost = rec.get("llm_cost_estimate_usd")
        if isinstance(est_cost, (int, float)):
            cumulative_token_estimated_cost_usd += float(est_cost)
        total_cost = rec.get("llm_cost_usd")
        if isinstance(total_cost, (int, float)):
            cumulative_cost_usd += float(total_cost)

        # Restore latest parent score snapshot, if present.
        pss = rec.get("parent_score_snapshot")
        if isinstance(pss, dict):
            restored: dict[tuple[int, int, int], float] = {}
            for k, v in pss.items():
                ck = _parse_cell_key(str(k))
                if ck is None:
                    continue
                if isinstance(v, (int, float)):
                    restored[ck] = float(v)
            if restored:
                parent_scores = restored

        # Rebuild archive elite contents from accepted insertions.
        if rec.get("inserted") is not True:
            continue
        cell = rec.get("insert_cell")
        eval_out = rec.get("eval")
        if (
            not isinstance(cell, list)
            or len(cell) != 3
            or not isinstance(eval_out, dict)
            or not eval_out.get("ok")
        ):
            continue
        desc = eval_out.get("descriptor_xyz")
        quality = eval_out.get("quality")
        if (
            not isinstance(desc, list)
            or len(desc) != 3
            or not isinstance(quality, (int, float))
        ):
            continue

        code = rec.get("code")
        if not isinstance(code, str) or not code.strip():
            # Bootstrap rows in older logs may not contain `code`; recover from seed list by index.
            if rec.get("mode") == "bootstrap":
                idx = rec.get("candidate_index")
                if isinstance(idx, int) and 1 <= idx <= len(default_seed_programs):
                    code = default_seed_programs[idx - 1]
        if not isinstance(code, str) or not code.strip():
            continue

        archive.insert(
            descriptor_xyz=(float(desc[0]), float(desc[1]), float(desc[2])),
            quality=float(quality),
            code=code,
            iteration=int(it) if isinstance(it, int) else 0,
            metadata={
                "resumed": True,
                "mode": rec.get("mode"),
                "parent_cells": rec.get("parent_cells"),
                "candidate_index": rec.get("candidate_index"),
            },
        )

    return {
        "archive": archive,
        "parent_scores": parent_scores,
        "next_iteration": max_iter + 1,
        "cumulative_prompt_tokens": cumulative_prompt_tokens,
        "cumulative_completion_tokens": cumulative_completion_tokens,
        "cumulative_provider_cost_usd": cumulative_provider_cost_usd,
        "cumulative_token_estimated_cost_usd": cumulative_token_estimated_cost_usd,
        "cumulative_cost_usd": cumulative_cost_usd,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="QD arm experiment with MAP-Elites over final (x,y,z).")
    ap.add_argument(
        "--checkpoint-path",
        "--checkpoint_path",
        type=Path,
        default=None,
        help="Resume an existing run directory (e.g. runs/qd_arm_xyz_...).",
    )
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--env-id", type=str, default="FetchReach-v4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model", type=str, default="moonshotai/kimi-k2-0905")
    ap.add_argument("--batch-size", type=int, default=1, help="Candidates generated/evaluated per iteration.")
    ap.add_argument(
        "--input-cost-per-1m",
        type=float,
        default=0.0,
        help="Estimated USD cost per 1M input tokens for the selected model/provider.",
    )
    ap.add_argument(
        "--output-cost-per-1m",
        type=float,
        default=0.0,
        help="Estimated USD cost per 1M output tokens for the selected model/provider.",
    )
    ap.add_argument("--recombine-prob", type=float, default=0.2)
    ap.add_argument(
        "--parent-tau",
        type=float,
        default=3.0,
        help="Softmax temperature for parent-score-biased sampling (higher = flatter).",
    )
    ap.add_argument(
        "--parent-eps",
        type=float,
        default=0.15,
        help="Exploration mix for parent sampling; final probs=(1-eps)*softmax + eps*uniform.",
    )
    ap.add_argument(
        "--parent-score-decay",
        type=float,
        default=0.99,
        help="Per-iteration multiplicative decay on parent scores (0,1].",
    )
    ap.add_argument("--bins-x", type=int, default=16)
    ap.add_argument("--bins-y", type=int, default=16)
    ap.add_argument("--bins-z", type=int, default=16)
    ap.add_argument("--min-x", type=float, default=1.0)
    ap.add_argument("--min-y", type=float, default=0.3)
    ap.add_argument("--min-z", type=float, default=0.3)
    ap.add_argument("--max-x", type=float, default=1.7)
    ap.add_argument("--max-y", type=float, default=1.1)
    ap.add_argument("--max-z", type=float, default=1.0)
    ap.add_argument("--w-v", type=float, default=1.0)
    ap.add_argument("--w-e", type=float, default=0.10)
    ap.add_argument("--w-u", type=float, default=0.05)
    ap.add_argument("--w-err", type=float, default=0.25)
    args = ap.parse_args()

    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1.")
    if not (args.parent_tau > 0):
        raise SystemExit("--parent-tau must be > 0.")
    if not (0.0 <= args.parent_eps <= 1.0):
        raise SystemExit("--parent-eps must be in [0,1].")
    if not (0.0 < args.parent_score_decay <= 1.0):
        raise SystemExit("--parent-score-decay must be in (0,1].")
    if not (0.0 <= args.recombine_prob <= 1.0):
        raise SystemExit("--recombine-prob must be in [0,1].")

    # Runtime configuration (possibly overridden by checkpoint meta).
    env_id = str(args.env_id)
    seed = int(args.seed)
    max_steps = int(args.max_steps)
    timeout_s = float(args.timeout)
    temperature = float(args.temperature)
    model = str(args.model)
    batch_size = int(args.batch_size)
    recombine_prob = float(args.recombine_prob)
    parent_tau = float(args.parent_tau)
    parent_eps = float(args.parent_eps)
    parent_score_decay = float(args.parent_score_decay)
    input_cost_per_1m = float(args.input_cost_per_1m)
    output_cost_per_1m = float(args.output_cost_per_1m)
    w_v = float(args.w_v)
    w_e = float(args.w_e)
    w_u = float(args.w_u)
    w_err = float(args.w_err)

    run_dir: Path
    meta_path: Path
    attempts_path: Path
    archive_events_path: Path
    archive_summary_path: Path
    archive: MapElitesArchive3D
    parent_scores: dict[tuple[int, int, int], float]
    start_step: int
    cumulative_prompt_tokens = 0
    cumulative_completion_tokens = 0
    cumulative_cost_usd = 0.0
    cumulative_provider_cost_usd = 0.0
    cumulative_token_estimated_cost_usd = 0.0

    if args.checkpoint_path is not None:
        run_dir = args.checkpoint_path.expanduser().resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            raise SystemExit(f"`--checkpoint-path` must be an existing run directory: {run_dir}")
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            raise SystemExit(f"Missing meta.json in checkpoint: {run_dir}")
        try:
            meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise SystemExit(f"Could not parse {meta_path}: {e}")
        if not isinstance(meta_obj, dict):
            raise SystemExit(f"{meta_path} did not contain a JSON object.")

        # Use checkpoint settings to guarantee consistent continuation.
        env_id = str(meta_obj.get("env_id", env_id))
        seed = int(meta_obj.get("seed", seed))
        max_steps = int(meta_obj.get("max_steps", max_steps))
        timeout_s = float(meta_obj.get("timeout_s", timeout_s))
        temperature = float(meta_obj.get("temperature", temperature))
        model = str(meta_obj.get("model", model))
        batch_size = int(meta_obj.get("batch_size", batch_size))
        recombine_prob = float(meta_obj.get("recombine_prob", recombine_prob))

        pricing = meta_obj.get("pricing")
        if isinstance(pricing, dict):
            input_cost_per_1m = float(pricing.get("input_cost_per_1m", input_cost_per_1m))
            output_cost_per_1m = float(pricing.get("output_cost_per_1m", output_cost_per_1m))

        ps = meta_obj.get("parent_sampling")
        if isinstance(ps, dict):
            parent_tau = float(ps.get("tau", parent_tau))
            parent_eps = float(ps.get("eps", parent_eps))
            parent_score_decay = float(ps.get("score_decay", parent_score_decay))

        qw = meta_obj.get("quality_weights")
        if isinstance(qw, dict):
            w_v = float(qw.get("w_v", w_v))
            w_e = float(qw.get("w_e", w_e))
            w_u = float(qw.get("w_u", w_u))
            w_err = float(qw.get("w_err", w_err))

        arc = meta_obj.get("archive")
        if not isinstance(arc, dict):
            raise SystemExit(f"meta.json missing archive config in {run_dir}")
        try:
            config = ArchiveConfig(
                bins_xyz=tuple(int(x) for x in arc["bins_xyz"]),
                min_xyz=tuple(float(x) for x in arc["min_xyz"]),
                max_xyz=tuple(float(x) for x in arc["max_xyz"]),
            )
        except Exception as e:
            raise SystemExit(f"Invalid archive config in {meta_path}: {e}")

        attempts_path = run_dir / "attempts.jsonl"
        archive_events_path = run_dir / "archive_events.jsonl"
        archive_summary_path = run_dir / "archive_summary.json"

        state = _load_checkpoint_state(run_dir, config=config, default_seed_programs=_seed_programs())
        archive = state["archive"]
        parent_scores = state["parent_scores"]
        start_step = int(state["next_iteration"])
        cumulative_prompt_tokens = int(state["cumulative_prompt_tokens"])
        cumulative_completion_tokens = int(state["cumulative_completion_tokens"])
        cumulative_provider_cost_usd = float(state["cumulative_provider_cost_usd"])
        cumulative_token_estimated_cost_usd = float(state["cumulative_token_estimated_cost_usd"])
        cumulative_cost_usd = float(state["cumulative_cost_usd"])
        append_archive_snapshot_json(archive_summary_path, archive)
    else:
        run_dir = _default_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        attempts_path = run_dir / "attempts.jsonl"
        archive_events_path = run_dir / "archive_events.jsonl"
        archive_summary_path = run_dir / "archive_summary.json"
        meta_path = run_dir / "meta.json"

        config = ArchiveConfig(
            bins_xyz=(args.bins_x, args.bins_y, args.bins_z),
            min_xyz=(args.min_x, args.min_y, args.min_z),
            max_xyz=(args.max_x, args.max_y, args.max_z),
        )
        archive = MapElitesArchive3D(config)

        meta = {
            "experiment": "qd_arm_map_elites_xyz",
            "env_id": env_id,
            "seed": int(seed),
            "iterations": int(args.iterations),
            "max_steps": int(max_steps),
            "timeout_s": float(timeout_s),
            "temperature": float(temperature),
            "model": model,
            "batch_size": int(batch_size),
            "pricing": {
                "input_cost_per_1m": float(input_cost_per_1m),
                "output_cost_per_1m": float(output_cost_per_1m),
            },
            "recombine_prob": float(recombine_prob),
            "parent_sampling": {
                "tau": float(parent_tau),
                "eps": float(parent_eps),
                "score_decay": float(parent_score_decay),
                "reward_insert": 1.0,
                "penalty_no_insert": -1.0,
            },
            "quality_weights": {"w_v": w_v, "w_e": w_e, "w_u": w_u, "w_err": w_err},
            "archive": asdict(config),
            "obs_signature": "def main(obs: list[float]) -> list[float]  # obs len 10, output len 3",
            "artifacts": {
                "attempts_path": attempts_path.name,
                "archive_events_path": archive_events_path.name,
                "archive_summary_path": archive_summary_path.name,
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        # Bootstrap with a few deterministic seeds to avoid a cold empty archive.
        for idx, code in enumerate(_seed_programs(), start=1):
            eval_out = evaluate_candidate_code(
                code,
                env_id=env_id,
                seed=seed,
                max_steps=max_steps,
                timeout_s=timeout_s,
                w_v=w_v,
                w_e=w_e,
                w_u=w_u,
                w_err=w_err,
            )
            inserted = False
            insert_reason = None
            insert_cell = None
            if eval_out.get("ok"):
                desc = tuple(float(x) for x in eval_out["descriptor_xyz"])
                quality = float(eval_out["quality"])
                ins = archive.insert(
                    descriptor_xyz=desc, quality=quality, code=code, iteration=0, metadata={"source": "bootstrap_seed"}
                )
                inserted = ins.inserted
                insert_reason = ins.reason
                insert_cell = list(ins.cell)
                if inserted:
                    seed_path = run_dir / f"seed_{idx:03d}.py"
                    seed_path.write_text(code, encoding="utf-8")
                    elite = archive.cells[ins.cell]
                    append_archive_event_jsonl(archive_events_path, iteration=0, elite=elite, reason=ins.reason)
            rec = {
                "iteration": 0,
                "candidate_index": idx,
                "mode": "bootstrap",
                "inserted": inserted,
                "insert_reason": insert_reason,
                "insert_cell": insert_cell,
                "eval": eval_out,
            }
            with attempts_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(jsonable(rec)) + "\n")
        append_archive_snapshot_json(archive_summary_path, archive)
        parent_scores = {}
        start_step = 1

    # Validate effective runtime config (including checkpoint overrides).
    if batch_size < 1:
        raise SystemExit("Effective batch_size must be >= 1.")
    if parent_tau <= 0:
        raise SystemExit("Effective parent_tau must be > 0.")
    if not (0.0 <= parent_eps <= 1.0):
        raise SystemExit("Effective parent_eps must be in [0,1].")
    if not (0.0 < parent_score_decay <= 1.0):
        raise SystemExit("Effective parent_score_decay must be in (0,1].")
    if not (0.0 <= recombine_prob <= 1.0):
        raise SystemExit("Effective recombine_prob must be in [0,1].")
    if max_steps < 1:
        raise SystemExit("Effective max_steps must be >= 1.")
    if timeout_s <= 0:
        raise SystemExit("Effective timeout_s must be > 0.")

    for step in range(start_step, start_step + args.iterations):
        # Decay historical parent credit so selection can adapt over time.
        if parent_scores:
            d = float(parent_score_decay)
            for k in list(parent_scores.keys()):
                parent_scores[k] = float(parent_scores[k]) * d

        def _prepare_candidate(candidate_index: int) -> dict[str, Any]:
            rng = random.Random(seed * 10_000 + step * 1_000 + candidate_index)
            have_archive = len(archive) > 0
            mode = "init"
            prompt = ""
            parent_cells: list[list[int]] = []

            can_recombine = len(archive) >= 2 and (rng.random() < recombine_prob)
            if can_recombine:
                picked_cells = _sample_two_parent_cells_weighted(
                    archive,
                    parent_scores,
                    rng=rng,
                    tau_parent=float(parent_tau),
                    eps_parent=float(parent_eps),
                )
                picked = (
                    (archive.cells[picked_cells[0]], archive.cells[picked_cells[1]])
                    if picked_cells is not None
                    else None
                )
                if picked is not None:
                    a, b = picked
                    mode = "recombine"
                    parent_cells = [list(a.cell), list(b.cell)]
                    prompt = _build_recombine_prompt(a.code, b.code)

            if not prompt and have_archive:
                parent_cell = _sample_parent_cell_weighted(
                    archive,
                    parent_scores,
                    rng=rng,
                    tau_parent=float(parent_tau),
                    eps_parent=float(parent_eps),
                )
                parent = archive.cells.get(parent_cell) if parent_cell is not None else None
                if parent is not None:
                    mode = "mutate"
                    parent_cells = [list(parent.cell)]
                    prompt = _build_mutate_prompt(parent.code)

            if not prompt:
                mode = "init"
                prompt = _build_mutate_prompt(None)

            return {
                "iteration": step,
                "candidate_index": candidate_index,
                "mode": mode,
                "parent_cells": parent_cells,
                "prompt": prompt,
            }

        candidates = [_prepare_candidate(i) for i in range(batch_size)]

        def _generate_one(rec: dict[str, Any]) -> dict[str, Any]:
            out = dict(rec)
            out["llm_response"] = ""
            out["code"] = ""
            out["llm_error"] = None
            out["llm_usage"] = {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "provider_cost_usd": None,
            }
            out["llm_provider_cost_usd"] = None
            out["llm_cost_estimate_usd"] = None
            out["llm_cost_usd"] = None
            try:
                resp = call_ai(
                    out["prompt"],
                    concurrent_calls=1,
                    temperature=temperature,
                    model=model,
                    max_completion_tokens=2048,
                )[0]
                content = resp.choices[0].message.content or ""
                out["llm_response"] = content
                out["code"] = _safe_parse_code(content)
                usage = _extract_usage_tokens(getattr(resp, "raw", None))
                out["llm_usage"] = usage
                out["llm_provider_cost_usd"] = usage.get("provider_cost_usd")
                out["llm_cost_estimate_usd"] = _estimate_cost_usd(
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    input_cost_per_1m=float(input_cost_per_1m),
                    output_cost_per_1m=float(output_cost_per_1m),
                )
                if isinstance(out["llm_provider_cost_usd"], (int, float)):
                    out["llm_cost_usd"] = float(out["llm_provider_cost_usd"])
                else:
                    out["llm_cost_usd"] = out["llm_cost_estimate_usd"]
            except Exception as e:
                out["llm_error"] = str(e)
            return out

        max_workers = min(max(1, batch_size), 8)
        if batch_size == 1:
            generated = [_generate_one(candidates[0])]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                generated = list(executor.map(_generate_one, candidates))

        def _eval_one(rec: dict[str, Any]) -> dict[str, Any]:
            out = dict(rec)
            if out.get("llm_error"):
                out["eval"] = {"ok": False, "error": "llm_error", "message": out.get("llm_error")}
                return out
            code = str(out.get("code") or "")
            try:
                eval_out = evaluate_candidate_code(
                    code,
                    env_id=env_id,
                    seed=seed + step + int(out["candidate_index"]),
                    max_steps=max_steps,
                    timeout_s=timeout_s,
                    w_v=w_v,
                    w_e=w_e,
                    w_u=w_u,
                    w_err=w_err,
                )
            except Exception as e:
                eval_out = {"ok": False, "error": "eval_exception", "message": str(e)}
            out["eval"] = eval_out
            return out

        if batch_size == 1:
            evaluated = [_eval_one(generated[0])]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                evaluated = list(executor.map(_eval_one, generated))

        # Insert sequentially to keep archive updates deterministic.
        inserted_batch: list[dict[str, Any]] = []
        for rec in sorted(evaluated, key=lambda r: int(r["candidate_index"])):
            usage = rec.get("llm_usage") or {}
            p_tok = usage.get("prompt_tokens")
            c_tok = usage.get("completion_tokens")
            if isinstance(p_tok, int) and p_tok >= 0:
                cumulative_prompt_tokens += p_tok
            if isinstance(c_tok, int) and c_tok >= 0:
                cumulative_completion_tokens += c_tok
            provider_cst = rec.get("llm_provider_cost_usd")
            if isinstance(provider_cst, (int, float)):
                cumulative_provider_cost_usd += float(provider_cst)
            est_cst = rec.get("llm_cost_estimate_usd")
            if isinstance(est_cst, (int, float)):
                cumulative_token_estimated_cost_usd += float(est_cst)
            cst = rec.get("llm_cost_usd")
            if isinstance(cst, (int, float)):
                cumulative_cost_usd += float(cst)

            eval_out = rec["eval"]
            rec["inserted"] = False
            rec["insert_reason"] = None
            rec["insert_cell"] = None
            rec["quality"] = None
            if eval_out.get("ok"):
                desc = tuple(float(x) for x in eval_out["descriptor_xyz"])
                quality = float(eval_out["quality"])
                rec["quality"] = quality
                ins = archive.insert(
                    descriptor_xyz=desc,
                    quality=quality,
                    code=str(rec.get("code") or ""),
                    iteration=step,
                    metadata={
                        "mode": rec["mode"],
                        "parent_cells": rec["parent_cells"],
                        "seed": seed + step + int(rec["candidate_index"]),
                        "step_errors": eval_out.get("step_errors"),
                        "candidate_index": int(rec["candidate_index"]),
                        "batch_size": int(batch_size),
                    },
                )
                rec["inserted"] = ins.inserted
                rec["insert_reason"] = ins.reason
                rec["insert_cell"] = list(ins.cell)
                if ins.inserted:
                    elite = archive.cells[ins.cell]
                    append_archive_event_jsonl(archive_events_path, iteration=step, elite=elite, reason=ins.reason)
                    inserted_batch.append(rec)

            # Parent credit update (+1 on insert, -1 otherwise).
            delta = 1.0 if rec["inserted"] else -1.0
            p_cells = rec.get("parent_cells") or []
            if rec.get("mode") == "recombine":
                delta_each = 0.5 * delta
                for c in p_cells[:2]:
                    if isinstance(c, list) and len(c) == 3:
                        ck = (int(c[0]), int(c[1]), int(c[2]))
                        parent_scores[ck] = float(parent_scores.get(ck, 0.0)) + delta_each
            elif rec.get("mode") == "mutate":
                if p_cells and isinstance(p_cells[0], list) and len(p_cells[0]) == 3:
                    c = p_cells[0]
                    ck = (int(c[0]), int(c[1]), int(c[2]))
                    parent_scores[ck] = float(parent_scores.get(ck, 0.0)) + delta

            with attempts_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        jsonable(
                            {
                                "iteration": step,
                                "candidate_index": rec["candidate_index"],
                                "mode": rec["mode"],
                                "parent_cells": rec["parent_cells"],
                                "prompt": rec["prompt"],
                                "llm_response": rec.get("llm_response"),
                                "llm_usage": rec.get("llm_usage"),
                                "llm_provider_cost_usd": rec.get("llm_provider_cost_usd"),
                                "llm_cost_estimate_usd": rec.get("llm_cost_estimate_usd"),
                                "llm_cost_usd": rec.get("llm_cost_usd"),
                                "code": rec.get("code"),
                                "llm_error": rec.get("llm_error"),
                                "eval": rec["eval"],
                                "inserted": rec["inserted"],
                                "insert_reason": rec["insert_reason"],
                                "insert_cell": rec["insert_cell"],
                                "parent_score_snapshot": {
                                    ",".join(str(v) for v in k): float(v) for k, v in parent_scores.items()
                                },
                                "archive": archive.to_summary(),
                            }
                        )
                    )
                    + "\n"
                )

        if inserted_batch:
            _write_iteration_bundle(run_dir, step, inserted_batch)
            append_archive_snapshot_json(archive_summary_path, archive)

        best = archive.best_elite()
        progress = {
            "iteration": step,
            "batch_size": int(batch_size),
            "inserted_count": sum(1 for r in evaluated if r.get("inserted")),
            "evaluated_count": len(evaluated),
            "coverage": archive.coverage(),
            "occupied_cells": len(archive),
            "best_quality": (best.quality if best is not None else None),
            "llm_errors": sum(1 for r in evaluated if r.get("llm_error")),
            "cumulative_prompt_tokens": cumulative_prompt_tokens,
            "cumulative_completion_tokens": cumulative_completion_tokens,
            "cumulative_provider_cost_usd": cumulative_provider_cost_usd,
            "cumulative_token_estimated_cost_usd": cumulative_token_estimated_cost_usd,
            "cumulative_cost_usd": cumulative_cost_usd,
            "parent_score_count": len(parent_scores),
            "parent_score_max": (max(parent_scores.values()) if parent_scores else None),
            "parent_score_min": (min(parent_scores.values()) if parent_scores else None),
        }
        print(json.dumps(jsonable(progress), indent=2))

    append_archive_snapshot_json(archive_summary_path, archive)


if __name__ == "__main__":
    main()
