from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

from program_synth.ai_code_env import validate_sandboxed_code
from program_synth.call_ai_utils import call_ai
from program_synth.embedding_utils import cosine_similarity, embed_texts
from program_synth.loops.bipedal_fitness_loop import (
    Attempt,
    EMBEDDING_SIM_THRESHOLD,
    _eval_policy_for_pool,
    _validate_main_exists,
    build_bipedal_prompt,
    evaluate_policy,
)
from program_synth.utils import jsonable


@dataclass
class LineageState:
    lineage_id: int
    history: list[Attempt]
    embeddings: list[list[float]]
    best_score_a: float
    best_score_b: float
    last_improvement_step: int


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"bipedal_islands_{ts}"


def _sample_candidate_for_lineage(
    *,
    lineage: LineageState,
    step: int,
    temperature: float,
    max_similarity_retries: int,
    slot_index: int,
) -> dict[str, Any]:
    """
    Generate a single candidate program for a lineage, with embedding-based
    similarity gating against that lineage's own canonical archive.
    """
    attempts_raw: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None

    for retry in range(max_similarity_retries + 1):
        seed = step * 10_000 + lineage.lineage_id * 1_000 + slot_index * 10 + retry
        prompt = build_bipedal_prompt(history=lineage.history, seed=seed)
        response = call_ai(prompt, concurrent_calls=1, temperature=temperature)[0]
        content = response.choices[0].message.content
        from program_synth.ai_code_env import extract_python_code, clean_generated_code

        code = clean_generated_code(extract_python_code(content))

        max_sim: float | None = None
        try:
            emb = embed_texts([code])[0]
            if lineage.embeddings:
                max_sim = max(cosine_similarity(emb, prev) for prev in lineage.embeddings)
            else:
                max_sim = 0.0
        except Exception:
            emb = None
            max_sim = None

        rec: dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "code": code,
            "embedding": emb,
            "max_cosine": max_sim,
        }
        attempts_raw.append(rec)

        if max_sim is None or max_sim <= EMBEDDING_SIM_THRESHOLD:
            accepted = rec
            break

    if accepted is None:
        viable = [r for r in attempts_raw if isinstance(r.get("max_cosine"), (int, float))]
        if viable:
            accepted = min(viable, key=lambda r: float(r["max_cosine"]))
        else:
            accepted = attempts_raw[0]

    return accepted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=50, help="Number of global iterations.")
    p.add_argument("--lineages", type=int, default=4, help="Number of independent lineages (islands).")
    p.add_argument("--concurrent", type=int, default=1, help="Candidates per lineage per iteration.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--env-id", default="BipedalWalker-v3")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument(
        "--stagnation-steps",
        type=int,
        default=15,
        help="If a lineage fails to improve for this many steps and is not globally best, it is restarted.",
    )
    args = p.parse_args()

    if args.iterations < 1:
        raise SystemExit("`--iterations` must be >= 1")
    if args.lineages < 1:
        raise SystemExit("`--lineages` must be >= 1")
    if args.concurrent < 1:
        raise SystemExit("`--concurrent` must be >= 1")

    env_id = str(args.env_id)
    seeds = list(args.seeds)
    max_steps = int(args.max_steps)
    timeout_s = float(args.timeout)
    stagnation_steps = int(args.stagnation_steps)
    temperature = float(args.temperature)

    run_dir = _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = run_dir / "attempts.jsonl"
    candidates_path = run_dir / "candidates.jsonl"
    meta_path = run_dir / "meta.json"

    meta_path.write_text(
        json.dumps(
            {
                "experiment": "bipedal_blackbox_islands",
                "env_id": env_id,
                "iterations": args.iterations,
                "seeds": seeds,
                "max_steps": max_steps,
                "timeout_s": timeout_s,
                "lineages": args.lineages,
                "temperature": temperature,
                "concurrent": args.concurrent,
                "stagnation_steps": stagnation_steps,
                "candidates_path": candidates_path.name,
                "spec": "def main(obs: list[float]) -> list[float]  # returns 4 floats in [-1,1]",
                "scores": ["score_a", "score_b"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Initialize lineages.
    lineages: list[LineageState] = []
    for i in range(args.lineages):
        lineages.append(
            LineageState(
                lineage_id=i,
                history=[],
                embeddings=[],
                best_score_a=float("-inf"),
                best_score_b=float("-inf"),
                last_improvement_step=0,
            )
        )

    global_best_score_a = float("-inf")
    global_best_score_b = float("-inf")

    max_similarity_retries = 4

    for step in range(1, args.iterations + 1):
        lineage_results: list[dict[str, Any]] = []

        # 1) Generate candidates in parallel across all (lineage, slot) pairs.
        jobs: list[tuple[LineageState, int]] = []
        for lineage in lineages:
            for slot_idx in range(args.concurrent):
                jobs.append((lineage, slot_idx))

        def _gen(job: tuple[LineageState, int]) -> dict[str, Any]:
            lineage, slot_idx = job
            cand = _sample_candidate_for_lineage(
                lineage=lineage,
                step=step,
                temperature=temperature,
                max_similarity_retries=max_similarity_retries,
                slot_index=slot_idx,
            )
            cand["lineage_id"] = lineage.lineage_id
            cand["candidate_index"] = slot_idx
            return cand

        with ThreadPoolExecutor(max_workers=len(jobs) or 1) as executor:
            cand_list = list(executor.map(_gen, jobs))

        # 2) Validate and prepare evaluation args.
        valid_cands: list[dict[str, Any]] = []
        for cand in cand_list:
            code = str(cand.get("code") or "")
            try:
                validate_sandboxed_code(
                    code,
                    allowed_import_roots={"math", "random", "itertools", "functools", "statistics"},
                )
                _validate_main_exists(code)
            except Exception as e:
                cand["eval_error"] = str(e)
                cand["eval_result"] = None
                continue
            valid_cands.append(cand)

        eval_results: list[dict[str, Any]] = []
        if valid_cands:
            eval_args = [
                (str(cand.get("code") or ""), env_id, seeds, max_steps, timeout_s)
                for cand in valid_cands
            ]
            max_workers = min(len(eval_args), args.lineages * args.concurrent)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                eval_results = list(executor.map(_eval_policy_for_pool, eval_args))

        eval_iter = iter(eval_results)

        # 3) Attach eval results, update per-lineage state, and log candidates.
        per_lineage_step_best: dict[int, dict[str, Any]] = {}

        for cand in cand_list:
            lineage_id = int(cand["lineage_id"])
            slot_idx = int(cand["candidate_index"])
            lineage = lineages[lineage_id]
            code = str(cand.get("code") or "")
            prompt = str(cand.get("prompt") or "")
            response = cand.get("response")

            result = None
            if cand in valid_cands:
                result = next(eval_iter, None)
            error: str | None = None
            step_errors: int | None = None
            episodes: list[dict[str, Any]] | None = None
            avg_return: float | None = None
            avg_distance: float | None = None
            avg_speed: float | None = None

            if result is None:
                error = cand.get("eval_error")
                score_a = float("-inf")
                score_b = float("-inf")
            else:
                if not result.get("ok"):
                    error = str(result.get("error") or "eval_failed")
                    score_a = float("-inf")
                    score_b = float("-inf")
                else:
                    avg_return = float(result.get("avg_return"))
                    avg_distance = result.get("avg_distance")
                    avg_distance = float(avg_distance) if avg_distance is not None else None
                    avg_speed = result.get("avg_speed")
                    avg_speed = float(avg_speed) if avg_speed is not None else None
                    score_a = avg_distance if avg_distance is not None else float("-inf")
                    score_b = avg_speed if avg_speed is not None else float("-inf")
                    step_errors = int(result.get("step_errors", 0))
                    episodes = list(result.get("episodes", []))

            # Append to lineage history.
            lineage.history.append(
                Attempt(
                    iteration=step,
                    score_a=score_a if math.isfinite(score_a) else -1e9,
                    score_b=score_b if math.isfinite(score_b) else -1e9,
                    code=code,
                    comment=None,
                )
            )
            # Update lineage embedding archive with this candidate.
            try:
                emb = embed_texts([code])[0]
                lineage.embeddings.append(emb)
            except Exception:
                pass

            cand_detail = {
                "avg_return": avg_return,
                "avg_distance": avg_distance,
                "avg_speed": avg_speed,
                "episodes": episodes,
                "step_errors": step_errors,
                "error": error,
                "embedding_max_cosine": cand.get("max_cosine"),
            }

            # Track per-lineage best for this step.
            curr_best = per_lineage_step_best.get(lineage_id)
            if curr_best is None or (score_a, score_b) > (curr_best["score_a"], curr_best["score_b"]):
                per_lineage_step_best[lineage_id] = {
                    "score_a": score_a,
                    "score_b": score_b,
                    "code": code,
                    "detail": cand_detail,
                }

            cand_record = {
                "iteration": step,
                "lineage_id": lineage_id,
                "candidate_index": slot_idx,
                "score_a": score_a,
                "score_b": score_b,
                "avg_return": avg_return,
                "avg_distance": avg_distance,
                "avg_speed": avg_speed,
                "episodes": episodes,
                "step_errors": step_errors,
                "error": error,
                "prompt": prompt,
                "response": jsonable(response),
                "code": code,
                "embedding_max_cosine": cand.get("max_cosine"),
            }
            with candidates_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(cand_record) + "\n")

        # 4) Update lineage-wide bests and assemble lineage_results for global selection.
        for lineage in lineages:
            step_best = per_lineage_step_best.get(lineage.lineage_id)
            if step_best is None:
                continue
            score_a = float(step_best["score_a"])
            score_b = float(step_best["score_b"])
            code = str(step_best["code"])
            detail = step_best["detail"]

            if (score_a, score_b) > (lineage.best_score_a, lineage.best_score_b):
                lineage.best_score_a = score_a
                lineage.best_score_b = score_b
                lineage.last_improvement_step = step

            lineage_results.append(
                {
                    "lineage_id": lineage.lineage_id,
                    "score_a": score_a,
                    "score_b": score_b,
                    "code": code,
                    "detail": detail,
                }
            )

        # Choose global best candidate this step across all lineages.
        best_lineage_entry = max(
            lineage_results,
            key=lambda r: (float(r["score_a"]), float(r["score_b"])),
        )
        best_lineage_id = int(best_lineage_entry["lineage_id"])
        best_code = str(best_lineage_entry["code"])
        best_score_a = float(best_lineage_entry["score_a"])
        best_score_b = float(best_lineage_entry["score_b"])
        best_detail = best_lineage_entry["detail"]

        if (best_score_a, best_score_b) > (global_best_score_a, global_best_score_b):
            global_best_score_a = best_score_a
            global_best_score_b = best_score_b

        # Write canonical program for this global step.
        code_path = run_dir / f"iter_{step:03d}.py"
        code_path.write_text(best_code, encoding="utf-8")

        # Stagnation-based restarts.
        restarted_lineages: list[int] = []
        for lineage in lineages:
            if (
                step - lineage.last_improvement_step >= stagnation_steps
                and (lineage.best_score_a, lineage.best_score_b) < (global_best_score_a, global_best_score_b)
            ):
                lineage.history.clear()
                lineage.embeddings.clear()
                lineage.best_score_a = float("-inf")
                lineage.best_score_b = float("-inf")
                lineage.last_improvement_step = step
                restarted_lineages.append(lineage.lineage_id)

        record: dict[str, Any] = {
            "iteration": step,
            "lineage_id": best_lineage_id,
            "score_a": best_score_a,
            "score_b": best_score_b,
            "code_path": code_path.name,
            "lineages": args.lineages,
            "stagnation_steps": stagnation_steps,
            **(best_detail or {}),
        }
        if restarted_lineages:
            record["restarted_lineages"] = restarted_lineages
            record.setdefault("comment", "")
            marker = " ".join(f"lineage {lid} <restart>" for lid in restarted_lineages)
            record["comment"] = (record["comment"] + " " + marker).strip()

        print(json.dumps(record, indent=2))
        with attempts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
