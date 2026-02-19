from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

import gymnasium as gym

from program_synth.ai_code_env import (
    clean_generated_code,
    extract_python_code,
    validate_sandboxed_code,
)
from program_synth.call_ai_utils import call_ai
from program_synth.embedding_utils import cosine_similarity, embed_texts


EMBEDDING_SIM_THRESHOLD = 0.99


@dataclass
class Attempt:
    iteration: int
    score: float
    code: str
    comment: str | None = None


@dataclass
class LineageState:
    lineage_id: int
    history: list[Attempt]
    embeddings: list[list[float]]
    best_score: float
    last_improvement_step: int


@dataclass
class GymTaskSpec:
    env_id: str
    obs_dim: int
    action_type: str  # "discrete" or "continuous"
    action_dim: int


def _default_run_dir(env_id: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_env = env_id.replace("/", "_").replace("-", "_")
    return Path("runs") / f"gym_islands_{safe_env}_{ts}"


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return _jsonable(dump())
        except Exception:
            pass
    asdict = getattr(obj, "__dict__", None)
    if isinstance(asdict, dict):
        try:
            return _jsonable(asdict)
        except Exception:
            pass
    return repr(obj)


def _infer_gym_spec(env_id: str) -> GymTaskSpec:
    env = gym.make(env_id)
    try:
        obs_space = env.observation_space
        act_space = env.action_space
    finally:
        env.close()

    # Observation: handle Box by flattening, Discrete by treating as single int.
    if hasattr(obs_space, "shape") and obs_space.shape is not None:
        obs_dim = 1
        for d in obs_space.shape:
            obs_dim *= int(d)
    else:
        # Fallback: treat as 1D scalar observation.
        obs_dim = 1

    # Action: support Box and Discrete.
    from gymnasium.spaces import Box, Discrete

    if isinstance(act_space, Box):
        if act_space.shape is None:
            action_dim = 1
        else:
            action_dim = 1
            for d in act_space.shape:
                action_dim *= int(d)
        action_type = "continuous"
    elif isinstance(act_space, Discrete):
        action_dim = int(act_space.n)
        action_type = "discrete"
    else:
        raise SystemExit(
            f"Unsupported action space type for env {env_id!r}: {type(act_space).__name__}. "
            "Only Box and Discrete are supported."
        )

    return GymTaskSpec(env_id=env_id, obs_dim=obs_dim, action_type=action_type, action_dim=action_dim)


def _validate_main_exists(code: str) -> None:
    # Minimal structural check: require a `main` function.
    import ast

    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            if len(node.args.args) != 1:
                raise ValueError("`main` must take exactly 1 positional arg: `obs`.")
            return
    raise ValueError("Generated code must define a `main(obs)` function.")


def _format_history_best_random_last(
    history: list[Attempt],
    *,
    seed: int,
    n_random: int = 3,
    n_last: int = 3,
) -> str:
    """
    Build a short history view: BEST overall, a few RANDOM, and a few LAST attempts.
    """
    if not history:
        return "No prior attempts.\n"

    best = max(history, key=lambda a: a.score)
    last = history[-n_last:] if n_last > 0 else []
    pool = [a for a in history if (a is not best and a not in last)]
    rng = random.Random(seed)
    sampled = rng.sample(pool, k=min(n_random, len(pool))) if pool and n_random > 0 else []

    chosen: list[tuple[str, Attempt]] = []
    chosen.append(("BEST", best))
    for i, a in enumerate(sampled, start=1):
        chosen.append((f"RANDOM_{i}", a))
    for i, a in enumerate(last, start=1):
        chosen.append((f"LAST_{i}", a))

    seen_ids: set[int] = set()
    out: list[str] = []
    kept = 0
    for label, a in chosen:
        if id(a) in seen_ids:
            continue
        seen_ids.add(id(a))
        snippet = "\n".join(a.code.strip().splitlines()[:12])
        header = f"{label}: reward={a.score:.6f}"
        if a.comment:
            out.append(f"{header}\n{a.comment}\n{snippet}\n")
        else:
            out.append(f"{header}\n{snippet}\n")
        kept += 1
        if kept >= 6:
            break
    return "\n".join(out)


def build_gym_prompt(*, history: list[Attempt], spec: GymTaskSpec, seed: int) -> str:
    if spec.action_type == "continuous":
        action_desc = (
            f"- Return a list of {spec.action_dim} floats.\n"
            "- The evaluator will clip floats to the valid action range.\n"
        )
        sig = "def main(obs: list[float]) -> list[float]:"
    else:
        action_desc = (
            "- Return a single integer action.\n"
            f"- Valid actions are integers from 0 to {spec.action_dim - 1}.\n"
        )
        sig = "def main(obs: list[float]) -> int:"

    base = (
        "Output exactly ONE fenced Python code block and nothing else. Do not write comments.\n"
        "Define exactly this function signature:\n"
        f"{sig}\n"
        "\n"
        "Contract:\n"
        "- `obs` is a list of floats representing the environment observation.\n"
        f"- The length of `obs` is {spec.obs_dim}.\n"
        f"{action_desc}"
        "- Imports: you may use `math` only.\n"
        "- Do not read/write files, do not use network, do not print.\n"
        "\n"
        "Goal: maximize the environment's episodic reward.\n"
        "Higher total return is better.\n"
    )

    # CarRacing-specific inductive bias.
    if "CarRacing" in spec.env_id:
        base += (
            "\n"
            "Environment details:\n"
            "- The observation is a 2D pixel image of shape (96, 96, 3), flattened into a list of length 27648.\n"
            "- It is recommended to first write 2 or 3 small feature-extraction functions that you apply across the\n"
            "  image using loops (for example, computing simple local statistics or edge-like responses), and then\n"
            "  base your control logic on those features.\n"
            "- Do not hard-code large weight matrices or giant numeric tables; instead compute features procedurally\n"
            "  using loops and simple math.\n"
        )

    return (
        base
        + "\n"
        "Previous attempts to build from (reward + previous code):\n"
        f"{_format_history_best_random_last(history, seed=seed, n_random=3, n_last=3)}\n"
    )


_GYM_AGENT_RUNNER = r"""
import importlib.util
import json
import sys


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("user_code", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    user_path = payload["user_path"]
    env_id = payload.get("env_id")
    seeds = payload.get("seeds", [0])
    max_steps = int(payload.get("max_steps", 1000))

    try:
        import gymnasium as gym
        from gymnasium.spaces import Box, Discrete
        import numpy as np
    except Exception as e:
        print(json.dumps({"ok": False, "error": "missing_gymnasium", "message": str(e)}))
        return 2

    def _make_env():
        try:
            return gym.make(env_id, max_episode_steps=max_steps)
        except TypeError:
            env = gym.make(env_id)
            try:
                from gymnasium.wrappers import TimeLimit

                return TimeLimit(env, max_episode_steps=max_steps)
            except Exception:
                return env

    try:
        mod = _load_module(user_path)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "load_failed", "message": str(e)}))
        return 3

    if not hasattr(mod, "main"):
        print(json.dumps({"ok": False, "error": "missing_main"}))
        return 4

    episodes = []
    step_errors = 0

    for s in seeds:
        env = _make_env()
        obs, _info = env.reset(seed=int(s))
        total = 0.0
        terminated = False
        truncated = False
        steps = 0

        act_space = env.action_space

        for _t in range(max_steps):
            try:
                obs_vec = np.asarray(obs, dtype=float).ravel().tolist()
                action = mod.main(obs_vec)
                if isinstance(act_space, Discrete):
                    a_int = int(action)
                    if a_int < 0:
                        a_int = 0
                    if a_int >= act_space.n:
                        a_int = act_space.n - 1
                    a = a_int
                elif isinstance(act_space, Box):
                    import numpy as np

                    arr = np.array(action, dtype=float).reshape(act_space.shape)
                    a = np.clip(arr, act_space.low, act_space.high)
                else:
                    raise ValueError("Unsupported action space type.")
            except Exception:
                step_errors += 1
                a = act_space.sample()

            obs, reward, terminated, truncated, _info = env.step(a)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break

        env.close()
        hit_max_steps = bool(steps >= max_steps and not terminated)
        episodes.append(
            {
                "seed": int(s),
                "return": total,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "hit_max_steps": hit_max_steps,
            }
        )

    returns = [e["return"] for e in episodes]
    avg_return = sum(returns) / max(1, len(returns))
    print(
        json.dumps(
            {
                "ok": True,
                "avg_return": avg_return,
                "episodes": episodes,
                "step_errors": step_errors,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def evaluate_gym_policy(
    code: str,
    *,
    env_id: str,
    seeds: list[int],
    max_steps: int,
    timeout_s: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gym_islands_eval_") as td:
        td_path = Path(td)
        user_path = td_path / "policy.py"
        runner_path = td_path / "runner.py"
        user_path.write_text(clean_generated_code(code), encoding="utf-8")
        runner_path.write_text(_GYM_AGENT_RUNNER, encoding="utf-8")

        try:
            here = Path(__file__).resolve()
            repo_root = here.parents[2]
            venv_python = repo_root / ".venv" / "bin" / "python"
            py = str(venv_python) if venv_python.exists() else "python3"

            proc = subprocess.run(
                [py, str(runner_path)],
                input=json.dumps(
                    {
                        "user_path": str(user_path),
                        "env_id": env_id,
                        "seeds": seeds,
                        "max_steps": max_steps,
                    }
                ),
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}
        except Exception as e:
            return {"ok": False, "error": f"subprocess_error: {e}"}

        if proc.returncode != 0:
            try:
                data = json.loads(proc.stdout.strip() or "{}")
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
            return {
                "ok": False,
                "error": "runner_failed",
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }

        try:
            data = json.loads(proc.stdout.strip() or "{}")
            if not isinstance(data, dict):
                raise ValueError("runner output was not a JSON object")
            return data
        except Exception as e:
            return {
                "ok": False,
                "error": f"bad_runner_json: {e}",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }


def _eval_gym_policy_for_pool(
    args_tuple: Tuple[str, str, List[int], int, float]
) -> dict[str, Any]:
    code_str, env, sd, ms, to = args_tuple
    return evaluate_gym_policy(
        code_str,
        env_id=env,
        seeds=sd,
        max_steps=ms,
        timeout_s=to,
    )


def _sample_candidate_for_lineage(
    *,
    lineage: LineageState,
    step: int,
    spec: GymTaskSpec,
    temperature: float,
    max_similarity_retries: int,
    slot_index: int,
) -> dict[str, Any]:
    """
    Generate a single candidate program for a lineage, with embedding-based
    similarity gating against that lineage's own archive.
    """
    attempts_raw: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    used_fallback = False

    for retry in range(max_similarity_retries + 1):
        seed = step * 10_000 + lineage.lineage_id * 1_000 + slot_index * 10 + retry
        prompt = build_gym_prompt(history=lineage.history, spec=spec, seed=seed)
        response = call_ai(prompt, concurrent_calls=1, temperature=temperature)[0]
        content = response.choices[0].message.content

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
            "comment": None,
        }
        attempts_raw.append(rec)

        # Accept immediately if similarity is unavailable or below threshold.
        if max_sim is None or max_sim <= EMBEDDING_SIM_THRESHOLD:
            accepted = rec
            break

    if accepted is None:
        # No low-similarity candidate found after all retries: pick the least similar
        # one we did get, and attach a warning so the history explicitly calls out
        # that we are making only minimal changes.
        viable = [r for r in attempts_raw if isinstance(r.get("max_cosine"), (int, float))]
        if viable:
            accepted = min(viable, key=lambda r: float(r["max_cosine"]))
            used_fallback = True
        else:
            accepted = attempts_raw[0]
            used_fallback = True

    # If we had to fall back to a high-similarity candidate, annotate it so that
    # the lineage history shown to the model carries a clear warning.
    if accepted is not None:
        max_cos = accepted.get("max_cosine")
        if used_fallback and isinstance(max_cos, (int, float)) and max_cos > EMBEDDING_SIM_THRESHOLD:
            warning = (
                "WARNING: minimal changes / near-identical solution detected; "
                "explore more diverse program structures."
            )
            accepted["comment"] = warning
            accepted["similarity_warning"] = True
        else:
            accepted.setdefault("comment", None)
            accepted.setdefault("similarity_warning", False)

    return accepted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=50, help="Number of global iterations.")
    p.add_argument("--lineages", type=int, default=4, help="Number of independent lineages (islands).")
    p.add_argument("--concurrent", type=int, default=1, help="Candidates per lineage per iteration.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--env-id", required=True, help="Gymnasium environment ID, e.g. LunarLander-v2.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-steps", type=int, default=1000)
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

    spec = _infer_gym_spec(env_id)

    run_dir = _default_run_dir(env_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = run_dir / "attempts.jsonl"
    candidates_path = run_dir / "candidates.jsonl"
    meta_path = run_dir / "meta.json"

    meta_path.write_text(
        json.dumps(
            {
                "experiment": "gym_blackbox_islands",
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
                "spec": (
                    "def main(obs: list[float]) -> list[float] or int  "
                    "# obs_dim="
                    f"{spec.obs_dim}, action_type={spec.action_type}, action_dim={spec.action_dim}"
                ),
                "scores": ["reward"],
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
                best_score=float("-inf"),
                last_improvement_step=0,
            )
        )

    global_best_score = float("-inf")
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
                spec=spec,
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
                eval_results = list(executor.map(_eval_gym_policy_for_pool, eval_args))

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
            comment: str | None = cand.get("comment")

            result = None
            if cand in valid_cands:
                result = next(eval_iter, None)
            error: str | None = None
            step_errors: int | None = None
            episodes: list[dict[str, Any]] | None = None
            avg_return: float | None = None

            if result is None:
                error = cand.get("eval_error")
                score = float("-inf")
            else:
                if not result.get("ok"):
                    error = str(result.get("error") or "eval_failed")
                    score = float("-inf")
                else:
                    avg_return = float(result.get("avg_return"))
                    score = avg_return
                    step_errors = int(result.get("step_errors", 0))
                    episodes = list(result.get("episodes", []))

            # Append to lineage history.
            lineage.history.append(
                Attempt(
                    iteration=step,
                    score=score if math.isfinite(score) else -1e9,
                    code=code,
                    comment=comment,
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
                "episodes": episodes,
                "step_errors": step_errors,
                "error": error,
                "embedding_max_cosine": cand.get("max_cosine"),
            }

            # Track per-lineage best for this step.
            curr_best = per_lineage_step_best.get(lineage_id)
            if curr_best is None or score > curr_best["score"]:
                per_lineage_step_best[lineage_id] = {
                    "score": score,
                    "code": code,
                    "detail": cand_detail,
                }

            cand_record = {
                "iteration": step,
                "lineage_id": lineage_id,
                "candidate_index": slot_idx,
                "score": score,
                "avg_return": avg_return,
                "episodes": episodes,
                "step_errors": step_errors,
                "error": error,
                 "comment": comment,
                 "prompt": prompt,
                 "response": _jsonable(response),
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
            score = float(step_best["score"])
            code = str(step_best["code"])
            detail = step_best["detail"]

            if score > lineage.best_score:
                lineage.best_score = score
                lineage.last_improvement_step = step

            lineage_results.append(
                {
                    "lineage_id": lineage.lineage_id,
                    "score": score,
                    "code": code,
                    "detail": detail,
                }
            )

        # Choose global best candidate this step across all lineages.
        best_lineage_entry = max(
            lineage_results,
            key=lambda r: float(r["score"]),
        )
        best_lineage_id = int(best_lineage_entry["lineage_id"])
        best_code = str(best_lineage_entry["code"])
        best_score = float(best_lineage_entry["score"])
        best_detail = best_lineage_entry["detail"]

        if best_score > global_best_score:
            global_best_score = best_score

        # Write canonical program for this global step.
        code_path = run_dir / f"iter_{step:03d}.py"
        code_path.write_text(best_code, encoding="utf-8")

        # Stagnation-based restarts.
        restarted_lineages: list[int] = []
        for lineage in lineages:
            if (
                step - lineage.last_improvement_step >= stagnation_steps
                and lineage.best_score < global_best_score
            ):
                lineage.history.clear()
                lineage.embeddings.clear()
                lineage.best_score = float("-inf")
                lineage.last_improvement_step = step
                restarted_lineages.append(lineage.lineage_id)

        record: dict[str, Any] = {
            "iteration": step,
            "lineage_id": best_lineage_id,
            "score": best_score,
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
