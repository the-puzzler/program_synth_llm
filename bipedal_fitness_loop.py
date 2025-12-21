from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_code_env import clean_generated_code, extract_python_code, validate_sandboxed_code
from call_ai_utils import call_ai


@dataclass
class Attempt:
    iteration: int
    score_a: float
    score_b: float
    code: str
    comment: str | None = None


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"bipedal_{ts}"

def _load_run_meta(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "meta.json"
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _load_attempt_history(run_dir: Path) -> list[Attempt]:
    """
    Reconstruct `history` from a prior run directory.
    This uses `attempts.jsonl` as the source of truth for iteration order/scores,
    and loads code from the recorded `code_path` (or `iter_XXX.py` fallback).
    """
    attempts_path = run_dir / "attempts.jsonl"
    if not attempts_path.exists():
        raise FileNotFoundError(f"Missing attempts file: {attempts_path}")

    history: list[Attempt] = []
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
        if not isinstance(it, int):
            continue
        code_path = rec.get("code_path")
        if isinstance(code_path, str) and code_path.endswith(".py"):
            code_file = run_dir / code_path
        else:
            code_file = run_dir / f"iter_{it:03d}.py"
        try:
            code = code_file.read_text(encoding="utf-8")
        except Exception:
            continue

        score_a = rec.get("score_a")
        score_b = rec.get("score_b")
        try:
            sa = float(score_a) if score_a is not None else float("-inf")
        except Exception:
            sa = float("-inf")
        try:
            sb = float(score_b) if score_b is not None else float("-inf")
        except Exception:
            sb = float("-inf")

        comment = rec.get("comment")
        comment = str(comment) if isinstance(comment, str) else None
        history.append(Attempt(iteration=int(it), score_a=sa, score_b=sb, code=code, comment=comment))

    history.sort(key=lambda a: a.iteration)
    return history


def _load_seen_programs(run_dir: Path) -> list[tuple[str, list[float], int, int]]:
    """
    Reconstruct `seen_programs` for minuscule-change warnings from `candidates.jsonl`.
    Best-effort: skips malformed records.
    """
    path = run_dir / "candidates.jsonl"
    if not path.exists():
        return []
    out: list[tuple[str, list[float], int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        code = rec.get("code")
        it = rec.get("iteration")
        idx = rec.get("candidate_index")
        if not isinstance(code, str) or not isinstance(it, int) or not isinstance(idx, int):
            continue
        try:
            sig, nums = _code_signature_and_numbers(code)
            out.append((sig, nums, int(it), int(idx)))
        except Exception:
            continue
    return out


def _max_existing_iter(run_dir: Path) -> int:
    """
    Determine the highest iteration index present in the run dir from iter_XXX.py files.
    """
    best = 0
    for p in run_dir.glob("iter_*.py"):
        m = re.match(r"^iter_(\d+)$", p.stem)
        if not m:
            continue
        try:
            best = max(best, int(m.group(1)))
        except Exception:
            continue
    return best


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


def _validate_main_exists(code: str) -> None:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            if len(node.args.args) != 1:
                raise ValueError("`main` must take exactly 1 positional arg: `obs`.")
            return
    raise ValueError("Generated code must define a `main(obs)` function.")


def _format_history_all(history: list[Attempt]) -> str:
    if not history:
        return "No prior attempts.\n"
    lines: list[str] = []
    for i, a in enumerate(history, start=1):
        snippet = "\n".join(a.code.strip().splitlines()[:12])
        lines.append(f"Attempt {i}: reward={a.reward:.4f}\n{snippet}\n")
    return "\n".join(lines)


class _NumericNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (int, float)):
            return ast.copy_location(ast.Constant(value=0), node)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
                return ast.copy_location(ast.Constant(value=0), node)
        return node


class _NumericExtractor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.values: list[float] = []

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
            if isinstance(node.operand.value, (int, float)):
                v = float(node.operand.value)
                if isinstance(node.op, ast.USub):
                    v = -v
                self.values.append(v)
                return
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, (int, float)):
            self.values.append(float(node.value))
        # Don't recurse.


def _code_signature_and_numbers(code: str) -> tuple[str, list[float]]:
    tree = ast.parse(code)
    extractor = _NumericExtractor()
    extractor.visit(tree)
    normalized = _NumericNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    sig = ast.dump(normalized, include_attributes=False)
    return sig, extractor.values


def _miniscule_param_change_warning(
    sig: str,
    nums: list[float],
    *,
    seen: list[tuple[str, list[float], int, int]],
) -> tuple[str | None, dict[str, Any] | None]:
    """
    If the structure (sig) matches a prior program and numeric constants changed only slightly,
    return a warning message and metadata about the match.
    """
    if not nums:
        return None, None

    # Heuristics for "minor numeric tweaks".
    max_rel_thresh = 0.05
    mean_rel_thresh = 0.02
    max_abs_thresh = 0.2
    mean_abs_thresh = 0.05

    best: tuple[float, float, int, int, float, float] | None = None  # (max_rel, mean_rel, step, idx, max_abs, mean_abs)
    for prev_sig, prev_nums, prev_step, prev_idx in seen:
        if prev_sig != sig:
            continue
        if len(prev_nums) != len(nums) or not prev_nums:
            continue
        rels: list[float] = []
        abss: list[float] = []
        for a, b in zip(prev_nums, nums):
            d = abs(a - b)
            abss.append(d)
            denom = max(1e-6, abs(a), abs(b))
            rel = d / denom
            rels.append(rel)
        max_rel = max(rels)
        mean_rel = sum(rels) / len(rels)
        max_abs = max(abss)
        mean_abs = sum(abss) / len(abss)

        # Identical program (or numerically identical constants).
        if max_abs <= 1e-12:
            return (
                "WARNING: only miniscule paramter change detected, inefficient. more topological change required",
                {"type": "identical", "similar_step": prev_step, "similar_candidate_index": prev_idx},
            )

        # "Miniscule" tweak heuristic.
        if (
            max_rel <= max_rel_thresh
            and mean_rel <= mean_rel_thresh
            and max_abs <= max_abs_thresh
            and mean_abs <= mean_abs_thresh
        ):
            if best is None or (max_rel, mean_rel, max_abs, mean_abs) < (
                best[0],
                best[1],
                best[4],
                best[5],
            ):
                best = (max_rel, mean_rel, prev_step, prev_idx, max_abs, mean_abs)

    if best is None:
        return None, None

    _max_rel, _mean_rel, prev_step, prev_idx, _max_abs, _mean_abs = best
    warning = "WARNING: only miniscule paramter change detected, inefficient. more topological change required"
    meta = {
        "type": "minuscule_params",
        "similar_step": prev_step,
        "similar_candidate_index": prev_idx,
        "max_rel": _max_rel,
        "mean_rel": _mean_rel,
        "max_abs": _max_abs,
        "mean_abs": _mean_abs,
    }
    return warning, meta


def _format_history_best_random_last(
    history: list[Attempt],
    *,
    seed: int,
    n_random: int = 3,
    n_last: int = 3,
) -> str:
    if not history:
        return "No prior attempts.\n"

    best = max(history, key=lambda a: (a.score_a, a.score_b))
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

    # De-dupe while preserving order (in case BEST is also in LAST_*).
    seen: set[int] = set()
    out: list[str] = []
    for label, a in chosen:
        key = id(a)
        if key in seen:
            continue
        seen.add(key)
        snippet = "\n".join(a.code.strip().splitlines()[:12])
        header = f"{label}: score_a={a.score_a:.6f} score_b={a.score_b:.6f}"
        if a.comment:
            out.append(f"{header}\n{a.comment}\n{snippet}\n")
        else:
            out.append(f"{header}\n{snippet}\n")
        if len(out) >= 6:
            break
    return "\n".join(out)


def build_bipedal_prompt(*, history: list[Attempt], seed: int) -> str:
    return (
        "Output exactly ONE fenced Python code block and nothing else. Do not write comments.\n"
        "Define exactly this function signature:\n"
        "def main(obs: list[float]) -> list[float]:\n"
        "\n"
        "Contract:\n"
        "- `obs` is a list of floats.\n" # besst runs did not specify 24
        "- Return a list of 4 floats.\n"
        "- The evaluator will clip floats to [-1, 1].\n"
        "- Imports: you may use `math` only.\n"
        "- Do not read/write files, do not use network, do not print.\n"
        #"Prioritize incremental changes to the functional form over parameter-only tuning.\n" # trialling this addition
        "\n"
        "Goal: maximize two scalar scores: `score_a` and `score_b`.\n"
        "Higher is better for both.\n"
        "\n"
        "Previous attempts to build from (scores + previous code):\n"
        f"{_format_history_best_random_last(history, seed=seed, n_random=3, n_last=3)}\n" #typically 3 and 3
    )


_AGENT_RUNNER_PY = r"""
import importlib.util
import json
import math
import sys


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("user_code", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clip(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return max(-1.0, min(1.0, float(x)))


def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    user_path = payload["user_path"]
    env_id = payload.get("env_id", "BipedalWalker-v3")
    seeds = payload.get("seeds", [0])
    max_steps = int(payload.get("max_steps", 1600))

    try:
        import gymnasium as gym
    except Exception as e:
        print(json.dumps({"ok": False, "error": "missing_gymnasium", "message": str(e)}))
        return 2

    def _make_env():
        # Override the default TimeLimit (BipedalWalker is commonly capped at 1600 steps).
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
        start_x = None
        end_x = None

        try:
            start_x = float(env.unwrapped.hull.position[0])
        except Exception:
            start_x = None

        for _t in range(max_steps):
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
            if terminated or truncated:
                break

        try:
            end_x = float(env.unwrapped.hull.position[0])
        except Exception:
            end_x = None

        env.close()
        # `truncated` is how Gymnasium reports TimeLimit; treat that as "hit max steps".
        hit_max_steps = bool(steps >= max_steps and not terminated)
        distance = None
        avg_speed = None
        if start_x is not None and end_x is not None:
            distance = float(end_x - start_x)
            if steps > 0:
                avg_speed = float(distance / steps)
        episodes.append(
            {
                "seed": int(s),
                "return": total,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "hit_max_steps": hit_max_steps,
                "distance": distance,
                "avg_speed": avg_speed,
            }
        )

    returns = [e["return"] for e in episodes]
    avg_return = sum(returns) / max(1, len(returns))
    distances = [e["distance"] for e in episodes if e.get("distance") is not None]
    avg_distance = (sum(distances) / max(1, len(distances))) if distances else None
    speeds = [e["avg_speed"] for e in episodes if e.get("avg_speed") is not None]
    avg_speed = (sum(speeds) / max(1, len(speeds))) if speeds else None
    print(
        json.dumps(
            {
                "ok": True,
                "avg_return": avg_return,
                "avg_distance": avg_distance,
                "avg_speed": avg_speed,
                "episodes": episodes,
                "step_errors": step_errors,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def evaluate_policy(
    code: str,
    *,
    env_id: str = "BipedalWalker-v3",
    seeds: list[int],
    max_steps: int = 1600,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="bipedal_eval_") as td:
        td_path = Path(td)
        user_path = td_path / "policy.py"
        runner_path = td_path / "runner.py"
        user_path.write_text(clean_generated_code(code), encoding="utf-8")
        runner_path.write_text(_AGENT_RUNNER_PY, encoding="utf-8")

        try:
            proc = subprocess.run(
                [str(Path(__file__).resolve().parent / ".venv" / "bin" / "python"), str(runner_path)],
                input=json.dumps(
                    {"user_path": str(user_path), "env_id": env_id, "seeds": seeds, "max_steps": max_steps}
                ),
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if not stdout:
            return {"ok": False, "error": "empty_stdout", "stderr": stderr, "returncode": proc.returncode}
        try:
            out = json.loads(stdout)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "error": "non_json_stdout",
                "stdout": stdout,
                "stderr": stderr,
                "returncode": proc.returncode,
            }
        out["returncode"] = proc.returncode
        if stderr:
            out["stderr"] = stderr
        return out


def main() -> None:
    DEFAULT_ENV_ID = "BipedalWalker-v3"
    DEFAULT_SEEDS = [0, 1, 2]
    DEFAULT_MAX_STEPS = 6000
    DEFAULT_TIMEOUT = 180.0
    DEFAULT_CONCURRENT = 1
    DEFAULT_TEMPERATURE = 1.0

    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint-path",
        "--checkpoint_path",
        type=Path,
        default=None,
        help="Resume an existing run dir (e.g. runs/bipedal_20251221T121844Z).",
    )
    p.add_argument("--iterations", type=int, default=20, help="Additional iterations to run.")
    p.add_argument(
        "--concurrent",
        type=int,
        default=None,
        help="Number of candidate programs per iteration (defaults to checkpoint meta.json or 1).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Model temperature (defaults to checkpoint meta.json or 1.0).",
    )
    p.add_argument("--env-id", default=None, help=f"Gym env id (defaults to checkpoint meta.json or {DEFAULT_ENV_ID}).")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Eval seeds (defaults to checkpoint meta.json).")
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=f"Episode cap (defaults to checkpoint meta.json or {DEFAULT_MAX_STEPS}).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"Per-eval timeout seconds (defaults to checkpoint meta.json or {DEFAULT_TIMEOUT}).",
    )
    args = p.parse_args()

    if args.iterations < 1:
        raise SystemExit("`--iterations` must be >= 1")

    run_dir: Path
    run_meta: dict[str, Any] = {}
    if args.checkpoint_path is not None:
        run_dir = args.checkpoint_path.expanduser().resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            raise SystemExit(f"`--checkpoint-path` must be an existing run directory: {run_dir}")
        run_meta = _load_run_meta(run_dir) or {}
    else:
        run_dir = _default_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)

    def _arg_or_meta(name: str, default: Any) -> Any:
        val = getattr(args, name)
        if val is not None:
            return val
        mv = run_meta.get(name)
        return mv if mv is not None else default

    concurrent = int(_arg_or_meta("concurrent", DEFAULT_CONCURRENT))
    temperature = float(_arg_or_meta("temperature", DEFAULT_TEMPERATURE))
    env_id = str(_arg_or_meta("env_id", DEFAULT_ENV_ID))

    seeds_val = args.seeds if args.seeds is not None else run_meta.get("seeds")
    if isinstance(seeds_val, list) and all(isinstance(x, int) for x in seeds_val):
        seeds = list(seeds_val)
    else:
        seeds = list(DEFAULT_SEEDS)

    max_steps_val = _arg_or_meta("max_steps", DEFAULT_MAX_STEPS)
    timeout_val = args.timeout if args.timeout is not None else run_meta.get("timeout_s", DEFAULT_TIMEOUT)
    max_steps = int(max_steps_val) if isinstance(max_steps_val, int) else int(DEFAULT_MAX_STEPS)
    timeout_s = float(timeout_val) if isinstance(timeout_val, (int, float)) else float(DEFAULT_TIMEOUT)

    if concurrent < 1:
        raise SystemExit("`--concurrent` must be >= 1")
    if max_steps < 1:
        raise SystemExit("`--max-steps` must be >= 1")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise SystemExit("`--timeout` must be a positive number")

    # Resume state if checkpoint provided.
    if args.checkpoint_path is not None:
        history = _load_attempt_history(run_dir)
        seen_programs = _load_seen_programs(run_dir)
        last_iter = max((a.iteration for a in history), default=0)
        last_file_iter = _max_existing_iter(run_dir)
        start_step = max(last_iter, last_file_iter) + 1
    else:
        history = []
        seen_programs = []
        start_step = 1

    iterations = args.iterations

    attempts_path = run_dir / "attempts.jsonl"
    candidates_path = run_dir / "candidates.jsonl"
    meta_path = run_dir / "meta.json"

    # Write meta only for brand-new runs; for checkpoints keep the existing meta.json as-is.
    if args.checkpoint_path is None and not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {
                    "experiment": "bipedal_blackbox",
                    "env_id": env_id,
                    "iterations": iterations,
                    "seeds": seeds,
                    "max_steps": max_steps,
                    "timeout_s": timeout_s,
                    "concurrent": concurrent,
                    "temperature": temperature,
                    "candidates_path": candidates_path.name,
                    "spec": "def main(obs: list[float]) -> list[float]  # returns 4 floats in [-1,1]",
                    "scores": ["score_a", "score_b"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for step in range(start_step, start_step + iterations):
        prompts = [
            build_bipedal_prompt(history=history, seed=(step * 10_000 + i))
            for i in range(concurrent)
        ]

        def _one_prompt(p: str):
            return call_ai(p, concurrent_calls=1, temperature=temperature)[0]

        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [executor.submit(_one_prompt, p) for p in prompts]
            responses = [f.result() for f in futures]

        best_reward = float("-inf")
        best_code: str | None = None
        best_idx: int | None = None
        best_detail: dict[str, Any] | None = None
        best_score_a = float("-inf")
        best_score_b = float("-inf")

        for idx, response in enumerate(responses):
            prompt = prompts[idx]
            content = response.choices[0].message.content
            code = clean_generated_code(extract_python_code(content))

            candidate_reward = float("-inf")
            error: str | None = None
            step_errors: int | None = None
            episodes: list[dict[str, Any]] | None = None
            avg_return: float | None = None
            avg_distance: float | None = None
            avg_speed: float | None = None
            comment: str | None = None
            similarity: dict[str, Any] | None = None

            try:
                validate_sandboxed_code(
                    code,
                    allowed_import_roots={"math", "random", "itertools", "functools", "statistics"},
                )
                _validate_main_exists(code)
                sig, nums = _code_signature_and_numbers(code)
                comment, similarity = _miniscule_param_change_warning(sig, nums, seen=seen_programs)
                result = evaluate_policy(
                    code,
                    env_id=env_id,
                    seeds=seeds,
                    max_steps=max_steps,
                    timeout_s=timeout_s,
                )
                if not result.get("ok"):
                    error = str(result.get("error") or "eval_failed")
                else:
                    avg_return = float(result.get("avg_return"))
                    avg_distance = result.get("avg_distance")
                    avg_distance = float(avg_distance) if avg_distance is not None else None
                    avg_speed = result.get("avg_speed")
                    avg_speed = float(avg_speed) if avg_speed is not None else None
                    # Two maximize-able scores (do not reveal their meaning in the prompt).
                    score_a = avg_distance if avg_distance is not None else float("-inf")
                    score_b = avg_speed if avg_speed is not None else float("-inf")
                    candidate_reward = score_a  # kept for backward-compat logging
                    step_errors = int(result.get("step_errors", 0))
                    episodes = list(result.get("episodes", []))
            except Exception as e:
                error = str(e)

            score_a = avg_distance if avg_distance is not None else float("-inf")
            score_b = avg_speed if avg_speed is not None else float("-inf")

            cand_record = {
                "iteration": step,
                "candidate_index": idx,
                "score_a": score_a,
                "score_b": score_b,
                "avg_return": avg_return,
                "avg_distance": avg_distance,
                "avg_speed": avg_speed,
                "episodes": episodes,
                "step_errors": step_errors,
                "error": error,
                "comment": comment,
                "similarity": similarity,
                "prompt": prompt,
                "response": _jsonable(response.raw),
                "code": code,
            }
            with candidates_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(cand_record) + "\n")

            if (score_a, score_b) > (best_score_a, best_score_b):
                best_score_a, best_score_b = score_a, score_b
                best_reward = score_a
                best_code = code
                best_idx = idx
                best_detail = {
                    "score_a": score_a,
                    "score_b": score_b,
                    "episodes": episodes,
                    "step_errors": step_errors,
                    "error": error,
                    "avg_return": avg_return,
                    "avg_distance": avg_distance,
                    "avg_speed": avg_speed,
                    "comment": comment,
                    "similarity": similarity,
                }

            # Track every candidate (not just the best) for similarity checks later.
            try:
                sig, nums = _code_signature_and_numbers(code)
                seen_programs.append((sig, nums, step, idx))
            except Exception:
                pass

        if best_code is None:
            best_code = "def main(obs: list[float]) -> list[float]:\n    return [0.0, 0.0, 0.0, 0.0]\n"
            best_reward = float("-inf")
            best_idx = None
            best_detail = {"error": "no_candidates"}

        history.append(
            Attempt(
                iteration=step,
                score_a=best_score_a if math.isfinite(best_score_a) else -1e9,
                score_b=best_score_b if math.isfinite(best_score_b) else -1e9,
                code=best_code,
                comment=(best_detail or {}).get("comment") if isinstance(best_detail, dict) else None,
            )
        )

        record = {
            "iteration": step,
            "best_candidate_index": best_idx,
            "score_a": best_score_a,
            "score_b": best_score_b,
            "code_path": f"iter_{step:03d}.py",
            "concurrent": concurrent,
            **(best_detail or {}),
        }
        if isinstance(record.get("episodes"), list):
            eps = record["episodes"]
            try:
                record["terminated_count"] = sum(1 for e in eps if e and e.get("terminated"))
                record["truncated_count"] = sum(1 for e in eps if e and e.get("truncated"))
                record["hit_max_steps_count"] = sum(1 for e in eps if e and e.get("hit_max_steps"))
                record["mean_steps"] = (
                    sum(float(e.get("steps", 0)) for e in eps) / max(1, len(eps))
                )
            except Exception:
                pass
        print(json.dumps(record, indent=2))
        (run_dir / f"iter_{step:03d}.py").write_text(best_code, encoding="utf-8")
        with attempts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
