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
    return Path("runs") / f"walker2d_{ts}"

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
    if not nums:
        return None, None

    max_rel_thresh = 0.05
    mean_rel_thresh = 0.02
    max_abs_thresh = 0.2
    mean_abs_thresh = 0.05

    best: tuple[float, float, int, int, float, float] | None = None
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
            rels.append(d / denom)
        max_rel = max(rels)
        mean_rel = sum(rels) / len(rels)
        max_abs = max(abss)
        mean_abs = sum(abss) / len(abss)

        if max_abs <= 1e-12:
            return (
                "WARNING: candidate repeats a previously seen program (or only tiny numeric changes).",
                {"type": "identical", "similar_step": prev_step, "similar_candidate_index": prev_idx},
            )

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
    warning = "WARNING: only miniscule parameter change detected; make a bigger structural change."
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


def _format_history_best_random_last(history: list[Attempt], *, seed: int) -> str:
    if not history:
        return "No prior attempts.\n"

    best = max(history, key=lambda a: (a.score_a, a.score_b))
    last = history[-3:]
    pool = [a for a in history if (a is not best and a not in last)]
    rng = random.Random(seed)
    sampled = rng.sample(pool, k=min(3, len(pool))) if pool else []

    chosen: list[tuple[str, Attempt]] = [("BEST", best)]
    for i, a in enumerate(sampled, start=1):
        chosen.append((f"RANDOM_{i}", a))
    for i, a in enumerate(last, start=1):
        chosen.append((f"LAST_{i}", a))

    seen: set[int] = set()
    seen_sigs: set[str] = set()
    out: list[str] = []
    kept = 0
    for label, a in chosen:
        key = id(a)
        if key in seen:
            continue
        seen.add(key)
        try:
            sig, _nums = _code_signature_and_numbers(a.code)
        except Exception:
            sig = ""
        if sig:
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
        else:
            # Fallback to exact-text de-dupe if parsing fails.
            text_key = f"code:{hash(a.code)}"
            if text_key in seen_sigs:
                continue
            seen_sigs.add(text_key)
        snippet = "\n".join(a.code.strip().splitlines()[:12])
        header = f"{label}: score_a={a.score_a:.6f} score_b={a.score_b:.6f}"
        if a.comment:
            out.append(f"{header}\n{a.comment}\n{snippet}\n")
        else:
            out.append(f"{header}\n{snippet}\n")
        kept += 1
        if kept >= 6:
            break
    return "\n".join(out)


def build_walker2d_prompt(*, history: list[Attempt], seed: int) -> str:
    return (
        "Output exactly ONE fenced Python code block and nothing else. Do not write comments.\n"
        "Define exactly this function signature:\n"
        "def main(obs: list[float]) -> list[float]:\n"
        "\n"
        "Contract:\n"
        "- `obs` is a list of 17 floats.\n"
        "- Return a list of 6 floats.\n"
        "- The evaluator will clip floats to [-1, 1].\n"
        "- Allowed imports: `math`, `random`, `itertools`, `functools`, `statistics`.\n"
        "- Do not import anything else.\n"
        "- Do not read/write files, do not use network, do not print.\n"
        "Prioritize changes to the functional form over parameter-only tuning.\n" #dont knwo if it helps
        "Search over a computation graph/program control flow space." # testing this
        "\n"
        "Goal: maximize two scalar scores: `score_a` and `score_b`.\n"
        "Higher is better for both.\n"
        "\n"
        "Previous attempts to build from (scores + previous code):\n"
        f"{_format_history_best_random_last(history, seed=seed)}\n"
    )


_AGENT_RUNNER_PY = r"""
import importlib.util
import json
import sys


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("user_code", path)
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
    user_path = payload["user_path"]
    env_id = payload.get("env_id", "Walker2d-v5")
    seeds = payload.get("seeds", [0])
    max_steps = int(payload.get("max_steps", 1000))

    try:
        import gymnasium as gym
    except Exception as e:
        print(json.dumps({"ok": False, "error": "missing_gymnasium", "message": str(e)}))
        return 2

    def _make_env():
        # Override the default TimeLimit.
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
            # MuJoCo: x position is usually qpos[0]
            start_x = float(env.unwrapped.data.qpos[0])
        except Exception:
            start_x = None

        for _t in range(max_steps):
            try:
                action = mod.main(list(obs))
                if not isinstance(action, (list, tuple)) or len(action) != 6:
                    raise ValueError("action must be a list/tuple of length 6")
                a = [_clip(action[i]) for i in range(6)]
            except Exception:
                step_errors += 1
                a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            obs, reward, terminated, truncated, _info = env.step(a)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break

        try:
            end_x = float(env.unwrapped.data.qpos[0])
        except Exception:
            end_x = None

        env.close()
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
    env_id: str = "Walker2d-v5",
    seeds: list[int],
    max_steps: int = 1000,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="walker2d_eval_") as td:
        td_path = Path(td)
        user_path = td_path / "policy.py"
        runner_path = td_path / "runner.py"
        user_path.write_text(clean_generated_code(code), encoding="utf-8")
        runner_path.write_text(_AGENT_RUNNER_PY, encoding="utf-8")

        py = str(Path(__file__).resolve().parent / ".venv" / "bin" / "python")
        if not Path(py).exists():
            py = "python3"

        try:
            proc = subprocess.run(
                [py, str(runner_path)],
                input=json.dumps({"user_path": str(user_path), "env_id": env_id, "seeds": seeds, "max_steps": max_steps}),
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
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint-path",
        "--checkpoint_path",
        type=Path,
        default=None,
        help="Resume an existing run dir (e.g. runs/walker2d_20251221T211922Z).",
    )
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--concurrent", type=int, default=1, help="Number of candidate programs per iteration.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--env-id", default="Walker2d-v5")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()

    if args.concurrent < 1:
        raise SystemExit("`--concurrent` must be >= 1")
    if args.iterations < 1:
        raise SystemExit("`--iterations` must be >= 1")

    run_meta: dict[str, Any] = {}
    if args.checkpoint_path is not None:
        run_dir = args.checkpoint_path.expanduser().resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            raise SystemExit(f"`--checkpoint-path` must be an existing run directory: {run_dir}")
        run_meta = _load_run_meta(run_dir) or {}
        history = _load_attempt_history(run_dir)
        seen_programs = _load_seen_programs(run_dir)
        last_iter = max((a.iteration for a in history), default=0)
        last_file_iter = _max_existing_iter(run_dir)
        start_step = max(last_iter, last_file_iter) + 1
    else:
        run_dir = _default_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        history = []
        seen_programs = []
        start_step = 1

    def _arg_or_meta(name: str, default: Any) -> Any:
        val = getattr(args, name)
        if val is not None:
            return val
        mv = run_meta.get(name)
        return mv if mv is not None else default

    iterations = int(args.iterations)
    env_id = str(_arg_or_meta("env_id", args.env_id))

    seeds_val = args.seeds if args.seeds is not None else run_meta.get("seeds")
    if isinstance(seeds_val, list) and all(isinstance(x, int) for x in seeds_val):
        seeds = list(seeds_val)
    else:
        seeds = list(args.seeds)

    max_steps_val = _arg_or_meta("max_steps", int(args.max_steps))
    max_steps = int(max_steps_val) if isinstance(max_steps_val, int) else int(args.max_steps)

    timeout_val = args.timeout if args.timeout is not None else run_meta.get("timeout_s", float(args.timeout))
    timeout_s = float(timeout_val) if isinstance(timeout_val, (int, float)) else float(args.timeout)
    attempts_path = run_dir / "attempts.jsonl"
    candidates_path = run_dir / "candidates.jsonl"
    meta_path = run_dir / "meta.json"
    if args.checkpoint_path is None and not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {
                    "experiment": "walker2d_blackbox",
                    "env_id": env_id,
                    "iterations": iterations,
                    "seeds": seeds,
                    "max_steps": max_steps,
                    "timeout_s": timeout_s,
                    "concurrent": args.concurrent,
                    "temperature": args.temperature,
                    "candidates_path": candidates_path.name,
                    "spec": "def main(obs: list[float]) -> list[float]  # returns 6 floats in [-1,1]",
                    "scores": ["score_a", "score_b"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for step in range(start_step, start_step + iterations):
        prompts = [build_walker2d_prompt(history=history, seed=(step * 10_000 + i)) for i in range(args.concurrent)]

        def _one_prompt(pr: str):
            return call_ai(pr, concurrent_calls=1, temperature=args.temperature)[0]

        with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            responses = [f.result() for f in [executor.submit(_one_prompt, pr) for pr in prompts]]

        best_code: str | None = None
        best_idx: int | None = None
        best_detail: dict[str, Any] | None = None
        best_score_a = float("-inf")
        best_score_b = float("-inf")

        for idx, response in enumerate(responses):
            prompt = prompts[idx]
            content = response.choices[0].message.content
            code = clean_generated_code(extract_python_code(content))

            error: str | None = None
            step_errors: int | None = None
            episodes: list[dict[str, Any]] | None = None
            avg_return: float | None = None
            avg_distance: float | None = None
            avg_speed: float | None = None
            comment: str | None = None
            similarity: dict[str, Any] | None = None

            score_a = float("-inf")
            score_b = float("-inf")

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
                    score_a = avg_distance if avg_distance is not None else float("-inf")
                    score_b = avg_speed if avg_speed is not None else float("-inf")
                    step_errors = int(result.get("step_errors", 0))
                    episodes = list(result.get("episodes", []))
            except Exception as e:
                error = str(e)

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
                best_code = code
                best_idx = idx
                best_detail = {
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
                }

            try:
                sig, nums = _code_signature_and_numbers(code)
                seen_programs.append((sig, nums, step, idx))
            except Exception:
                pass

        if best_code is None:
            best_code = "def main(obs: list[float]) -> list[float]:\n    return [0.0]*6\n"
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
            "concurrent": args.concurrent,
            **(best_detail or {}),
        }
        print(json.dumps(record, indent=2))
        (run_dir / f"iter_{step:03d}.py").write_text(best_code, encoding="utf-8")
        with attempts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
