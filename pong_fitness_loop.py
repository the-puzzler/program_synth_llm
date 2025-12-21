from __future__ import annotations

import argparse
import ast
import json
import math
import random
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
    score: float  # avg_return
    code: str
    comment: str | None = None


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"pong_{ts}"


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


def _format_history_best_random_last(
    history: list[Attempt],
    *,
    seed: int,
    n_random: int = 3,
    n_last: int = 3,
) -> str:
    if not history:
        return "No prior attempts.\n"

    best = max(history, key=lambda a: a.score)
    last = history[-n_last:] if n_last > 0 else []
    pool = [a for a in history if (a is not best and a not in last)]
    rng = random.Random(seed)
    sampled = rng.sample(pool, k=min(n_random, len(pool))) if pool and n_random > 0 else []

    chosen: list[tuple[str, Attempt]] = [("BEST", best)]
    for i, a in enumerate(sampled, start=1):
        chosen.append((f"RANDOM_{i}", a))
    for i, a in enumerate(last, start=1):
        chosen.append((f"LAST_{i}", a))

    seen: set[int] = set()
    out: list[str] = []
    for label, a in chosen:
        key = id(a)
        if key in seen:
            continue
        seen.add(key)
        snippet = "\n".join(a.code.strip().splitlines()[:16])
        header = f"{label}: score={a.score:.6f}"
        if a.comment:
            out.append(f"{header}\n{a.comment}\n{snippet}\n")
        else:
            out.append(f"{header}\n{snippet}\n")
        if len(out) >= 6:
            break
    return "\n".join(out)


def build_pong_prompt(*, history: list[Attempt], seed: int) -> str:
    return (
        "Output exactly ONE fenced Python code block and nothing else. Do not write comments.\n"
        "Define exactly this function signature:\n"
        "def main(obs: list[list[float]]) -> int:\n"
        "\n"
        "Contract:\n"
        "- `obs` is a 2D list of floats representing the current frame (grayscale), shape (210, 160).\n"
        "- Values are in [0, 1].\n"
        "- Return an integer action in {0,1,2,3,4,5}.\n"
        "- Allowed imports: `math`, `itertools`, `functools`, `statistics`, `numpy`.\n"
        "- Do not import anything else.\n"
        "- Do not read/write files, do not use network, do not print.\n"
        "\n"
        "Goal: maximize a scalar score.\n"
        "Higher is better.\n"
        "Prioritize changes to the functional form over parameter-only tuning.\n"
        "Hint: develop a kernel function.\n"
        "Previous attempts to build from (scores + previous code):\n"
        f"{_format_history_best_random_last(history, seed=seed, n_random=3, n_last=3)}\n"
    )


_PONG_RUNNER_PY = r"""
import importlib.util
import json
import sys


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("user_code", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _as_int_action(x, n=6):
    try:
        if isinstance(x, bool):
            return 0
        if isinstance(x, int):
            v = x
        else:
            v = int(float(x))
        if v < 0:
            v = 0
        if v >= n:
            v = n - 1
        return v
    except Exception:
        return 0


def _preprocess(frame, prev, size=16):
    # Returns grayscale as a 2D list of floats in [0,1], shape (H,W).
    try:
        import numpy as np
    except Exception:
        np = None

    if frame is None:
        return [], None

    if np is None:
        # Slow fallback.
        h = len(frame)
        w = len(frame[0]) if h else 0
        out = []
        for y in range(h):
            row = frame[y]
            out_row = []
            for x in range(w):
                px = row[x]
                r, g, b = float(px[0]), float(px[1]), float(px[2])
                out_row.append((r + g + b) / (3.0 * 255.0))
            out.append(out_row)
        return out, None

    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray_full = (arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]) / (3.0 * 255.0)
    else:
        gray_full = arr.astype(np.float32) / 255.0

    return gray_full.tolist(), None


def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    user_path = payload["user_path"]
    env_id = payload.get("env_id", "ALE/Pong-v5")
    seeds = payload.get("seeds", [0])
    max_steps = int(payload.get("max_steps", 20000))

    try:
        import gymnasium as gym
    except Exception as e:
        print(json.dumps({"ok": False, "error": "missing_gymnasium", "message": str(e)}))
        return 2

    # Gymnasium doesn't always auto-register ALE env ids; ensure they're registered when ale_py is available.
    try:
        from gymnasium.envs.registration import registry

        if env_id not in registry:
            import ale_py.registration as _ale_reg

            _ale_reg.register_v5_envs()
    except Exception:
        pass

    def _make_env():
        try:
            return gym.make(env_id, obs_type="rgb", frameskip=1, repeat_action_probability=0.0, full_action_space=False)
        except TypeError:
            return gym.make(env_id)

    try:
        mod = _load_module(user_path)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "load_failed", "message": str(e)}))
        return 3

    if not hasattr(mod, "main"):
        print(json.dumps({"ok": False, "error": "missing_main"}))
        return 4

    episodes = []
    total_step_errors = 0

    for s in seeds:
        env = _make_env()
        obs, _info = env.reset(seed=int(s))
        total = 0.0
        terminated = False
        truncated = False
        steps = 0
        step_errors = 0

        for _t in range(max_steps):
            frame2d, _ = _preprocess(obs, None, size=0)
            try:
                a = mod.main(frame2d)
                action = _as_int_action(a, n=6)
            except Exception:
                step_errors += 1
                action = 0

            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break

        env.close()
        total_step_errors += step_errors
        hit_max_steps = bool(steps >= max_steps and not terminated)
        episodes.append(
            {
                "seed": int(s),
                "return": total,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "hit_max_steps": hit_max_steps,
                "step_errors": step_errors,
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
                "step_errors": total_step_errors,
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
    env_id: str = "ALE/Pong-v5",
    seeds: list[int],
    max_steps: int = 20000,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pong_eval_") as td:
        td_path = Path(td)
        user_path = td_path / "policy.py"
        runner_path = td_path / "runner.py"
        user_path.write_text(clean_generated_code(code), encoding="utf-8")
        runner_path.write_text(_PONG_RUNNER_PY, encoding="utf-8")

        py = str(Path(__file__).resolve().parent / ".venv" / "bin" / "python")
        if not Path(py).exists():
            py = "python3"

        try:
            proc = subprocess.run(
                [py, str(runner_path)],
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
            return {"ok": False, "error": "non_json_stdout", "stdout": stdout, "stderr": stderr, "returncode": proc.returncode}
        out["returncode"] = proc.returncode
        if stderr:
            out["stderr"] = stderr
        return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--concurrent", type=int, default=1, help="Candidates per iteration.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--env-id", default="ALE/Pong-v5")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args()

    if args.concurrent < 1:
        raise SystemExit("`--concurrent` must be >= 1")
    if args.iterations < 1:
        raise SystemExit("`--iterations` must be >= 1")

    run_dir = _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = run_dir / "attempts.jsonl"
    candidates_path = run_dir / "candidates.jsonl"
    meta_path = run_dir / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "experiment": "pong_blackbox",
                "env_id": args.env_id,
                "iterations": args.iterations,
                "seeds": list(args.seeds),
                "max_steps": args.max_steps,
                "timeout_s": args.timeout,
                "concurrent": args.concurrent,
                "temperature": args.temperature,
                "candidates_path": candidates_path.name,
                "spec": "def main(obs: list[list[float]]) -> int  # obs is a grayscale frame (210,160), action in {0..5}",
                "score": "avg_return",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    history: list[Attempt] = []
    seen_programs: list[tuple[str, list[float], int, int]] = []
    for step in range(1, args.iterations + 1):
        prompts = [build_pong_prompt(history=history, seed=(step * 10_000 + i)) for i in range(args.concurrent)]

        def _one_prompt(pr: str):
            return call_ai(pr, concurrent_calls=1, temperature=args.temperature)[0]

        with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            responses = [f.result() for f in [executor.submit(_one_prompt, pr) for pr in prompts]]

        best_score = float("-inf")
        best_step_errors = 10**9
        best_code: str | None = None
        best_idx: int | None = None
        best_detail: dict[str, Any] | None = None

        for idx, response in enumerate(responses):
            prompt = prompts[idx]
            content = response.choices[0].message.content
            code = clean_generated_code(extract_python_code(content))

            error: str | None = None
            avg_return: float | None = None
            episodes: list[dict[str, Any]] | None = None
            step_errors: int | None = None
            comment: str | None = None
            similarity: dict[str, Any] | None = None

            score = float("-inf")

            try:
                validate_sandboxed_code(
                    code,
                    allowed_import_roots={"math", "random", "itertools", "functools", "statistics", "numpy"},
                )
                _validate_main_exists(code)
                sig, nums = _code_signature_and_numbers(code)
                comment, similarity = _miniscule_param_change_warning(sig, nums, seen=seen_programs)
                result = evaluate_policy(
                    code,
                    env_id=args.env_id,
                    seeds=list(args.seeds),
                    max_steps=args.max_steps,
                    timeout_s=args.timeout,
                )
                if not result.get("ok"):
                    error = str(result.get("error") or "eval_failed")
                else:
                    avg_return = float(result.get("avg_return"))
                    episodes = list(result.get("episodes", []))
                    step_errors = int(result.get("step_errors", 0))
                    score = avg_return
            except Exception as e:
                error = str(e)

            cand_record = {
                "iteration": step,
                "candidate_index": idx,
                "score": score,
                "avg_return": avg_return,
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

            se = int(step_errors) if isinstance(step_errors, int) else 10**9
            if (score, -se) > (best_score, -best_step_errors):
                best_score = score
                best_step_errors = se
                best_code = code
                best_idx = idx
                best_detail = {
                    "score": score,
                    "avg_return": avg_return,
                    "episodes": episodes,
                    "step_errors": step_errors,
                    "error": error,
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
            best_code = "def main(obs: list[list[float]]) -> int:\n    return 0\n"
            best_idx = None
            best_detail = {"error": "no_candidates"}

        history.append(
            Attempt(
                iteration=step,
                score=best_score if math.isfinite(best_score) else -1e9,
                code=best_code,
                comment=(best_detail or {}).get("comment") if isinstance(best_detail, dict) else None,
            )
        )

        record = {
            "iteration": step,
            "best_candidate_index": best_idx,
            "score": best_score,
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
