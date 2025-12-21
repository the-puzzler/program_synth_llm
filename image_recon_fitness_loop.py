from __future__ import annotations

import argparse
import ast
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ai_code_env import clean_generated_code, extract_python_code, run_code_main_batch, validate_sandboxed_code
from call_ai_utils import call_ai


@dataclass
class Attempt:
    score_a: float
    score_b: float
    mean_mse: float
    tail_mse: float
    code: str


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"image_{ts}"


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


def _validate_main_exists_and_arity(code: str) -> None:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            if len(node.args.args) != 2:
                raise ValueError("`main` must take exactly 2 positional args: (x, y).")
            return
    raise ValueError("Generated code must define a `main(x, y)` function.")


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
    out: list[str] = []
    for label, a in chosen:
        key = id(a)
        if key in seen:
            continue
        seen.add(key)
        snippet = "\n".join(a.code.strip().splitlines()[:12])
        out.append(
            f"{label}: score_a={a.score_a:.6f} score_b={a.score_b:.6f}\n{snippet}\n"
        )
        if len(out) >= 6:
            break
    return "\n".join(out)


def build_prompt(*, history: list[Attempt], seed: int) -> str:
    return (
        "Output exactly ONE fenced Python code block and nothing else. Do not write comments.\n"
        "Define exactly this function signature:\n"
        "def main(x: float, y: float) -> list[float]:\n"
        "\n"
        "Contract:\n"
        "- Inputs `x` and `y` are floats in [0, 1].\n"
        "- Return a list of 3 floats [a, b, c], each ideally in [0, 1].\n"
        "- Imports: you may `import math` only.\n"
        "- Do not read/write files, do not use network, do not print.\n"
        "\n"
        "Goal: maximize two scalar scores: `score_a` and `score_b` (higher is better for both).\n"
        "\n"
        "Previous attempts to build from (scores + code):\n"
        f"{_format_history_best_random_last(history, seed=seed)}\n"
    )


def _load_image_rgb01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3) in [0,1]
    return arr


def _sample_points(
    rgb: np.ndarray, *, n: int, seed: int
) -> tuple[list[list[float]], np.ndarray]:
    h, w, _c = rgb.shape
    rng = random.Random(seed)
    inputs: list[list[float]] = []
    targets = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        px = rng.randrange(0, w)
        py = rng.randrange(0, h)
        x = 0.0 if w == 1 else px / (w - 1)
        y = 0.0 if h == 1 else py / (h - 1)
        inputs.append([float(x), float(y)])
        targets[i, :] = rgb[py, px, :]
    return inputs, targets


def _preds_to_array(preds: list[Any]) -> tuple[np.ndarray, int]:
    out = np.empty((len(preds), 3), dtype=np.float32)
    invalid = 0
    for i, p in enumerate(preds):
        try:
            if not isinstance(p, (list, tuple)) or len(p) != 3:
                raise TypeError("bad shape")
            r, g, b = float(p[0]), float(p[1]), float(p[2])
            if not (math.isfinite(r) and math.isfinite(g) and math.isfinite(b)):
                raise ValueError("non-finite")
            out[i, 0] = np.clip(r, 0.0, 1.0)
            out[i, 1] = np.clip(g, 0.0, 1.0)
            out[i, 2] = np.clip(b, 0.0, 1.0)
        except Exception:
            invalid += 1
            out[i, :] = 0.0
    return out, invalid


def _mean_mse_and_tail_mse(
    pred: np.ndarray, targets: np.ndarray, *, tail_frac: float
) -> tuple[float, float]:
    dif = pred - targets
    per_point = np.mean(dif * dif, axis=1)  # (N,)
    mean_mse = float(np.mean(per_point))

    n = int(per_point.shape[0])
    if n == 0:
        return mean_mse, float("inf")

    frac = float(tail_frac)
    if not math.isfinite(frac) or frac <= 0:
        frac = 0.05
    frac = min(1.0, frac)
    k = max(1, int(math.ceil(frac * n)))

    # Mean of the largest k errors.
    idx = n - k
    tail = np.partition(per_point, idx)[idx:]
    tail_mse = float(np.mean(tail))
    return mean_mse, tail_mse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, default=Path("image.jpeg"))
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--concurrent", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--samples", type=int, default=2048)
    p.add_argument("--tail-frac", type=float, default=0.05, help="Top fraction for worst-case error score.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args()

    if args.concurrent < 1:
        raise SystemExit("`--concurrent` must be >= 1")
    if not args.image.exists():
        raise SystemExit(f"Missing image file: {args.image} (add it to the repo root, or pass --image)")

    rgb = _load_image_rgb01(args.image)

    run_dir = _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = run_dir / "attempts.jsonl"
    candidates_path = run_dir / "candidates.jsonl"
    meta_path = run_dir / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "experiment": "image_recon_blackbox",
                "image_path": str(args.image),
                "iterations": args.iterations,
                "concurrent": args.concurrent,
                "temperature": args.temperature,
                "samples": args.samples,
                "tail_frac": args.tail_frac,
                "seed": args.seed,
                "spec": "def main(x: float, y: float) -> list[float]  # returns [r,g,b] in [0,1]",
                "score": "maximize score = -MSE on hidden samples",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    history: list[Attempt] = []

    for step in range(1, args.iterations + 1):
        batch_inputs, targets = _sample_points(rgb, n=args.samples, seed=args.seed + step)

        prompts = [build_prompt(history=history, seed=(step * 10_000 + i)) for i in range(args.concurrent)]

        def _one_prompt(prompt: str):
            return call_ai(prompt, concurrent_calls=1, temperature=args.temperature)[0]

        with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            futures = [executor.submit(_one_prompt, pr) for pr in prompts]
            responses = [f.result() for f in futures]

        best_score = float("-inf")
        best_score_b = float("-inf")
        best_mean_mse = float("inf")
        best_tail_mse = float("inf")
        best_idx: int | None = None
        best_code: str | None = None

        for idx, response in enumerate(responses):
            prompt = prompts[idx]
            code = clean_generated_code(extract_python_code(response.choices[0].message.content))

            error: str | None = None
            invalid = 0
            mean_mse = float("inf")
            tail_mse = float("inf")
            score_a = float("-inf")
            score_b = float("-inf")

            try:
                validate_sandboxed_code(code, allowed_import_roots={"math"})
                _validate_main_exists_and_arity(code)
                run_result = run_code_main_batch(code, batch_inputs=batch_inputs, timeout_s=args.timeout)
                if not run_result.get("ok"):
                    error = str(run_result.get("error") or "run_failed")
                else:
                    preds = run_result.get("results", [])
                    pred_arr, invalid = _preds_to_array(preds)
                    mean_mse, tail_mse = _mean_mse_and_tail_mse(
                        pred_arr, targets, tail_frac=args.tail_frac
                    )
                    score_a = -mean_mse
                    score_b = -tail_mse
            except Exception as e:
                error = str(e)

            cand_record = {
                "iteration": step,
                "candidate_index": idx,
                "score_a": score_a,
                "score_b": score_b,
                "mean_mse": mean_mse,
                "tail_mse": tail_mse,
                "invalid": invalid,
                "error": error,
                "prompt": prompt,
                "response": _jsonable(response.raw),
                "code": code,
            }
            with candidates_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(cand_record) + "\n")

            if (score_a, score_b) > (best_score, best_score_b):
                best_score = score_a
                best_score_b = score_b
                best_mean_mse = mean_mse
                best_tail_mse = tail_mse
                best_idx = idx
                best_code = code

        if best_code is None:
            best_code = "def main(x: float, y: float) -> list[float]:\n    return [0.0, 0.0, 0.0]\n"
            best_score = float("-inf")
            best_score_b = float("-inf")
            best_mean_mse = float("inf")
            best_tail_mse = float("inf")
            best_idx = None

        history.append(
            Attempt(
                score_a=best_score,
                score_b=best_score_b,
                mean_mse=best_mean_mse,
                tail_mse=best_tail_mse,
                code=best_code,
            )
        )

        record = {
            "iteration": step,
            "best_candidate_index": best_idx,
            "score_a": best_score,
            "score_b": best_score_b,
            "mean_mse": best_mean_mse,
            "tail_mse": best_tail_mse,
            "concurrent": args.concurrent,
            "code_path": f"iter_{step:03d}.py",
        }
        print(json.dumps(record, indent=2))
        (run_dir / record["code_path"]).write_text(best_code, encoding="utf-8")
        with attempts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
