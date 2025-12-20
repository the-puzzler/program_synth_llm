from __future__ import annotations

import json
import random
import ast
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_code_env import (
    clean_generated_code,
    extract_python_code,
    run_code_main_batch,
    validate_sandboxed_code,
)
from call_ai_utils import call_ai


@dataclass
class Attempt:
    reward: float
    code: str


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"xor_{ts}"


def make_hidden_xor_dataset(n: int = 200, *, seed: int = 0) -> list[tuple[float, float, int]]:
    rng = random.Random(seed)
    data: list[tuple[float, float, int]] = []
    for _ in range(n):
        x0 = rng.uniform(-1.0, 1.0)
        x1 = rng.uniform(-1.0, 1.0)
        y = int((x0 > 0) ^ (x1 > 0))
        data.append((x0, x1, y))
    return data


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


def reward_accuracy(dataset: list[tuple[float, float, int]], preds: list[Any]) -> float:
    correct = 0
    for i, (_x0, _x1, y) in enumerate(dataset):
        p = preds[i] if i < len(preds) else None
        try:
            pi = int(p)
        except Exception:
            continue
        if pi not in (0, 1):
            continue
        if pi == y:
            correct += 1
    return correct / len(dataset)


def count_invalid_preds(dataset: list[tuple[float, float, int]], preds: list[Any]) -> int:
    invalid = 0
    for i in range(len(dataset)):
        p = preds[i] if i < len(preds) else None
        try:
            pi = int(p)
        except Exception:
            invalid += 1
            continue
        if pi not in (0, 1):
            invalid += 1
    return invalid


def _format_history(history: list[Attempt], *, last_k: int = 6) -> str:
    if not history:
        return "No prior attempts.\n"
    lines = []
    for i, a in enumerate(history[-last_k:], start=max(1, len(history) - last_k + 1)):
        snippet = "\n".join(a.code.strip().splitlines()[:10])
        lines.append(f"Attempt {i}: reward={a.reward:.4f}\n{snippet}\n")
    return "\n".join(lines)


def _format_history_best_plus_random(history: list[Attempt], *, seed: int, k_random: int = 2) -> str:
    if not history:
        return "No prior attempts.\n"

    best = max(history, key=lambda a: a.reward)
    pool = [a for a in history if a is not best]
    rng = random.Random(seed)
    sampled = rng.sample(pool, k=min(k_random, len(pool))) if pool else []

    def fmt(label: str, a: Attempt) -> str:
        snippet = "\n".join(a.code.strip().splitlines()[:10])
        return f"{label} (reward={a.reward:.4f})\n{snippet}\n"

    out = [fmt("BEST", best)]
    for i, a in enumerate(sampled, start=1):
        out.append(fmt(f"RANDOM_{i}", a))
    return "\n".join(out)


def _validate_main_exists_and_arity(code: str, *, n_args: int) -> None:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            if len(node.args.args) != n_args:
                raise ValueError(f"`main` must take {n_args} positional args; got {len(node.args.args)}.")
            return
    raise ValueError("Generated code must define a `main` function.")


def build_blackbox_prompt(*, history: list[Attempt], step: int) -> str:
    return (
        "Output exactly ONE fenced Python code block and nothing else. Do not write comments.\n"
        "Define exactly this function signature:\n"
        "def main(x0: float, x1: float) -> int:\n"
        "Return value requirement: `main` MUST return an integer 0 or 1 (no other values).\n"
        "Imports: you may `import math` only.\n"
        "Do not read/write files, do not use network, do not print.\n"
        "\n"
        "Goal: maximize a scalar reward. Never return the same solution as one in history.\n"
        "Reward: accuracy on a hidden dataset (higher is better).\n"
        "\n"
        "History (reward + previous code):\n"
        f"{_format_history(history, last_k=len(history))}\n"
    )


def main() -> None:
    dataset = make_hidden_xor_dataset(n=400, seed=0)

    history: list[Attempt] = []
    iterations = 20
    run_dir = _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = run_dir / "attempts.jsonl"
    responses_path = run_dir / "llm_responses.jsonl"
    meta_path = run_dir / "meta.json"
    dataset_path = run_dir / "dataset.jsonl"
    meta_path.write_text(
        json.dumps(
            {
                "experiment": "xor_blackbox",
                "iterations": iterations,
                "dataset_n": len(dataset),
                "dataset_path": dataset_path.name,
                "responses_path": responses_path.name,
                "spec": {"signature": "def main(x0: float, x1: float) -> int", "return_values": [0, 1]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_path.write_text(
        "".join(json.dumps({"x0": x0, "x1": x1, "y": y}) + "\n" for (x0, x1, y) in dataset),
        encoding="utf-8",
    )

    for step in range(1, iterations + 1):
        prompt = build_blackbox_prompt(history=history, step=step)
        response = call_ai(prompt, concurrent_calls=1, temperature=1)[0]
        with responses_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "iteration": step,
                        "prompt": prompt,
                        "request": {"concurrent_calls": 1, "temperature": 1},
                        "response": _jsonable(response),
                    }
                )
                + "\n"
            )
        content = response.choices[0].message.content
        code = clean_generated_code(extract_python_code(content))

        error: str | None = None
        reward = 0.0
        invalid_preds: int | None = None
        batch_errors: int | None = None
        try:
            validate_sandboxed_code(code, allowed_import_roots={"math"})
            _validate_main_exists_and_arity(code, n_args=2)

            batch_inputs = [[x0, x1] for (x0, x1, _y) in dataset]
            run_result = run_code_main_batch(code, batch_inputs=batch_inputs, timeout_s=20)
            if run_result.get("ok"):
                preds = run_result.get("results", [])
                reward = reward_accuracy(dataset, preds)
                invalid_preds = count_invalid_preds(dataset, preds)
                batch_errors = int(run_result.get("errors") or 0)
            else:
                error = str(run_result.get("error") or "run_failed")
        except Exception as e:
            error = str(e)

        history.append(Attempt(reward=reward, code=code))
        best = max(a.reward for a in history)
        payload = {"iteration": step, "reward": reward, "best": best}
        if error:
            payload["error"] = error
        if invalid_preds is not None:
            payload["invalid_preds"] = invalid_preds
        if batch_errors is not None:
            payload["batch_errors"] = batch_errors
        print(json.dumps(payload, indent=2))

        code_file = f"iter_{step:03d}.py"
        (run_dir / code_file).write_text(code, encoding="utf-8")
        with attempts_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "iteration": step,
                        "reward": reward,
                        "best": best,
                        "error": error,
                        "code_path": code_file,
                        "invalid_preds": invalid_preds,
                        "batch_errors": batch_errors,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
