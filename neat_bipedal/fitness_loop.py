from __future__ import annotations

import argparse
import ast
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_code_env import clean_generated_code, extract_python_code, validate_sandboxed_code
from bipedal_fitness_loop import evaluate_policy
from call_ai_utils import call_ai
from neat_bipedal.config import DEFAULT_CONFIG, NeatBipedalConfig


@dataclass
class Attempt:
    generation: int
    score_a: float
    score_b: float
    avg_return: float | None
    avg_distance: float | None
    avg_speed: float | None
    episodes: list[dict[str, Any]] | None
    step_errors: int | None
    error: str | None
    code_path: str
    code: str


@dataclass
class Individual:
    id: int
    history: list[Attempt]
    species_id: int | None = None
    parent_ids: tuple[int, int] | None = None  # (p1, p2) for crossover, (p1, -1) for mutation
    operator: str | None = None  # "init" | "mutate" | "crossover"

    @property
    def latest(self) -> Attempt | None:
        return self.history[-1] if self.history else None

    @property
    def code_path(self) -> str | None:
        a = self.latest
        return a.code_path if a else None

    def score_tuple(self) -> tuple[float, float]:
        a = self.latest
        if not a:
            return (float("-inf"), float("-inf"))
        return (float(a.score_a), float(a.score_b))


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"neat_bipedal_{ts}"


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


def _node_type_hist(code: str) -> Counter[str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return Counter()
    return Counter(type(n).__name__ for n in ast.walk(tree))


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


def _signature_and_numbers(code: str) -> tuple[str, list[float]]:
    tree = ast.parse(code)
    extractor = _NumericExtractor()
    extractor.visit(tree)
    normalized = _NumericNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    sig = ast.dump(normalized, include_attributes=False)
    return sig, extractor.values


def _cosine_distance(a: Counter[str], b: Counter[str]) -> float:
    if not a and not b:
        return 0.0
    keys = set(a) | set(b)
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k in keys:
        va = float(a.get(k, 0))
        vb = float(b.get(k, 0))
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na <= 0 or nb <= 0:
        return 1.0
    sim = dot / (math.sqrt(na) * math.sqrt(nb))
    sim = max(0.0, min(1.0, sim))
    return 1.0 - sim


def _numeric_distance(nums_a: list[float], nums_b: list[float]) -> float:
    if not nums_a and not nums_b:
        return 0.0
    if not nums_a or not nums_b:
        return 1.0
    n = min(len(nums_a), len(nums_b))
    if n <= 0:
        return 1.0
    diffs = [abs(float(x) - float(y)) for x, y in zip(nums_a[:n], nums_b[:n])]
    mean_abs = sum(diffs) / n
    # Squash to [0,1) smoothly.
    return float(mean_abs / (mean_abs + 1.0))


@dataclass(frozen=True)
class _CodeFingerprint:
    sig: str
    nums: list[float]
    node_hist: Counter[str]


def _fingerprint(code: str) -> _CodeFingerprint:
    try:
        sig, nums = _signature_and_numbers(code)
    except Exception:
        sig, nums = ("", [])
    return _CodeFingerprint(sig=sig, nums=nums, node_hist=_node_type_hist(code))


def _distance(a: _CodeFingerprint, b: _CodeFingerprint, *, alpha_struct: float) -> float:
    if a.sig and b.sig and a.sig == b.sig:
        # Same structure: only numeric constant drift matters.
        return _numeric_distance(a.nums, b.nums)
    struct = _cosine_distance(a.node_hist, b.node_hist)
    num = _numeric_distance(a.nums, b.nums)
    alpha = float(alpha_struct)
    if not math.isfinite(alpha):
        alpha = 0.7
    alpha = max(0.0, min(1.0, alpha))
    return alpha * struct + (1.0 - alpha) * num


@dataclass
class Species:
    id: int
    representative_id: int
    member_ids: list[int]
    rep_fingerprint: _CodeFingerprint


def _assign_species(
    individuals: list[Individual],
    code_by_id: dict[int, str],
    *,
    threshold: float,
    alpha_struct: float,
) -> list[Species]:
    species_list: list[Species] = []
    next_species_id = 1

    for ind in individuals:
        code = code_by_id.get(ind.id, "")
        fp = _fingerprint(code)
        chosen: Species | None = None
        best_d = float("inf")
        for s in species_list:
            d = _distance(fp, s.rep_fingerprint, alpha_struct=alpha_struct)
            if d < best_d:
                best_d = d
                chosen = s
        if chosen is None or best_d > threshold:
            s = Species(
                id=next_species_id,
                representative_id=ind.id,
                member_ids=[ind.id],
                rep_fingerprint=fp,
            )
            next_species_id += 1
            species_list.append(s)
            ind.species_id = s.id
        else:
            chosen.member_ids.append(ind.id)
            ind.species_id = chosen.id

    return species_list


def _format_history(history: list[Attempt], *, seed: int, n_random: int, n_last: int) -> str:
    if not history:
        return "No prior attempts.\n"

    best = max(history, key=lambda a: (a.score_a, a.score_b))
    last = history[-n_last:] if n_last > 0 else []
    pool = [a for a in history if (a is not best and a not in last)]
    rng = random.Random(seed)
    sampled = rng.sample(pool, k=min(n_random, len(pool))) if pool and n_random > 0 else []

    chosen: list[tuple[str, Attempt]] = [("BEST", best)]
    for i, a in enumerate(sampled, start=1):
        chosen.append((f"RANDOM_{i}", a))
    for i, a in enumerate(last, start=1):
        chosen.append((f"LAST_{i}", a))

    seen: set[tuple[int, int]] = set()
    out: list[str] = []
    for label, a in chosen:
        key = (a.generation, hash(a.code_path))
        if key in seen:
            continue
        seen.add(key)
        snippet = "\n".join((a.code or "").strip().splitlines()[:30])
        out.append(f"{label}: score_a={a.score_a:.6f} score_b={a.score_b:.6f}")
        out.append("```python")
        out.append(snippet)
        out.append("```")
        if len(out) >= 6:
            break
    return "\n".join(out) + "\n"


def _mutation_prompt(history: list[Attempt], *, seed: int, cfg: NeatBipedalConfig) -> str:
    return (
        "Output exactly ONE fenced Python code block and nothing else.\n"
        "Do not write comments.\n"
        "Define exactly this function signature:\n"
        "def main(obs: list[float]) -> list[float]:\n"
        "\n"
        "Contract:\n"
        "- `obs` is a list of floats.\n"
        "- Return a list of 4 floats (will be clipped to [-1, 1]).\n"
        "- Imports: you may `import math` only.\n"
        "- Do not read/write files, do not use network, do not print.\n"
        "\n"
        "Goal: maximize two scalar scores: `score_a` and `score_b`.\n"
        "Higher is better for both.\n"
        "\n"
        "You are evolving a policy; make a meaningful topological change (not just tiny constant tweaks).\n"
        "Previous attempts (labels + scores + code):\n"
        f"{_format_history(history, seed=seed, n_random=cfg.mutation_prompt_n_random, n_last=cfg.mutation_prompt_n_last)}\n"
    )


def _crossover_prompt(
    parent_a: tuple[str, tuple[float, float]],
    parent_b: tuple[str, tuple[float, float]],
) -> str:
    code_a, (sa, sb) = parent_a
    code_b, (ta, tb) = parent_b
    return (
        "Output exactly ONE fenced Python code block and nothing else.\n"
        "Do not write comments.\n"
        "Define exactly this function signature:\n"
        "def main(obs: list[float]) -> list[float]:\n"
        "\n"
        "Contract:\n"
        "- `obs` is a list of floats.\n"
        "- Return a list of 4 floats (will be clipped to [-1, 1]).\n"
        "- Imports: you may `import math` only.\n"
        "- Do not read/write files, do not use network, do not print.\n"
        "\n"
        "Task: Create a new policy by CROSSING OVER the two parent programs below.\n"
        "- Preserve any good structural ideas from each parent.\n"
        "- Make the result stable (avoid domain errors, divide-by-zero).\n"
      
        "\n"
        f"PARENT_A: score_a={sa:.6f} score_b={sb:.6f}\n"
        "```python\n"
        f"{code_a}\n"
        "```\n"
        f"PARENT_B: score_a={ta:.6f} score_b={tb:.6f}\n"
        "```python\n"
        f"{code_b}\n"
        "```\n"
    )


def _write_code(run_dir: Path, *, generation: int, individual_id: int, variant: str, code: str) -> Path:
    gen_dir = run_dir / f"gen_{generation:03d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    path = gen_dir / f"ind_{individual_id:03d}_{variant}.py"
    path.write_text(clean_generated_code(code), encoding="utf-8")
    return path


def _select_top(
    individuals: list[Individual],
    *,
    n: int,
    key,
) -> list[Individual]:
    return sorted(individuals, key=key, reverse=True)[: max(0, int(n))]


def _sample_weighted(rng: random.Random, items: list[Individual], weights: list[float]) -> Individual:
    # fallback to uniform if all weights are non-positive
    total = sum(w for w in weights if w > 0)
    if not math.isfinite(total) or total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    acc = 0.0
    for it, w in zip(items, weights):
        if w <= 0 or not math.isfinite(w):
            continue
        acc += w
        if acc >= r:
            return it
    return items[-1]


def _adjusted_tuple(score: tuple[float, float], *, species_size: int) -> tuple[float, float]:
    s = max(1, int(species_size))
    return (score[0] / s, score[1] / s)


def _load_state(run_dir: Path) -> tuple[NeatBipedalConfig, list[Individual], int]:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state file: {state_path}")
    obj = json.loads(state_path.read_text(encoding="utf-8"))
    cfg_obj = obj.get("config") if isinstance(obj, dict) else None
    cfg = DEFAULT_CONFIG
    if isinstance(cfg_obj, dict):
        try:
            cfg = NeatBipedalConfig(**cfg_obj)
        except Exception:
            cfg = DEFAULT_CONFIG

    pop_obj = obj.get("population") if isinstance(obj, dict) else None
    population: list[Individual] = []
    if isinstance(pop_obj, list):
        for one in pop_obj:
            if not isinstance(one, dict):
                continue
            ind_id = one.get("id")
            if not isinstance(ind_id, int):
                continue
            hist: list[Attempt] = []
            hist_obj = one.get("history")
            if isinstance(hist_obj, list):
                for h in hist_obj:
                    if not isinstance(h, dict):
                        continue
                    try:
                        hist.append(
                            Attempt(
                                generation=int(h["generation"]),
                                score_a=float(h["score_a"]),
                                score_b=float(h["score_b"]),
                                avg_return=h.get("avg_return"),
                                avg_distance=h.get("avg_distance"),
                                avg_speed=h.get("avg_speed"),
                                episodes=h.get("episodes"),
                                step_errors=h.get("step_errors"),
                                error=h.get("error"),
                                code_path=str(h["code_path"]),
                                code=str(h.get("code") or ""),
                            )
                        )
                    except Exception:
                        continue
            population.append(Individual(id=int(ind_id), history=hist))
    gen = obj.get("generation")
    generation = int(gen) if isinstance(gen, int) else 0
    return cfg, population, generation


def _save_state(run_dir: Path, *, cfg: NeatBipedalConfig, population: list[Individual], generation: int) -> None:
    state_path = run_dir / "state.json"
    payload = {
        "generation": int(generation),
        "config": _jsonable(cfg.__dict__),
        "population": [
            {
                "id": ind.id,
                "species_id": ind.species_id,
                "parent_ids": ind.parent_ids,
                "operator": ind.operator,
                "history": [_jsonable(a.__dict__) for a in ind.history],
            }
            for ind in population
        ],
    }
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

def _safe_mean(xs: list[float]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _best_attempt_overall(population: list[Individual]) -> tuple[Attempt, int] | None:
    best: Attempt | None = None
    best_id: int | None = None
    for ind in population:
        for a in ind.history:
            if best is None or (a.score_a, a.score_b) > (best.score_a, best.score_b):
                best = a
                best_id = ind.id
    if best is None or best_id is None:
        return None
    return best, best_id


def _append_stats(
    run_dir: Path,
    *,
    generation: int,
    population: list[Individual],
    species: list[Species],
) -> None:
    scores_a = [ind.score_tuple()[0] for ind in population]
    scores_b = [ind.score_tuple()[1] for ind in population]

    # Use avg_distance/avg_speed when present (same as score_a/score_b in this experiment).
    avg_distances: list[float] = []
    avg_speeds: list[float] = []
    for ind in population:
        a = ind.latest
        if not a:
            continue
        if isinstance(a.avg_distance, (int, float)) and math.isfinite(float(a.avg_distance)):
            avg_distances.append(float(a.avg_distance))
        if isinstance(a.avg_speed, (int, float)) and math.isfinite(float(a.avg_speed)):
            avg_speeds.append(float(a.avg_speed))

    best = max(population, key=lambda ind: ind.score_tuple())
    best_attempt = best.latest
    best_so_far = _best_attempt_overall(population)

    species_sizes = sorted((len(s.member_ids) for s in species), reverse=True)
    payload: dict[str, Any] = {
        "generation": int(generation),
        "population_size": int(len(population)),
        "species_count": int(len(species)),
        "species_sizes": species_sizes,
        "mean_score_a": _safe_mean(scores_a),
        "mean_score_b": _safe_mean(scores_b),
        "mean_avg_distance": _safe_mean(avg_distances),
        "mean_avg_speed": _safe_mean(avg_speeds),
        "best_in_generation": {
            "id": int(best.id),
            "species_id": int(best.species_id or 0),
            "score_a": float(best.score_tuple()[0]),
            "score_b": float(best.score_tuple()[1]),
            "code_path": best.code_path,
            "attempt": _jsonable(best_attempt.__dict__) if best_attempt else None,
        },
        "best_overall": None,
    }
    if best_so_far is not None:
        a, ind_id = best_so_far
        payload["best_overall"] = {
            "id": int(ind_id),
            "generation": int(a.generation),
            "score_a": float(a.score_a),
            "score_b": float(a.score_b),
            "code_path": a.code_path,
        }

    with (run_dir / "stats.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _validate_policy_code(code: str) -> None:
    validate_sandboxed_code(code, allowed_import_roots={"math"})
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            if len(node.args.args) != 1:
                raise ValueError("`main` must take exactly 1 positional arg: obs")
            return
    raise ValueError("Missing `main(obs)`")


def _evaluate_and_record(
    code: str,
    *,
    cfg: NeatBipedalConfig,
    run_dir: Path,
    generation: int,
    individual_id: int,
    variant: str,
    meta: dict[str, Any],
) -> Attempt:
    cleaned = clean_generated_code(extract_python_code(code))
    code_path = _write_code(
        run_dir,
        generation=generation,
        individual_id=individual_id,
        variant=variant,
        code=cleaned,
    )

    result: dict[str, Any] | None = None
    error: str | None = None
    try:
        _validate_policy_code(cleaned)
        result = evaluate_policy(
            cleaned,
            env_id=cfg.env_id,
            seeds=list(cfg.seeds),
            max_steps=cfg.max_steps,
            timeout_s=cfg.timeout_s,
        )
        if not result.get("ok"):
            error = str(result.get("error") or "eval_failed")
    except Exception as e:
        error = str(e)
        result = {"ok": False, "error": "exception", "message": error}

    avg_distance = (result or {}).get("avg_distance")
    avg_speed = (result or {}).get("avg_speed")
    avg_return = (result or {}).get("avg_return")
    try:
        score_a = float(avg_distance) if avg_distance is not None else float("-inf")
    except Exception:
        score_a = float("-inf")
    try:
        score_b = float(avg_speed) if avg_speed is not None else float("-inf")
    except Exception:
        score_b = float("-inf")

    attempt = Attempt(
        generation=generation,
        score_a=score_a,
        score_b=score_b,
        avg_return=float(avg_return) if isinstance(avg_return, (int, float)) else None,
        avg_distance=float(avg_distance) if isinstance(avg_distance, (int, float)) else None,
        avg_speed=float(avg_speed) if isinstance(avg_speed, (int, float)) else None,
        episodes=(result or {}).get("episodes") if isinstance((result or {}).get("episodes"), list) else None,
        step_errors=int((result or {}).get("step_errors", 0)) if (result or {}).get("step_errors") is not None else None,
        error=error,
        code_path=str(code_path.relative_to(run_dir)),
        code=cleaned,
    )

    rec = {
        "generation": generation,
        "individual_id": individual_id,
        "variant": variant,
        "meta": meta,
        "attempt": _jsonable(attempt.__dict__),
        "eval": _jsonable(result),
    }
    with (run_dir / "population.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return attempt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", type=Path, default=None, help="Resume from an existing neat_bipedal run dir.")
    p.add_argument("--generations", type=int, default=None, help="Override generations in config.")
    args = p.parse_args()

    cfg = DEFAULT_CONFIG
    run_dir: Path
    population: list[Individual]
    start_generation = 0

    if args.checkpoint_path is not None:
        run_dir = args.checkpoint_path.expanduser().resolve()
        cfg, population, start_generation = _load_state(run_dir)
    else:
        run_dir = _default_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        population = [Individual(id=i + 1, history=[], operator="init") for i in range(cfg.population_size)]
        (run_dir / "population.jsonl").write_text("", encoding="utf-8")
        (run_dir / "stats.jsonl").write_text("", encoding="utf-8")

    if args.generations is not None:
        cfg = NeatBipedalConfig(**{**cfg.__dict__, "generations": int(args.generations)})

    # Snapshot meta/config.
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {
                    "experiment": "neat_bipedal_blackbox",
                    "run_dir": str(run_dir),
                    "config": _jsonable(cfg.__dict__),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    rng = random.Random(0)

    def _ensure_pop_size(pop: list[Individual]) -> list[Individual]:
        pop = pop[: cfg.population_size]
        while len(pop) < cfg.population_size:
            pop.append(Individual(id=len(pop) + 1, history=[]))
        return pop

    population = _ensure_pop_size(population)

    for gen in range(start_generation + 1, start_generation + 1 + cfg.generations):
        # (1) Initialization: if an individual has no attempt yet, create + eval it now.
        for ind in population:
            if ind.latest is not None:
                continue
            seed = gen * 10_000 + ind.id
            prompt = _mutation_prompt(ind.history, seed=seed, cfg=cfg)
            resp = call_ai(
                prompt,
                concurrent_calls=cfg.llm_samples_per_child,
                temperature=cfg.temperature,
                model=cfg.model,
            )[0]
            attempt = _evaluate_and_record(
                resp.choices[0].message.content,
                cfg=cfg,
                run_dir=run_dir,
                generation=gen,
                individual_id=ind.id,
                variant="init",
                meta={"operator": "init", "seed": seed},
            )
            ind.history.append(attempt)
            ind.operator = "init"

        # Build code map for current population.
        code_by_id: dict[int, str] = {ind.id: (ind.latest.code if ind.latest else "") for ind in population}

        # (2) Speciate current population.
        species_list = _assign_species(
            population,
            code_by_id,
            threshold=cfg.species_threshold,
            alpha_struct=cfg.distance_alpha_struct,
        )
        species_size_by_id = {s.id: len(s.member_ids) for s in species_list}

        # (3) Selection: raw elites + adjusted survival (fitness sharing).
        elites = _select_top(population, n=cfg.n_elites, key=lambda ind: ind.score_tuple())
        adjusted_ranked = sorted(
            population,
            key=lambda ind: _adjusted_tuple(
                ind.score_tuple(),
                species_size=species_size_by_id.get(ind.species_id or 0, 1),
            ),
            reverse=True,
        )
        n_survive = max(1, int(math.ceil(cfg.population_size * cfg.survival_frac)))
        survivors = adjusted_ranked[:n_survive]

        kept_ids: set[int] = set()
        parents: list[Individual] = []
        for ind in elites + survivors:
            if ind.id in kept_ids:
                continue
            kept_ids.add(ind.id)
            parents.append(ind)

        # (4) Build next generation: carry parents unchanged + spawn new children.
        next_pop: list[Individual] = []
        for ind in parents:
            next_pop.append(
                Individual(
                    id=ind.id,
                    history=list(ind.history[-cfg.max_history_per_individual :]),
                    operator="carry",
                    species_id=ind.species_id,
                )
            )

        needed = cfg.population_size - len(next_pop)
        if needed < 0:
            next_pop = next_pop[: cfg.population_size]
            needed = 0

        parent_weights: list[float] = []
        for ind in parents:
            size = species_size_by_id.get(ind.species_id or 0, 1)
            w = _adjusted_tuple(ind.score_tuple(), species_size=size)[0]
            parent_weights.append(w if math.isfinite(w) else 0.0)

        next_id = max((ind.id for ind in next_pop), default=0) + 1

        for _ in range(needed):
            use_cx = rng.random() < cfg.crossover_prob and len(parents) >= 2
            child = Individual(id=next_id, history=[], operator="crossover" if use_cx else "mutate")
            next_id += 1

            if use_cx:
                p1 = _sample_weighted(rng, parents, parent_weights)
                p2 = _sample_weighted(rng, parents, parent_weights)
                tries = 0
                while p2.id == p1.id and tries < 10:
                    p2 = _sample_weighted(rng, parents, parent_weights)
                    tries += 1
                prompt = _crossover_prompt((code_by_id.get(p1.id, ""), p1.score_tuple()), (code_by_id.get(p2.id, ""), p2.score_tuple()))
                resp = call_ai(
                    prompt,
                    concurrent_calls=cfg.llm_samples_per_child,
                    temperature=cfg.temperature,
                    model=cfg.model,
                )[0]
                attempt = _evaluate_and_record(
                    resp.choices[0].message.content,
                    cfg=cfg,
                    run_dir=run_dir,
                    generation=gen,
                    individual_id=child.id,
                    variant="cx",
                    meta={"operator": "crossover", "parents": [p1.id, p2.id]},
                )
                inherited = (p1.history[-cfg.max_history_per_individual // 2 :] if p1.history else []) + (
                    p2.history[-cfg.max_history_per_individual // 2 :] if p2.history else []
                )
                child.history = inherited[-cfg.max_history_per_individual :] + [attempt]
                child.parent_ids = (p1.id, p2.id)
            else:
                p1 = _sample_weighted(rng, parents, parent_weights)
                seed = gen * 10_000 + child.id
                prompt = _mutation_prompt(p1.history[-cfg.max_history_per_individual :], seed=seed, cfg=cfg)
                resp = call_ai(
                    prompt,
                    concurrent_calls=cfg.llm_samples_per_child,
                    temperature=cfg.temperature,
                    model=cfg.model,
                )[0]
                attempt = _evaluate_and_record(
                    resp.choices[0].message.content,
                    cfg=cfg,
                    run_dir=run_dir,
                    generation=gen,
                    individual_id=child.id,
                    variant="mut",
                    meta={"operator": "mutate", "parent": p1.id, "seed": seed},
                )
                child.history = list(p1.history[-cfg.max_history_per_individual :]) + [attempt]
                child.parent_ids = (p1.id, -1)

            next_pop.append(child)

        # (5) Speciate the next population (for state + summary).
        next_code_by_id = {ind.id: (ind.latest.code if ind.latest else "") for ind in next_pop}
        next_species = _assign_species(
            next_pop,
            next_code_by_id,
            threshold=cfg.species_threshold,
            alpha_struct=cfg.distance_alpha_struct,
        )

        _append_stats(run_dir, generation=gen, population=next_pop, species=next_species)

        best = max(next_pop, key=lambda ind: ind.score_tuple())
        print(
            json.dumps(
                {
                    "generation": gen,
                    "species_count": len(next_species),
                    "best": {
                        "id": best.id,
                        "species_id": best.species_id,
                        "score_a": best.score_tuple()[0],
                        "score_b": best.score_tuple()[1],
                        "code_path": best.code_path,
                    },
                },
                indent=2,
            )
        )

        _save_state(run_dir, cfg=cfg, population=next_pop, generation=gen)
        population = _ensure_pop_size(next_pop)


if __name__ == "__main__":
    main()
