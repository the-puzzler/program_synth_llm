from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NeatBipedalConfig:
    # Core run sizes
    population_size: int = 20
    generations: int = 50

    # Selection
    survival_frac: float = 0.20
    n_elites: int = 2

    # Reproduction
    crossover_prob: float = 0.30
    llm_samples_per_child: int = 1  # "concurrent requests per population member"
    mutation_prompt_n_random: int = 2
    mutation_prompt_n_last: int = 2
    max_history_per_individual: int = 30  # cap history carried into children for prompting/state size

    # Speciation / novelty protection
    species_threshold: float = 0.3
    distance_alpha_struct: float = 0.70  # weight on structural distance (vs numeric distance)

    # Evaluation
    env_id: str = "BipedalWalker-v3"
    seeds: tuple[int, ...] = (0, 1, 2)
    max_steps: int = 6000
    timeout_s: float = 180.0

    # Model
    model: str = "moonshotai/kimi-k2-instruct-0905"
    temperature: float = 1.0

    # Parallelism
    eval_workers: int = 4


DEFAULT_CONFIG = NeatBipedalConfig()
