# Program Synthesis LLM

This repo runs simple black-box “program evolution” loops where an LLM proposes small Python programs and we score them in an environment.

**Bipedal (best GIF + policy)**

![best_biped.gif](best_biped.gif)

`policy.py` (the policy used to render `best_biped.gif`):

```python
def main(obs: list[float]) -> list[float]:
    import math
    if len(obs) < 4:
        obs = obs + [0.0] * (4 - len(obs))
    a, b, c, d = obs[:4]
    f1 = math.tanh(10*a + 8*c + 8.5*b*d + 7.9*a*b + 7.8*c*d)
    f2 = math.tanh(8.5*b - 7.9*c + 8.8*d + 8.3*a*c + 7.9*c*d + 7.6*a*d)
    f3 = math.tanh(7.9*a*c + 7.9*b*d + 7.8*a - 7.2*b + 8.4*c + 7.8*a*b + 8.1*c*d)
    f4 = math.tanh(8.1*c*d - 7.2*a*b - 8.9*d + 7.5*b - 8*a*c)
    out0 = math.tanh(f1 + 7.4*f2 + 8.5*f3 + 3.2*f4 + 8.5*a*d - 2.5*b*c)
    out1 = math.tanh(8*f1 - 8.2*f2 + 8.8*f3 - 3.1*f4 + 2.5*d)
    out2 = math.tanh(8.3*f1 * f3 - 8*f2 * f4 + 2.5*c - 2.5*a*d)
    out3 = math.tanh(f1 - f2 + f3 - f4)
    return [out0, out1, out2, out3]
```

**XOR (reference solution)**

```python
def main(x0: float, x1: float) -> int:
    return 1 if (x0 > 0) != (x1 > 0) else 0
```

![XOR-boundries.png](XOR-boundries.png)

Quickstart

- Install deps (once per machine):
  - `uv sync`
- Set API key:
  - `export OPENROUTER_API_KEY="..."` (or put it in `.env`)

Black-box experiment (XOR)

- Iteratively asks the model for a classifier `main(x0: float, x1: float) -> int` and only returns a scalar reward (accuracy) plus history:
  - `uv run xor_fitness_loop.py`
  - After a run, render one combined figure (`all_boundaries.png`) with subplots for every `iter_###.py`:
  - Latest run: `uv run visualize_boundaries.py`
  - Specific run: `uv run visualize_boundaries.py runs/xor_YYYYMMDDTHHMMSSZ`

Black-box experiment (BipedalWalker)

- Iteratively asks the model for a policy `main(obs: list[float]) -> list[float]` (4-dim action) and scores average episode return:
  - Install env deps: `uv add "gymnasium[box2d]"` (or otherwise install Gymnasium + Box2D)
  - Run: `uv run bipedal_fitness_loop.py`
  - Render GIFs (per-iteration) after a run:
    - `uv run visualize_bipedal_gifs.py` (defaults to latest `runs/bipedal_*`)
    - Collage (sequential) GIF: `uv run visualize_bipedal_gifs.py --collage --collage-limit 16`
  - Render a one-off GIF from a specific policy:
    - `uv run render_bipedal_gif.py --code-path policy.py`

NEAT-inspired experiment (BipedalWalker) (currently not effective/working)

- A population-based loop with selection, crossover, and speciation (fitness sharing):
  - Run: `uv run neat_bipedal_fitness_loop.py`
  - Resume: `uv run neat_bipedal_fitness_loop.py --checkpoint-path runs/neat_bipedal_YYYYMMDDTHHMMSSZ`

Black-box experiment (Atari Pong)

- Iteratively asks the model for a policy `main(obs: list[float]) -> int` and scores average episode return:
  - Install env deps: `uv add "gymnasium[atari,accept-rom-license]"`
  - Run: `uv run pong_fitness_loop.py`

## How It Works (Very Briefly)

- The LLM outputs a single Python function `main(...)` that defines an agent/program.
- We run that code in a sandbox and score it (XOR accuracy, or Bipedal distance/speed).
- We keep a history of attempts and feed a small slice back into the prompt to bias the next proposal.
- In the NEAT-style variant, we keep a population, select survivors/elites, create new programs via mutation/crossover prompts, and use a simple AST-based distance to form species and apply fitness sharing.
