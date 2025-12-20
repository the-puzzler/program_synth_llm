Quickstart

- Set `GROQ_API_KEY` in your environment.
- Generate code and run it:
  - `uv run run_ai_codegen.py`

What this does

- Builds a "code-only" prompt requiring the model to output a single Python code block defining `main(...)`.
- Extracts the code block, validates that `main` exists and has the right number of inputs, then runs it in a subprocess and returns JSON.

Black-box experiment (XOR)

- Iteratively asks the model for a classifier `main(x0: float, x1: float) -> int` and only returns a scalar reward (accuracy) plus history:
  - `uv run xor_fitness_loop.py`

Visualize decision boundaries

- After a run, render one combined figure (`all_boundaries.png`) with subplots for every `iter_###.py`:
  - Latest run: `uv run visualize_boundaries.py`
  - Specific run: `uv run visualize_boundaries.py runs/xor_YYYYMMDDTHHMMSSZ`
