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

Black-box experiment (BipedalWalker)

- Iteratively asks the model for a policy `main(obs: list[float]) -> list[float]` (4-dim action) and scores average episode return:
  - Install env deps: `uv add "gymnasium[box2d]"` (or otherwise install Gymnasium + Box2D)
  - Run: `uv run bipedal_fitness_loop.py`
  - Render GIFs (per-iteration) after a run:
    - `uv run visualize_bipedal_gifs.py` (defaults to latest `runs/bipedal_*`)
    - Collage (sequential) GIF: `uv run visualize_bipedal_gifs.py --collage --collage-limit 16`

Visualize decision boundaries

- After a run, render one combined figure (`all_boundaries.png`) with subplots for every `iter_###.py`:
  - Latest run: `uv run visualize_boundaries.py`
  - Specific run: `uv run visualize_boundaries.py runs/xor_YYYYMMDDTHHMMSSZ`

Black-box experiment (Image Reconstruction)

- Add an `image.jpeg` in the repo root, then run an LLM loop that learns `main(x: float, y: float) -> [r,g,b]` from a hidden dataset scored by MSE:
  - `uv run image_recon_fitness_loop.py`
  - Visualize all reconstructions in a run as a subplot collage:
    - `uv run visualize_image_recon.py runs/image_YYYYMMDDTHHMMSSZ`
    - Or latest run: `uv run visualize_image_recon.py`
