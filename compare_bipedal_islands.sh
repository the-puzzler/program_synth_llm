#!/usr/bin/env bash
set -euo pipefail

# Fixed experiment setup:
# - 10 runs per variant
# - 20 iterations per run
# - Regular: concurrent=10
# - Islands: 10 lineages, concurrent=1, stagnation_steps=5

REPS=10
STEPS=20

PYTHON_CMD="uv run python"

OUT_CSV="bipedal_islands_comparison.csv"
echo "run,variant,best_score_a" > "${OUT_CSV}"

for variant in regular islands; do
  for i in $(seq 1 "${REPS}"); do
    if [ "${variant}" = "regular" ]; then
      PYTHONPATH=src ${PYTHON_CMD} -m program_synth.loops.bipedal_fitness_loop --iterations "${STEPS}" --concurrent 10
      run_dir=$(ls -dt runs/bipedal_* | head -n 1)
    else
      PYTHONPATH=src ${PYTHON_CMD} -m program_synth.loops.bipedal_fitness_loop_islands --iterations "${STEPS}" --lineages 10 --concurrent 1 --stagnation-steps 5
      run_dir=$(ls -dt runs/bipedal_islands_* | head -n 1)
    fi

    best_score=$(
      python - "$run_dir" << 'EOF'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
attempts = run_dir / "attempts.jsonl"
best = float("-inf")
if not attempts.exists():
    print(best)
    sys.exit(0)

with attempts.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        v = obj.get("score_a")
        if v is None:
            continue
        try:
            v = float(v)
        except Exception:
            continue
        if v > best:
            best = v

print(best)
EOF
    )

    echo "${i},${variant},${best_score}" | tee -a "${OUT_CSV}"
  done
done

echo "Results written to ${OUT_CSV}"
