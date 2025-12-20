from __future__ import annotations

import json

from ai_code_env import (
    FunctionSpec,
    build_code_only_prompt,
    extract_python_code,
    run_code_main,
    validate_generated_code,
)
from call_ai_utils import call_ai


def main() -> None:
    spec = FunctionSpec(input_types=("int", "int"), output_types=("int",))
    task = "Return the greatest common divisor (GCD) of the two inputs."
    prompt = build_code_only_prompt(task, spec)

    responses = call_ai(prompt, concurrent_calls=1, temperature=0.2)
    code = extract_python_code(responses[0].choices[0].message.content)
    validate_generated_code(code, spec)

    run_result = run_code_main(code, inputs=[54, 24], timeout_s=20)
    print(json.dumps(run_result, indent=2))


if __name__ == "__main__":
    main()

