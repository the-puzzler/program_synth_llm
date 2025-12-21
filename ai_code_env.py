from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_JOINED_KEYWORD_RE = re.compile(r"(?i)(\d)(and|or|not|in|is)\b")


@dataclass(frozen=True)
class FunctionSpec:
    input_types: tuple[str, ...]
    output_types: tuple[str, ...]

    def signature_hint(self) -> str:
        inputs = ", ".join(f"x{i}: {t}" for i, t in enumerate(self.input_types))
        if len(self.output_types) == 0:
            out = "None"
        elif len(self.output_types) == 1:
            out = self.output_types[0]
        else:
            out = f"tuple[{', '.join(self.output_types)}]"
        return f"def main({inputs}) -> {out}:"

    def contract_text(self) -> str:
        return (
            "Return outputs exactly matching the spec:\n"
            f"- inputs:  {list(self.input_types)}\n"
            f"- outputs: {list(self.output_types)}\n"
        )


def build_code_only_prompt(task: str, spec: FunctionSpec) -> str:
    return (
        "Write Python code only.\n"
        "Output exactly ONE fenced code block, and nothing else.\n"
        "Inside the code block:\n"
        "- Define a function named `main` with this signature:\n"
        f"  {spec.signature_hint()}\n"
        "- Do not read input() and do not print.\n"
        "- Imports are allowed only from a small safe whitelist (e.g. `math`, `numpy`).\n"
        "- Do not import `os`, `sys`, `subprocess`, or anything that touches files/network.\n"
        "- Put all logic inside functions; `main` should call them.\n"
        "- `main` must return the required number of outputs.\n"
        "\n"
        f"{spec.contract_text()}\n"
        "Task:\n"
        f"{task}\n"
    )


def extract_python_code(text: str) -> str:
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return clean_generated_code(match.group(1))
    return clean_generated_code(text)


def clean_generated_code(code: str) -> str:
    """
    Best-effort cleanup for common LLM artifacts like a stray leading `py`/`python` line.
    Keeps behavior minimal and predictable: only removes a single leading language tag line.
    """
    normalized = code.replace("\ufeff", "").strip()
    # Fix a common tokenization artifact that can trigger `SyntaxWarning: invalid decimal literal`,
    # e.g. `0and` / `1or` / `2in` (missing whitespace after a number).
    normalized = _JOINED_KEYWORD_RE.sub(r"\1 \2", normalized)
    lines = normalized.splitlines()

    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines):
        first = lines[i].strip().lower()
        if first in {"py", "python"}:
            lines.pop(i)

    cleaned = "\n".join(lines).strip() + "\n"

    # Always provide `math` to generated code (keep any `from __future__ import ...` at the top).
    out_lines = cleaned.splitlines()
    has_math_import = any(
        ln.strip() == "import math" or ln.strip().startswith("import math,") or ln.strip().startswith("import math ")
        for ln in out_lines
    )
    if not has_math_import:
        insert_at = 0
        while insert_at < len(out_lines) and out_lines[insert_at].startswith("from __future__ import "):
            insert_at += 1
        out_lines.insert(insert_at, "import math")
        cleaned = "\n".join(out_lines).strip() + "\n"

    return cleaned


def validate_generated_code(code: str, spec: FunctionSpec) -> None:
    tree = ast.parse(code)
    main_def = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_def = node
            break
    if main_def is None:
        raise ValueError("Generated code must define a `main` function.")

    if len(main_def.args.args) != len(spec.input_types):
        raise ValueError(
            f"`main` must take {len(spec.input_types)} positional args; got {len(main_def.args.args)}."
        )


def validate_sandboxed_code(code: str, *, allowed_import_roots: set[str] | None = None) -> None:
    tree = ast.parse(code)

    if allowed_import_roots is None:
        allowed_import_roots = {
            "math",
            "random",
            "itertools",
            "functools",
            "statistics",
            "numpy",
        }

    banned_calls = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "print",
    }
    banned_names = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "shutil",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed_import_roots:
                    raise ValueError(f"Import `{alias.name}` is not allowed in generated code.")
        if isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                raise ValueError("Relative imports are not allowed in generated code.")
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root not in allowed_import_roots:
                raise ValueError(f"Import `from {module} ...` is not allowed in generated code.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in banned_calls:
                raise ValueError(f"Call to `{node.func.id}` is not allowed in generated code.")
        if isinstance(node, ast.Name) and node.id in banned_names:
            raise ValueError(f"Name `{node.id}` is not allowed in generated code.")


def run_code_main(
    code: str,
    *,
    inputs: list[Any],
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    runner_src = r"""
import importlib.util
import json
import sys


def _load_module(path: str):
    try:
        spec = importlib.util.spec_from_file_location("user_code", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {"ok": True, "mod": mod}
    except Exception as e:
        return {"ok": False, "error": "load_failed", "message": str(e)}


def main() -> int:
    user_path = sys.argv[1]
    payload = json.loads(sys.stdin.read() or "{}")
    inputs = payload.get("inputs", None)
    batch_inputs = payload.get("batch_inputs", None)

    loaded = _load_module(user_path)
    if not loaded.get("ok"):
        print(json.dumps(loaded))
        return 2

    mod = loaded["mod"]
    if not hasattr(mod, "main"):
        print(json.dumps({"ok": False, "error": "missing_main"}))
        return 3

    try:
        if inputs is not None and batch_inputs is not None:
            print(json.dumps({"ok": False, "error": "ambiguous_inputs"}))
            return 4
        if batch_inputs is not None:
            results = []
            errors = 0
            for one in batch_inputs:
                try:
                    results.append(mod.main(*one))
                except Exception:
                    results.append(None)
                    errors += 1
            print(json.dumps({"ok": True, "results": results, "errors": errors}))
            return 0
        if inputs is not None:
            result = mod.main(*inputs)
            print(json.dumps({"ok": True, "result": result}))
            return 0
        print(json.dumps({"ok": False, "error": "missing_inputs"}))
        return 5
    except Exception as e:
        print(json.dumps({"ok": False, "error": "exception", "message": str(e)}))
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
"""

    with tempfile.TemporaryDirectory(prefix="ai_code_env_") as td:
        td_path = Path(td)
        user_path = td_path / "user_code.py"
        runner_path = td_path / "runner.py"

        user_path.write_text(clean_generated_code(code), encoding="utf-8")
        runner_path.write_text(runner_src, encoding="utf-8")

        venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
        py = str(venv_python) if venv_python.exists() else sys.executable

        proc = subprocess.run(
            [py, str(runner_path), str(user_path)],
            input=json.dumps({"inputs": inputs}),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )

        out = proc.stdout.strip()
        err = proc.stderr.strip()
        payload: dict[str, Any]

        if out:
            try:
                payload = json.loads(out)
            except json.JSONDecodeError:
                payload = {"ok": False, "error": "non_json_stdout", "stdout": out, "stderr": err}
        else:
            payload = {"ok": False, "error": "empty_stdout", "stdout": out, "stderr": err}

        payload.setdefault("returncode", proc.returncode)
        if err:
            payload.setdefault("stderr", err)
        return payload


def run_code_main_batch(
    code: str,
    *,
    batch_inputs: list[list[Any]],
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    runner_src = r"""
import importlib.util
import json
import sys


def _load_module(path: str):
    try:
        spec = importlib.util.spec_from_file_location("user_code", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {"ok": True, "mod": mod}
    except Exception as e:
        return {"ok": False, "error": "load_failed", "message": str(e)}


def main() -> int:
    user_path = sys.argv[1]
    payload = json.loads(sys.stdin.read() or "{}")
    inputs = payload.get("inputs", None)
    batch_inputs = payload.get("batch_inputs", None)

    loaded = _load_module(user_path)
    if not loaded.get("ok"):
        print(json.dumps(loaded))
        return 2

    mod = loaded["mod"]
    if not hasattr(mod, "main"):
        print(json.dumps({"ok": False, "error": "missing_main"}))
        return 3

    try:
        if inputs is not None and batch_inputs is not None:
            print(json.dumps({"ok": False, "error": "ambiguous_inputs"}))
            return 4
        if batch_inputs is not None:
            results = []
            errors = 0
            for one in batch_inputs:
                try:
                    results.append(mod.main(*one))
                except Exception:
                    results.append(None)
                    errors += 1
            print(json.dumps({"ok": True, "results": results, "errors": errors}))
            return 0
        if inputs is not None:
            result = mod.main(*inputs)
            print(json.dumps({"ok": True, "result": result}))
            return 0
        print(json.dumps({"ok": False, "error": "missing_inputs"}))
        return 5
    except Exception as e:
        print(json.dumps({"ok": False, "error": "exception", "message": str(e)}))
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
"""

    with tempfile.TemporaryDirectory(prefix="ai_code_env_") as td:
        td_path = Path(td)
        user_path = td_path / "user_code.py"
        runner_path = td_path / "runner.py"

        user_path.write_text(clean_generated_code(code), encoding="utf-8")
        runner_path.write_text(runner_src, encoding="utf-8")

        venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
        py = str(venv_python) if venv_python.exists() else sys.executable

        proc = subprocess.run(
            [py, str(runner_path), str(user_path)],
            input=json.dumps({"batch_inputs": batch_inputs}),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )

        out = proc.stdout.strip()
        err = proc.stderr.strip()
        payload: dict[str, Any]

        if out:
            try:
                payload = json.loads(out)
            except json.JSONDecodeError:
                payload = {"ok": False, "error": "non_json_stdout", "stdout": out, "stderr": err}
        else:
            payload = {"ok": False, "error": "empty_stdout", "stdout": out, "stderr": err}

        payload.setdefault("returncode", proc.returncode)
        if err:
            payload.setdefault("stderr", err)
        return payload

