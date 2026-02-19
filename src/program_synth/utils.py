from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return jsonable(dump())
        except Exception:
            pass
    asdict = getattr(obj, "__dict__", None)
    if isinstance(asdict, dict):
        try:
            return jsonable(asdict)
        except Exception:
            pass
    return repr(obj)


def repo_python_from_file(file_path: str) -> str:
    """
    Prefer project-root virtualenv Python (`.venv/bin/python`) for a module file path.
    Falls back to the current interpreter.
    """
    here = Path(file_path).resolve()
    repo_root = here.parents[2]
    venv_python = repo_root / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable
