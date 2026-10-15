"""Standalone subprocess worker for functional LiveCodeBench cases; stdlib only."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

# Mirrors the convenient standard-library names available in coding contest evaluators.
PRELUDE = "\n".join(
    f"from {module} import *\nimport {module}"
    for module in (
        "string",
        "re",
        "datetime",
        "collections",
        "heapq",
        "bisect",
        "copy",
        "math",
        "random",
        "statistics",
        "itertools",
        "functools",
        "operator",
        "typing",
    )
)
PRELUDE += "\nfrom builtins import *"


def main() -> None:
    request_file, result_file = map(Path, sys.argv[1:3])
    request = json.loads(request_file.read_text())
    namespace = {"__name__": "__solution__"}
    output = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            # Compile separately so a solution's __future__ imports remain valid.
            exec(compile(PRELUDE, "<prelude>", "exec"), namespace)
            exec(compile(request["code"], "solution.py", "exec"), namespace)
            function = request["func_name"]
            if "Solution" in namespace:
                method = getattr(namespace["Solution"](), function)
            else:
                method = namespace[function]
            value = method(*[json.loads(line) for line in request["input"].splitlines()])
            output = {"value": value}
    except BaseException:
        output = {"error": traceback.format_exc(limit=5)}
    result_file.write_text(json.dumps(output))


if __name__ == "__main__":
    main()
