"""Invoke an existing local upstream evaluator in a disposable subprocess."""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text())
    # Keep handles before the upstream evaluator applies any reliability guard.
    output = open(sys.argv[2], "w")
    dumps = json.dumps
    try:
        spec = importlib.util.spec_from_file_location("lcb_testing_util", request["module"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        results, metadata = module.run_test(
            request["sample"],
            test=request["code"],
            debug=False,
            timeout=request["timeout"],
        )
        # Numpy scalar results need conversion for JSON serialization.
        normalized = [value.item() if hasattr(value, "item") else value for value in results]
        output.write(dumps({"results": normalized, "metadata": metadata}, default=str))
    except BaseException:
        output.write(dumps({"error": traceback.format_exc(limit=8)}))
    finally:
        output.close()


if __name__ == "__main__":
    main()
