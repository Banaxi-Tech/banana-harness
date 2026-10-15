from __future__ import annotations

import base64
import io
import json
import pickle
import zlib

from banana_harness.types import HarnessError, Task


class _StringUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise HarnessError("Private test pickle contains executable objects, not plain test data.")


def decode_cases(value) -> list[dict]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        # Upstream private tests use base64(zlib(pickle(JSON string))).
        raw = zlib.decompress(base64.b64decode(value))
        decoded = _StringUnpickler(io.BytesIO(raw)).load()
        if not isinstance(decoded, str):
            raise HarnessError("Expected a JSON string inside the private test pickle.") from None
        result = json.loads(decoded)
    if not isinstance(result, list):
        raise HarnessError("LiveCodeBench test cases must be a list.")
    return result


def livecode_task(row: dict) -> Task:
    public = decode_cases(row.get("public_test_cases", "[]"))
    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    starter = row.get("starter_code") or ""
    prompt = row["question_content"]
    if starter:
        prompt += "\n\nStarter code:\n```python\n" + starter + "\n```"
    prompt += (
        "\n\nWrite the complete Python solution to solution.py. "
        "For stdin tasks read standard input and write standard output. "
        "For functional tasks preserve the starter interface."
    )
    if public:
        prompt += "\n\nPublic examples:\n" + json.dumps(public, ensure_ascii=False)
    return Task(
        id=str(row["question_id"]),
        benchmark="livecodebench-v6",
        kind="livecode",
        prompt=prompt,
        workdir=".",
        metadata={
            "release": "release_v6",
            "platform": row.get("platform"),
            "contest_date": row.get("contest_date"),
            "protocol": "bash-agent/pass@1 (host evaluator)",
        },
        grading={
            "public": public,
            "private_encoded": row.get("private_test_cases", "[]"),
            "func_name": metadata.get("func_name"),
        },
    )
