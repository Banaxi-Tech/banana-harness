from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from banana_harness.adapters.harbor import load_harbor
from banana_harness.adapters.livecode import livecode_task
from banana_harness.adapters.swebench import swebench_task
from banana_harness.types import HarnessError, Task

BENCHMARKS = (
    "swebench-verified",
    "swebench-pro",
    "livecodebench-v6",
    "deepswe",
    "terminal-bench-2.1",
)


def load_tasks(
    benchmark: str,
    source: str | Path | None = None,
    *,
    allow_downloads: bool = False,
    revision: str | None = None,
    split: str = "test",
    dataset_config: str | None = None,
) -> Iterator[Task]:
    if benchmark not in (*BENCHMARKS, "custom"):
        raise HarnessError(f"Unknown benchmark: {benchmark}")
    if source is not None:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            yield from load_harbor(path, benchmark)
            return
        if not path.is_file():
            raise HarnessError(f"Task source does not exist: {path}")
        with path.open() as file:
            for index, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    task = row_to_task(benchmark, row)
                    if task.repo and not task.repo.startswith(("https://", "http://", "git@")):
                        task.repo = str((path.parent / task.repo).resolve())
                    yield task
                except (KeyError, TypeError, ValueError) as error:
                    raise HarnessError(f"Invalid task on {path}:{index}: {error}") from error
        return
    if not allow_downloads:
        raise HarnessError(
            "Supply --source for local tasks, or explicitly enable --allow-downloads."
        )
    if benchmark not in {"swebench-verified", "livecodebench-v6"}:
        raise HarnessError(f"Use banana fetch {benchmark} --dest PATH, then --source PATH.")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise HarnessError(
            "Install banana-harness[datasets] to load Hugging Face datasets."
        ) from error
    dataset_id = "princeton-nlp/SWE-bench_Verified"
    config = dataset_config
    if benchmark == "livecodebench-v6":
        dataset_id, config = "livecodebench/code_generation_lite", "release_v6"
    # Parquet configs avoid deprecated trust_remote_code dataset scripts.
    dataset = load_dataset(dataset_id, name=config, split=split, revision=revision, streaming=True)
    for row in dataset:
        task = row_to_task(benchmark, row)
        task.metadata.update(dataset=dataset_id, dataset_config=config, dataset_revision=revision)
        yield task


def row_to_task(benchmark: str, row: dict) -> Task:
    if benchmark == "swebench-verified":
        return swebench_task(row)
    if benchmark == "livecodebench-v6":
        return livecode_task(row)
    if benchmark == "swebench-pro" and "instance_id" in row:
        raise HarnessError(
            "SWE-bench Pro JSON rows omit the complete verifier. Supply its Harbor task "
            "directory (v2/tasks), or a normalized Task JSONL with explicit test_command."
        )
    fields = dict(row)
    fields.setdefault("benchmark", benchmark)
    if fields["benchmark"] != benchmark:
        raise HarnessError("Task benchmark disagrees with selected benchmark.")
    return Task(**fields)


def select_tasks(
    tasks: Iterable[Task],
    ids: set[str] | None = None,
    limit: int | None = None,
) -> Iterator[Task]:
    if limit is not None and limit < 1:
        raise HarnessError("limit must be positive")
    count = 0
    seen: set[str] = set()
    found: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise HarnessError(f"Duplicate task ID: {task.id}")
        seen.add(task.id)
        if ids and task.id not in ids:
            continue
        found.add(task.id)
        yield task
        count += 1
        if limit is not None and count >= limit:
            return
        if ids and found == ids:
            return
    if ids and found != ids:
        raise HarnessError(f"Task IDs not found: {sorted(ids - found)}")
