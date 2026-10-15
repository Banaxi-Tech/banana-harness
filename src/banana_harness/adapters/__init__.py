from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import quote

from banana_harness.adapters.harbor import load_harbor
from banana_harness.adapters.klein import SUBSETS, klein_task
from banana_harness.adapters.livecode import livecode_task
from banana_harness.adapters.swebench import swebench_task
from banana_harness.types import HarnessError, Task

BENCHMARKS = (
    "klein-code-bench",
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
    if benchmark not in {"swebench-verified", "livecodebench-v6", "klein-code-bench"}:
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
        if dataset_config not in {None, config}:
            raise HarnessError("livecodebench-v6 supports only dataset_config='release_v6'")
        if split != "test":
            raise HarnessError("LiveCodeBench release_v6 provides only the 'test' split")
        # Calling load_dataset with the repo ID discovers its legacy Python script,
        # even with name='release_v6'. Match that release's upstream file mapping
        # explicitly and use the built-in JSON reader instead (datasets 4+).
        base_url = (
            f"https://huggingface.co/datasets/{dataset_id}/resolve/"
            f"{quote(revision or 'main', safe='')}"
        )
        filenames = ["test.jsonl", *(f"test{i}.jsonl" for i in range(2, 7))]
        dataset = load_dataset(
            "json",
            data_files={"test": [f"{base_url}/{name}" for name in filenames]},
            split=split,
            streaming=True,
        )
    else:
        if benchmark == "klein-code-bench":
            dataset_id, config = "klein-ml/klein-code-bench", dataset_config or "default"
            if config not in (*SUBSETS, "default"):
                raise HarnessError(
                    "Klein-Code-Bench config must be default, easy, intermediate, hard or max"
                )
            if split != "test":
                raise HarnessError("Klein-Code-Bench provides only the 'test' split")
        dataset = load_dataset(
            dataset_id, name=config, split=split, revision=revision, streaming=True
        )
    for row in dataset:
        task = row_to_task(benchmark, row)
        task.metadata.update(dataset=dataset_id, dataset_config=config, dataset_revision=revision)
        yield task


def row_to_task(benchmark: str, row: dict) -> Task:
    if benchmark == "klein-code-bench":
        return klein_task(row)
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
    *,
    random_limit: int | None = None,
    seed: int = 0,
) -> Iterator[Task]:
    """Select a prefix, or sample up to random_limit tasks reproducibly by task ID."""
    if random_limit is not None:
        if limit is not None:
            raise HarnessError("limit and random_limit are mutually exclusive")
        if random_limit < 1:
            raise HarnessError("random_limit must be positive")
        # Validate the full source and apply ID filtering before sampling. Sorting makes
        # the subset independent of source ordering; use a private RNG for model runs.
        candidates = sorted(select_tasks(tasks, ids=ids), key=lambda task: task.id)
        yield from random.Random(seed).sample(candidates, min(random_limit, len(candidates)))
        return
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
