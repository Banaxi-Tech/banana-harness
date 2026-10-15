from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

from banana_harness.adapters import BENCHMARKS, load_tasks, select_tasks
from banana_harness.agent import FINISH, AgentConfig, LiteLLMModel, ScriptedModel
from banana_harness.download import SOURCES, fetch
from banana_harness.runner import RunConfig, Runner
from banana_harness.transformers_model import TransformersModel
from banana_harness.types import HarnessError, Task
from banana_harness.util import digest, write_json


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Host coding benchmarks without a Docker daemon")
    root.add_argument("--version", action="version", version="banana-harness 0.1.0")
    sub = root.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Read-only diagnostics; no downloads or image runs")
    doctor.add_argument("--json", action="store_true")
    smoke = sub.add_parser("smoke", help="Run a synthetic native task offline, without an API key")
    smoke.add_argument("--output", type=Path, default=Path("jobs/smoke"))
    provision = sub.add_parser("fetch", help="Explicitly download benchmark task definitions")
    provision.add_argument("benchmark", choices=list(SOURCES))
    provision.add_argument("--dest", type=Path, required=True)
    provision.add_argument("--revision")
    provision.add_argument("--allow-downloads", action="store_true")
    listing = sub.add_parser("list", help="Inspect local task metadata, without downloading")
    listing.add_argument("benchmark", choices=(*BENCHMARKS, "custom"))
    listing.add_argument("--source", required=True)
    list_limit = listing.add_mutually_exclusive_group()
    list_limit.add_argument("--limit", type=int, help="First N tasks (default: 20)")
    list_limit.add_argument(
        "--random-limit", type=int, help="Sample up to N tasks without replacement"
    )
    listing.add_argument("--seed", type=int, default=0, help="Seed for --random-limit (default: 0)")
    run = sub.add_parser("run", help="Run tasks; remote provisioning is opt-in")
    run.add_argument("benchmark", choices=(*BENCHMARKS, "custom"))
    run.add_argument("--source", help="Local Harbor task directory or JSONL")
    model = run.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", help="Provider/model name or Transformers model ID/local path")
    model.add_argument("--replay", type=Path, help="JSON array of model responses (offline)")
    run.add_argument("--model-runtime", choices=("litellm", "transformers"), default="litellm")
    run.add_argument("--model-options", type=Path, help="JSON object of model runtime options")
    run.add_argument(
        "--trust-remote-code", action="store_true",
        help="Allow custom tokenizer/model Python code with the Transformers runtime",
    )
    run.add_argument("--backend", choices=("auto", "native", "rootfs"), default="auto")
    run.add_argument("--output", type=Path, default=Path("jobs/default"))
    run.add_argument("--cache", type=Path, default=Path(".banana/udocker"))
    run.add_argument("--allow-downloads", action="store_true")
    run.add_argument("--revision", help="Hugging Face dataset revision")
    run.add_argument("--split", default="test")
    run.add_argument("--dataset-config")
    run.add_argument("--id", action="append", dest="ids")
    run_limit = run.add_mutually_exclusive_group()
    run_limit.add_argument("--limit", type=int, help="First N tasks in source order")
    run_limit.add_argument(
        "--random-limit", type=int, help="Sample up to N tasks without replacement"
    )
    run.add_argument("--seed", type=int, default=0, help="Seed for --random-limit (default: 0)")
    run.add_argument("--max-steps", type=int, default=100)
    run.add_argument("--command-timeout", type=float, default=120)
    run.add_argument("--max-cost", type=float)
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--no-venv", action="store_true")
    run.add_argument("--rootfs-mode", choices=("P1", "P2"), default="P1")
    run.add_argument(
        "--lcb-evaluator",
        type=Path,
        help="Existing local LiveCodeBench checkout; uses its official run_test",
    )
    return root


def diagnostics() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "commands": {name: shutil.which(name) for name in ("bash", "git", "udocker")},
        "optional_modules": {
            name: importlib.util.find_spec(name) is not None
            for name in (
                "litellm", "transformers", "torch", "accelerate",
                "datasets", "swebench", "klein_code_bench",
            )
        },
        "nested_container": Path("/.dockerenv").exists(),
        "notebook": "ipykernel" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG")),
        "rootfs_note": "P1/P2 require working ptrace. No root or Docker daemon required.",
        "downloads_attempted": False,
    }


def smoke(output: Path) -> dict:
    replies = [
        "```bash\nprintf 'host harness works\\n' > answer.txt\n```",
        f"```bash\n{FINISH}\n```",
    ]
    task = Task(
        id="synthetic-smoke",
        benchmark="custom",
        prompt="Write 'host harness works' to answer.txt.",
        test_command="test \"$(cat answer.txt)\" = 'host harness works'",
    )
    runner = Runner(
        lambda: ScriptedModel(replies),
        RunConfig(output=output, backend="native", use_venv=False),
        model_identity=digest(replies),
    )
    return runner.run([task])


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            report = diagnostics()
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(f"Python {report['python']} on {report['platform']}")
                for name, path in report["commands"].items():
                    print(f"{name}: {path or 'not installed'}")
                for name, available in report["optional_modules"].items():
                    print(f"{name}: {'installed' if available else 'optional, not installed'}")
                print(report["rootfs_note"])
                print("No downloads or benchmark execution performed.")
            return 0
        if args.command == "fetch":
            print(
                fetch(
                    args.benchmark,
                    args.dest,
                    allow_downloads=args.allow_downloads,
                    revision=args.revision,
                )
            )
            return 0
        if args.command == "list":
            limit = args.limit
            if limit is None and args.random_limit is None:
                limit = 20
            tasks = select_tasks(
                load_tasks(args.benchmark, args.source), limit=limit,
                random_limit=args.random_limit, seed=args.seed,
            )
            for task in tasks:
                print(
                    json.dumps(
                        {
                            "id": task.id,
                            "benchmark": task.benchmark,
                            "kind": task.kind,
                            "image": task.image,
                        }
                    )
                )
            return 0
        if args.command == "smoke":
            summary = smoke(args.output)
        else:
            options = json.loads(args.model_options.read_text()) if args.model_options else {}
            if not isinstance(options, dict):
                raise HarnessError("--model-options must contain a JSON object")
            if args.replay and args.model_runtime != "litellm":
                raise HarnessError("--model-runtime applies to --model, not --replay")
            if args.trust_remote_code and (args.replay or args.model_runtime != "transformers"):
                raise HarnessError("--trust-remote-code requires --model-runtime transformers")
            if args.replay:
                replies = json.loads(args.replay.read_text())
                if not isinstance(replies, list) or not all(isinstance(x, str) for x in replies):
                    raise HarnessError("--replay must contain a JSON array of response strings")

                def factory():
                    return ScriptedModel(replies)

                identity = digest(replies)
            elif args.model_runtime == "transformers":
                if "allow_downloads" in options:
                    raise HarnessError("Use --allow-downloads instead of a model-options entry")
                if args.trust_remote_code:
                    options["trust_remote_code"] = True
                try:
                    local_model = TransformersModel(
                        args.model, allow_downloads=args.allow_downloads, **options
                    )
                except TypeError as error:
                    raise HarnessError(f"Invalid Transformers model options: {error}") from error

                def factory():
                    return local_model

                identity = digest(
                    {"runtime": "transformers", "model": args.model, "options": options}
                )
            else:

                def factory():
                    return LiteLLMModel(args.model, **options)

                identity = digest({"model": args.model, "options": options})
            tasks = select_tasks(
                load_tasks(
                    args.benchmark,
                    args.source,
                    allow_downloads=args.allow_downloads,
                    revision=args.revision,
                    split=args.split,
                    dataset_config=args.dataset_config,
                ),
                ids=set(args.ids) if args.ids else None,
                limit=args.limit,
                random_limit=args.random_limit,
                seed=args.seed,
            )
            config = RunConfig(
                output=args.output,
                backend=args.backend,
                cache=args.cache,
                allow_downloads=args.allow_downloads,
                resume=not args.no_resume,
                use_venv=not args.no_venv,
                rootfs_mode=args.rootfs_mode,
                livecode_evaluator=args.lcb_evaluator,
                agent=AgentConfig(
                    max_steps=args.max_steps,
                    command_timeout=args.command_timeout,
                    max_cost=args.max_cost,
                ),
            )
            summary = Runner(factory, config, model_identity=identity).run(tasks)
        write_json(args.output / "summary.json", summary)
        print(
            f"{summary['passed']}/{summary['tasks']} passed; {summary['graded']} graded. "
            f"Artifacts: {args.output.resolve()}"
        )
        return 0 if summary["tasks"] and summary["passed"] == summary["tasks"] else 1
    except (HarnessError, ValueError, OSError) as error:
        print(f"banana: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("banana: interrupted; saved artifacts can be resumed", file=sys.stderr)
        return 130
