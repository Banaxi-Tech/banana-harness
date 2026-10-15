# Banana Harness

A Python package for running coding agents and benchmark verifiers on a Linux host,
including notebook runtimes and machines already inside a container. No Docker daemon,
Docker CLI, privileged container, or nested Docker is required.

The agent follows mini SWE Agent's core design: one bash action per response, a linear
message history, independent shells, and a final submission command. It is an independent
implementation, rather than the upstream mini SWE Agent package.

**Status: experimental 0.1.0.** The package is tested with synthetic offline tasks. Real
benchmark images, complete suites, API model calls, and Colab/Molab execution have not
been validated in this repository. Packaging is prepared for PyPI; it has not been published.

## Install and try it

Python 3.10+ on Linux, with `bash` and `git` available:

```bash
# From this repository, before publication:
pip install -e .
banana doctor
banana smoke
```

After publication, the intended notebook installation is:

```python
%pip install "banana-harness[all]"
```

The core package has no runtime dependencies on Python 3.11+. Optional extras:

| Extra | Purpose |
| --- | --- |
| `models` | LiteLLM API providers and compatible local model servers |
| `transformers` | Direct local Hugging Face Transformers inference with PyTorch and Accelerate |
| `datasets` | Streaming Hugging Face dataset loading |
| `rootfs` | udocker 1.3.17 for daemon-free image execution |
| `swebench` | Pinned SWE-bench 4.1.0 evaluation scripts and parsers |
| `livecode` | NumPy, needed by a separately supplied upstream LiveCodeBench evaluator |
| `all` | All of the above |
| `dev` | Tests, lint, build, and package validation |

Imports, `doctor`, `list`, and `smoke` do not fetch benchmark data, images, or model weights.
Remote dataset loading, repository cloning, image pulls, and Transformers model downloads
require `--allow-downloads` or
`RunConfig(allow_downloads=True)`. Local task sources and cached images work without that
flag. This switch controls harness provisioning; it does not firewall agent commands,
user-supplied setup scripts, verifier scripts, or model API requests.

## Local Transformers models

Run a causal chat model directly in the harness process, without LiteLLM, an API key, or
an inference server. Install only the extras you need:

```bash
pip install -e '.[transformers]'
banana run custom --source /path/to/tasks.jsonl \
  --model-runtime transformers --model /path/to/local-chat-model \
  --model-options /path/to/model-options.json --output jobs/local
```

For example, `model-options.json` can contain:

```json
{
  "device_map": "auto",
  "dtype": "auto",
  "max_new_tokens": 1024,
  "generation_kwargs": {"do_sample": false},
  "chat_template_kwargs": {"enable_thinking": false}
}
```

The default device is CPU; set `device` to `cuda` or `cuda:0` for one GPU, or use
`device_map: "auto"` for Accelerate placement. With a device map, the `device` option is
ignored. `chat_template_kwargs` passes model-specific options to the tokenizer; omit it
when unnecessary. The runtime uses the tokenizer's
[chat template](https://huggingface.co/docs/transformers/chat_templating) and decodes only
the generated continuation. Models must support the agent's system/user/assistant chat
history. Context overflow raises an error instead of silently truncating history.

`--model` also accepts a Hugging Face model ID. Only local/cached files are loaded by
default; add `--allow-downloads` to fetch weights/tokenizer files. Set `revision` in the
options file to pin the model independently of the dataset's `--revision`. For models
with custom Python implementations, add `--trust-remote-code` or set
`"trust_remote_code": true` in the options file. This executes the model repository's
Python code. The default is false and the option applies to both model and tokenizer.

Python/notebook usage keeps one model instance across tasks:

```python
from pathlib import Path
from banana_harness import RunConfig, Runner, TransformersModel, load_tasks

model = TransformersModel(
    "/path/to/local-chat-model",
    device_map="auto",
    max_new_tokens=1024,
    trust_remote_code=False,
)
runner = Runner(lambda: model, RunConfig(output=Path("jobs/local")))
summary = runner.run(load_tasks("custom", "/path/to/tasks.jsonl"))
```

Weights load on the first query and are reused across tasks. Generation defaults to greedy
decoding; sampling options go in `generation_kwargs`. Each trajectory records prompt and
completion token counts. Local inference records zero API cost; hardware costs are not
estimated. Task time limits are checked between generations/commands, so an in-flight
generation may exceed the deadline. CPU inference is supported; practical model size and
speed depend on available RAM/VRAM. Only the tiny synthetic local checkpoint is tested
offline; full coding checkpoints and GPU placement still require integration validation.

## Execution paths

`native` runs bash directly in a separate workspace and creates a task virtualenv by default.
It is suitable for LiveCodeBench, custom tasks, and Harbor tasks with an explicit native
profile. A virtualenv separates Python packages; it does not pin the host OS or installed
compilers. `--no-venv` uses the existing notebook runtime.

`rootfs` downloads or reuses an OCI image, extracts it to ordinary files, and executes its
programs with udocker's PRoot P1/P2 modes. These are host processes, with translated filesystem
paths and simulated root, preserving the image's tools and libraries. They need neither
root nor a Docker daemon. Images are still necessary for this path. `auto` chooses rootfs
when the task has an image, otherwise native. Dockerfiles are never built or executed by the
harness.

Install the PRoot execution tools explicitly, once, using the same cache as the harness:

```bash
pip install -e '.[rootfs,models,datasets,swebench]'
export UDOCKER_DIR="$PWD/.banana/udocker"
udocker install
# If the enclosing notebook/container runs as root: udocker --allow-root install
```

`udocker install` is an explicit network operation. The harness disables udocker's automatic
tool downloads. If binaries cannot be downloaded, install an offline udockertools archive
using udocker's `UDOCKER_TARBALL` setting. See the
[udocker installation manual](https://indigo-dc.github.io/udocker/installation_manual.html).

PRoot needs usable `ptrace`. Some notebook policies deny it. P2 can help with P1/seccomp
compatibility, but cannot bypass a host policy that prohibits tracing. In that case, use native
profiles and supply the dependencies already available on the host. No backend here offers
a security sandbox, network isolation, or the official container CPU/memory/storage limits.
Requested resources and these runtime differences are saved in results.

## Benchmark adapters

| Suite | Task source | Execution and grading |
| --- | --- | --- |
| SWE-bench Verified | Local official JSONL or `princeton-nlp/SWE-bench_Verified` | Published instance image, patch replay into a fresh rootfs, official 4.1.0 evaluation script and report parser |
| SWE-bench Pro | Current public V2 Harbor tasks (`v2/tasks`) | Task image and original verifier; native profiles optional |
| LiveCodeBench v6 | Local official JSONL or `livecodebench/code_generation_lite`, `release_v6` | Python stdin/functional tasks, private tests decoded only during grading; optional existing upstream evaluator checkout |
| DeepSWE | Datacurve DeepSWE Harbor tasks | Original collect hooks and separate fresh verifier environment |
| Terminal-Bench 2.1 | Its dedicated 2.1 Harbor task repository | Task image, agent filesystem retained for verification, `/tests/test.sh` and reward artifacts |

The Pro adapter targets public V2 tasks. V1 JSON rows alone lack complete verifier files;
they are rejected instead of guessing tests. An explicitly normalized custom Task with
setup/test commands can represent other versions. DeepSWE refers to the
[Datacurve benchmark](https://github.com/datacurve-ai/deep-swe).

### Reuse a random subset across models

`--limit 100` selects the first 100 tasks in source order. Use `--random-limit 100 --seed 42`
to sample up to 100 distinct tasks reproducibly instead:

```bash
banana run swebench-verified --source /path/to/verified.jsonl \
  --model-runtime transformers --model /path/to/model-a --backend rootfs \
  --random-limit 100 --seed 42 --output jobs/model-a
```

Repeat with the same source, random limit, and seed for each model, changing `--model` and
`--output`. Selection is independent of the model and source ordering: eligible tasks are
sorted by ID before sampling. Keep the task population unchanged (or pin `--revision` for
a remote dataset). The default seed is `0`. `--limit` and `--random-limit` are mutually
exclusive; repeated `--id` filters apply before random sampling. Sampling reads the full
task source before execution. If fewer than N tasks are available, all are selected.

Preview the same subset without loading a model or running tasks:

```bash
banana list swebench-verified --source /path/to/verified.jsonl --random-limit 100 --seed 42
```

Python also supports `select_tasks(tasks, random_limit=100, seed=42)`.

### SWE-bench Verified

```bash
export YOUR_PROVIDER_API_KEY=...
banana run swebench-verified \
  --source /path/to/verified.jsonl \
  --model provider/model --limit 1 --backend rootfs \
  --output jobs/verified --allow-downloads
```

Omit `--source` to stream the Hugging Face dataset with explicit download opt-in. Use
`--revision COMMIT` to pin the dataset. The default instance image naming follows the
published x86_64 SWE-bench images. A JSONL row's `image` field can supply an immutable digest
or an alternative image. The evaluator is pinned to 4.1.0; newer SWE-bench dataset schemas
or unsupported repositories require adaptation. The optional evaluator dependency may
install the Docker Python SDK, but this harness never creates a Docker client.

### Pro, DeepSWE, and Terminal-Bench

Fetch definitions only when you choose to do so, or point directly at existing task files:

```bash
banana fetch deepswe --dest datasets/deepswe --allow-downloads
banana fetch swebench-pro --dest datasets/pro --allow-downloads
banana fetch terminal-bench-2.1 --dest datasets/tb21 --allow-downloads

banana run deepswe --source datasets/deepswe/tasks \
  --model provider/model --limit 1 --allow-downloads --output jobs/deepswe
banana run swebench-pro --source datasets/pro/v2/tasks \
  --model provider/model --limit 1 --allow-downloads --output jobs/pro
banana run terminal-bench-2.1 --source datasets/tb21 \
  --model provider/model --limit 1 --allow-downloads --output jobs/tb21
```

Fetches default to the upstream repository's current branch. Pass `--revision TAG_OR_SHA`
to choose a release. `banana-source.json` records the resolved commit. The Terminal-Bench
source is the separate **2.1 repository**, not the 2.0 repository. Harbor task definitions,
verifier files, image metadata, and model configuration contribute to saved provenance.

DeepSWE's separate verifier receives the original collect artifacts. The harness commits
the final workspace before running collect commands so uncommitted edits and new files are
included. Neither `solution/` nor held-out `tests/` is copied into the agent rootfs. Ordinary
Harbor tasks are verified after the agent finishes; Terminal-Bench changes outside git
repositories remain available to its verifier.

### LiveCodeBench v6

```bash
banana run livecodebench-v6 --source /path/to/lcb-v6.jsonl \
  --model provider/model --backend native --limit 1 --output jobs/lcb
```

Without `--source`, `--allow-downloads` streams the six upstream JSONL files that define
`release_v6` (`test.jsonl` through `test6.jsonl`) using the built-in JSON reader. This avoids
the repository's legacy dataset script, which Datasets 4+ no longer supports. `--revision`
applies to all six file URLs. Only the `test` split and `release_v6` config are supported.
Random sampling reads all six files before selecting a subset, including encoded private
tests; allow sufficient download time and memory. The agent writes `solution.py`. Private cases are
decoded in the driver after generation and are never put in the prompt or workspace.
The restricted pickle decoder accepts plain data and rejects executable pickle globals.

The built-in evaluator supports stdin, top-level functions, `Solution` methods, standard
contest imports, per-case timeouts, and a total verifier timeout. Its comparison behavior
is a host approximation. To use the upstream grading implementation instead, supply an
existing local checkout:

```bash
pip install -e '.[livecode,models]'
banana run livecodebench-v6 --source /path/to/lcb-v6.jsonl \
  --lcb-evaluator /path/to/LiveCodeBench --model provider/model --output jobs/lcb
```

This invokes the checkout's `testing_util.py:run_test` in a subprocess and records its
content hash. It never downloads that checkout. The default generation protocol uses a
bash agent and one candidate per problem, so its scores differ from the usual single-turn
LiveCodeBench protocol even when using the upstream evaluator. All summaries explicitly
record `official_score: false`.

## Python and notebooks

```python
from pathlib import Path
from banana_harness import Runner, RunConfig
from banana_harness.adapters import load_tasks, select_tasks
from banana_harness.agent import LiteLLMModel

tasks = select_tasks(load_tasks("deepswe", "/path/to/deep-swe/tasks"), limit=1)
runner = Runner(
    lambda: LiteLLMModel("provider/model"),
    RunConfig(output=Path("jobs/notebook"), allow_downloads=False),
)
summary = runner.run(tasks)  # Reuses already provisioned images; no implicit pulls.
```

See [the notebook](examples/notebook_quickstart.ipynb), which starts with an offline smoke
test and defaults benchmark execution/download switches to false. Persistent notebook
storage can hold both the output directory and rootfs cache so a runtime restart can
resume completed tasks. Each task is run serially to reduce memory pressure.

## Native Harbor profiles

For environments without PRoot, add `banana.toml` next to a task's `task.toml`:

```toml
repo = "/path/to/local/repository"
base_commit = "YOUR_PINNED_COMMIT"
setup = []  # Explicit setup commands, executed in the task virtualenv.
test_command = "python .banana-tests/verify_native.py"

[env]
PYTHONHASHSEED = "0"
```

```bash
banana run terminal-bench-2.1 --source /path/to/task \
  --backend native --model provider/model --output jobs/native
```

The native verifier runs in a fresh checkout with the model patch replayed and held-out
files copied to `.banana-tests` only at grading time. Native profiles need to adapt absolute
paths and host dependencies explicitly. Separate Harbor verifier environments currently
require rootfs. This is not an automatic translation of arbitrary Dockerfiles.

## Artifacts, budgets, and resume

Each task saves `result.json` and an attempt directory containing `task.json`,
`trajectory.json`, `model.patch` where applicable, verifier output, and reward reports.
Jobs save `summary.json` and SWE-style `predictions.jsonl`. DeepSWE/Harbor logs also appear
under `logs/verifier` and collected files under `logs/artifacts`.

Passed and failed tasks resume only when task/config/model identities match. Errors and
timeouts retry in new attempt directories. Partial trajectories are saved after every step.
Only one writer can hold a job output directory. Default API model requests have a 120-second
request timeout; the overall task deadline is checked between requests and commands.

Use `--max-steps`, `--command-timeout`, and `--max-cost` for limits. A cost limit can be
exceeded by the final API request already in flight; no further action runs after that
request. Unknown provider pricing stops a cost-limited run instead of silently counting
it as free. Results retain the agent exit status and may grade partial work after a limit.
API keys matching common credential environment names are removed from child command
environments. As this is a host harness, commands can still access host files and network.

## Development and release

```bash
pip install -e '.[dev]'
pytest -q
ruff check src tests
python -m build
twine check dist/*
```

CI runs synthetic tests, builds the wheel/source distribution, checks metadata, and runs
an offline smoke task from the installed wheel. It never downloads benchmark tasks/images.
See [release notes and remaining validation](docs/RELEASING.md). Publication is a separate,
explicit step and no publishing workflow runs automatically.

Sources: [mini SWE Agent](https://github.com/SWE-agent/mini-swe-agent),
[SWE-bench](https://github.com/SWE-bench/SWE-bench),
[SWE-bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os),
[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench),
[DeepSWE](https://github.com/datacurve-ai/deep-swe),
[Terminal-Bench 2.1](https://github.com/harbor-framework/terminal-bench-2-1), and
[udocker](https://indigo-dc.github.io/udocker/user_manual.html).
