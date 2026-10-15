from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from banana_harness.types import CommandResult, HarnessError

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def read_toml(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)[:80] + "-" + digest(value)[:10]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(data, file, indent=2, default=str, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def run_process(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 60,
    output_limit: int | None = 100_000,
    input_data: bytes | None = None,
) -> CommandResult:
    """Spool output to disk; kill the entire process group on timeout/cancellation."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    start = time.monotonic()
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as input_file:
        if input_data is not None:
            input_file.write(input_data)
            input_file.seek(0)
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdin=input_file if input_data is not None else subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            raise HarnessError(f"Cannot execute {argv[0]}: {error}") from error
        expired = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            expired = True
        finally:
            # Also dispose of background children after an otherwise successful command.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        size = output.tell()
        output.seek(0)
        if output_limit is not None and size > output_limit:
            half = output_limit // 2
            head = output.read(half)
            output.seek(-half, os.SEEK_END)
            data = head + f"\n... {size - 2 * half} bytes omitted ...\n".encode() + output.read()
        else:
            data = output.read()
    return CommandResult(
        data.decode(errors="replace"),
        124 if expired else process.returncode,
        expired,
        time.monotonic() - start,
    )


def checked(argv: list[str], **kwargs: Any) -> str:
    kwargs.setdefault("env", child_env())
    result = run_process(argv, **kwargs)
    if result.returncode:
        raise HarnessError(f"Command failed ({result.returncode}): {argv[0]}\n{result.output}")
    return result.output.strip()


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    # Credentials used by the model driver should not be inherited accidentally by commands.
    result = {
        key: value
        for key, value in os.environ.items()
        if not re.search(r"(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", key, re.I)
    }
    result.update(PAGER="cat", MANPAGER="cat", PIP_PROGRESS_BAR="off", TQDM_DISABLE="1")
    result.update(extra or {})
    return result
