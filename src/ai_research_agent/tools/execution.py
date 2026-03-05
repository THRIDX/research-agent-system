"""Code execution tools: Docker sandboxed and local subprocess execution."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    return_code: int
    duration_seconds: float
    timed_out: bool = False


def run_local(
    code: str,
    *,
    working_dir: Optional[Path] = None,
    timeout: float = 60.0,
    env: Optional[dict[str, str]] = None,
) -> ExecutionResult:
    """Execute Python code in a local subprocess.

    The code is written to a temporary script file and executed with the
    current Python interpreter.
    """
    import os
    import sys
    import tempfile

    working_dir = working_dir or Path.cwd()
    working_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        dir=working_dir,
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(code)
        script_path = f.name

    run_env = dict(os.environ)
    if env:
        run_env.update(env)

    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
            env=run_env,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        return_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return_code = -1
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass

    duration = time.monotonic() - start
    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        return_code=return_code,
        duration_seconds=duration,
        timed_out=timed_out,
    )


def run_in_docker(
    code: str,
    *,
    image: str = "python:3.11-slim",
    working_dir: Optional[Path] = None,
    timeout: float = 120.0,
    memory_limit: str = "512m",
    cpu_quota: int = 50000,  # 50% of one CPU
    extra_volumes: Optional[dict[str, dict[str, str]]] = None,
) -> ExecutionResult:
    """Execute Python code inside a Docker container (sandboxed).

    Requires Docker daemon to be running and the `docker` Python SDK installed.
    """
    import docker  # type: ignore[import-untyped]
    import tempfile
    import os

    working_dir = working_dir or Path(tempfile.mkdtemp())
    working_dir.mkdir(parents=True, exist_ok=True)

    script_path = working_dir / "_run.py"
    script_path.write_text(code, encoding="utf-8")

    volumes: dict[str, dict[str, str]] = {
        str(working_dir.resolve()): {"bind": "/workspace", "mode": "rw"}
    }
    if extra_volumes:
        volumes.update(extra_volumes)

    client = docker.from_env()
    start = time.monotonic()
    timed_out = False
    try:
        container = client.containers.run(
            image,
            command=["python", "/workspace/_run.py"],
            volumes=volumes,
            working_dir="/workspace",
            mem_limit=memory_limit,
            cpu_quota=cpu_quota,
            network_disabled=False,
            remove=False,
            detach=True,
        )
        try:
            result = container.wait(timeout=timeout)
            return_code = result.get("StatusCode", -1)
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
        except Exception:
            timed_out = True
            container.kill()
            stdout = ""
            stderr = "Container execution timed out."
            return_code = -1
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass
    except docker.errors.DockerException as exc:
        duration = time.monotonic() - start
        return ExecutionResult(
            stdout="",
            stderr=f"Docker error: {exc}",
            return_code=-1,
            duration_seconds=duration,
        )

    duration = time.monotonic() - start
    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        return_code=return_code,
        duration_seconds=duration,
        timed_out=timed_out,
    )
