"""Docker sandbox manager — creates and manages isolated containers per user."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import docker
from docker.errors import APIError, ImageNotFound, NotFound
from docker.models.containers import Container
from sqlalchemy.orm import Session

from anzar.sandbox.limits import ResourceLimits, get_limits

logger = logging.getLogger("anzar.sandbox")

DOCKERFILE_DIR = Path(__file__).parent
IMAGE_TAG = "anzar-sandbox:latest"
CONTAINER_PREFIX = "anzar-user"


class DockerSandboxManager:
    """Manages Docker containers for user workspaces.

    Falls back to subprocess execution when Docker is unavailable (local dev).
    """

    def __init__(self, docker_host: str = "unix:///var/run/docker.sock"):
        self._docker_host = docker_host
        self._client: docker.DockerClient | None = None
        self._has_docker: bool | None = None

    def _get_client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def check_docker(self) -> bool:
        """Check if Docker daemon is reachable."""
        if self._has_docker is not None:
            return self._has_docker
        import socket
        # Quick check: is Docker socket/pipe reachable at all?
        if self._docker_host.startswith("unix://"):
            sock_path = self._docker_host.replace("unix://", "")
            if not Path(sock_path).exists():
                self._has_docker = False
                logger.warning("Docker socket %s not found — subprocess mode", sock_path)
                return False
        try:
            self._client = docker.DockerClient(base_url=self._docker_host, timeout=3)
            self._client.ping()
            self._has_docker = True
            logger.info("Docker daemon connected")
        except Exception as e:
            self._has_docker = False
            self._client = None
            logger.warning("Docker not available: %s — falling back to subprocess", e)
        return self._has_docker

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------

    def ensure_image(self) -> str:
        """Build the sandbox image if it doesn't exist. Returns the image tag."""
        if not self.check_docker():
            raise RuntimeError("Docker is not available")

        client = self._get_client()
        try:
            client.images.get(IMAGE_TAG)
            logger.debug("Image %s already exists", IMAGE_TAG)
        except ImageNotFound:
            logger.info("Building sandbox image %s ...", IMAGE_TAG)
            client.images.build(
                path=str(DOCKERFILE_DIR),
                tag=IMAGE_TAG,
                rm=True,
            )
            logger.info("Image %s built successfully", IMAGE_TAG)
        return IMAGE_TAG

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def create_container(
        self,
        user_id: str,
        plan: str = "free",
        disk_path: str | None = None,
    ) -> dict[str, Any]:
        """Create a new container for a user. Returns container info dict."""
        if not self.check_docker():
            raise RuntimeError("Docker is not available")

        limits = get_limits(plan)
        self.ensure_image()

        container_name = f"{CONTAINER_PREFIX}-{user_id[:8]}"
        workspace_host = disk_path or str(Path.home() / ".anzar" / "workspaces" / user_id)

        # Ensure host directory exists
        Path(workspace_host).mkdir(parents=True, exist_ok=True)

        client = self._get_client()

        # Remove existing container with same name (if stopped)
        self._remove_if_exists(container_name)

        container = client.containers.create(
            image=IMAGE_TAG,
            name=container_name,
            detach=True,
            volumes={
                workspace_host: {
                    "bind": "/home/sandbox/workspace",
                    "mode": "rw",
                },
            },
            mem_limit=limits.memory,
            cpu_quota=int(limits.cpu * 100000),
            network_disabled=False,
            working_dir="/home/sandbox/workspace",
            user="sandbox",
        )

        logger.info(
            "Created container %s for user %s (plan=%s)",
            container.short_id,
            user_id[:8],
            plan,
        )

        return {
            "container_id": container.id,
            "container_name": container_name,
            "short_id": container.short_id,
            "image": IMAGE_TAG,
            "disk_path": workspace_host,
            "memory_limit": limits.memory,
            "cpu_limit": limits.cpu,
        }

    def start_container(self, container_id: str) -> dict[str, Any]:
        """Start a stopped container."""
        if not self.check_docker():
            raise RuntimeError("Docker is not available")

        client = self._get_client()
        container = client.containers.get(container_id)
        container.start()

        logger.info("Started container %s", container.short_id)
        return self._container_info(container)

    def stop_container(self, container_id: str, timeout: int = 10) -> dict[str, Any]:
        """Gracefully stop a container."""
        if not self.check_docker():
            raise RuntimeError("Docker is not available")

        client = self._get_client()
        container = client.containers.get(container_id)
        container.stop(timeout=timeout)

        logger.info("Stopped container %s", container.short_id)
        return self._container_info(container)

    def destroy_container(self, container_id: str) -> None:
        """Remove a container (files on host are kept)."""
        if not self.check_docker():
            raise RuntimeError("Docker is not available")

        client = self._get_client()
        try:
            container = client.containers.get(container_id)
            container.remove(force=True)
            logger.info("Destroyed container %s", container.short_id)
        except NotFound:
            logger.warning("Container %s not found, already removed", container_id[:12])

    def _remove_if_exists(self, name: str) -> None:
        """Remove a container by name if it exists (force stop + remove)."""
        try:
            client = self._get_client()
            container = client.containers.get(name)
            container.remove(force=True)
            logger.debug("Removed existing container %s", name)
        except NotFound:
            pass

    # ------------------------------------------------------------------
    # Status & monitoring
    # ------------------------------------------------------------------

    def container_status(self, container_id: str) -> dict[str, Any]:
        """Get detailed status of a container."""
        if not self.check_docker():
            return {"state": "unavailable", "error": "Docker not available"}

        client = self._get_client()
        try:
            container = client.containers.get(container_id)
            return self._container_info(container)
        except NotFound:
            return {"state": "not_found"}

    def _container_info(self, container: Container) -> dict[str, Any]:
        """Extract status info from a Docker container object."""
        container.reload()
        state = container.status  # created, running, paused, restarting, exited, dead

        info: dict[str, Any] = {
            "container_id": container.id,
            "short_id": container.short_id,
            "name": container.name,
            "state": state,
            "image": str(container.image.tags) if container.image else None,
            "created": container.attrs.get("Created"),
            "started_at": container.attrs.get("State", {}).get("StartedAt"),
        }

        # Resource usage (only available for running containers)
        if state == "running":
            try:
                stats = container.stats(stream=False)
                cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                    stats["precpu_stats"]["cpu_usage"]["total_usage"]
                system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                    stats["precpu_stats"]["system_cpu_usage"]
                num_cpus = stats["cpu_stats"]["online_cpus"]
                cpu_percent = (cpu_delta / system_delta * num_cpus * 100.0) if system_delta > 0 else 0.0

                mem_usage = stats["memory_stats"].get("usage", 0)
                mem_limit = stats["memory_stats"].get("limit", 1)
                mem_percent = (mem_usage / mem_limit * 100.0) if mem_limit > 0 else 0.0

                info["cpu_percent"] = round(cpu_percent, 2)
                info["memory_usage_mb"] = round(mem_usage / (1024 * 1024), 2)
                info["memory_limit_mb"] = round(mem_limit / (1024 * 1024), 2)
                info["memory_percent"] = round(mem_percent, 2)
            except Exception:
                pass

        return info

    def health_check_all(self, db: Session) -> list[dict[str, Any]]:
        """Check all running containers, update DB for crashed ones, return results."""
        from anzar.db.models import Workspace

        results = []
        workspaces = db.query(Workspace).filter(
            Workspace.status.in_(["running", "starting"])
        ).all()

        for ws in workspaces:
            if not ws.container_id:
                continue

            status = self.container_status(ws.container_id)
            state = status.get("state", "unknown")

            if state == "running":
                ws.last_active_at = datetime.now(timezone.utc)
            elif state in ("exited", "dead", "not_found"):
                ws.status = "stopped"
                ws.stopped_at = datetime.now(timezone.utc)
                logger.warning(
                    "Container %s for user %s is %s — marking stopped",
                    ws.container_id[:12],
                    str(ws.user_id)[:8],
                    state,
                )
            elif state == "unavailable":
                # Docker not reachable — don't change status, just log
                logger.warning("Docker unavailable during health check")

            results.append({
                "workspace_id": str(ws.id),
                "user_id": str(ws.user_id)[:8],
                "container_id": ws.container_id[:12] if ws.container_id else None,
                "state": state,
            })

        db.commit()
        return results

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def exec_command(
        self,
        container_id: str,
        command: str,
        timeout: int = 30,
        working_dir: str | None = None,
    ) -> dict[str, Any]:
        """Execute a command inside a container. Returns stdout, stderr, exit code."""
        if not self.check_docker():
            return self._subprocess_exec(command, timeout)

        client = self._get_client()
        container = client.containers.get(container_id)

        if container.status != "running":
            raise RuntimeError(f"Container is not running (status: {container.status})")

        exit_code, output = container.exec_run(
            cmd=["bash", "-c", command],
            workdir=working_dir or "/home/sandbox/workspace",
            user="sandbox",
            demux=True,
            timeout=timeout,
        )

        stdout = (output[0] or b"").decode("utf-8", errors="replace")
        stderr = (output[1] or b"").decode("utf-8", errors="replace")

        return {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }

    def _subprocess_exec(self, command: str, timeout: int = 30) -> dict[str, Any]:
        """Fallback: run command on host via subprocess (no Docker)."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path.home()),
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
            }

    # ------------------------------------------------------------------
    # Workspace-scoped execution (used by the agent's run_command tool)
    # ------------------------------------------------------------------

    def exec_workspace_command(
        self,
        command: str,
        workspace_path: str,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Execute a command scoped to a workspace directory.

        Prefers the running container that mounts the workspace (true
        isolation). When Docker is unavailable or no matching container
        exists, falls back to a host subprocess whose working directory is
        the workspace. The returned dict carries an ``executor`` key
        (``"container"`` or ``"subprocess"``).
        """
        if self.check_docker():
            container = self._container_for_workspace(workspace_path)
            if container is not None:
                try:
                    result = self.exec_command(container.id, command, timeout=timeout)
                    result["executor"] = "container"
                    return result
                except Exception as e:
                    logger.warning("Container exec failed (%s) — falling back to subprocess", e)
        return self._subprocess_exec_cwd(command, cwd=workspace_path, timeout=timeout)

    def _container_for_workspace(self, workspace_path: str) -> Container | None:
        """Find a running container whose workspace bind matches the host path."""
        if not self.check_docker():
            return None
        client = self._get_client()
        wanted = os.path.normpath(os.path.abspath(workspace_path))
        if os.name == "nt":
            wanted = wanted.lower()
        try:
            for container in client.containers.list(filters={"status": "running"}):
                for mount in container.attrs.get("Mounts", []):
                    source = mount.get("Source")
                    if not source:
                        continue
                    source = os.path.normpath(os.path.abspath(source))
                    if os.name == "nt":
                        source = source.lower()
                    if source == wanted:
                        return container
        except Exception as e:
            logger.warning("Failed to scan containers: %s", e)
        return None

    def _subprocess_exec_cwd(
        self,
        command: str,
        cwd: str,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Host subprocess with bounded output capture and tree-kill on timeout."""
        import threading

        if not os.path.isdir(cwd):
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"workspace directory not found: {cwd}",
            }

        cap = 1_000_000
        stdout_buf: list[str] = []
        stderr_buf: list[str] = []
        stdout_overflow = {"value": False}
        stderr_overflow = {"value": False}

        def pump(stream: Any, buf: list[str], overflow: dict[str, bool]) -> None:
            used = 0
            for line in iter(stream.readline, ""):
                remaining = cap - used
                if remaining <= 0:
                    overflow["value"] = True
                    continue
                buf.append(line[:remaining])
                used += len(line)

        popen_kwargs: dict[str, Any] = dict(
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            encoding="utf-8",
            errors="replace",
        )
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(command, **popen_kwargs)
        t_out = threading.Thread(target=pump, args=(proc.stdout, stdout_buf, stdout_overflow))
        t_err = threading.Thread(target=pump, args=(proc.stderr, stderr_buf, stderr_overflow))
        t_out.start()
        t_err.start()

        timed_out = False
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_tree(proc)
            try:
                returncode = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                returncode = proc.wait(timeout=5)
        finally:
            t_out.join(timeout=10)
            t_err.join(timeout=10)
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

        stdout = "".join(stdout_buf)
        stderr = "".join(stderr_buf)
        if stdout_overflow["value"]:
            stdout += "\n[stdout truncated]"
        if stderr_overflow["value"]:
            stderr += "\n[stderr truncated]"

        if timed_out:
            return {
                "exit_code": -1,
                "stdout": stdout,
                "stderr": f"Command timed out after {timeout}s",
                "timed_out": True,
            }
        return {"exit_code": returncode, "stdout": stdout, "stderr": stderr}

    def _kill_tree(self, proc: subprocess.Popen) -> None:
        """Terminate a process and its children (best-effort)."""
        if proc.poll() is not None:
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                proc.kill()
        else:
            try:
                import signal

                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_all(self) -> int:
        """Remove all Anzar containers. Returns count removed."""
        if not self.check_docker():
            return 0

        client = self._get_client()
        containers = client.containers.list(
            all=True,
            filters={"name": CONTAINER_PREFIX},
        )
        count = 0
        for c in containers:
            try:
                c.remove(force=True)
                count += 1
            except Exception as e:
                logger.error("Failed to remove %s: %s", c.short_id, e)
        return count


# Module-level singleton
_manager: DockerSandboxManager | None = None


def get_manager() -> DockerSandboxManager:
    """Get or create the singleton sandbox manager."""
    global _manager
    if _manager is None:
        _manager = DockerSandboxManager()
    return _manager
