"""This is the client that communicates with the tool runtime running inside a Docker container."""

import itertools
import os
import re
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Iterator, Literal, Self, Sequence, cast

import docker
import httpx
import tenacity
from docker.errors import BuildError
from docker.errors import NotFound as DockerNotFoundError
from docker.models.containers import Container
from docker.models.images import Image
from docker.types import DeviceRequest
from docker.types import Mount as DockerMount
from docker.utils.json_stream import json_stream
from loguru import logger
from pydantic import BaseModel, Field, TypeAdapter

from toolarena.definition import ArgumentType, Mount
from toolarena.utils import ROOT_DIR, join_paths, rmdir

TOOL_DOCKERFILE: Final[Path] = ROOT_DIR / "docker" / "tool.Dockerfile"
DEFAULT_TOOL_IMAGE_NAME: Final[str] = "toolarena-tool"
DOCKER_CONTAINER_PORT: Final[str] = "8000/tcp"


@dataclass
class ToolInstallError(Exception):
    """Raised when the tool installation fails due to an error in the install.sh script."""

    message: str
    build_log: str

    def __str__(self) -> str:
        return (
            self.message
            + "\nNote: this may be a cached exception -- clear the cache if you are unsure..."
            + f"\nBuild log:\n{self.build_log}"
        )


class ToolResult(BaseModel):
    return_code: int
    result: Any
    stdout: str
    type: Literal["result"] = "result"

    @property
    def status(self) -> Literal["success", "failure"]:
        return "success" if self.return_code == 0 else "failure"


class ToolRunResult(ToolResult):
    output_dir: Path


class FailedToolInstall(BaseModel):
    message: str
    build_log: str
    type: Literal["failed_tool_install"] = "failed_tool_install"

    @classmethod
    def from_exception(cls, e: ToolInstallError) -> Self:
        return cls(message=e.message, build_log=e.build_log)

    def to_exception(self) -> ToolInstallError:
        return ToolInstallError(self.message, self.build_log)


CachedToolResult = Annotated[
    ToolResult | FailedToolInstall, Field(discriminator="type")
]
cached_tool_result_adapter = TypeAdapter(CachedToolResult)


@dataclass(frozen=True, kw_only=True)
class HTTPToolClient:
    host: str
    port: int

    http_client = httpx.Client(timeout=None)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def run(self, **kwargs: ArgumentType) -> ToolResult:
        logger.info(
            f"Running tool with arguments: {', '.join(f'{k}={v!r}' for k, v in kwargs.items())}"
        )
        response = self.http_client.post(f"{self.url}/run", json=kwargs)
        return ToolResult.model_validate_json(response.text)

    def is_alive(self) -> bool:
        try:
            response = self.http_client.get(f"{self.url}/alive")
            return (
                response.status_code == 200
                and response.json().get("status", None) == "ok"
            )
        except httpx.HTTPError:
            return False

    def wait_for_alive(self, timeout: float | None = 10.0) -> Self:
        logger.debug(f"Waiting for runtime client {self.name} to become ready")
        if timeout is None:
            timeout = float("inf")

        try:
            tenacity.retry(
                retry=tenacity.retry_if_result(False.__eq__),
                stop=tenacity.stop_after_delay(timeout),
                wait=tenacity.wait_fixed(1),
            )(self.is_alive)()
        except tenacity.RetryError:
            raise RuntimeError(
                f"Runtime client did not become ready after {timeout} seconds. You may want to inspect the container logs using `docker logs {self.name}`"
            )
        logger.debug(f"Runtime client {self.name} is ready")
        return self


def get_docker() -> docker.DockerClient:
    return docker.from_env(timeout=480)  # increase timeout to avoid timeout errors


@dataclass(frozen=True)
class Mounts:
    input: Path | None = None  # folder to mount as input
    output: Path | None = None  # folder to mount as output
    data_dir: Path | None = None  # folder to copy input data from
    input_mapping: Sequence[Mount] = ()  # mapping of input data to mount

    def to_docker(self) -> list[DockerMount]:
        mounts = []
        if self.input is not None:
            mounts.append(
                DockerMount(
                    target="/mount/input",
                    source=str(Path(self.input).resolve()),
                    type="bind",
                    read_only=False,  # TODO: make read-only?
                )
            )
        if self.output is not None:
            mounts.append(
                DockerMount(
                    target="/mount/output",
                    source=str(Path(self.output).resolve()),
                    type="bind",
                    read_only=False,
                )
            )
        return mounts

    def setup(self) -> None:
        """Setup the input and output mounts by copying data."""
        if self.output:
            rmdir(self.output)
            self.output.mkdir(parents=True, exist_ok=True)

        if self.input:
            rmdir(self.input)
            self.input.mkdir(parents=True, exist_ok=True)
            if not self.input_mapping:
                logger.debug("Not creating any input mounts...")
                return
            if not self.data_dir:
                raise ValueError("data_dir is required")

            for mount in self.input_mapping:
                src_path = join_paths(self.data_dir, mount.source)
                dst_path = join_paths(self.input, mount.target)
                logger.debug(f"Copying {src_path} to {dst_path}")
                if not src_path.exists():
                    raise FileNotFoundError(f"Input data not found: {src_path}")
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if src_path.is_dir():
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy(src_path, dst_path)


def build_image(
    repository: str = DEFAULT_TOOL_IMAGE_NAME,
    *,
    tag: str,
    context: Path | str,
    dockerfile: Path | str = TOOL_DOCKERFILE,
    buildargs: Mapping[str, str] = {},
) -> tuple[Image, Iterator[Mapping[str, str]]]:
    """Build an image using Docker BuildKit via the low-level Docker API.

    The implementation follows the implementation of `DockerClient.images.build()`.
    This function streams the build output to the console while the build is running.
    """
    logger.debug(f"Building image {repository}:{tag} using Docker BuildKit")
    resp = get_docker().api.build(
        path=str(context),
        dockerfile=str(dockerfile),
        tag=f"{repository}:{tag}",
        buildargs={
            "DOCKER_BUILDKIT": "1",
            # "BUILDKIT_PROGRESS": "plain",
            **buildargs,
        },
    )

    if isinstance(resp, str):
        # The low-level API may return the image ID as a string when
        # `resp` is not a generator. In this case no build logs are
        # available, so return an empty iterator for the log stream.
        return get_docker().images.get(resp), iter(())
    last_event = None
    image_id = None
    internal_stream, result_stream = itertools.tee(json_stream(resp))
    for chunk in internal_stream:
        if "error" in chunk:
            logger.error(f"Build error: {chunk['error']}")
            raise BuildError(chunk["error"], result_stream)
        if "stream" in chunk:
            print(chunk["stream"], end="", file=sys.stderr)
            match = re.search(
                r"(^Successfully built |sha256:)([0-9a-f]+)$", chunk["stream"]
            )
            if match:
                image_id = match.group(2)
        last_event = chunk
    if image_id:
        logger.info(f"Built image {repository}:{tag} using Docker BuildKit")
        return get_docker().images.get(image_id), result_stream
    raise BuildError(last_event or "Unknown", result_stream)


@dataclass(frozen=True, kw_only=True)
class DockerRuntimeClient(HTTPToolClient):
    name: str  # the name of the container
    image: Image  # the docker image

    @classmethod
    @tenacity.retry(
        stop=tenacity.stop_after_delay(5),
        wait=tenacity.wait_fixed(1),
        reraise=True,
    )
    def _get_host_port(cls, container: Container) -> int:
        container.reload()
        if not container.ports:
            raise RuntimeError("Container is not running")
        try:
            port = container.ports[DOCKER_CONTAINER_PORT][0]["HostPort"]
        except KeyError as e:
            raise RuntimeError("Container is not running") from e
        return int(port.split("/")[0])  # may be "1234/tcp"

    @classmethod
    def _start_container(
        cls,
        image: Image,
        name: str,
        port: int | None = None,  # None lets SDK choose port
        mounts: Mounts | None = None,
        cuda: bool = False,
        env: Mapping[str, str] = {},
    ) -> Container:
        device_requests = []
        if cuda:
            gpus = os.getenv("CUDA_VISIBLE_DEVICES", "").split(",")
            gpus = [gpu for gpu in gpus if gpu]
            gpus = gpus or ["0"]
            logger.debug(f"Using GPUs {gpus}")
            device_requests.append(
                DeviceRequest(device_ids=gpus, capabilities=[["gpu"]])
            )
            env |= {"CUDA_VISIBLE_DEVICES": ",".join(map(str, range(len(gpus))))}
        else:
            logger.debug("Not using GPUs")
        logger.debug("Starting container...")
        logger.debug(
            f"CUDA_VISIBLE_DEVICES inside the container: {env.get('CUDA_VISIBLE_DEVICES', 'not set')}"
        )

        # Start a container with the supplied name
        container = get_docker().containers.run(
            image=image,
            name=name,
            detach=True,
            ports={DOCKER_CONTAINER_PORT: port},
            tty=True,
            mounts=(mounts or Mounts()).to_docker(),
            mem_limit="100g",
            shm_size="10g",
            device_requests=device_requests,
            environment=dict(env),
        )
        logger.info(
            f"Started runtime client {container.name} on port {cls._get_host_port(container)} from image {image.tags}"
        )
        return container

    @classmethod
    def create(
        cls,
        name: str,
        image: str | Image,
        port: int | None = None,
        timeout: float | None = 10.0,  # max wait time for the runtime to become ready
        mounts: Mounts | None = None,
        # Below, cuda works as follows:
        # - if True: use GPUs specified in CUDA_VISIBLE_DEVICES, or use GPU 0 if CUDA_VISIBLE_DEVICES is not set
        # - if False: do not use GPUs
        cuda: bool = False,
        env: Mapping[str, str] = {},
    ) -> Self:
        """Create a new runtime client by building the image and starting the container."""
        client = get_docker()
        docker_image = cast(
            Image,
            image if isinstance(image, Image) else client.images.get(image),
        )

        try:
            container: Container = client.containers.get(name)
            logger.info(f"Found existing container {name}, removing it")
            container.remove(force=True)
        except DockerNotFoundError:
            pass

        container = cls._start_container(
            image=docker_image,
            name=name,
            mounts=mounts,
            cuda=cuda,
            env=env,
            port=port,
        )
        return cls(
            host="localhost",
            port=cls._get_host_port(
                container
            ),  # (can't use port directly because it may be None, but we want to use the port that was assigned)
            name=container.name,  # type: ignore
            image=docker_image,
        ).wait_for_alive(timeout=timeout)

    def stop(self):
        container = get_docker().containers.get(self.name)
        container.stop()
        container.remove(force=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
