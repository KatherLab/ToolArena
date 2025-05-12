from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Self, assert_never

import dotenv
import yaml
from docker.errors import BuildError
from docker.models.images import Image
from loguru import logger
from pydantic import BaseModel, ConfigDict

from toolarena.definition import ToolDefinition, ToolInvocation
from toolarena.runtime import (
    DockerRuntimeClient,
    FailedToolInstall,
    Mounts,
    ToolInstallError,
    ToolResult,
    ToolRunResult,
    build_image,
    cached_tool_result_adapter,
)
from toolarena.utils import RUNS_DIR


def build_tool(
    definition: ToolDefinition, install_script: str, code_implementation: str
) -> ToolImplementation:
    environment = definition.repo.resolve_env()
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.debug(
            f"Building tool image {definition.name} in {temp_dir} with environment {environment}"
        )
        temp_dir_path = Path(temp_dir)
        temp_dir_path.joinpath("task.yaml").write_text(
            yaml.dump(definition.model_dump())
        )
        temp_dir_path.joinpath("install.sh").write_text(install_script)
        temp_dir_path.joinpath("implementation.py").write_text(code_implementation)
        temp_dir_path.joinpath(".env").touch()
        for key, value in environment.items():
            dotenv.set_key(temp_dir_path / ".env", key, value)
        try:
            image, logs = build_image(
                tag=definition.name,
                context=temp_dir_path,
                buildargs={
                    "TAG": definition.requires
                },  # will build CUDA image if required
            )
        except BuildError as e:
            if ">>>START INSTALL<<<" in e.msg and "returned a non-zero code" in e.msg:
                raise ToolInstallError(
                    message=f"Failed to install {definition.name}. {e.msg}",
                    build_log="".join(
                        chunk.get("stream", "") + chunk.get("error", "")
                        for chunk in e.build_log
                    ),
                )
            raise
        return ToolImplementation(
            definition=definition,
            image=image,
            install_script=install_script,
            code_implementation=code_implementation,
        )


class ToolImplementation(BaseModel):
    definition: ToolDefinition
    image: Image
    install_script: str
    code_implementation: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def run(
        self,
        invocation: ToolInvocation,
        *,
        data_dir: Path,
        cache_root: Path | None = RUNS_DIR,
    ) -> ToolResult:
        return ToolRunner(
            definition=self.definition,
            invocation=invocation,
            data_dir=data_dir,
            install_script=self.install_script,
            code_implementation=self.code_implementation,
            _cache_root=cache_root,
        ).run(self.image)


class ToolRunner(BaseModel):
    definition: ToolDefinition
    invocation: ToolInvocation
    data_dir: Path
    install_script: str
    code_implementation: str
    _cache_root: Path = RUNS_DIR

    @classmethod
    def from_paths(
        cls,
        *,
        task_file: Path,
        invocation: ToolInvocation,
        data_dir: Path,
        install_script: Path,
        code_implementation: Path,
    ) -> Self:
        return cls(
            definition=ToolDefinition.from_yaml(task_file),
            invocation=invocation,
            data_dir=data_dir,
            install_script=install_script.read_text(),
            code_implementation=code_implementation.read_text(),
        )

    def hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "definition": self.definition.model_dump(),
                    "invocation": self.invocation.model_dump(),
                    "data_dir": str(self.data_dir.resolve().absolute()),
                    "install_script": self.install_script,
                    "code_implementation": self.code_implementation,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @property
    def run_dir(self) -> Path:
        return self._cache_root / self.definition.name / self.hash()

    @property
    def input_dir(self) -> Path:
        return self.run_dir / "input"

    @property
    def output_dir(self) -> Path:
        return self.run_dir / "output"

    @property
    def cache_file(self) -> Path:
        return self.run_dir / "result.json"

    def start_client(self, image: Image | None = None) -> DockerRuntimeClient:
        """Start a client for the tool. If image is not provided, build it."""
        if image is None:
            logger.debug("Image not provided, building it")
            image = build_tool(
                definition=self.definition,
                install_script=self.install_script,
                code_implementation=self.code_implementation,
            ).image
        mounts = Mounts(
            input=self.input_dir,
            output=self.output_dir,
            data_dir=self.data_dir,
            input_mapping=self.invocation.mount,
        )
        mounts.setup()
        return DockerRuntimeClient.create(
            name=self.definition.name,
            image=image,
            mounts=mounts,
            env=self.definition.repo.resolve_env(),
            cuda=self.definition.requires == "cuda",
        )

    def run_without_cache(self, image: Image | None = None) -> ToolResult:
        """Run tool without using cache. If image is not provided, build it."""
        client = self.start_client(image)
        return client.run(**{arg.name: arg.value for arg in self.invocation.arguments})

    def run(self, image: Image | None = None) -> ToolRunResult:
        """Build image and run tool, using cache if available."""
        if self.is_cached():
            return self.read_cache()
        try:
            result = self.run_without_cache(image)
        except ToolInstallError as e:
            result = FailedToolInstall.from_exception(e)
        self.write_cache(result)
        return self.read_cache()

    def write_cache(self, result: ToolResult | FailedToolInstall):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(result.model_dump_json(indent=2))
        self.run_dir.joinpath("install.sh").write_text(self.install_script)
        self.run_dir.joinpath("implementation.py").write_text(self.code_implementation)
        self.run_dir.joinpath("task.yaml").write_text(
            yaml.dump(self.definition.model_dump())
        )

    def read_cache(self) -> ToolRunResult:
        logger.debug(
            f"Retrieving cached result for {self.definition.name} at {self.cache_file}"
        )
        match cached_tool_result_adapter.validate_json(self.cache_file.read_text()):
            case FailedToolInstall() as err:
                raise err.to_exception()
            case ToolResult() as result:
                return ToolRunResult(**result.model_dump(), output_dir=self.output_dir)
            case _ as unreachable:
                assert_never(unreachable)

    def is_cached(self) -> bool:
        return self.cache_file.exists()
