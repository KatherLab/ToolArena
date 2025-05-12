"""Utilities for task tests."""

import inspect
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from pytest_lazy_fixtures import lf
from toolarena import (
    ToolDefinition,
    ToolInvocation,
    ToolRunner,
    ToolRunResult,
)
from toolarena.utils import DATA_DIR, DEFINITIONS_DIR

type ToolFixture = Callable[[ToolInvocation], ToolRunResult]
type InvocationFixture = Callable[[ToolFixture], ToolRunResult]


def _invocation_fixture(
    tool_name: str,
    invocation: ToolInvocation,
    module: str,
) -> InvocationFixture:
    def fixture(
        candidate_impl_dir: Path, request: pytest.FixtureRequest
    ) -> ToolRunResult:
        runner = ToolRunner.from_paths(
            task_file=DEFINITIONS_DIR / tool_name / "task.yaml",
            invocation=invocation,
            data_dir=DATA_DIR / tool_name / "data",
            install_script=candidate_impl_dir / tool_name / "install.sh",
            code_implementation=candidate_impl_dir / tool_name / "implementation.py",
        )
        if (
            request.config.getoption("--skip-uncached", False)
            and not runner.is_cached()
        ):
            pytest.skip(
                f"Skipping uncached invocation {invocation.name} for {tool_name}"
            )
        return runner.run()

    fixture.__name__ = invocation.name
    fixture.__doc__ = f"Test invocation {invocation.name} for {tool_name}."
    fixture.__module__ = module

    return pytest.fixture(
        fixture,
        scope="module",
        params=[
            pytest.param(
                None,
                marks=[
                    pytest.mark.tool_invocation(
                        tool=tool_name, invocation=invocation.name
                    )
                ],
            )
        ],
        ids=[tool_name],
    )


def parametrize_invocation(
    *invocations: str | InvocationFixture,
) -> pytest.MarkDecorator:
    return pytest.mark.parametrize(
        "invocation",
        [lf(getattr(invocation, "__name__", invocation)) for invocation in invocations],
    )


def _get_fixtures(tool_name: str) -> Mapping[str, ToolFixture | InvocationFixture]:
    module = f"tasks.{tool_name}.tests"
    fixtures = {}
    definition = ToolDefinition.from_yaml(DEFINITIONS_DIR / tool_name / "task.yaml")
    for invocation in definition.test_invocations:
        fixtures[invocation.name] = _invocation_fixture(
            tool_name, invocation, module=module
        )
    return fixtures


def initialize() -> None:
    """Initialize the fixtures for the current test module.

    This function should be called from each test module, where the name of the test file is `<tool_name>/tests.py`.
    It will populate the global namespace with the fixtures for the tool.
    """
    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back
        file_path = Path(caller_frame.f_code.co_filename)
        assert file_path.name == "tests.py", (
            "This function should be called from a test file"
        )
        tool_name = file_path.parent.name
        caller_frame.f_globals.update(_get_fixtures(tool_name))
    finally:
        del frame  # Break reference cycle
