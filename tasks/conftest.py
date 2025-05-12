from collections.abc import Callable
from pathlib import Path

import pytest
from toolarena.definition import ToolInvocation
from toolarena.run import ToolRunResult

type ToolFixture = Callable[[ToolInvocation], ToolRunResult]
type InvocationFixture = Callable[[ToolFixture], ToolRunResult]


def pytest_addoption(parser):
    parser.addoption(
        "--implementation",
        action="store",
        default=None,
        help="Path to the root folder with candidate implementations",
    )
    parser.addoption(
        "--skip-uncached",
        action="store_true",
        default=False,
        help="Skip uncached invocations",
    )


@pytest.fixture(scope="session")
def candidate_impl_dir(request) -> Path:
    """
    Returns the user-supplied path to a directory containing candidate tasks, e.g.:
      my_candidate_implementation/
        conch_extract_features/
          install.sh
          implementation.py
        some_other_task/
          install.sh
          implementation.py
    """
    path_str = request.config.getoption("--implementation")
    if not path_str:
        path_str = Path(__file__).parent  # use tasks/ folder as default
    path = Path(path_str).resolve()
    if not path.is_dir():
        raise ValueError(f"Candidate implementation directory does not exist: {path}")
    return path


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        for marker in item.iter_markers(name="tool_invocation"):
            for key in ("tool", "invocation"):
                item.user_properties.append((key, marker.kwargs[key]))
