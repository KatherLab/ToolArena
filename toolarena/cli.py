import os
import shlex
import shutil
from builtins import print as builtins_print
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich import print
from rich.panel import Panel
from rich.text import Text

from toolarena.definition import ToolDefinition
from toolarena.run import ToolImplementation, ToolRunner
from toolarena.runtime import get_docker
from toolarena.utils import (
    DATA_DIR,
    DEFINITIONS_DIR,
    ROOT_DIR,
    RUNS_DIR,
    TEMPLATE_DIR,
    run_and_stream_container,
)

TESTS_DIR = DEFINITIONS_DIR
REFERENCE_IMPLEMENTATIONS_DIR = DEFINITIONS_DIR
app = typer.Typer()


@app.command()
def signature(
    name: Annotated[str, typer.Argument(help="The name of the tool")],
) -> None:
    """Print the signature of a tool."""
    definition = ToolDefinition.from_yaml(DEFINITIONS_DIR / name / "task.yaml")
    builtins_print(definition.python_signature)


@app.command()
def init(name: Annotated[str, typer.Argument(help="The name of the tool")]) -> None:
    """Initialize a new tool."""
    task_dir = DEFINITIONS_DIR / name
    if task_dir.exists():
        print(f"Task directory [bold]{task_dir}[/bold] already exists!")
        raise typer.Abort()
    shutil.copytree(TEMPLATE_DIR / "task", task_dir)
    print(f"Initialized tool [bold]{name}[/bold] at [bold]{task_dir}[/bold]")


@app.command()
def generate(name: Annotated[str, typer.Argument(help="The name of the tool")]) -> None:
    """Generate the starting files for a new tool, after it has been initialized and the `task.yaml` file has been populated."""
    definition_path = DEFINITIONS_DIR / name / "task.yaml"
    if not definition_path.exists():
        print(f"Task definition [bold]{definition_path}[/bold] does not exist")
        raise typer.Abort()
    definition = ToolDefinition.from_yaml(definition_path)

    code_file = REFERENCE_IMPLEMENTATIONS_DIR / name / "implementation.py"
    install_script = REFERENCE_IMPLEMENTATIONS_DIR / name / "install.sh"
    tests_file = TESTS_DIR / name / "tests.py"

    if not code_file.exists():
        code_file.write_text(definition.python_signature)
        print(f"[green]Created[/green] [bold]{code_file}[/bold]")
    else:
        print(f"[yellow]Skipped[/yellow] [bold]{code_file}[/bold] (already exists)")

    if not install_script.exists():
        install_script.write_text(
            f"#! /bin/bash\n"
            f"set -e\n\n"
            f"{definition.repo.git_clone_command}\n\n"
            f"# Insert commands here to install dependencies and setup the environment...\n"
        )
        print(f"[green]Created[/green] [bold]{install_script}[/bold]")
    else:
        print(
            f"[yellow]Skipped[/yellow] [bold]{install_script}[/bold] (already exists)"
        )

    if not tests_file.exists():
        tests_file.write_text(f"""import pytest
from pytest_lazy_fixtures import lf
from tasks.utils import initialize, parametrize_invocation
from toolarena.run import ToolRunResult

initialize()


@parametrize_invocation({", ".join(('"' + invocation.name + '"') for invocation in definition.test_invocations)})
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"

# TODO: Add more tests here. You may take inspiration from the tests in the other tasks' `tests.py` files.
""")
        print(f"[green]Created[/green] [bold]{tests_file}[/bold]")
    else:
        print(f"[yellow]Skipped[/yellow] [bold]{tests_file}[/bold] (already exists)")


@app.command()
def build(
    name: Annotated[str, typer.Argument(help="The name of the tool")],
    implementation: Annotated[
        Path | None,
        typer.Option(
            help="Path to the folder where the implementation is stored (optional, default use the implementation in the task directory). This folder should contain an install.sh script and a implementation.py file."
        ),
    ] = None,
) -> ToolImplementation:
    """Build a tool."""
    implementation_dir = implementation or (REFERENCE_IMPLEMENTATIONS_DIR / name)
    definition = ToolDefinition.from_yaml(DEFINITIONS_DIR / name / "task.yaml")
    image = definition.build(
        install_script=implementation_dir.joinpath("install.sh").read_text(),
        code_implementation=implementation_dir.joinpath(
            "implementation.py"
        ).read_text(),
    )
    docker_run_command = " ".join(
        (
            "docker run",
            "-it",
            "--rm",
            *(["--env-file .env"] if definition.repo.env else []),
            *(["--gpus all"] if definition.requires == "cuda" else []),
            image.image.tags[0],
            "/bin/bash",
        )
    )
    print(f"To start an interactive shell, run: [bold]{docker_run_command}[/bold]")
    return image


@app.command()
def run(
    name: Annotated[str, typer.Argument(help="The name of the tool")],
    implementation: Annotated[
        Path | None,
        typer.Option(
            help="Path to the folder where the implementation is stored (optional, default use the implementation in the task directory). This folder should contain an install.sh script and a implementation.py file."
        ),
    ] = None,
    invocation: Annotated[
        str | None,
        typer.Argument(
            help="The invocation to run (optional, default run all invocations)"
        ),
    ] = None,
    cache: Annotated[
        Path | None,
        typer.Option(
            help="The root directory for caching tool runs (optional, default use the default cache root)"
        ),
    ] = RUNS_DIR,
) -> None:
    """Run a tool."""
    tool = build(name=name, implementation=implementation)

    if invocation:
        invocations = [tool.definition.get_invocation(invocation)]
    else:
        invocations = [
            tool.definition.example,
            *tool.definition.test_invocations,
        ]
    for invocation in invocations:
        logger.info(f"Running {invocation.name} for {name}")
        result = tool.run(
            invocation, data_dir=DATA_DIR / name / "data", cache_root=cache
        )
        status_style = "green" if result.status == "success" else "red"
        print(
            f"Tool [bold]{name}[/bold] invocation [bold]{invocation.name}[/bold] finished with status {Text(result.status, style=f'bold {status_style}').markup}"
        )
        print(
            Panel(
                Text(result.stdout),
                title="Standard output",
                title_align="left",
                border_style=status_style,
            )
        )
        print(
            Panel(
                Text(repr(result.result)),
                title="Result",
                title_align="left",
                border_style=status_style,
            )
        )


@app.command()
def debug(
    name: Annotated[str, typer.Argument(help="The name of the tool")],
    implementation: Annotated[
        Path | None,
        typer.Option(
            help="Path to the folder where the implementation is stored (optional, default use the implementation in the task directory). This folder should contain an install.sh script and a implementation.py file."
        ),
    ] = None,
    invocation: Annotated[
        str,
        typer.Argument(
            help="The invocation to run (optional, default run example invocation)"
        ),
    ] = "example",
) -> None:
    """Debug a tool."""
    implementation_dir = implementation or (REFERENCE_IMPLEMENTATIONS_DIR / name)
    task_file = DEFINITIONS_DIR / name / "task.yaml"
    definition = ToolDefinition.from_yaml(task_file)
    runner = ToolRunner.from_paths(
        task_file=task_file,
        invocation=definition.get_invocation(invocation),
        data_dir=DATA_DIR / name / "data",
        install_script=implementation_dir / "install.sh",
        code_implementation=implementation_dir / "implementation.py",
    )
    client = runner.start_client()

    print(
        Panel.fit(
            f"""\
To [green]start[/green] an interactive shell, run: [bold]docker exec -it --env-file .env {client.name} /bin/bash[/bold].
To [red]stop[/red] the container, run: [bold]docker stop {client.name}[/bold].
To [red]remove[/red] the container, run: [bold]docker rm {client.name}[/bold].

You can [blue link=https://code.visualstudio.com/docs/devcontainers/attach-container]attach[/blue link] VS Code to this container as follows:
1. Install the [italic link=https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers]Dev Containers[/italic link] extension.
2. Open the Command Palette ([italic]Ctrl+Shift+P[/italic] or [italic]Cmd+Shift+P[/italic] on Mac).
3. Search for [italic]Dev Containers: Attach to Running Container...[/italic]
4. Select the container [bold]{client.name}[/bold] to attach to it.""",
            title=f"""Container [bold]{client.name}[/bold] [green]started[/green]""",
        )
    )


@app.command()
def download(
    name: Annotated[str | None, typer.Argument(help="The name of the tool")] = None,
    force: Annotated[
        bool, typer.Option(help="Force download even if already downloaded")
    ] = False,
) -> None:
    """Download data files for a tool."""
    # If no name is provided, download data for all tools
    if not name:
        for i, task in enumerate(tasks := list(DEFINITIONS_DIR.glob("*/task.yaml"))):
            name = task.parent.name
            print(f"Task {i + 1}/{len(tasks)}: [bold]{name}[/bold]")
            download(name, force=force)
        return

    # Download data for the given tool
    data_dir = DATA_DIR / name / "data"
    download_script = data_dir / "download.sh"
    if not download_script.exists():
        print(
            f"[yellow]Skipped[/yellow] downloading data for [bold]{name}[/bold] (nothing to download)"
        )
        return
    if data_dir.joinpath(".downloaded").exists() and not force:
        print(
            f"[yellow]Skipped[/yellow] downloading data for [bold]{name}[/bold] (already downloaded)"
        )
        return

    print(f"Downloading data for [bold]{name}[/bold]...")
    container_repo_dir = Path("/repo")
    container_data_dir = container_repo_dir / data_dir.relative_to(ROOT_DIR)
    command = (
        f"git config --global --add safe.directory {shlex.quote(str(container_repo_dir))} && "
        f"cd {shlex.quote(str(container_repo_dir))} && "
        f"git clean -fdX {shlex.quote(str(container_data_dir))} && "
        f"cd {shlex.quote(str(container_data_dir))} && "
        f"{shlex.quote(str(container_data_dir / 'download.sh'))} && "
        f"chown -R {os.getuid()}:{os.getgid()} {shlex.quote(str(container_data_dir))} && "
        f"touch {shlex.quote(str(container_data_dir / '.downloaded'))}"
    )
    print(f"Running [bold]{command}[/bold] in container...")
    container = get_docker().containers.run(
        name=f"toolarena-download-{name}",
        image="ghcr.io/astral-sh/uv:bookworm",
        command=f"/bin/bash -c {shlex.quote(command)}",
        volumes={
            str(ROOT_DIR.resolve().absolute()): {
                "bind": str(container_repo_dir),
                "mode": "ro",
            },
            str(data_dir.resolve().absolute()): {
                "bind": str(container_repo_dir / data_dir.relative_to(ROOT_DIR)),
                "mode": "rw",
            },
        },
        stdout=True,
        stderr=True,
        detach=True,
        tty=False,
        remove=True,
    )
    for chunk in run_and_stream_container(container):
        print(chunk, end="")
    print(f"[green]Downloaded[/green] data for [bold]{name}[/bold]")
