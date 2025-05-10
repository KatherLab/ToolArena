# Contributing a new task to ToolArena

Thank you for your interest in contributing a new task to ToolArena! This guide walks you through the steps to add a new task (tool) to this benchmark repository by opening a pull request.

## Overview

ToolArena is a benchmark for LLM-based agentic "tool creation." 

Specifically, the purpose of this benchmark is to assess how well LLM agents can create LLM-compatible "tools" from GitHub repositories.
You can think of a tool as a Python function that performs a given task: it has input arguments, and returns some sort of output.

> [!NOTE]  
> The terms "task" and "tool" are used interchangeably in this guide.

So, what makes a good task?
**A good task for ToolArena is a well-scoped, functional wrapper around an open-source GitHub repository that is associated with a research paper.**
The task should be meaningful for LLM-based tool generation -- simple enough to be learnable from documentation and repo code, yet non-trivial enough to require some reasoning and code synthesis.
Importantly, it must also be *testable*, i.e. you can write code to assess whether the task was successful.

When you create a new task, you place these files in a directory under `tasks/`. For example, if your task is named `my_nifty_task`, your files would live under `tasks/my_nifty_task/`.
Each task in ToolArena consists of:
1. A **task definition** (`task.yaml`) containing basic metadata, inputs/outputs, test invocations, etc.
2. A **reference solution** (i.e. implementation), consisting of:
   - An **installation script** (`install.sh`) that installs all necessary dependencies to run the tool.
   - An **code implementation** (`implementation.py`) containing the Python implementation for the tool itself.
3. Any **data** files needed to invoke the tool (`data/` and `data/tests`).
4. Several **unit tests** to assess the correctness of a candidate implementation (`tests.py`).

> This guide will walk you through creating all the necessary files to define a task. By the end of this guide, you will have produced the following directory structure:
> ```
> tasks/
>   └── my_nifty_task/
>       ├── __init__.py
>       ├── task.yaml
>       ├── install.sh
>       ├── implementation.py
>       ├── tests.py
>       └── data/
>           ├── .gitignore
>           ├── download.sh  # (optional script to download external data)
>           ├── ...          # (optional) (your data files)
>           └── tests/
>               └── ...      # (optional) (additional data files for tests)

---

## Table of Contents

1. [Fork the repository](#1-fork-the-repository)
2. [Install ToolArena](#2-install-toolarena)
3. [Create a new task](#3-create-a-new-task)
4. [Fill in your task definition (`task.yaml`)](#4-fill-in-your-task-definition-taskyaml)
5. [Add data required to run the task](#5-add-data-required-to-run-the-task)
6. [Define the example invocation](#6-define-the-example-invocation)
7. [Generate `install.sh` and `implementation.py`](#7-generate-installsh-and-implementationpy)
8. [Write the `install.sh` script](#8-write-the-installsh-script)
9. [Write `implementation.py`](#9-write-implementationpy)
10. [Check that the example invocation works](#10-check-that-the-example-invocation-works)
11. [Write tests](#11-write-tests)
12. [Run your tests](#12-run-your-tests)
13. [Submit a Pull Request!](#13-submit-a-pull-request)

---

## 1. Fork the repository

1. Click the "Fork" button in the GitHub UI to create your personal copy of ToolArena.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/ToolArena.git
   cd ToolArena
   ```

## 2. Install ToolArena
1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
2. Install this project in a virtual environment:
   ```bash
   uv sync --all-groups
   ```
   This will create a virtual environment at `.venv/`.
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
4. Check that the installation succeeded by verifying that the `toolarena` command exists:
   ```bash
   toolarena --help
   ```
5. Install [Docker](https://docs.docker.com/desktop) and pull the latest ToolArena image:
   ```bash
   docker pull ghcr.io/katherlab/toolarena:cpu
   docker pull ghcr.io/katherlab/toolarena:cuda  # (only required if your task requires GPU)
   ```
6. Create a `.env` file at the root of the repository (it can be empty for now):
   ```bash
   touch .env
   ```

## 3. Create a new task

Use the following command to create a new task:
```bash
toolarena init my_nifty_task
```
This will create a new task at `tasks/my_nifty_task` with a `task.yaml` file to get you started, alongside a `data/` directory which you will later use to store data files necessary for your task.
> Your directory structure will now look as follows:
> ```
> tasks/
>   └── my_nifty_task/
>       ├── __init__.py  # (empty file)
>       ├── task.yaml
>       └── data/
>           ├── .gitignore
>           └── download.sh
> ```

## 4. Fill in your task definition (`task.yaml`)

You will find an automatically generated `task.yaml` file in your task folder (`tasks/my_nifty_task/task.yaml`). 
Edit the `task.yaml` file to define your task. 
Take special care to come up with a task `description`, and make sure all `arguments` and `returns` are specified.

> [!NOTE]  
> At any point, you can run the following command to see how the Python function signature of your task will look like:
> ```bash
> toolarena signature my_nifty_task
> ```

> [!TIP]  
> You can use an existing task in `tasks/` as a reference (for instance, [`tasks/conch_extract_features/task.yaml`](tasks/conch_extract_features/task.yaml)).

### Key sections to include
At this point, you should at least define the following parts (the `task.yaml` file includes extensive documentation in the form of comments to help you):
1. **`name`**: matches the folder name (e.g., `mynifty_task`).
2. **`description`**: One or two sentences describing what your tool does.
3. **`repo`**: your should wrap functionality from a GitHub repository, so please specify:
   ```yaml
   repo:
     name: MyRepo
     url: "https://github.com/username/MyRepo"
     commit: abc123
     branch: main
     env:  # (optional)
       - name: MY_TOKEN
         value: "${env:MY_TOKEN}"
   ```
4. **`arguments`**: inputs your tool expects.
5. **`returns`**: outputs your tool produces.

> [!IMPORTANT]  
> If your task requires a GPU, ensure you set `requires: cuda` in the `task.yaml` file.

### Define environment variables (optional)
In the `env` section of `repo`, you may optionally define environment variables that should be set for the tool.
For example, to set the environment variable `MY_ENV_VAR` to the value `"abc"`, you would write:
```yaml
repo:
  env:
    - name: MY_ENV_VAR
      value: "abc"
``` 
The main use case for environment variables is to supply **secret tokens** such as API keys. For this, you may use a special syntax (`"${env:MY_TOKEN}"`) to insert environment variables from your *local environment*, defined in the top-level `.env` file.

> [!IMPORTANT]
> The `.env` file should be placed at the **top level** of the ToolArena repository, **not** within your task folder `tasks/my_nifty_task/.env`!

For example, if you set the following in your `.env` file:
> **`.env`:**
> ```bash
> HF_TOKEN=my_huggingface_token
> ```
Then you can tell ToolArena that the local `HF_TOKEN` environment variable defined in `.env` should be available to the tool as an environment variable, also named `HF_TOKEN`:
> **`tasks/my_nifty_task/task.yaml`:**
> ```yaml
> repo:
>   env:
>     - name: HF_TOKEN
>       value: "${env:HF_TOKEN}"
> ``` 


## 5. Add data required to run the task
If your task requires external data as input, add these files to the `data/` folder within your task. You may place any files / datasets in there that your tool requires as input for the example invocation or for the tests later on.

If your task requires large files (e.g. larger than a few MB), please add these files to the `data/.gitignore` file and add commands in the `data/download.sh` script so they can be downloaded by the user of the benchmark.
> Your directory structure will now look as follows:
> ```
> tasks/
>   └── my_nifty_task/
>       ├── __init__.py
>       ├── task.yaml
>       └── data/
>           ├── .gitignore
>           ├── download.sh
>           └── ...  # your data files
> ```

## 6. Define the example invocation
In the `example` section of `task.yaml`, define an *example invocation* by providing a concrete value for each input argument.
Later on, you will verify that your tool implementation works by calling your tool using this example invocation.

## 7. Generate `install.sh` and `implementation.py`
ToolArena provides an automated mechanism to generate starter code in the files needed for the implementation:
```bash
toolarena generate my_nifty_tool
```
> Now, your folder structure will look like this:
> ```
> tasks/
>   └── my_nifty_task/
>       ├── __init__.py
>       ├── task.yaml
>       ├── implementation.py
>       ├── install.sh
>       ├── tests.py
>       └── data/
>           ├── .gitignore
>           ├── download.sh
>           └── ...  # your data files
> ```

Importantly, ToolArena will automatically place starter code in the `implementation.py` file. 
This code contains the Python function signature of the task you defined in `task.yaml`.

> [!CAUTION]  
> If you need to change the `description`, `arguments` or `returns` in your `task.yaml`, be sure to re-generate the Python function signature and update it in `implementation.py`:
> ```bash
> toolarena signature my_nifty_task
> ```

## 8. Write the `install.sh` script
Now, write the `install.sh` script to install all necessary dependencies for your tool.
The `toolarena generate` command which you executed earlier will already include the `git clone` command in your install script to get you started.

To try out the install script, simply run:
```bash
toolarena build my_nifty_task
```

This will create a Docker image using your `install.sh` script to install all dependencies.
If this succeeds without errors, you can start an interactive shell in a container based on this image using:
```bash
docker run -it --rm --env-file .env toolarena-tool:my_nifty_task /bin/bash
# Or, if your task requires CUDA:
docker run -it --rm --env-file .env --gpus all toolarena-tool:my_nifty_task /bin/bash

```

> [!TIP]  
> You can just start with an empty `install.sh` file including just the `git clone` command, and then run:
> ```bash
> toolarena build my_nifty_task
> docker run -it --rm --env-file .env toolarena-tool:my_nifty_task /bin/bash
> ```
> In the interactive shell, you can just run the commands you need in order to install all the dependencies.
> Simply take note of the commands you ran and put them in the `install.sh` file!

> **Example:**
> ```bash
> #! /bin/bash
> set -e
>
> git clone https://github.com/username/MyRepo /workspace/MyRepo
> cd /workspace/MyRepo && git checkout abc123
>
> apt-get install somepackage
> pip install -e /workspace/MyRepo
> pip install someotherpackage
> ```

## 9. Write `implementation.py`

This file must define **one function** whose name matches `name` in `task.yaml` and arguments are as defined in `task.yaml`.

> **Example:**
> ```python
> def my_nifty_task(a: float) -> dict:
>     """
>     Round the input to the nearest integer.
>
>     Args:
>         a: the number to be rounded
> 
>     Returns:
>         dict with the following structure:
>         {
>           'rounded': int  # The rounded number
>         }
>     """
>
>     import math # Note: we're importing modules inside the function, not globally!
>     rounded = int(math.round(a))
>     return {"rounded": rounded}
> ```

> [!NOTE]  
> Do **not** define imports globally; instead put your import statements in the body of the function, as shown in the example above.

## 10. Check that the example invocation works
To check that the example invocation runs as expected, run the following command:
```bash
toolarena run my_nifty_example example
```
Check that the result and standard output are as expected for your task.
If there is an error, you should modify the `install.sh` and/or `implementation.py` files to work correctly, and then re-run the `toolarena run my_nifty_example example` command.

> [!TIP]  
> If your tool takes very long to run, you can inspect the output by running the following in a separate shell:
> ```bash
> docker logs -f my_nifty_tool
> ```
> This will show you the standard output of your tool while it is running.

> [!TIP]  
> You can also attach VS Code to your task's Docker container by running:
> ```bash
> toolarena debug my_nifty_task example
> ```
> This will start a docker container for the `example` invocation (specified in the `example` section of `task.yaml`). This command will provide instructions on how to attach VS code to the container.

## 11. Write tests
Define 2-3 test invocations in the `invocation` section of `task.yaml`.
These invocations are defined similarly to how you defined the `example` invocation.

Then, write unit tests for these invocations in the `tests.py` file.
You can use the unit tests at [`tasks/conch_extract_features/tests.py`](tasks/conch_extract_features/tests.py) (or any of the other tasks) as an example how to define these tests.

> [!TIP]  
> You can see the outputs that your tool implementation produces for each of your test invocations by running:
> ```bash
> toolarena run my_nifty_tool my_invocation  # (replace `my_invocation` with the name of the invocation which you provided in `task.yaml`)
> ```

## 12. Run your tests
Run the tests for your tool using the following command, and check that they all pass for your implementation:
```bash
pytest tasks/my_nifty_task
```

## 13. Submit a Pull Request!
Nice work, you have successfully created a new task for the ToolArena benchmark. Now, open a [new Pull Request](https://github.com/KatherLab/ToolArena/pulls) on GitHub to submit this task!

Check out [this PR](https://github.com/KatherLab/ToolArena/pull/1) as an example.

## Tips and best practices

- **Pin your dependencies** to ensure reproducibility
- **Use specific commits** for GitHub repos instead of floating branches
- **Minimize storage**: Avoid checking in large data files
- **Use env vars** like `HF_TOKEN` for secure model downloading
- **Refer to existing tasks** as examples if unsure
- **Try to write at least 2-3 test cases** to validate outputs

---

Thank you for contributing to ToolArena! Feel free to reach out via GitHub Issues if you have any questions.
