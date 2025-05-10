# ToolArena

ToolArena is a **benchmark** for evaluating how well Large Language Model (LLM) agents can create "tools" from GitHub repositories. 
Each *tool* (or *task*) corresponds to a Python function with defined inputs and outputs, wrapped in a containerised environment for reproducibility and testability.

[![Dataset](https://img.shields.io/badge/Dataset-gray?logo=huggingface)](https://huggingface.co/datasets/KatherLab/ToolArena)

## Overview
ToolArena contains multiple tasks under the [`tasks/`](tasks/) directory. Each task is:
- **Self-contained**: it has its own `task.yaml` describing its name, inputs, outputs, example invocation, and test invocations.
- **Implementation-agnostic**: we provide a *reference* implementation in `implementation.py` with a companion `install.sh` that sets up dependencies.  
- **Tested**: A `tests.py` script ensures that any candidate implementation is correct.

By default, each task's folder contains:
```
tasks/<TASK_NAME>/
  ├── task.yaml
  ├── implementation.py
  ├── install.sh
  ├── tests.py
  └── data/
      ├── download.sh
      └── ...
```

You can use this benchmark to evaluate how well your LLM agent can create implementations for the tasks defined in ToolArena.
You can also propose your own new tasks to contribute to the benchmark. 

## Contributing
We welcome new tasks and improvements to existing ones. 
See [CONTRIBUTING.md](CONTRIBUTING.md) for a full guide on how to contribute a new task.

---

## Installation
1. **Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/)**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Sync dependencies** for all environment groups. In the root of this repo, run:
   ```bash
   uv sync --all-groups
   ```
   This creates a virtual environment at `.venv/` and installs all necessary Python dependencies.
3. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```
4. **Verify installation** to check the `toolarena` command is available:
   ```bash
   toolarena --help
   ```
5. **Install [Docker](https://docs.docker.com/desktop)**, then **pull the latest ToolArena images**:
   ```bash
   docker pull ghcr.io/katherlab/toolarena:cpu
   docker pull ghcr.io/katherlab/toolarena:cuda
   ```

### Environment setup
Create a `.env` file in the repository’s root directory with at least these variables:
```bash
CUDA_VISIBLE_DEVICES=0  # If you have a GPU, set the device ID here; otherwise, you can leave it blank
HF_TOKEN=hf_...  # Replace with your HuggingFace token
OPENAI_API_KEY=sk-...  # Replace with your OpenAI API key
```
You will need to **request access** to the following HuggingFace repositories, which are required by some of the tasks:
- [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) (for [`uni_extract_features`](tasks/uni_extract_features/))
- [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) (for [`conch_extract_features`](tasks/conch_extract_features/))
- [xiangjx/musk](https://huggingface.co/xiangjx/musk) (for [`musk_extract_features`](tasks/musk_extract_features/))
- [KatherLab/COBRA](https://huggingface.co/KatherLab/COBRA) (for [`cobra_extract_features`](tasks/cobra_extract_features/) and [`cobra_heatmaps`](tasks/cobra_heatmaps))
- [pixas/MedSSS_Policy](https://huggingface.co/pixas/MedSSS_Policy) (for [`medsss_generate`](tasks/medsss_generate/))
- [YukunZhou/RETFound_mae_natureCFP](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP) (for [`retfound_feature_vector`](tasks/retfound_feature_vector/))
- [KatherLab/MoPaDi](https://huggingface.co/KatherLab/MoPaDi) (for [`mopadi_generate_counterfactuals`](tasks/mopadi_generate_counterfactuals/))

An OpenAI API key is required for the [`textgrad_medical_qa_optimize`](tasks/textgrad_medical_qa_optimize) task.

## Tasks
All tasks live under the [`tasks/`](tasks/) directory. Each contains:
- A **task definition** (`task.yaml`) describing:
  - **Name**  
  - **Repository** (source GitHub repo & commit)  
  - **Inputs** and **outputs**  
  - **Example** and **test** invocations, including any **data** supplied as input for these invocations
- A **tests file** (`tests.py`) with `pytest` tests verifying correctness of the implementation.
- A **reference implementation** consisting of:
  - An **installation script** (`install.sh`) that clones the repository and installs necessary dependencies.
  - A **Python implementation** (`implementation.py`) that provides a reference function for the given task.


## Downloading external data
Some tasks require large external data files as input (e.g. images, datasets, etc.).
Each task may specify a `data/download.sh` script, which downloads all required external data from publicly available repositories.
While may execute these download scripts yourself, ToolArena provides a convenient command to download all data for you:
```bash
toolarena download
```
If you want to download the data for one specific task, simply run `toolarena download <TASK_NAME>` instead.

## Running a candidate implementation
A candidate implementation must have:
1. **`install.sh`**  
   Installs all necessary dependencies for the tool (including cloning the associated repository)
2. **`implementation.py`**  
   Contains the Python function that matches the signature defined in the `task.yaml`.

Suppose you have created a folder `<IMPLEMENTATION_DIR>` (outside of `tasks/`) that looks like:
```
<IMPLEMENTATION_DIR>/
  └── <TASK_NAME>/
      ├── install.sh
      └── implementation.py
```
Here's how to run it:
```bash
toolarena run <TASK_NAME> <INVOCATION> --implementation <IMPLEMENTATION_DIR>
```
Where:
- `<TASK_NAME>` is the name of your task (folder name in `tasks/`).
- `<INVOCATION>` is either `example` (the example in `task.yaml`) or any named test invocation.
- `<IMPLEMENTATION_DIR>` is the name of the aforementioned implementations directory you created.

> [!NOTE]
> If you omit `<INVOCATION>`, ToolArena runs **all** invocations (the example plus every test invocation) for that task.

Because ToolArena caches results, repeated runs with the same inputs do not require rerunning the entire tool.

## Running the benchmark
To run the benchmark (i.e., the entire battery of tests) on your candidate implementations:
```bash
pytest tasks --implementation <IMPLEMENTATION_DIR>
```

> [!TIP]
> You can refine your test runs:
> - **Skip uncached** invocations:
>   ```bash
>   pytest tasks --implementation <IMPLEMENTATION_DIR> --skip-uncached
>   ```
>   Only tests for which you already have cached results will run.
> - **Run only one task**:
>   ```bash
>   pytest tasks/<TASK_NAME> --implementation <IMPLEMENTATION_DIR>
>   ```

## Running the reference implementation
Each task provides a human-generated *reference implementation* to prove that the task is possible.
The reference implementations are supplied alongside the task definitions in the [`tasks/`](tasks/) directory.

To run the reference implementation for any task, simply omit the `--implementation` flag:
```bash
# Run a single invocation (example invocation):
toolarena run <TASK_NAME> example

# Or run all invocations for that task:
toolarena run <TASK_NAME>
```
And to run **all** tasks' tests (the entire benchmark) with reference implementations:
```bash
pytest tasks
```
## Debugging implementations

If you need to inspect a running container or attach a debugger:

1. **Debug a specific invocation**:
   ```bash
   toolarena debug <TASK_NAME> <INVOCATION_NAME> --implementation <IMPLEMENTATION_DIR>
   ```
   This starts the container and provides instructions to attach VS Code or open a bash session in the container.

2. **Check logs** directly in Docker:
   ```bash
   toolarena run <TASK_NAME> <INVOCATION> --implementation <IMPLEMENTATION_DIR>
   docker logs -f <TASK_NAME>
   ```
   This streams any output from the tool in real-time. Under the hood, the `toolarena run` command creates a Docker container with the same name as the task.

---

Feel free to open Issues or Pull Requests if you encounter problems or want to propose improvements. Happy tool-building!