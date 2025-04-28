# Installation

## Step 1: Create a virtual environment

Create and activate a virtual environment to install the current package, e.g. using
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (please refer to their
site for questions about creating the environment):

```bash
conda activate # ensures you are in the base environment
conda create -n neurogym python=3.11 -y
conda activate neurogym
```

## Step 2: Install NeuroGym

You can install the latest stable release of `neurogym` using pip:

```bash
pip install neurogym
```

If you plan to use reinforcement learning (RL) features based on Stable-Baselines3, install the RL extra dependencies:

```bash
pip install neurogym[rl]
```

### Step 2b: Install in Editable/Development Mode

If you want to contribute to NeuroGym or always work with the latest updates from the source code, install it in editable mode:

```bash
git clone https://github.com/neurogym/neurogym.git
cd neurogym
pip install -e .
```

This links your local code changes directly to your Python environment without needing to reinstall after every edit.

If you also want RL and development tools (for testing, linting, and documentation), install with:

```bash
pip install -e .[rl,dev]
```

## Step 3 (Optional): Psychopy installation

**NOTE**: psycohopy installation is currently not working

If you need psychopy for your project, additionally run

```bash
pip install psychopy
```
