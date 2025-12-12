# Installation

## 1. Create a Virtual Environment

Create and activate a virtual environment to install the current package, e.g. using
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (please refer to their
site for questions about creating the environment):

```bash
conda activate # ensures you are in the base environment
conda create -n neurogym python=3.11 -y
conda activate neurogym
```

## 2. Install NeuroGym

Install the latest stable release of `neurogym` using pip:

```bash
pip install neurogym
```

### 2.1 Reinforcement Learning Support

NeuroGym includes optional reinforcement learning (RL) features via Stable-Baselines3.
To install these, choose one of the two options below depending on your hardware setup:

#### Option A — CPU-only (recommended for most users):

NeuroGym includes optional reinforcement learning (RL) features via Stable-Baselines3.
To install these, choose one of the two options below depending on your hardware setup:

```bash
pip install neurogym[rl]
```

**NOTE for Linux/WSL users:** If you do not have access to a CUDA-capable NVIDIA GPU (which is the case for most users),
above line will install up to 1.5GB of unnecessary GPU libraries. To avoid excessive overhead, we recommend first
installing the CPU-only version of [PyTorch](https://pytorch.org/get-started/locally/):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install neurogym[rl]
```

### 2.2 Editable/Development Mode

To contribute to NeuroGym or run it from source with live code updates:

```bash
git clone https://github.com/neurogym/neurogym.git
cd neurogym
pip install -e .
```

This installs the package in editable mode, so changes in source files are reflected without reinstalling.

To include both RL and development tools (e.g., for testing, linting, documentation):

```bash
pip install -e .[rl,dev]
```

## 3. Psychopy Installation (Optional)

**NOTE**: psycohopy installation is currently not working

If you need psychopy for your project, additionally run

```bash
pip install psychopy
```
