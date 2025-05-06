# Installation

## 1: Create a virtual environment

Create and activate a virtual environment to install the current package, e.g. using
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (please refer to their
site for questions about creating the environment):

```bash
conda activate # ensures you are in the base environment
conda create -n neurogym python=3.11 -y
conda activate neurogym
```

## 2: Install NeuroGym

Install the latest stable release of `neurogym` using pip:

```bash
pip install neurogym
```

### 2a: Reinforcement Learning Support

NeuroGym includes optional reinforcement learning (RL) features via Stable-Baselines3. To install these:

If you do not have a CUDA-capable NVIDIA GPU (which is the case for most users), we recommend installing the CPU-only
version of [PyTorch](https://pytorch.org/get-started/locally/) to avoid downloading unnecessary GPU libraries (up to 1.5GB):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install neurogym[rl]
```

If you do have a properly configured CUDA setup, you can install the GPU-compatible version:

```bash
pip install neurogym[rl]
```

### 2b: Editable/Development Mode

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

## Step 3 (Optional): Psychopy installation

**NOTE**: psycohopy installation is currently not working

If you need psychopy for your project, additionally run

```bash
pip install psychopy
```
