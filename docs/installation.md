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

## Step 2: Install neurogym

Then install the latest version of `neurogym` as follows:

```bash
pip install neurogym
```

### Step 2b: Install editable package

Alternatively, get the latest updates by cloning the repo and installing the editable version of neurogym, by replacing
step 2 above by:

```bash
git clone https://github.com/neurogym/neurogym.git
cd neurogym
pip install -e .
```

## Step 3 (Optional): Psychopy installation

**NOTE**: psycohopy installation is currently not working

If you need psychopy for your project, additionally run

```bash
pip install psychopy
```
