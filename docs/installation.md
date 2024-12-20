# Installation

Create and activate a virtual environment to install the current package, e.g. using
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (please refer to their
site for questions about creating the environment):

```bash
conda activate # ensures you are in the base environment
conda create -n neurogym python=3.11
conda activate neurogym
```

Then install neurogym as follows:

```bash
git clone https://github.com/neurogym/neurogym.git
cd neurogym
pip install -e .
```

## Psychopy installation

If you need psychopy for your project, additionally run

```bash
pip install psychopy
```
