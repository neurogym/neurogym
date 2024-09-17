import os
import sys

from setuptools import find_packages, setup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "neurogym"))
from version import VERSION

if sys.version_info.major != 3:
    print(
        "This Python is only compatible with Python 3, but you are running "
        f"Python {sys.version_info.major}. The installation will likely fail.",
    )

# Environment-specific dependencies.
extras = {
    "psychopy": ["psychopy"],
}

extras["tutorials"] = [
    "jupyter",
]

# For developers
extras["ci"] = ["pytest", "ruff", "mypy"]

extras["docs"] = [
    "sphinx",
    "sphinx_rtd_theme",
    "sphinxcontrib.katex",
    "nbsphinx",
]

extras["dev"] = extras["ci"] + extras["docs"]

# All extra dependencies.
extras["all"] = [item for group in extras.values() for item in group]

setup(
    name="neurogym",
    packages=[package for package in find_packages() if package.startswith("neurogym")],
    install_requires=[
        "numpy",
        "gymnasium>=0.29.1",
        "matplotlib",
        "stable-baselines3>=2.3.2",
        "scipy",
    ],
    extras_require=extras,
    description="NeuroGym: Gymnasium-style cognitive neuroscience tasks",
    author="Manuel Molano, Guangyu Robert Yang, and contributors",
    url="https://github.com/ANNUBS/neurogym",
    version=VERSION,
)
