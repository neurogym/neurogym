import re
from setuptools import setup, find_packages
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neurogym'))
from version import VERSION


if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


# Environment-specific dependencies.
extras = {
  'psychopy': ['psychopy'],
}

# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

# For developers
extras['dev'] = extras['all'] + [
    'jupyter', 'pytest', 'sphinx', 'sphinx_rtd_theme', 'sphinxcontrib.katex',
    'nbsphinx'
]

setup(name='neurogym',
      packages=[package for package in find_packages()
                if package.startswith('neurogym')],
      install_requires=[
          'numpy',
          'gym',
          'matplotlib',
      ],
      extras_require=extras,
      description='NeuroGym: Gym-style cognitive neuroscience tasks',
      author='Manuel Molano, Guangyu Robert Yang, and contributors',
      url='https://github.com/neurogym/neurogym',
      version=VERSION)

