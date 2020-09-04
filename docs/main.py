import os

import docs.source.make_envs_rst as make_envs_rst

make_envs_rst.main()
os.system('make html')
os.system('open _build/html/index.html')
# Requires neurogym and neurogym.github.io to be in the same directory
os.system('cp -r ./_build/html/ ../../neurogym.github.io')