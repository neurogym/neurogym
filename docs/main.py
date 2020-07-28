import os

import docs.source.make_envs_rst as make_envs_rst

make_envs_rst.main()
os.system('make html')
os.system('open _build/html/index.html')