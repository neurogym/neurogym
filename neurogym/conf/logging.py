from loguru import logger

from neurogym.conf.conf import conf

# Logger configuration
# ==================================================
logger.remove()
logger.configure(**conf.monitor.log.make_conf())

# Enable colour tags in messages.
logger = logger.opt(colors=True)
