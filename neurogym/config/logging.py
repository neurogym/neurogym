from loguru import logger

from neurogym.config.config import config

# Logger configuration
# ==================================================
logger.remove()
logger.configure(**config.monitor.log.make_config())

# Enable colour tags in messages.
logger = logger.opt(colors=True)
