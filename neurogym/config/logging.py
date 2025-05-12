from loguru import logger

from neurogym.config.config import config

# Logger configuration
# ==================================================
logger.remove()
logger.configure(**config.monitor.log.build_log_config())
