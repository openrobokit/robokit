import logging

import rich
from rich.logging import RichHandler


# --- logging ---------------------------------------------------------------
rich.reconfigure(log_path=False)

logger = logging.getLogger("robokit")
logger.propagate = False
logger.setLevel("INFO")
if not logger.handlers:
    logger.addHandler(RichHandler(console=rich.get_console(), show_path=False, log_time_format="[%X]"))


__version__ = "1.0.0rc1"

__all__ = ["__version__"]
