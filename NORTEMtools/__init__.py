"""
NORTEMtools

Code for simplifying data analysis at NORTEM.

To add your own code, see the `/Example` or the repository README.md.

"""

import logging


def make_logger(log_name: str = "log.txt", formatter=None) -> logging.Logger:
    __default_fmt__ = "[%(asctime)s - %(levelname)s - %(name)s - %(pathname)s:%(lineno)s - %(funcName)20s() ]:\n%(message)s\n"

    f"""Create a logger that logs to both console and a file.

    Args:
        log_name (str, optional): The name of the log file. Defaults to 'log.txt'.
        formatter (Union[None, str], optional): The log message format. Defaults to None which will use a detailed format: "{__default_fmt__}".
        
    Returns:
        logging.Logger: Configured logger instance.
    
    """

    logger = logging.getLogger("NORTEM tools logger")
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # Set default formatter
    if formatter is None:
        formatter = logging.Formatter(__default_fmt__, datefmt="%d-%m-%Y %H:%M:%S")
    else:
        formatter = logging.Formatter(fmt=formatter, datefmt="%d-%m-%Y %H:%M:%S")

    # Add stream handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Add file handler
    fh = logging.FileHandler(log_name, "w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


logger = make_logger()

# Submodule imports
from . import Emil
from . import Example

logger.debug(f"Loaded NORTEMtools module")
