"""
NORTEMtools

Code for simplifying data analysis at NORTEM.

To add your own code, see the `/Example` or the repository README.md.

"""

import logging
from pathlib import Path


def make_logger(log_file: str = "log.txt", formatter=None) -> logging.Logger:
    __default_fmt__ = "[%(asctime)s - %(levelname)s - %(name)s - %(pathname)s:%(lineno)s - %(funcName)20s() ]:\n%(message)s\n"

    f"""Create a logger that logs to both console and a file.

    Args:
        log_file (str, optional): The name of the log file. Defaults to 'log.txt'.
        formatter (Union[None, str], optional): The log message format. Defaults to None which will use a detailed format: "{__default_fmt__}".
        
    Returns:
        logging.Logger: Configured logger instance.
    
    """

    logger = logging.getLogger(__file__)
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
    fh = logging.FileHandler(log_file, "w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


logger = make_logger()


def add_log_handler(logfile: str, logger: logging.Logger = logger):
    """
    Add a file handler to the logger

    :param logfile: The file to add to the logger
    :type logfile: str
    :param logger: The logger to add the file to
    :type logger: logging.Logger
    """

    fh = logging.FileHandler(logfile, "w", encoding="utf-8")
    fh.setLevel(logger.level)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            fh.setFormatter(handler.formatter)
            break
    logger.debug(f'Adding file handler to stream "{Path(fh.stream.name).absolute()}"')
    logger.addHandler(fh)


def remove_log_handler(logfile: str, logger: logging.Logger = logger):
    """
    Docstring for remove_log_handler

    :param logfile: Description
    :type logfile: str
    :param logger: Description
    :type logger: logging.Logger
    """
    logfile = Path(logfile).absolute()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            stream_path = Path(handler.stream.name)
            if stream_path == logfile:
                logger.debug(f'Removing file handler to stream "{stream_path}"')
                handler.close()
                logger.removeHandler(handler)


# Submodule imports
from . import Emil
from . import Example

logger.debug(f"Loaded NORTEMtools module")
