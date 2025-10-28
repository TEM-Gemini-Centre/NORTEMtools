"""
This module contains io tools for loading and saving template matching related files, such as template banks or neatly formatted crystal maps.
"""

from NORTEMtools.Emil.utils import MyPath
from NORTEMtools import logger
import pickle


def load_template(template_path: MyPath) -> object:
    """ "Load a template from a given path using pickle.
    Args:
        template_path (MyPath): Path to the template file.

    Returns:
        template: Loaded template object.
    """
    try:
        with template_path.open("rb") as f:
            logger.debug("Loading template using pickle")
            template = pickle.load(f)
            logger.debug(f"Template loaded: {template!s}")
    except Exception as e:
        logger.error(f"Failed to load template: {e}")
        raise
    return template


def save_template(template: object, template_path: MyPath) -> None:
    """Save a template to a given path using pickle.
    Args:
        template (object): The template object to save.
        template_path (MyPath): Path to save the template file.

    Returns:
        None
    """
    try:
        with template_path.open("wb") as f:
            logger.debug("Saving template using pickle")
            pickle.dump(template, f)
            logger.debug(f"Template saved to: {template_path}")
    except Exception as e:
        logger.error(f"Failed to save template: {e}")
        raise
