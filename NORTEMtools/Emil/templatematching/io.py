"""
This module contains io tools for loading and saving template matching related files, such as template banks or neatly formatted crystal maps.
"""

import NORTEMtools.Emil.utils as _emutils
from NORTEMtools import _logger
import pickle


def load_template(template_path: _emutils.MyPath) -> object:
    """ "Load a template from a given path using pickle.
    Args:
        template_path (_emutils.MyPath): Path to the template file.

    Returns:
        template: Loaded template object.
    """
    try:
        with template_path.open("rb") as f:
            _logger.debug("Loading template using pickle")
            template = pickle.load(f)
            _logger.debug(f"Template loaded: {template!s}")
    except Exception as e:
        _logger.error(f"Failed to load template: {e}")
        raise
    return template


def save_template(template: object, template_path: _emutils.MyPath) -> None:
    """Save a template to a given path using pickle.
    Args:
        template (object): The template object to save.
        template_path (_emutils.MyPath): Path to save the template file.

    Returns:
        None
    """
    try:
        with template_path.open("wb") as f:
            _logger.debug("Saving template using pickle")
            pickle.dump(template, f)
            _logger.debug(f"Template saved to: {template_path}")
    except Exception as e:
        _logger.error(f"Failed to save template: {e}")
        raise
