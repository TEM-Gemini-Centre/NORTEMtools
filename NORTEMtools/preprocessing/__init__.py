from NORTEMtools import logger
from NORTEMtools.utils import Signal, log_shift, MyPath, load, load_metadata_from_json, set_metadata, NORTEMError, Error

from typing import Union, Dict
import hyperspy.api as hs
import pyxem as pxm
import matplotlib.pyplot as plt
import numpy as np
from json import load as jload


class PreprocessingError(Error):
    pass

def make_navigation_mask(signal: Signal, width: Union[None, int] = None) -> hs.signals.Signal2D:
    """
    Make a navigation mask frame

    :param signal: The signal to make a mask for
    :param width: the width of the mask frame
    :returns: navigation mask

    :type signal: Signal
    :type width: Union[None, int]
    :rtype: hs.signals.Signal2D
    """
    navigation_mask = hs.signals.Signal2D(np.zeros(signal.axes_manager.navigation_shape, dtype=bool).T).T
    navigation_mask.metadata.General.title = 'Navigation mask'
    if width is None:
        width = 0
    logger.debug(f'Making navigation mask with a frame width of {width} pixels set to True')
    navigation_mask.inav[width:-width, width:-width] = True    
    return navigation_mask
    
def center_direct_beam(signal: Signal, com_mask: Union[None, tuple] = None, estimate_linear_shift: bool = False, plot_results: bool = False, estimate_linear_shifts_kwargs: Union[None, Dict] = None, **kwargs) -> pxm.signals.ElectronDiffraction2D:
    """
    Center the direct beam of a signal

    :param signal: The signal to center the direct beam for
    :param com_mask: The region of the diffraction patterns to calculate COM within
    :param estimate_linear_shift: Whether to estimate linear shifts or not
    :param plot_results: Whether to plot the results or not
    :param estimate_linear_shifts_kwargs: Keyword arguments passed to linear shift estimation
    :kwargs: Optional keyword arguments passed to signal.center_direct_beam.
    :returns: Centered. The centered signal.

    :type signal: Signal
    :type com_mask: Union[None, tuple]
    :type estimate_linear_shift: bool
    :type plot_results: bool
    :type estimate_linear_shift_kwargs: Dict
    :rtype: Signal
    """


    kwargs['inplace'] = kwargs.get('inplace', False) #Set default inplace to False
    _ = kwargs.pop('method', None) #Remove any `method` as it is not compatible with the shifts we will specify later on.

    logger.debug(f'Calculating maximum through-stack before centering')
    max_before = signal.max(axis=[0, 1])
    max_before.metadata.General.title = 'Before'
    try:
        max_before.compute()
    except Exception:
        pass

    centering_metadata = {} #Metadata dict for centering parameters

    if 'shifts' not in kwargs:
        logger.debug('No shifts provided to centering algorithm. Will calculate shifts using `center_of_mass`')
        if com_mask is None:
            nx, ny = signal.axes_manager.signal_size
            cx, cy = nx//2, ny//2
            com_mask = (cx, cy, 15)
            logger.debug(f'No mask for COM analysis provided, using default {com_mask}')

        logger.debug(f'Finding direct beam position through COM analysis within {com_mask}')
        centering_metadata['COM_mask'] = com_mask

        shifts = signal.get_direct_beam_position(method='center_of_mass', mask=com_mask)
        try:
            shifts.compute()
        except Exception:
            pass
        kwargs['shifts'] = shifts
        logger.debug(f'Found direct beam positions {shifts}')

    if estimate_linear_shift:
        logger.debug(f'Estimating linear shifts with input: {estimate_linear_shifts_kwargs}')
        linear_shift = kwargs['shifts'].get_linear_plane(**estimate_linear_shifts_kwargs)
        kwargs['shifts'] = linear_shift
        logger.debug(f'Estimated linear shifts: {linear_shift}')
        centering_metadata['estimate_linear_shift'] = estimate_linear_shifts_kwargs

    centering_metadata['Shifts'] = kwargs['shifts']
    logger.debug(f'Centering direct beam using arguments: {kwargs}')
    centered = signal.center_direct_beam(**kwargs)
    logger.debug(f'Finished centering direct beam')

    logger.debug(f'Calculating maximum through-stack after centering')
    max_after = centered.max(axis=[0, 1])
    max_after.metadata.General.title = 'After'

    try:
        max_after.compute()
    except Exception:
        pass
        
    centering_metadata['Max_before'] = max_before
    centering_metadata['Max_after'] = max_after

    #Add centering metadata to signal metadata.
    centered.metadata.add_dictionary({
        'Preprocessing': {
            'Centering': centering_metadata
        }
    })

    if plot_results:
        hs.plot.plot_images([max_before, max_after], overlay=True, alphas=[1, 0.75], colors=['w', 'r'])
    
    logger.debug(f"Finished centering direct beam. Centering metadata:\n{centered.metadata.Preprocessing.Centering}")

    return centered

def set_calibrations(signal: Signal, x: Union[None, float], y: Union[None, float], kx:Union[None, float], ky:Union[None, float]) -> None:
    """
    Sets the calibration of provided signal

    :param signal: The signal to calibrate
    :param x: x-axis scan step size calibration
    :param y: y-axis scan step size calibration
    :param kx: x-axis diffraction calibration
    :param ky: y-axis diffraction calibration
    :returns: None

    :type x: Union[None, float]
    :type y: Union[None, float]
    :type kx: Union[None, float]
    :type ky: Union[None, float]
    :rtype: None
    """
    if x is None and y is None:
        logger.warn('No scan calibrations provided!')
    else:
        if x is None:
            logger.warn(f'No scan calibration in x-axis provided, using calibration for y-axis = {x}')
            x = y
        elif y is None:
            logger.warn(f'No scan calibration in y-axis provided, using calibration for x-axis = {y}')
            y = x
        logger.info(f'Setting scan step size calibrations:\n\tx={x}\n\ty={y}')
        signal.set_scan_calibration(x)
        signal.axes_manager['y'].scale = y

    if kx is None and ky is None:
        logger.warn('No diffraction calibrations provided!')
    else:
        if kx is None:
            logger.warn(f'No diffraction calibration in x-axis provided, using calibration for y-axis = {ky}')
            kx = ky
        elif ky is None:
            logger.warn(f'No diffraction calibration in y-axis provided, using calibration for x-axis = {kx}')
            ky = kx
        logger.info(f'Setting diffraction calibrations:\n\tkx={kx}\n\tky={ky}')
        signal.set_diffraction_calibration(kx)
        signal.axes_manager['ky'].scale =ky

    logger.info(f'Calibrated axes manager:\n{signal.axes_manager}')