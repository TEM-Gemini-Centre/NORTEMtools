from NORTEMtools import logger
import hyperspy.api as hs
hs.set_log_level('INFO')

import numpy as np
import matplotlib.pyplot as plt
import pyxem as pxm
from pathlib import Path
from typing import Union, Dict
from zarr import NestedDirectoryStore, ZipStore
from json import load as jload

Signal = Union[pxm.signals.ElectronDiffraction2D, pxm.signals.LazyDiffraction2D]

class MyPath(Path):  # helpful for appending suffixes to filenames
    """
    A subclass of pathlib.Path with an append method to add suffixes to filenames.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize MyPath instance.

        Args:
            *args: Arguments to pass to the Path constructor.
            **kwargs: Keyword arguments to pass to the Path constructor.
        
        Returns:
            None
        """
        super().__init__(*args, **kwargs)

    def append(self, s: str, suffix: Union[None, str] = None, delimiter: str = '_') -> 'MyPath':
        """Append a string to the stem of the filename.
        Args:
            s (str): The string to append.
            suffix (Union[None, str], optional): The file extension to use. Defaults to None.
            delimiter (str, optional): The delimiter to use between the original stem and the appended string. Defaults to '_'.
                
        Returns:
            MyPath: A new MyPath instance with the modified filename.
        """
        if suffix:
            return self.with_name(f'{self.stem}{delimiter}{s}.{suffix}')
        else:
            return self.with_stem(f'{self.stem}{delimiter}{s}')

def load(path: MyPath, *args, **kwargs) -> Signal:
    """
    Docstring for load
    
    :param path: Description
    :type path: MyPath
    :return: Description
    :rtype: Signal
    """

    logger.info(f'Loading data from "{path}"')
    if path.suffix.lower() == '.zspy':
        try:
            logger.debug('Detected .zspy file; loading using NestedDirectoryStore.')
            store = NestedDirectoryStore(str(path))
            signal = hs.load(store, *args, **kwargs)
        except Exception as e:    
            logger.debug('Detected .zspy file; loading using zarr ZipStore.')
            store = ZipStore(str(path))
            signal = hs.load(store, *args, **kwargs)
    else:
        logger.debug('Loading data')
        signal = hs.load(str(path), *args, **kwargs)

    logger.info(f'Loaded data: {signal}\n'\
                f'Data loaded with shape {signal.data.shape} and type {type(signal)}\n'\
                f'Current signal axes:\n{signal.axes_manager}'
                )
    
    logger.debug(f'Current metadata:\n{signal.metadata.as_dictionary()!s}')
    
    return signal

def frame_string(s: str) -> str:  # helper function to make a header string
    """Create a header string with asterisks.

    Args:
        s (str): The string to create a header for.

    Returns:
        str: The formatted header string.
    """
    header = '*' * (len(s) + 8)
    return f'{header}\n*** {s} ***\n{header}'

def compute(s: Signal) -> Signal:  # helper function to compute dask arrays with error handling
    """Compute a Signal object, handling exceptions gracefully.

    Args:
        s (Signal): The Signal object to compute.

    Returns:
        Signal: The computed Signal object.

    """
    try:
        logger.debug('Computing signal...')
        s.compute()
        logger.debug('Signal computed successfully.')
    except Exception as e:
        logger.debug(f'Error during computation: {e}\n Continuing without halting execution.')
    finally:
        return s

def log_shift(image:np.ndarray, shift: float=1.) -> np.ndarray:
    """
    Apply a logarithmic shift to an image.

    Args:
        image (np.ndarray): The input image.
        shift (float, optional): The shift value to avoid log(0). Defaults to 1.0.
    
    Returns:
        np.ndarray: The log-shifted image.

    """
    return np.log10(image+shift) - np.log10(shift) 

def get_random_coordinates(signal: Signal, n: int = 1, seed: int = 197405) -> list:
    """
    Pick n random coordinates from a 4DSTEM signal.

    Args:
        signal (pxm.signals.ElectronDiffraction2D): The input 4DSTEM signal.
        n (int, optional): The number of random coordinates to pick. Defaults to 1.
        seed (int, optional): The random seed for reproducibility. Defaults to 197405.
    
    Returns:
        list: A list of tuples representing the random coordinates.
    """

    nx, ny = signal.axes_manager.navigation_shape

    if not n%2 == 0:
        logger.debug('Number of random coordinates requested is not even; incrementing by 1 to make it even.')
        n += 1
    
    if n > nx*ny:
        logger.debug(f'Number of random coordinates requested ({n}) exceeds total number of available coordinates ({nx*ny}).')
        n = nx*ny
        logger.debug(f'Using max coordinates instead: n={n}.') 

    logger.debug(f'Picking {n} random coordinates with seed {seed}')
    
    np.random.seed(seed)
    coords = []
    for _ in range(n):
        x = np.random.randint(0, nx-1)
        y = np.random.randint(0, ny-1)
        coords.append((x, y))
    
    logger.debug(f'Picked random coordinates: {coords!s}')
    return coords

def pick_random(signal: pxm.signals.ElectronDiffraction2D, *args, show: bool = True,
                output_path: Union[None, MyPath] = None, **kwargs) -> pxm.signals.ElectronDiffraction2D:
    """
    Pick random coordinates from a 4DSTEM signal and return a new signal with those patterns.

    Args:
        signal (pxm.signals.ElectronDiffraction2D): The input 4DSTEM signal.
        *args: Additional arguments to pass to get_random_coordinates.
        show (bool, optional): Whether to display and save a figure of the selected coordinates. Defaults to True.
        output_path (Union[None, MyPath], optional): Path to save the figure. Defaults to None.
        **kwargs: Additional keyword arguments to pass to get_random_coordinates.
    
    Returns:
        pxm.signals.ElectronDiffraction2D: A new signal containing the selected random patterns.
       
    """
    # Get random coordinates
    coords = get_random_coordinates(signal, *args, **kwargs)

    if show:
        logger.debug('Generating VBF coordinates figure...')
        xs = signal.axes_manager[0].axis
        ys = signal.axes_manager[1].axis
        cx, cy = xs[len(xs) // 2], ys[len(ys) // 2]  # Pick middle of the axis as center.
        r = max(xs) * 0.02  # 2 % of total maximum width
        logger.debug(f'Center for VBF circle ROI: ({cx:.2f}, {cy:.2f}), radius: {r:.2f}')
        try:
            logger.debug('Computing VBF for coordinate visualization...')
            vbf = signal.get_integrated_intensity(hs.roi.CircleROI(cx, cy, r))
            compute(vbf)
        except Exception as e:
            logger.error(f'Failed to compute VBF: {e}.\n Using total sum of pattern as image instead.')
            vbf = signal.nansum(axis=[2, 3])
            compute(vbf)

        plt.figure()
        plt.imshow(vbf.data, cmap='gray')
        for i, c in zip(np.arange(len(coords)), coords):
            plt.scatter(c[0], c[1], color='red', marker='*')
            plt.text(c[0] - 3, c[1] - 5, s=str(i), color='red')
        plt.tight_layout()
        plt.axis('off')
        try:
            if output_path is None:
                output_path = MyPath('vbf_coords.png')
                logger.warning(
                    f'Output path not provided; saving VBF coordinates figure to current directory instead: "{output_path}".')
            else:
                pass

            logger.info(f'Saving VBF coordinates figure to "{output_path}"')
            plt.savefig(output_path, dpi=300)
        except Exception as e:
            logger.error(f'Failed to save VBF coordinates figure: {e}')
        plt.close()

    selection = [signal.inav[c[0], c[1]].data for c in coords]
    # NB The length of coords must be an even number for the code below to work.
    selection = np.reshape(selection, (2, -1, 256, 256))

    logger.debug('Creating ElectronDiffraction2D signal from selected patterns...')
    selection = pxm.signals.ElectronDiffraction2D(selection)

    logger.debug('Adding metadata to selected patterns signal...')
    selection.metadata.add_dictionary(signal.metadata.as_dictionary())
    selection.metadata.General.title = 'Selected random patterns for template matching'
    selection.metadata.add_dictionary({'Processing': {'Coordinates': coords}})

    logger.debug('Setting axes for selected patterns signal...')
    for ax in range(signal.axes_manager.signal_dimension + signal.axes_manager.navigation_dimension):
        selection.axes_manager[ax].name = signal.axes_manager[ax].name
        selection.axes_manager[ax].offset = signal.axes_manager[ax].offset
        selection.axes_manager[ax].scale = signal.axes_manager[ax].scale
        selection.axes_manager[ax].units = signal.axes_manager[ax].units

    logger.debug('Computing selected patterns...')
    compute(selection)

    return selection

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

def preprocess(filename: Union[str, Path],
            lazy: bool = True, 
            com_mask: tuple = (128, 128, 12), 
            estimate_linear_shift: bool = False, 
            width: int = 20, 
            calibrations: Dict={},
            nav_chunks: int = 32,
            sig_chunks: int = 32):
    """
    Preprocess a 4D STEM data file

    :param filename: The path to the data file.
    :param lazy: Whether to work lazily. Default is True
    :param com_mask: The region of the diffraction pattern to calculate COM within given in pixel coordinates (x, y, r). Default is (128, 128, 12)
    :param estimate_linear_shift: Whether to estimate the linear shift based on COM results or not. Default is False
    :param calibrations: The calibrations of the data as a dictionary `{"x": x, "y": y, "kx": kx, "ky": ky}`. Default is empty (no calibration performed)
    :param width: The width of the navigation frame to use when estimating the linear descan shift. Default is False
    :param nav_chunks: The chunking to use in the navigation dimension
    :param sig_chunks: The chunking to use in the signal dimension
    
    :type filename: Union[str, Path]
    :type lazy: bool
    :type com_mask: Union[None, tuple]
    :type estimate_linear_shift: bool
    :type calibrations: Dict
    :type width: int
    :type nav_chunks: int
    :type sig_chunks: int
    
    :return: preprocessed_signal
    :rtype: Signal
    """
    filename=Path(filename)

    logger.debug(
        f'Preprocessing function got the following arguments:'\
        f'\n\tfilename: {filename!r}'\
        f'\n\tlazy: {lazy!r}'\
        f'\n\tcom_mask: {com_mask!r}'\
        f'\n\testimate_linear_shift: {estimate_linear_shift!r}'
        f'\n\twidth: {width!r}'\
        f'\n\tcalibrations: {calibrations!r}'\
        f'\n\tnav_chunks: {nav_chunks!r}'\
        f'\n\tsig_chunks: {sig_chunks!r}'
        )

    filename = Path(filename)
    logger.info(f'Loading data from "{filename}"')

    # Load data
    if filename.suffix == '.zspy':
        logger.debug('Loading data lazily with zarr ZipStore')
        try:
            store = ZipStore(filename)
            signal = hs.load(store, lazy=lazy)
        except Exception as e:
            logger.error(f'Exception when loading zspy file with ZipStore: {e}. Trying NestedDirectoryStore instead.')
            store = NestedDirectoryStore(filename)
            signal = hs.load(store, lazy=lazy)
    else:
        logger.debug(f'Loading data')
        signal = hs.load(filename, lazy=lazy)
    logger.debug(f'Loaded data')
    if not isinstance(signal, pxm.signals.ElectronDiffraction2D):
        logger.warning(
            f'Only ElectronDiffraction2D signals can be preprocessed. I got {signal!r} of type {type(signal)}')
    
    if isinstance(signal, pxm.signals.LazyElectronDiffraction2D):
        logger.debug(f'Rechunking data with `nav_chunks={nav_chunks}`, `sig_chunks={sig_chunks}`')
        signal.rechunk(nav_chunks=nav_chunks, sig_chunks=sig_chunks)

    # Center data
    logger.info('Centering dataset')
    estimate_linear_shift_kwargs = {}
    if estimate_linear_shift:
        estimate_linear_shift_kwargs['mask'] = make_navigation_mask(signal, width)

    signal = center_direct_beam(signal, com_mask=com_mask, estimate_linear_shift=estimate_linear_shift, plot_results=False, estimate_linear_shifts_kwargs=estimate_linear_shift_kwargs)

    # Add metadata
    metadata_dict = jload(filename.with_name("metadata.json").open('r'))
    logger.debug(f'Loaded metadata file "{filename.with_name("metadata.json")}":\n{metadata_dict}')
    
    experimental_parameters = metadata_dict.get('experimental_parameters', {})
    logger.debug(f'Setting experimental parameters:\n{experimental_parameters}')
    signal.set_experimental_parameters(**experimental_parameters)

    signal.metadata.add_dictionary({'Custom': metadata_dict})
    signal.original_metadata.add_dictionary({'Custom': metadata_dict})
    logger.info(f'Signal metadata:\n{signal.metadata!s}')

    # Set calibration
    x, y, kx, ky = [calibrations.get(f'{ax}', metadata_dict.get('axes', {}).get(f'{ax}', {}).get('scale', None)) for ax in ["x", "y", "kx", "ky"]]
    set_calibrations(signal, x, y, kx, ky)

    # Make VBF and maximum through-stack
    logger.info(f'Preparing VBF')
    vbf = signal.get_integrated_intensity(hs.roi.CircleROI(cx=0., cy=0., r_inner=0., r=0.07))
    signal.metadata.add_dictionary({
        'Preprocessing': {'VBF': vbf}
    })

    logger.info('Preparing maximum through-stack')
    maximums = signal.max(axis=[0, 1])
    signal.metadata.add_dictionary({
        'Preprocessing': {'Maximums': maximums}
    })

    # Save the VBF and maximums
    logger.info(f'Saving VBF and maximums as images')
    plt.imsave(filename.with_name(f'{filename.stem}_preprocessed_vbf.png'), vbf.data)
    plt.imsave(filename.with_name(f'{filename.stem}_preprocessed_maximums.png'), maximums.data)

    return signal