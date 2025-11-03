"""
This module contains utility functions for working with pyxem and hyperspy. It does not add new functionality or analysis tools, but simplifies code by defining commonly and frequently used functions that serve as wrappers around other analysis tools/code. For more information, please read the original documentation of hyperspy and pyxem.
"""

from NORTEMtools import logger
import hyperspy.api as hs

hs.set_log_level("INFO")

import numpy as np
import matplotlib.pyplot as plt
import pyxem as pxm
from pathlib import Path
from typing import Union, Dict
from zarr import NestedDirectoryStore, ZipStore
from json import load as jload
import argparse
import logging

Signal = Union[pxm.signals.ElectronDiffraction2D, pxm.signals.LazyDiffraction2D]


def log_with_header(header, *messages, logger=logger):
    """
    Write to logger on INFO with a framed header.

    Used to write "pretty-ish" information to a logger. This is essentially just a wrapper around the `frame_string()` function and calls to `logger.info()`.

    :param header: The header to frame before the messages.
    :param messages: optional positional argument with messages to follow the header.
    :param logger: The logger to use when logging
    """
    header = frame_string(header)
    if len(messages) > 0:
        message = "\n" + "\n".join([message for message in messages])
    else:
        message = ""
    logger.info(f"{header}{message}")


def dict2string(dictionary: Dict) -> str:
    """
    Format a dictionary into a string

    :param dictionary: Description
    :type dictionary: Dict
    :return: Description
    :rtype: str
    """

    string = ""
    for key in dictionary:
        value = dictionary[key]
        if isinstance(value, dict):
            value = dict2string(value).replace(
                "\n\t", "\n\t\t", count=1
            )  # add a new tab for each level
        string += f"\n\t{key}: {value}"
    return string


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

    def append(
        self, s: str, suffix: Union[None, str] = None, delimiter: str = "_"
    ) -> "MyPath":
        """Append a string to the stem of the filename.
        Args:
            s (str): The string to append.
            suffix (Union[None, str], optional): The file extension to use. Defaults to None.
            delimiter (str, optional): The delimiter to use between the original stem and the appended string. Defaults to '_'.

        Returns:
            MyPath: A new MyPath instance with the modified filename.
        """
        if suffix:
            return self.with_name(f"{self.stem}{delimiter}{s}.{suffix}")
        else:
            return self.with_stem(f"{self.stem}{delimiter}{s}")


def set_log_level(logger, level: int = 0):
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min([level, len(log_levels) - 1])]
    logger.setLevel(log_level)


def args2string(arguments: argparse.Namespace) -> None:
    """
    Docstring for print_parser

    :param parser: Description
    :type parser: argparse.ArgumentParser
    """

    parser_description = "Input parser arguments:"
    parser_description_header = "*" * len(parser_description)
    parser_description = (
        parser_description_header
        + "\n"
        + parser_description
        + "\n"
        + parser_description_header
        + "\n"
    )
    parser_description += "\n".join(
        [f"{arg}: {getattr(arguments, arg)}" for arg in vars(arguments)]
    )
    return parser_description


def load_metadata_from_json(filename: Union[None, MyPath]) -> Dict:
    """
    Load metadata from a json file

    :param filename: The json file containing the metadata
    :type filename: MyPath
    :return: The metadata.
    :rtype: Dict
    """

    if filename is None:
        filename = "metadata.json"

    if not filename.suffix == ".json":
        logger.warning(
            f'Metadata file name "{filename}" is not a json file. I will change the suffix to .json instead before attempting to load.'
        )
        filename = filename.with_suffix(".json")

    metadata = jload(filename.open("r"))
    logger.debug(f'Loaded metadata file "{filename}":\n{metadata}')

    return metadata


def metadata2json(filename: Union[None, MyPath], metadata: Dict):
    """
    Docstring for metadata2json

    :param filename: Description
    :type filename: Union[None, MyPath]
    :param metadata: Description
    :type metadata: Dict
    """
    if filename is None:
        filename = MyPath("metadata.json")
    filename = MyPath(filename)

    log_with_header(
        "Writing metadata to json file",
        f'File path: "{filename}"',
        f"Metadata: {dict2string(metadata)}",
    )

    logger.debug(f'Writing metadata file to "{filename}"')


def set_metadata(signal, metadata: Dict, metadata_key: str = "Custom"):
    """
    Set experimental parameters and metadata from a json file

    :param signal: The hyperspy signal to set metadata and experimental parameters for.
    :param filename: The file containing the metadata to use
    :type filename: MyPath
    """

    debug_string = "\n\t".join([f"{key}={metadata[key]}" for key in metadata])

    logger.debug(
        f"Setting metadata of signal {signal} under the {metadata_key} key:\n\t{debug_string}"
    )

    logger.debug(f"Signal metadata before:\n{signal.metadata}")
    signal.metadata.add_dictionary({metadata_key: metadata})
    logger.debug(f"Signal metadata after:\n{signal.metadata}")

    logger.debug(f"Signal original metadata before:\n{signal.original_metadata}")
    signal.original_metadata.add_dictionary({metadata_key: metadata})
    logger.debug(f"Signal original metadata after:\n{signal.original_metadata}")

    logger.info(
        f'Added custom signal metadata under key "{metadata_key}":\n{signal.metadata.as_dict()[metadata_key]!s}'
    )


def set_experimental_parameters(signal, parameters: Dict) -> None:
    """
    Docstring for set_experimental_parameters

    :param signal: Description
    :param parameters: Description
    :type parameters: Dict
    """
    logger.info(frame_string("Setting experimental parameters"))

    # Log some infor for debugging
    debug_string = "\n\t".join([f"{key}={parameters[key]}" for key in parameters])
    logger.debug(
        f"Got parameters for setting experimental parameters:\n\t{debug_string}"
    )

    # Set one experimental parameter at a time.
    for key in parameters:
        try:
            logger.debug(f"Setting experimental parameter {key}={parameters[key]}")
            signal.set_experimental_parameters(
                **{key: parameters[key]}
            )  # Set one experimental parameter
            logger.info(
                f"Set experimental parameter of signal: {key}={parameters[key]}"
            )
        except Exception as e:
            logger.warning(
                f"Could not set experimental parameter {key}={parameters[key]} due to error: {e}. Ignoring and continuing, but you might want to double check this."
            )
    logger.debug(f"Finished setting experimental parameters.")


def load(path: MyPath, *args, **kwargs) -> Signal:
    """
    Docstring for load

    :param path: Description
    :type path: MyPath
    :return: Description
    :rtype: Signal
    """
    path = MyPath(path)
    log_with_header("Loading data", f'Loading data from "{path}"')
    if path.suffix.lower() == ".zspy":
        try:
            logger.debug("Detected .zspy file; loading using NestedDirectoryStore.")
            store = NestedDirectoryStore(str(path))
            signal = hs.load(store, *args, **kwargs)
        except Exception as e:
            logger.debug("Detected .zspy file; loading using zarr ZipStore.")
            store = ZipStore(str(path))
            signal = hs.load(store, *args, **kwargs)
    else:
        logger.debug("Loading data")
        signal = hs.load(str(path), *args, **kwargs)

    log_with_header(
        "Loaded data",
        f"{signal}",
        f"Shape {signal.data.shape} and type {type(signal)}",
        f"Current signal axes:\n{signal.axes_manager}",
    )

    logger.debug(f"Current metadata:\n{signal.metadata.as_dictionary()!s}")

    return signal


def frame_string(s: str) -> str:  # helper function to make a header string
    """Create a header string with asterisks.

    Args:
        s (str): The string to create a header for.

    Returns:
        str: The formatted header string.
    """
    header = "*" * (len(s) + 8)
    return f"{header}\n*** {s} ***\n{header}"


def compute(
    s: Signal,
) -> Signal:  # helper function to compute dask arrays with error handling
    """Compute a Signal object, handling exceptions gracefully.

    Args:
        s (Signal): The Signal object to compute.

    Returns:
        Signal: The computed Signal object.

    """
    try:
        logger.debug("Computing signal...")
        s.compute()
        logger.debug("Signal computed successfully.")
    except Exception as e:
        logger.debug(
            f"Error during computation: {e}\n Continuing without halting execution."
        )
    finally:
        return s


def log_shift(image: np.ndarray, shift: float = 1.0) -> np.ndarray:
    """
    Apply a logarithmic shift to an image.

    Args:
        image (np.ndarray): The input image.
        shift (float, optional): The shift value to avoid log(0). Defaults to 1.0.

    Returns:
        np.ndarray: The log-shifted image.

    """
    return np.log10(image + shift) - np.log10(shift)


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

    if not n % 2 == 0:
        logger.debug(
            "Number of random coordinates requested is not even; incrementing by 1 to make it even."
        )
        n += 1

    if n > nx * ny:
        logger.debug(
            f"Number of random coordinates requested ({n}) exceeds total number of available coordinates ({nx*ny})."
        )
        n = nx * ny
        logger.debug(f"Using max coordinates instead: n={n}.")

    logger.debug(f"Picking {n} random coordinates with seed {seed}")

    np.random.seed(seed)
    coords = []
    for _ in range(n):
        x = np.random.randint(0, nx - 1)
        y = np.random.randint(0, ny - 1)
        coords.append((x, y))

    logger.debug(f"Picked random coordinates: {coords!s}")
    return coords


def pick_random(
    signal: pxm.signals.ElectronDiffraction2D,
    *args,
    show: bool = True,
    output_path: Union[None, MyPath] = None,
    **kwargs,
) -> pxm.signals.ElectronDiffraction2D:
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
        logger.debug("Generating VBF coordinates figure...")
        xs = signal.axes_manager[0].axis
        ys = signal.axes_manager[1].axis
        cx, cy = (
            xs[len(xs) // 2],
            ys[len(ys) // 2],
        )  # Pick middle of the axis as center.
        r = max(xs) * 0.02  # 2 % of total maximum width
        logger.debug(
            f"Center for VBF circle ROI: ({cx:.2f}, {cy:.2f}), radius: {r:.2f}"
        )
        try:
            logger.debug("Computing VBF for coordinate visualization...")
            vbf = signal.get_integrated_intensity(hs.roi.CircleROI(cx, cy, r))
            compute(vbf)
        except Exception as e:
            logger.error(
                f"Failed to compute VBF: {e}.\n Using total sum of pattern as image instead."
            )
            vbf = signal.nansum(axis=[2, 3])
            compute(vbf)

        plt.figure()
        plt.imshow(vbf.data, cmap="gray")
        ax = plt.gca()
        for i, c in zip(np.arange(len(coords)), coords):
            ax.scatter(
                c[0], c[1], color="red", marker="*", label=f"{i}: ({c[0]},{c[1]})"
            )
            ax.text(c[0] - 3, c[1] - 5, s=f"{i}", color="red")
        ax.legend(
            fontsize="x-small",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        plt.tight_layout()
        plt.axis("off")
        try:
            if output_path is None:
                output_path = MyPath("vbf_coords.png")
                logger.warning(
                    f'Output path not provided; saving VBF coordinates figure to current directory instead: "{output_path}".'
                )
            else:
                pass

            logger.info(f'Saving VBF coordinates figure to "{output_path}"')
            plt.savefig(output_path, dpi=300)
        except Exception as e:
            logger.error(f"Failed to save VBF coordinates figure: {e}")
        plt.close()
    logger.debug("Selecting pixels from dataset")
    selection = []
    for c in coords:
        logger.debug(f"Getting signal at pixel coordinate ({c[0]}, {c[1]})")
        selection.append(signal.inav[c[0], c[1]].data)
    logger.debug(f"Got {len(selection)} signals from signal")
    # for s, c in zip(selection, coords):
    #    assert s == signal.inav[c].data, f'Data at coordinate {c} is not equal to the corresponding data in the selection'

    # NB The length of coords must be an even number for the code below to work.
    new_shape = (2, -1, 256, 256)
    logger.debug(f"Reshaping selected signals to shape {new_shape}")
    selection = np.reshape(selection, new_shape)
    logger.debug(f"Reshaped signal has shape {selection.shape}")
    # for i, s, c in enumerate(zip(selection, coords)):
    #    logger.debug(f'Verifying that data at location {i} is equal to data at coordinate {c}')
    #    assert s == signal.inav[c[0], c[1]].data, f'Data at coordinate {c} is not equal to the corresponding data in the selection'
    #    assert selection[i] == signal.inav[c[0], c[1]].data, f'Data at location {i} in selected signal should be equal to data at coordinate {c} in signal'

    logger.debug("Creating ElectronDiffraction2D signal from selected patterns...")
    selection = pxm.signals.ElectronDiffraction2D(selection)
    logger.debug(f"Cast reshaped signal into hyperspy signal: {selection}")

    logger.debug("Adding metadata to selected patterns signal...")
    selection.metadata.add_dictionary(signal.metadata.as_dictionary())
    selection.metadata.General.title = "Selected random patterns for template matching"
    selection.metadata.add_dictionary({"Processing": {"Coordinates": coords}})

    logger.debug("Setting axes for selected patterns signal...")
    for ax in range(
        signal.axes_manager.signal_dimension + signal.axes_manager.navigation_dimension
    ):
        selection.axes_manager[ax].name = signal.axes_manager[ax].name
        selection.axes_manager[ax].offset = signal.axes_manager[ax].offset
        selection.axes_manager[ax].scale = signal.axes_manager[ax].scale
        selection.axes_manager[ax].units = signal.axes_manager[ax].units

    logger.debug("Computing selected patterns...")
    compute(selection)

    return selection


def make_navigation_mask(
    signal: Signal, width: Union[None, int] = None
) -> hs.signals.Signal2D:
    """
    Make a navigation mask frame

    :param signal: The signal to make a mask for
    :param width: the width of the mask frame
    :returns: navigation mask

    :type signal: Signal
    :type width: Union[None, int]
    :rtype: hs.signals.Signal2D
    """
    navigation_mask = hs.signals.Signal2D(
        np.zeros(signal.axes_manager.navigation_shape, dtype=bool).T
    ).T
    navigation_mask.metadata.General.title = "Navigation mask"
    if width is None:
        width = 0
    logger.debug(
        f"Making navigation mask with a frame width of {width} pixels set to True"
    )
    navigation_mask.inav[width:-width, width:-width] = True
    return navigation_mask


def center_direct_beam(
    signal: Signal,
    com_mask: Union[None, tuple] = None,
    estimate_linear_shift: bool = False,
    plot_results: bool = False,
    estimate_linear_shifts_kwargs: Union[None, Dict] = None,
    **kwargs,
) -> pxm.signals.ElectronDiffraction2D:
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

    kwargs["inplace"] = kwargs.get("inplace", False)  # Set default inplace to False
    _ = kwargs.pop(
        "method", None
    )  # Remove any `method` as it is not compatible with the shifts we will specify later on.

    logger.debug(f"Calculating maximum through-stack before centering")
    max_before = signal.max(axis=[0, 1])
    max_before.metadata.General.title = "Before"
    try:
        max_before.compute()
    except Exception:
        pass

    centering_metadata = {}  # Metadata dict for centering parameters

    if "shifts" not in kwargs:
        logger.debug(
            "No shifts provided to centering algorithm. Will calculate shifts using `center_of_mass`"
        )
        if com_mask is None:
            nx, ny = signal.axes_manager.signal_size
            cx, cy = nx // 2, ny // 2
            com_mask = (cx, cy, 15)
            logger.debug(f"No mask for COM analysis provided, using default {com_mask}")

        logger.debug(
            f"Finding direct beam position through COM analysis within {com_mask}"
        )
        centering_metadata["COM_mask"] = com_mask

        shifts = signal.get_direct_beam_position(method="center_of_mass", mask=com_mask)
        try:
            shifts.compute()
        except Exception:
            pass
        kwargs["shifts"] = shifts
        logger.debug(f"Found direct beam positions {shifts}")

    if estimate_linear_shift:
        logger.debug(
            f"Estimating linear shifts with input: {estimate_linear_shifts_kwargs}"
        )
        linear_shift = kwargs["shifts"].get_linear_plane(
            **estimate_linear_shifts_kwargs
        )
        kwargs["shifts"] = linear_shift
        logger.debug(f"Estimated linear shifts: {linear_shift}")
        centering_metadata["estimate_linear_shift"] = estimate_linear_shifts_kwargs

    centering_metadata["Shifts"] = kwargs["shifts"]
    logger.debug(f"Centering direct beam using arguments: {kwargs}")
    centered = signal.center_direct_beam(**kwargs)
    logger.debug(f"Finished centering direct beam")

    logger.debug(f"Calculating maximum through-stack after centering")
    max_after = centered.max(axis=[0, 1])
    max_after.metadata.General.title = "After"

    try:
        max_after.compute()
    except Exception:
        pass

    centering_metadata["Max_before"] = max_before
    centering_metadata["Max_after"] = max_after

    # Add centering metadata to signal metadata.
    centered.metadata.add_dictionary(
        {"Preprocessing": {"Centering": centering_metadata}}
    )

    if plot_results:
        hs.plot.plot_images(
            [max_before, max_after], overlay=True, alphas=[1, 0.75], colors=["w", "r"]
        )

    logger.debug(
        f"Finished centering direct beam. Centering metadata:\n{centered.metadata.Preprocessing.Centering}"
    )

    return centered


def set_calibrations(
    signal: Signal,
    x: Union[None, float],
    y: Union[None, float],
    kx: Union[None, float],
    ky: Union[None, float],
) -> None:
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
        logger.warn("No scan calibrations provided!")
    else:
        if x is None:
            logger.wawarning(
                f"No scan calibration in x-axis provided, using calibration for y-axis = {x}"
            )
            x = y
        elif y is None:
            logger.warning(
                f"No scan calibration in y-axis provided, using calibration for x-axis = {y}"
            )
            y = x
        logger.info(f"Setting scan step size calibrations:\n\tx={x}\n\ty={y}")
        signal.set_scan_calibration(x)
        signal.axes_manager["y"].scale = y

    if kx is None and ky is None:
        logger.warn("No diffraction calibrations provided!")
    else:
        if kx is None:
            logger.warning(
                f"No diffraction calibration in x-axis provided, using calibration for y-axis = {ky}"
            )
            kx = ky
        elif ky is None:
            logger.warning(
                f"No diffraction calibration in y-axis provided, using calibration for x-axis = {kx}"
            )
            ky = kx
        logger.info(f"Setting diffraction calibrations:\n\tkx={kx}\n\tky={ky}")
        signal.set_diffraction_calibration(kx)
        signal.axes_manager["ky"].scale = ky

    logger.info(f"Calibrated axes manager:\n{signal.axes_manager}")


def make_navigation_mask(
    signal: Signal, width: Union[None, int] = None
) -> hs.signals.Signal2D:
    """
    Make a navigation mask frame

    :param signal: The signal to make a mask for
    :param width: the width of the mask frame
    :returns: navigation mask

    :type signal: Signal
    :type width: Union[None, int]
    :rtype: hs.signals.Signal2D
    """
    navigation_mask = hs.signals.Signal2D(
        np.zeros(signal.axes_manager.navigation_shape, dtype=bool).T
    ).T
    navigation_mask.metadata.General.title = "Navigation mask"
    if width is None:
        width = 0
    logger.debug(
        f"Making navigation mask with a frame width of {width} pixels set to True"
    )
    navigation_mask.inav[width:-width, width:-width] = True
    return navigation_mask


def center_direct_beam(
    signal: Signal,
    com_mask: Union[None, tuple] = None,
    estimate_linear_shift: bool = False,
    plot_results: bool = False,
    estimate_linear_shifts_kwargs: Union[None, Dict] = None,
    **kwargs,
) -> pxm.signals.ElectronDiffraction2D:
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

    kwargs["inplace"] = kwargs.get("inplace", False)  # Set default inplace to False
    _ = kwargs.pop(
        "method", None
    )  # Remove any `method` as it is not compatible with the shifts we will specify later on.

    logger.debug(f"Calculating maximum through-stack before centering")
    max_before = signal.max(axis=[0, 1])
    max_before.metadata.General.title = "Before"
    try:
        max_before.compute()
    except Exception:
        pass

    centering_metadata = {}  # Metadata dict for centering parameters

    if "shifts" not in kwargs:
        logger.debug(
            "No shifts provided to centering algorithm. Will calculate shifts using `center_of_mass`"
        )
        if com_mask is None:
            nx, ny = signal.axes_manager.signal_size
            cx, cy = nx // 2, ny // 2
            com_mask = (cx, cy, 15)
            logger.debug(f"No mask for COM analysis provided, using default {com_mask}")

        logger.debug(
            f"Finding direct beam position through COM analysis within {com_mask}"
        )
        centering_metadata["COM_mask"] = com_mask

        shifts = signal.get_direct_beam_position(method="center_of_mass", mask=com_mask)
        try:
            shifts.compute()
        except Exception:
            pass
        kwargs["shifts"] = shifts
        logger.debug(f"Found direct beam positions {shifts}")

    if estimate_linear_shift:
        logger.debug(
            f"Estimating linear shifts with input: {estimate_linear_shifts_kwargs}"
        )
        linear_shift = kwargs["shifts"].get_linear_plane(
            **estimate_linear_shifts_kwargs
        )
        kwargs["shifts"] = linear_shift
        logger.debug(f"Estimated linear shifts: {linear_shift}")
        centering_metadata["estimate_linear_shift"] = estimate_linear_shifts_kwargs

    centering_metadata["Shifts"] = kwargs["shifts"]
    logger.debug(f"Centering direct beam using arguments: {kwargs}")
    centered = signal.center_direct_beam(**kwargs)
    logger.debug(f"Finished centering direct beam")

    logger.debug(f"Calculating maximum through-stack after centering")
    max_after = centered.max(axis=[0, 1])
    max_after.metadata.General.title = "After"

    try:
        max_after.compute()
    except Exception:
        pass

    centering_metadata["Max_before"] = max_before
    centering_metadata["Max_after"] = max_after

    # Add centering metadata to signal metadata.
    centered.metadata.add_dictionary(
        {"Preprocessing": {"Centering": centering_metadata}}
    )

    if plot_results:
        hs.plot.plot_images(
            [max_before, max_after], overlay=True, alphas=[1, 0.75], colors=["w", "r"]
        )

    logger.debug(
        f"Finished centering direct beam. Centering metadata:\n{centered.metadata.Preprocessing.Centering}"
    )

    return centered


def set_calibrations(
    signal: Signal,
    x: Union[None, float],
    y: Union[None, float],
    kx: Union[None, float],
    ky: Union[None, float],
) -> None:
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
        logger.warn("No scan calibrations provided!")
    else:
        if x is None:
            logger.warn(
                f"No scan calibration in x-axis provided, using calibration for y-axis = {x}"
            )
            x = y
        elif y is None:
            logger.warn(
                f"No scan calibration in y-axis provided, using calibration for x-axis = {y}"
            )
            y = x
        logger.info(f"Setting scan step size calibrations:\n\tx={x}\n\ty={y}")
        signal.set_scan_calibration(x)
        signal.axes_manager["y"].scale = y

    if kx is None and ky is None:
        logger.warn("No diffraction calibrations provided!")
    else:
        if kx is None:
            logger.warn(
                f"No diffraction calibration in x-axis provided, using calibration for y-axis = {ky}"
            )
            kx = ky
        elif ky is None:
            logger.warn(
                f"No diffraction calibration in y-axis provided, using calibration for x-axis = {kx}"
            )
            ky = kx
        logger.info(f"Setting diffraction calibrations:\n\tkx={kx}\n\tky={ky}")
        signal.set_diffraction_calibration(kx)
        signal.axes_manager["ky"].scale = ky

    logger.info(f"Calibrated axes manager:\n{signal.axes_manager}")


__all__ = [
    "Signal",
    "MyPath",
    "set_log_level",
    "args2string",
    "load_metadata_from_json",
    "set_metadata",
    "set_experimental_parameters",
    "load",
    "frame_string",
    "compute",
    "log_shift",
    "get_random_coordinates",
    "pick_random",
    "make_navigation_mask",
    "center_direct_beam",
    "set_calibrations",
    "make_navigation_mask",
    "center_direct_beam",
    "set_calibrations",
    "logger",
    "log_with_header",
]
