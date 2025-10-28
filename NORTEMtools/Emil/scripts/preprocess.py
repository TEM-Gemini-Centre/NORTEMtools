"""
This script can be used to preprocess datasets using commonly and frequently used steps. Please see the help call of `main()` for more information on how to use this script (see the example below).

```bash
python NORTEMtools/scripts/preprocess.py --help
```
"""

import argparse
from zarr import ZipStore
from NORTEMtools import logger
from NORTEMtools.Emil.utils import (
    MyPath,
    args2string,
    set_log_level,
    load,
    load_metadata_from_json,
    set_calibrations,
    set_metadata,
    center_direct_beam,
    make_navigation_mask,
    compute,
    set_experimental_parameters,
)
from typing import Union, Dict
import matplotlib.pyplot as plt
import pyxem as pxm
import hyperspy.api as hs


def preprocess(
    filename: Union[str, MyPath],
    lazy: bool = True,
    com_mask: tuple = (128, 128, 12),
    estimate_linear_shift: bool = False,
    width: int = 20,
    calibrations: Dict = {},
    nav_chunks: int = 32,
    sig_chunks: int = 32,
    metadata_file: Union[None, MyPath] = None,
):
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
    :param metadata_file: The name or path to a json file containing metadata for the signal

    :type filename: Union[str, Path]
    :type lazy: bool
    :type com_mask: Union[None, tuple]
    :type estimate_linear_shift: bool
    :type calibrations: Dict
    :type width: int
    :type nav_chunks: int
    :type sig_chunks: int
    :type metadata_file: Union[None, MyPath]

    :return: preprocessed_signal
    :rtype: Signal
    """

    filename = MyPath(filename)
    # Load data
    signal = load(filename, lazy=lazy)

    # Check that data is ElectronDiffraction2D, if not, raise a warning and try to change the signal type
    if not isinstance(signal, pxm.signals.ElectronDiffraction2D):
        logger.warning(
            f"Only ElectronDiffraction2D signals can be preprocessed. I got {signal!r} of type {type(signal)}"
        )
        signal.set_signal_type("electron_diffraction")

    if isinstance(signal, pxm.signals.LazyElectronDiffraction2D):
        logger.debug(
            f"Rechunking data with `nav_chunks={nav_chunks}`, `sig_chunks={sig_chunks}`"
        )
        signal.rechunk(nav_chunks=nav_chunks, sig_chunks=sig_chunks)

    # Center data
    logger.info("Centering dataset")
    estimate_linear_shift_kwargs = {}
    if estimate_linear_shift:
        estimate_linear_shift_kwargs["mask"] = make_navigation_mask(signal, width)

    signal = center_direct_beam(
        signal,
        com_mask=com_mask,
        estimate_linear_shift=estimate_linear_shift,
        plot_results=False,
        estimate_linear_shifts_kwargs=estimate_linear_shift_kwargs,
    )

    # Add metadata
    try:
        metadata = load_metadata_from_json(metadata_file)
    except Exception as e:
        logger.warning(f"Could not load metadata from json file due to error: {e}")
        metadata = {}
    else:
        set_metadata(signal, metadata)
        set_experimental_parameters(signal, metadata.get("experimental_parameters"))

    # Set calibration
    x, y, kx, ky = [
        calibrations.get(
            f"{ax}", metadata.get("axes", {}).get(f"{ax}", {}).get("scale", None)
        )
        for ax in ["x", "y", "kx", "ky"]
    ]
    set_calibrations(signal, x, y, kx, ky)

    # Make VBF and maximum through-stack
    logger.info(f"Preparing VBF")
    vbf = signal.get_integrated_intensity(
        hs.roi.CircleROI(cx=0.0, cy=0.0, r_inner=0.0, r=0.07)
    )
    compute(vbf)
    signal.metadata.add_dictionary({"Preprocessing": {"VBF": vbf}})

    logger.info("Preparing maximum through-stack")
    maximums = signal.max(axis=[0, 1])
    compute(maximums)
    signal.metadata.add_dictionary({"Preprocessing": {"Maximums": maximums}})

    # Save the VBF and maximums
    logger.info(f"Saving VBF and maximums as images")
    plt.imsave(filename.with_name(f"{filename.stem}_preprocessed_vbf.png"), vbf.data)
    plt.imsave(
        filename.with_name(f"{filename.stem}_preprocessed_maximums.png"), maximums.data
    )

    return signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", type=MyPath, help="Path to a 4D-STEM dataset to preprocess"
    )
    parser.add_argument(
        "-x",
        "--cx",
        dest="cx",
        default=128,
        type=int,
        help="Center X position of COM mask",
    )
    parser.add_argument(
        "-y",
        "--cy",
        dest="cy",
        default=128,
        type=int,
        help="Center Y position of COM mask",
    )
    parser.add_argument(
        "-r",
        dest="r",
        default=12,
        type=float,
        help="COM mask radius for shift correction",
    )
    parser.add_argument(
        "-w",
        "--width",
        dest="width",
        default=20,
        type=int,
        help="Width of navigation frame to use when estimating linear shift",
    )
    parser.add_argument(
        "--estimate_linear_shift",
        dest="estimate_linear_shift",
        action="store_true",
        help="Whether to estimate linear shifts based on the COM results when centering the direct beam",
    )
    parser.add_argument(
        "--calibrations",
        dest="calibrations",
        default=None,
        type=float,
        nargs=4,
        metavar=("x", "y", "kx", "ky"),
        help="Calibrations for the x, y, kx, and ky axes in nm/px and Å/px. If not given, calibrations from the `metadata.json` file in the same location as the raw data will be used",
    )
    parser.add_argument(
        "-l", "--lazy", dest="lazy", action="store_true", help="Work on the data lazily"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosity",
        default=0,
        action="count",
        help="Set verbose level",
    )
    parser.add_argument(
        "-f",
        "--formats",
        dest="formats",
        type=str,
        default=".hspy",
        nargs="+",
        help="The dataformats to save the preprocessed data to",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Whether to overwrite data or not.",
    )
    parser.add_argument(
        "--chunks",
        dest="chunks",
        type=int,
        nargs=2,
        default=[32, 32],
        metavar=("nav_chunks", "sig_chunks"),
        help="Chunksize for navigation and signal axes",
    )
    arguments = parser.parse_args()

    set_log_level(logger, arguments.verbosity)  # Set log level

    logger.debug(args2string(arguments))  # Log the input values

    calibrations = arguments.calibrations
    if calibrations is None:
        calibrations = {}
    else:
        calibrations = {
            "x": calibrations[0],
            "y": calibrations[1],
            "kx": calibrations[2],
            "ky": calibrations[3],
        }

    logger.debug(
        f"I will save preprocessed signals in these formats: {arguments.formats}"
    )

    signal = preprocess(
        arguments.filename,
        lazy=arguments.lazy,
        com_mask=(arguments.cx, arguments.cy, arguments.r),
        estimate_linear_shift=arguments.estimate_linear_shift,
        width=arguments.width,
        calibrations=calibrations,
        nav_chunks=arguments.chunks[0],
        sig_chunks=arguments.chunks[1],
    )

    # Save the signal
    for f in arguments.formats:
        preprocessed_filename = arguments.filename.with_name(
            f"{arguments.filename.stem}_preprocessed{f}"
        )
        logger.info(f'Saving preprocessed data to "{preprocessed_filename.absolute()}"')
        chunks = (arguments.chunks[0],) * signal.axes_manager.navigation_dimension + (
            arguments.chunks[1],
        ) * signal.axes_manager.signal_dimension
        try:
            if f == ".zspy":
                logger.info(f"Saving data as a ZipStore with chunks: {chunks}")
                store = ZipStore(preprocessed_filename)
                signal.save(store, overwrite=arguments.overwrite, chunks=chunks)
            elif f in [".hspy", ".hdf5"]:
                logger.info(f"Saving data with chunks: {chunks}")
                signal.save(
                    preprocessed_filename, overwrite=arguments.overwrite, chunks=chunks
                )
            else:
                signal.save(preprocessed_filename, overwrite=arguments.overwrite)
        except Exception as e:
            logger.error(
                f"Exception when saving preprocessed signal with format {f}: \n{e}. \nSkipping format and continuing."
            )


if __name__ == "__main__":
    main()
