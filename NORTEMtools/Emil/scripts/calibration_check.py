from NORTEMtools import logger, add_log_handler, remove_log_handler

from NORTEMtools.Emil.utils import (
    MyPath,
    compute,
    frame_string,
    log_shift,
    pick_random,
    Signal,
    load,
)
from NORTEMtools.Emil.templatematching.io import load_template
from NORTEMtools.Emil.templatematching.postprocessing import (
    result2DataFrame,
    summarize_results,
)

import numpy as np
from typing import Union, Callable
import time
import hyperspy.api as hs
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import argparse
import gc
import datetime as dt
import json


def test_calibration(
    signal: Signal,
    simulations: object,
    start: float,
    end: float,
    n: int,
    intensity_transform_function: Union[None, Callable] = None,
    npt: Union[None, int] = None,
) -> pd.DataFrame:
    """
    Test calibration of a 4DSTEM signal using template matching over a range of calibration factors.

    :param signal: The input 4DSTEM signal to test calibration on.
    :type signal: Signal
    :param simulations: The simulated templates to use for matching.
    :type simulations: object
    :param start: The start of the calibration range (as a fraction of current calibration).
    :type start: float
    :param end: The end of the calibration range (as a fraction of current calibration).
    :type end: float
    :param n: The number of calibration points to evaluate between start and end.
    :type n: int
    :param intensity_transform_function: An optional function to transform intensities of the radial integration before template matching.
    :type intensity_transform_function: Union[None, function]
    :param npt: Number of points for radial integration. If None, defaults to the half-diagonal of the pattern.
    :type npt: Union[None, int]
    :return: A DataFrame containing the results of the calibration test.
    :rtype: pd.DataFrame
    """

    logger.debug("Starting calibration test...")
    results = pd.DataFrame()  # Placeholder for actual results

    hs.set_log_level("ERROR")  # Suppress hyperspy logs during calibration test

    cal_min, cal_max = (
        signal.axes_manager[-1].scale * start,
        signal.axes_manager[-1].scale * end,
    )
    logger.info(
        f"Testing calibrations from {cal_min:.3e} to {cal_max:.3e} with {n} points."
    )

    cal = np.linspace(cal_min, cal_max, num=n)
    logger.debug(f"Calibration values to test: {cal!s}")

    if npt is None:
        nx, ny = signal.axes_manager.signal_shape
        npt = int(np.sqrt((nx / 2) ** 2 + (ny / 2) ** 2))
        logger.debug(
            f"Number of points for radial integration not provided; using half-diagonal of pattern: npt={npt}"
        )
    else:
        logger.debug(
            f"Using provided number of points for radial integration: npt={npt}"
        )

    results = pd.DataFrame()
    tic = time.time()
    for i, c in enumerate(cal):
        # Log calibration being tested
        logger.info(frame_string(f"Calibration test {i+1}/{n}: calibration = {c:.4e}"))

        logger.debug(f"Setting diffraction calibration to {c:.4e}...")
        signal.set_diffraction_calibration(c)

        logger.debug("Performing radial integration of the signal...")
        radial_integration = signal.get_azimuthal_integral2d(
            npt=npt, show_progressbar=False
        )  # , radial_range=(0.066, reciprocal_radius))
        if intensity_transform_function is not None:
            logger.debug(
                "Applying intensity transformation function to radial integration..."
            )
            radial_integration.map(
                function=intensity_transform_function,
                inplace=True,
                show_progressbar=False,
            )
        compute(radial_integration)

        logger.debug("Getting orientation using template matching...")
        res = radial_integration.get_orientation(
            simulations,
            n_best=simulations.rotations.size,
            frac_keep=1.0,
            show_progressbar=False,
        )

        logger.debug("Creating DataFrame from template matching result...")
        df = result2DataFrame(res, signal)

        logger.debug("Appending results to overall DataFrame...")
        results = pd.concat([results, df], ignore_index=True)
    toc = time.time()
    elapsed = toc - tic
    logger.info(
        f"Calibration test completed in {elapsed:.2f} seconds.\nTime per calibration: {elapsed/n:.2f} seconds."
    )

    hs.set_log_level(logger.level)  # Restore previous hyperspy log level

    logger.debug("Calibration test completed.")

    logger.debug(f"Results DataFrame:\n{results.to_string()}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calibration check for 4DSTEM data using template matching."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input 4DSTEM data file"
    )
    parser.add_argument(
        "--template_file",
        type=str,
        default=None,
        help='Path to the template file for matching. Pickle format. If not provided (default), I will look for a a file with "_template.pkl" appended to the name in same directory as 4DSTEM data file',
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        dest="verbosity",
        default=0,
        action="count",
        help="Set verbose level: -v for INFO, -vv for DEBUG",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help='Directory to save output results. Default is None, which will put outputs in a subfolder "calibration_check" at the same location as input data',
    )
    parser.add_argument(
        "--n_random",
        type=int,
        default=5,
        help="Number of random coordinates to pick for analysis",
    )
    parser.add_argument(
        "--seed", type=int, default=197405, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--lazy", action="store_true", help="Load data lazily using dask"
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.9,
        help="Start of the calibration range. Given as a fraction of the current calibration.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=1.1,
        help="End of the calibration range. Given as a fraction of the current calibration.",
    )
    parser.add_argument(
        "--guess",
        type=float,
        default=None,
        help="Initial guess for the diffraction calibration in Å^-1 (overrides current calibration in data)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of calibration points to evaluate between start and end",
    )
    parser.add_argument(
        "--npt",
        type=int,
        default=None,
        help="Number of points for radial integration. If None, defaults to half-diagonal of the pattern.",
    )
    parser.add_argument(
        "--log_shift",
        action="store_true",
        help="Wether to use a log shift (with shift=1) as an intensity transformation to apply before template matching. Helps to enhance weak reflections. Default is False.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name for the output files. Output files are timestamped. If not provided, the name of the input file is used.",
    )
    args = parser.parse_args()

    # Get the input path of the data file. Log files will be saved in the same directory.
    input_path = MyPath(args.input_file)
    if len(args.name) == 0:
        output_dir_name = f"{input_path.stem}"
    else:
        output_dir_name = f"{args.name}"
    output_dir_name = f'{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}_{output_dir_name}'
    if args.output_dir is None:
        output_dir = (
            input_path.absolute().parent / "calibration_check" / output_dir_name
        )
    else:
        output_dir = MyPath(args.output_dir) / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    add_log_handler(output_dir / "log.txt")

    # Print parser description and arguments to log
    parser_description = "Calibration check script arguments:"
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
        [f"{arg}: {getattr(args, arg)}" for arg in vars(args)]
    )
    logger.info(f"Parser description:\n{parser_description}")

    # Write parser arguments to json file in output directory for record keeping
    args_path = output_dir / "arguments.json"
    with open(args_path, "w") as f:
        json.dump({arg: getattr(args, arg) for arg in vars(args)}, f, indent=4)

    # Set up paths
    if args.template_file is None:
        template_path = input_path.append("template", "pkl")
        logger.debug(f"No template file provided; using default: {template_path}")
    else:
        template_path = MyPath(args.template_file)

    # Load data
    signal = load(input_path)

    # Set initial guess for calibration if provided
    if args.guess is not None:
        signal.set_diffraction_calibration(args.guess)
        logger.debug(f"Setting initial guess for calibration to {args.guess:.4e}")

    # Load template
    logger.info(f"Loading template from {template_path}")
    template = load_template(template_path)

    # Pick random coordinates
    logger.info(f"Picking {args.n_random} random coordinates from data for analysis")
    random_signal = pick_random(
        signal,
        n=args.n_random,
        seed=args.seed,
        show=True,
        output_path=output_dir / "vbf_coordinates.png",
    )

    # Perform calibration check
    logger.info("Performing calibration check...")
    results = test_calibration(
        random_signal,
        template,
        start=args.start,
        end=args.end,
        n=args.n,
        intensity_transform_function=log_shift if args.log_shift else None,
        npt=args.npt,
    )
    logger.info("Calibration check completed.")

    # Save results
    logger.info(f"Saving results to {output_dir}")
    results_path = output_dir / f"calibration_results.csv"
    with open(results_path, "w", encoding="utf-8") as f:
        results.to_csv(f, index=False)
    logger.info(f"Results saved to {results_path}")

    # Generate and save plot
    logger.info("Generating calibration plot...")
    plt.figure()
    try:
        sb.lineplot(data=results, x="Calibration", y="Correlation", hue="Location")
    except Exception as e:
        logger.error(
            f"Failed to generate calibration plot: {e}\nGenerating plot without hue."
        )
        sb.lineplot(data=results, x="Calibration", y="Correlation")
    plt.title(
        f"{args.name} Calibration Check" if len(args.name) > 0 else "Calibration Check"
    )
    plt.tight_layout()
    plt.savefig(results_path.with_suffix(".png"), dpi=300)
    logger.info(f'Calibration plot saved to {results_path.with_suffix(".png")}')
    plt.close()

    # Summarize results
    summary_path = output_dir / "calibration_summary.png"
    summary = summarize_results(
        results,
        summary_path,
        title=(
            f"{args.name} Calibration Summary"
            if len(args.name) > 0
            else "Calibration Summary"
        ),
    )
    logger.info(f"Summary plot saved to {summary_path}")
    summary_csv_path = summary_path.with_suffix(".csv")
    with open(summary_csv_path, "w", encoding="utf-8") as f:
        summary.to_csv(f, index=False)
    logger.info(f"Summary statistics saved to {summary_csv_path}")

    logger.info("Calibration check process completed successfully.")

    remove_log_handler(output_dir / "log.txt")
    gc.collect()


if __name__ == "__main__":
    main()
