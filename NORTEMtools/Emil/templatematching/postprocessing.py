"""
This module contains tools to postprocess templatematching results.
"""

from NORTEMtools.Emil.utils import MyPath, frame_string
from NORTEMtools import logger
from typing import Union, Dict
import pandas as pd
import pyxem as pxm
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs


_default_plot_kwargs = {
    "cmap": "magma_r",
    "norm": "symlog",
    "colorbar": None,
    "axes_off": True,
}

_default_marker_kwargs = {
    "include_intensity": False,
    "annotate": True,
    "permanent": True,
}

_default_vector_kwargs = {"annotate": True, "fast": False}


def show_results(
    result: pxm.signals.indexation_results.OrientationMap,
    signal: pxm.signals.ElectronDiffraction2D,
    n_best: int = 1,
    plot_kwargs: Union[None, Dict] = None,
    **kwargs,
):
    f"""
    Plot the signal and add markers for the template matching results.
    
    :param result: The template matching results
    :type result: pxm.signals.indexation_results.OrientationMap
    :param signal: The signal to show the results for
    :type signal: pxm.signals.ElectronDiffraction2D
    :param n_best: Which `n_best` to show
    :type n_best: int
    :param plot_kwargs: Optional keyword arguments passed to `signal.plot()`. Default values are {_default_plot_kwargs}
    :type plot_kwargs: Union[None, Dict]
    :param kwargs: Optional keyword arguments passed to `result.to_markers()`
    """

    if plot_kwargs is None:
        plot_kwargs = {}

    _ = [
        plot_kwargs.update(key, kwargs.get(key, _default_plot_kwargs[key]))
        for key in _default_plot_kwargs
    ]
    _ = [
        kwargs.update(key, _default_marker_kwargs.get(key, _default_marker_kwargs[key]))
        for key in _default_marker_kwargs
    ]

    signal = signal.deepcopy()
    signal.plot(**plot_kwargs)
    signal.add_marker(result.to_markers(n_best=n_best, **kwargs))


def save_pixel_results(
    result: pxm.signals.indexation_results.OrientationMap,
    signal: pxm.signals.ElectronDiffraction2D,
    plot_kwargs: Union[None, Dict] = None,
    vector_kwargs: Union[None, Dict] = None,
    output_dir: Union[None, MyPath] = None,
    dpi: int = 300,
    max_pixels: int = 50,
):
    f"""
    Show template matching results for individual pixels.

    This will generate two plots for each pixel: a navigator plot and a signal plot. This should be avoided for large datasets.

    :param result: The orientation mapping results.
    :type result: pxm.signals.indexation_results.OrientationMap
    :param signal: The signal to plot the results for.
    :type signal: pxm.signals.ElectronDiffraction2D
    :param plot_kwargs: Optional keyword arguments passed to signal.plot(), Default values are {_default_plot_kwargs!r}.
    :type plot_kwargs: Union[None, Dict]
    :param vector_kwargs: The keyword arguments passed to `result.to_single_phase_markers()`. Default values are {_default_vector_kwargs!r}.
    :type vector_kwargs: Union[None, Dict]
    :param output_dir: The directory to save the pixel results.
    :type output_dir: Union[None, MyPath]
    :param dpi: The DPI of the plots to be saved.
    :type dpi: int
    :param max_pixels: The maximum number of pixels to generate plots for.
    :type max_pixels: int

    :returns: 2-tuple with the signal figure and the navigation figure handles.
    :rtype: tuple
    """

    if output_dir is None:
        output_dir = MyPath(".")
    else:
        output_dir = MyPath(output_dir)
    if output_dir.is_dir():
        pass
    else:
        logger.warning(
            f'Output directory "{output_dir}" is not a directory. Using the parent directory instead: {output_dir.parent}'
        )
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(
        f"Plotting results over signal with following parameters:\n\tvector_kwargs: {vector_kwargs!r}\n\tplot_kwargs: {plot_kwargs!r}"
    )
    result.plot_over_signal(
        signal,
        add_ipf_correlation_heatmap=True,
        vector_kwargs=vector_kwargs,
        **plot_kwargs,
    )

    logger.debug("Iterating through plots to save individual pixel results")
    signal_fig = plt.gcf()
    navigator_fig = plt.figure(signal_fig.number - 1)

    counter = 0
    for i in range(signal.axes_manager.navigation_shape[0]):
        for j in range(signal.axes_manager.navigation_shape[1]):
            if counter >= max_pixels:
                logger.info(
                    f"Maximum number of images ({max_pixels}) generated. To generate more images, please increase the `max_pixels` parameter"
                )
                return signal_fig, navigator_fig

            output_path = output_dir / f"{i}_{j}.png"
            logger.debug(
                f'Changing indices to ({i}, {j}) and saving plots to "{output_path}"'
            )
            signal.axes_manager.indices = (i, j)
            signal_fig.savefig(output_path.append("signal"), dpi=dpi)
            navigator_fig.savefig(output_path.append("navigator"), dpi=dpi)
            counter += 1

    return signal_fig, navigator_fig


def result2DataFrame(
    result: pxm.signals.indexation_results.OrientationMap,
    signal: pxm.signals.ElectronDiffraction2D,
) -> pd.DataFrame:
    """
    Convert a template matching result into a pandas DataFrame.

    If the signal metadata contains coordinates under 'Processing' -> 'Coordinates', those will be included in the DataFrame as both 'Position' (index in the 4DSTEM data) and 'Pixels' (actual pixel coordinates).

    :param result: The result dictionary from template matching.
    :type result: pxm.signals.indexation_results.OrientationMap
    :param signal: The original 4DSTEM signal.
    :type signal: pxm.signals.ElectronDiffraction2D
    :return: A DataFrame containing the template matching results for every pixel in the 4DSTEM data.
    :rtype: pd.DataFrame
    """
    logger.debug("Converting result dictionary to DataFrame...")

    coords = (
        signal.metadata.as_dictionary().get("Processing", {}).get("Coordinates", None)
    )
    if coords is None:
        logger.warning(
            "No coordinates found in signal metadata; proceeding without position information."
        )
    else:
        logger.debug(f"Using coordinates from metadata: {coords!s}")
        order = "F"
        navigation_shape = signal.axes_manager.navigation_shape
        coords_array = np.array(coords)
        logger.debug(
            f'Reshaping coordinate array {coords_array}:\nnavigation shape: {navigation_shape}\norder: "{order}"'
        )
        xs = coords_array[:, 0].reshape(navigation_shape, order=order)
        ys = coords_array[:, 1].reshape(navigation_shape, order=order)
        logger.debug(f"xs={xs}\nys={ys}")

    results = pd.DataFrame()
    for i in range(result.axes_manager[0].size):
        for j in range(result.axes_manager[1].size):
            df = pd.DataFrame(
                data={
                    "Template index": result.inav[i, j].isig[0, 0].data[0],
                    "Correlation": result.inav[i, j].isig[1, 0].data[0],
                    "Rotation": result.inav[i, j].isig[2, 0].data[0],
                    "Factor": result.inav[i, j].isig[3, 0].data[0],
                    "Calibration": signal.axes_manager[-1].scale,
                },
                index=[0],
            )
            location = i + j * result.axes_manager[0].size
            position = f"({i}, {j})"
            pixels = (
                (f"({int(xs[i, j])}, {int(ys[i, j])})") if coords is not None else None
            )
            df["Location"] = location
            df["Position"] = position
            df["Pixels"] = pixels

            results = pd.concat([results, df], ignore_index=True)
            logger.debug(
                f"Dataframe for location {location} = {position} (pixels {pixels}):\n{str(df)}"
            )

    logger.debug(f"Result DataFrame:\n{results.to_string()}")
    return results


def summarize_results(
    results: pd.DataFrame,
    output_path: MyPath,
    groupby_column: str = "Pixels",
    title: str = "",
) -> pd.DataFrame:
    """
    Summarize calibration test results by identifying the best calibration for each position and computing summary statistics.

    :param results: The DataFrame containing calibration test results.
    :type results: pd.DataFrame
    :param output_path: Path to save the summary plot.
    :type output_path: MyPath
    :param title: Title for the summary plot.
    :type title: str
    :return: A DataFrame containing summary statistics of the best calibrations.
    :rtype: DataFrame
    """
    try:
        logger.debug(f'Grouping dataframe by column "{groupby_column}"...')
        group = results.groupby(groupby_column)
    except Exception as e:
        logger.error(
            f'Cannot summarize results due to error when grouping dataframe by column "{groupby_column}": {e}.\nDataframe is:\n{results.head()}\nSkipping summary and returning an empty dataframe with matching columns'
        )
        return pd.DataFrame(columns=results.columns)

    best_rows = group["Correlation"].idxmax()
    best_calibrations = results.loc[best_rows]

    # Save best calibrations
    csv_path = output_path.with_name("best_calibrations.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        best_calibrations.to_csv(f, index=False)

    best_calibrations.sort_values("Correlation", inplace=True)
    # Save best calibrations (sorted)
    csv_path = output_path.with_name("best_calibrations_sorted.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        best_calibrations.to_csv(f, index=False)

    # Print best calibrations
    logger.info(
        f"{frame_string('Best Calibrations for Each Position')}\n"
        f"{best_calibrations.to_string()}"
    )

    # Compute summary statistics
    best_calibrations_array = np.array(best_calibrations["Calibration"])
    summary = pd.DataFrame(
        {
            "Mean": [best_calibrations_array.mean()],
            "Std": [best_calibrations_array.std()],
            "Median": [np.median(best_calibrations_array)],
            "Max:": [best_calibrations_array.max()],
            "Min": [best_calibrations_array.min()],
        }
    )

    # Print summary
    logger.info(
        f"{frame_string('Calibration Summary Statistics')}\n"
        "Statistics computed over best calibrations for each position:\n"
        f"{summary.to_string()}"
    )

    # Generate summary plot
    logger.debug("Generating summary plot...")
    plt.figure()
    x = np.arange(len(best_calibrations_array))
    sb.set_style("ticks")
    mean = np.full_like(best_calibrations_array, best_calibrations_array.mean())
    median = np.full_like(best_calibrations_array, np.median(best_calibrations_array))
    max_val = np.full_like(best_calibrations_array, best_calibrations_array.max())
    min_val = np.full_like(best_calibrations_array, best_calibrations_array.min())

    ax = sb.scatterplot(
        data=best_calibrations, x=groupby_column, y="Calibration", marker="x"
    )
    sb.lineplot(data=best_calibrations, x=groupby_column, y="Calibration")
    line = ax.lines[0]
    line_color = line.get_color()
    ax.yaxis.label.set_color(line_color)
    sb.lineplot(
        data=best_calibrations, x=groupby_column, y=mean, label="Mean", linestyle="--"
    )
    sb.lineplot(
        data=best_calibrations,
        x=groupby_column,
        y=median,
        label="Median",
        linestyle="--",
    )
    sb.lineplot(
        data=best_calibrations, x=groupby_column, y=max_val, label="Max", linestyle="--"
    )
    sb.lineplot(
        data=best_calibrations, x=groupby_column, y=min_val, label="Min", linestyle="--"
    )

    twinax = ax.twinx()
    twin_color = "r"
    sb.scatterplot(
        data=best_calibrations,
        x=groupby_column,
        y="Correlation",
        marker="x",
        color=twin_color,
        ax=twinax,
    )
    sb.lineplot(
        data=best_calibrations,
        x=groupby_column,
        y="Correlation",
        color=twin_color,
        ax=twinax,
    )
    twinax.yaxis.label.set_color(twin_color)

    ax.set_xlabel(f"{groupby_column}")
    ax.set_ylabel("Best Calibration")
    _ = ax.set_xticks(
        x, labels=best_calibrations[groupby_column], rotation=45, ha="right"
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    plt.close()
    return summary
