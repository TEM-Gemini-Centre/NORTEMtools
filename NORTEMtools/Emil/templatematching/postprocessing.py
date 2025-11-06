"""
This module contains tools to postprocess templatematching results.
"""

from NORTEMtools.Emil.utils import MyPath, frame_string
from NORTEMtools import logger
from typing import Union
import pandas as pd
import pyxem as pxm
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs


def show_results(
    result: pxm.signals.indexation_results.OrientationMap,
    signal: pxm.signals.ElectronDiffraction2D,
    n_best: int = 1,
    x: int = 0,
    y: int = 0,
    **kwargs,
):

    signal = signal.deepcopy()
    signal.plot(**kwargs)
    signal.add_marker(
        result.to_markers(n_best=n_best, include_intensity=True, annotate=True),
        permanent=True,
    )


def result2DataFrame(
    result: pxm.signals.indexation_results.OrientationMap,
    signal: pxm.signals.ElectronDiffraction2D,
    save_frames: bool = False,
    out_image_dir: Union[None, str, MyPath] = None,
    title=Union[None, str],
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
        order = "A"
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

            if save_frames:
                frame = signal.inav[i, j]
                if out_image_dir is None:
                    out_image_dir = MyPath(".")
                else:
                    out_image_dir = MyPath(out_image_dir)

                if out_image_dir.is_dir:
                    pass
                else:
                    logger.warning(
                        f"Expected a path to a directory to put frames, not a complete path {out_image_dir}. Continuing using the parent of the provided path: {out_image_dir.parent}"
                    )
                    out_image_dir = out_image_dir.parent

                if title is None:
                    title = signal.metadata.General.title

                out_image_path = (
                    out_image_dir
                    / f"{location}_{i}_{j}_{xs[i,j]}_{ys[i,j]}_{title}.hspy"
                )

                out_image_path.append(f"{signal.axes_manager[-1].scale}")
                out_image_path.parent.mkdir(exist_ok=True)
                logger.info(f'Saving image data to "{out_image_path}"')
                frame.save(out_image_path, overwrite=True)

                hs.plot.plot_images(frame)
                fig = plt.gcf()
                logger.info(f'Saving image plot to "{out_image_path}"')
                fig.savefig(out_image_path.with_suffix(".png"), dpi=300)
                plt.close(fig)

    logger.debug(f"Result DataFrame:\n{results.to_string()}")
    return results


def summarize_results(
    results: pd.DataFrame,
    output_path: MyPath,
    groupby_column: str = "Location",
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
