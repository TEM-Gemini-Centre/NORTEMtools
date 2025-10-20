from NORTEMtools.utils import MyPath, frame_string
from NORTEMtools import logger
import pickle
import pandas as pd
import pyxem as pxm
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt



def load_template(template_path: MyPath) -> object:
    """"Load a template from a given path using pickle.
    Args:
        template_path (MyPath): Path to the template file.
    
    Returns:
        template: Loaded template object.    
        """
    try:
        with template_path.open('rb') as f:
            logger.debug('Loading template using pickle')
            template = pickle.load(f)
            logger.debug(f'Template loaded: {template!s}')
    except Exception as e:
        logger.error(f'Failed to load template: {e}')
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
        with template_path.open('wb') as f:
            logger.debug('Saving template using pickle')
            pickle.dump(template, f)
            logger.debug(f'Template saved to: {template_path}')
    except Exception as e:
        logger.error(f'Failed to save template: {e}')
        raise

def result2DataFrame(result: pxm.signals.indexation_results.OrientationMap, signal: pxm.signals.ElectronDiffraction2D) -> pd.DataFrame:

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
    logger.debug('Converting result dictionary to DataFrame...')

    coords = signal.metadata.as_dictionary().get('Processing', {}).get('Coordinates', None)
    if coords is None:
        logger.warning('No coordinates found in signal metadata; proceeding without position information.')
    else:
        logger.debug(f'Using coordinates from metadata: {coords!s}')
        xs = np.array(coords)[:, 0].reshape(signal.axes_manager.navigation_shape)
        ys = np.array(coords)[:, 1].reshape(signal.axes_manager.navigation_shape)

    results = pd.DataFrame()
    for i in range(result.axes_manager[0].size):
        for j in range(result.axes_manager[1].size):
            df = pd.DataFrame(
                data={
                    'Template index': result.inav[i,j].isig[0,0].data[0], 
                    'Correlation': result.inav[i,j].isig[1, 0].data[0], 
                    'Rotation':result.inav[i,j].isig[2,0].data[0], 
                    'Factor':result.inav[i,j].isig[3,0].data[0], 
                    'Calibration': signal.axes_manager[-1].scale, 
                }, index=[0])
            
            if coords is not None:
                df['Position'] = f"({i}, {j})", 
                df['Pixels'] = f"({int(xs[i, j])}, {int(ys[i, j])})"

            results = pd.concat([results, df], ignore_index=True)

    logger.debug(f'Result DataFrame:\n{results.to_string()}')
    return results

def summarize_results(results: pd.DataFrame, output_path: MyPath, groupby_column: str = 'Position', title: str='') -> pd.DataFrame:
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
        logger.error(f'Cannot summarize results due to error when grouping dataframe by column "{groupby_column}": {e}.\nDataframe is:\n{results.head()}\nSkipping summary and returning an empty dataframe with matching columns')
        return pd.DataFrame(columns=results.columns)

    best_rows = group['Correlation'].idxmax()
    best_calibrations = results.loc[best_rows]

    #Print best calibrations
    logger.info(f'{frame_string('Best Calibrations for Each Position')}\n'\
                f'{best_calibrations.to_string()}')

    #Compute summary statistics
    best_calibrations_array = np.array(best_calibrations['Calibration'])
    summary = pd.DataFrame({
        'Mean': [best_calibrations_array.mean()],
        'Std': [best_calibrations_array.std()],
        'Median': [np.median(best_calibrations_array)],
        'Max:': [best_calibrations_array.max()],
        'Min': [best_calibrations_array.min()]
    })

    #Print summary
    logger.info(f'{frame_string('Calibration Summary Statistics')}\n'\
                'Statistics computed over best calibrations for each position:\n'\
                f'{summary.to_string()}')

    #Generate summary plot
    logger.debug('Generating summary plot...')
    plt.figure()
    x=np.arange(len(best_calibrations_array))
    sb.set_style('whitegrid')
    mean = np.full_like(best_calibrations_array, best_calibrations_array.mean())
    median = np.full_like(best_calibrations_array, np.median(best_calibrations_array))
    max = np.full_like(best_calibrations_array, best_calibrations_array.max())
    min = np.full_like(best_calibrations_array, best_calibrations_array.min())

    ax = sb.scatterplot(x=x, y=best_calibrations_array, marker='x')
    sb.lineplot(x=x, y=best_calibrations_array)
    sb.lineplot(x=x, y=mean, label='Mean', linestyle='--')
    sb.lineplot(x=x, y=median, label='Median', linestyle='--')
    sb.lineplot(x=x, y=max, label='Max', linestyle='--')
    sb.lineplot(x=x, y=min, label='Min', linestyle='--')

    ax.set_xlabel('Position number')
    ax.set_ylabel('Best Calibration')
    _ = ax.set_xticks(x, labels=best_calibrations.index, rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    plt.close()
    return summary