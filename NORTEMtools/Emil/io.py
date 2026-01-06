from NORTEMtools.Emil.jh5 import load_JEOL_worksheet, load_jh5
from NORTEMtools.Emil.utils import MyPath, get_files_in_directory, load, Signal
from NORTEMtools import _logger

from typing import Union, List, Tuple


def load_all(
    directory: Union[str, MyPath],
    file_types: List[str] = [".jh5", ".prz"],
    save: bool = False,
    overwrite: bool = False,
) -> Tuple[List[Signal], List[str], List[str]]:
    """
    Load all files in a directory into a dictionary.

    :param directory: Path to the directory containing data files. If None, raises a ValueError.
    :param file_types: The file types to load. Defaults to [".jh5", ".prz"].
    :param save: Whether to save the loaded data to disk.
    :param overwrite: Whether to overwrite existing files when saving.
    :returns: A dictionary with filenames as keys and loaded data as values

    :type string: Union[str, MyPath, None]
    :type file_types: List[str]
    :type save: bool
    :type overwrite: bool
    :raises ValueError: If directory is None or not a directory

    :returns: A tuple containing:
        - A list of loaded Signal objects.
        - A list of filenames corresponding to the loaded signals.
        - A list of files that failed to load.
    :rtype: Tuple[List[Signal], List[str], List[str]]
    """
    if directory is None:
        raise ValueError("Input path cannot be None")

    path = MyPath(directory)
    if not path.is_dir():
        raise ValueError(f"Input path {path} is not a directory")

    if ".jh5" in file_types and ".jfw" in file_types:
        _logger.warning(
            "Both .jh5 and .jfw file types are requested for loading. This may lead to duplicated data as .jfw files are treated especially and usually refer to .jh5 data."
        )

    files = []
    for suffix in file_types:
        _logger.debug(f"Getting all files with suffix {suffix} in directory {path}")
        files.extend(get_files_in_directory(path, suffix=suffix, recursive=True))
    _logger.info(
        f'Found {len(files)} files with specified file types in {directory}:\n\t{"\n\t".join([str(file) for file in files])}'
    )

    filenames = []
    signal_list = []
    failed_files = []

    for file in files:
        _logger.debug(f"Loading file {file}")
        try:
            if (
                file.suffix == ".jfw"
            ):  # Treat JEOL worksheets especially. This might lead to duplicating data!
                _logger.debug(f"Loading .jfw file {file} using `load_JEOL_worksheet()`")
                worksheet = load_JEOL_worksheet(file)
                _logger.debug(
                    f"Adding worksheet data with keys {list(worksheet.keys())} to loaded_data dictionary"
                )

                signal_list.append([value for value in worksheet.values()])
                filenames.append(
                    f"{str(file)}::{key}" for key in worksheet.keys()
                )  # Append key to filename to avoid duplicates
            elif file.suffix == ".jh5":
                _logger.debug(f"Loading .jh5 file {file} using `load_jh5()`")
                signal_list.append(load_jh5(file))
                filenames.append(str(file))
            else:
                signal_list.append(load(file))
                filenames.append(str(file))
        except Exception as e:
            _logger.error(f"Failed to load file {file}: {e}")
            failed_files.append(file)
        else:
            _logger.info(f"Successfully loaded file {file}: {signal_list[-1]}")

    _logger.info(
        f"Loaded {len(signal_list)} signals successfully ({len(failed_files)} files were unsuccessful)."
    )

    if save:
        for s, fname in zip(signal_list, filenames):
            fname = MyPath(fname)
            _logger.debug(
                f"Saving signal {s} from file {fname} to {fname.with_suffix('.hspy')}"
            )
            s.save(fname.with_suffix(".hspy"), overwrite=overwrite)
            _logger.debug(f'Saved signal {s} to {fname.with_suffix(".hspy")}')
    return signal_list, filenames, failed_files
