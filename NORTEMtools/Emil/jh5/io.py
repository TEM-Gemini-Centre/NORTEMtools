import hyperspy.api as hs
import pyxem as pxm
from NORTEMtools.Emil import MyPath
import numpy as np
import json
from pathlib import Path
from typing import Union, Dict, Any, List
import h5py
import numpy as np
import tabulate
from natsort import natsorted

from NORTEMtools.Emil.utils import str2dict, unravel_dictionary
from NORTEMtools import logger


def _convert_bytes_to_str(value):
    """
    Recursively convert bytes objects to strings, handling nested data structures.

    This function processes various data types and converts any bytes or numpy.bytes_
    objects to UTF-8 decoded strings. It handles nested structures like lists,
    tuples, NumPy arrays, and dictionaries by recursively applying the conversion
    to their elements or values.

    Parameters
    ----------
    value : Any
        The value to process, which may be bytes, numpy.bytes_, a list/tuple/ndarray
        containing bytes, a dictionary with bytes values, or other types.

    Returns
    -------
    Any
        The processed value with all bytes objects converted to strings where applicable.
        For complex structures, returns a new structure with converted elements.

    Examples
    --------
    >>> _convert_bytes_to_str(b'hello')
    'hello'

    >>> _convert_bytes_to_str([b'item1', b'item2'])
    ['item1', 'item2']

    >>> _convert_bytes_to_str({'key': b'value'})
    {'key': 'value'}
    """
    # Check if the current value is a bytes object
    if isinstance(value, bytes):
        # Decode bytes to a UTF-8 string
        return value.decode("utf-8")

    # Check if the current value is a numpy.bytes_ object (common in HDF5 files)
    elif isinstance(value, np.bytes_):
        return value.decode("utf-8")

    # Check if the value is a list, tuple, or NumPy array
    elif isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 1:
            return _convert_bytes_to_str(value[0])
        else:
            # Otherwise, recursively process each element in the container
            return [_convert_bytes_to_str(item) for item in value]
    elif isinstance(value, dict):
        return {k: _convert_bytes_to_str(v) for k, v in value.items()}

    # If none of the above conditions match, return the value unchanged
    else:
        return value


class JH5Reader:
    """
    Reader for .jh5 files (HDF5-based format).

    This class provides methods to read and explore .jh5 files,
    which are typically used in scientific data applications.
    """

    _signal_type_mapping = {"EdsSpectrum": "EDS_TEM", "EelsCube": "EELS"}
    _units_mapping = {"Nanometer": "nm", "KeV": "keV"}

    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize the JH5Reader with a file path.

        This constructor checks if the provided file exists and has the correct extension (.jh5).

        Parameters:
            filepath (str or Path): Path to the .jh5 file
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file extension is not .jh5.
        """
        # Store the file path as a Path object for easier manipulation
        self.filepath = Path(filepath)

        # Check if the file exists at the given path
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        # Verify that the file has the correct extension
        if self.filepath.suffix != ".jh5":
            raise ValueError(
                f"File should be a '.jh5' file, not '{self.filepath.suffix}'"
            )

    def read_file(self) -> Dict[str, Any]:
        """
        Read the entire .jh5 file and return its contents as a dictionary.

        Opens the HDF5 file in read-only mode and uses the internal method `_read_group`
        to recursively extract all data and metadata into a nested dictionary structure.
        Returns:
            dict: Dictionary containing all data and metadata from the file
        """
        try:
            # Open the file using h5py in read mode
            with h5py.File(self.filepath, "r") as f:
                # Call the recursive reader on the root group
                return self._read_group(f)
        except Exception as e:
            # Raise an IOError with a descriptive message if something goes wrong
            raise IOError(f"Error reading file {self.filepath}: {str(e)}")

    def _read_group(self, group: h5py.Group) -> Dict[str, Any]:
        """
        Recursively read an HDF5 group and its contents.

        This helper method traverses the HDF5 hierarchy and extracts data and attributes
        from datasets and groups. It handles special cases such as string data and
        nested structures, ensuring that all data is properly formatted for Python usage.
        Parameters:
            group (h5py.Group): The HDF5 group to read

        Returns:
            dict: Dictionary representation of the group, including all subgroups and datasets
        """
        # Initialize an empty dictionary to store the contents of this group
        data = {}

        # Iterate over all items (datasets and subgroups) in the current group
        for key, item in group.items():
            # If the item is a dataset (i.e., actual data)
            if isinstance(item, h5py.Dataset):
                # Handle string data (fixed-length or variable-length)
                if item.dtype.kind == "S":  # Fixed-length byte strings
                    # Decode each string and store as a list
                    data[key] = [
                        s.decode("utf-8") if isinstance(s, bytes) else s
                        for s in item[...]
                    ]
                elif item.dtype.kind == "U":  # Unicode strings
                    # Directly use the string data
                    data[key] = item[...]
                else:
                    # For numerical or other types, simply load the data
                    data[key] = item[...]

                # Process dataset attributes if they exist
                if len(item.attrs) > 0:
                    # Get the attributes dictionary
                    attrs = dict(item.attrs) if len(item.attrs) > 0 else {}

                    # Convert any byte strings or nested dictionaries in attributes
                    attrs = {
                        key: str2dict(_convert_bytes_to_str(value))
                        for key, value in attrs.items()
                    }
                    # Store attributes under a special key name (e.g., "dataset_name_attrs")
                    data[f"{key}_attrs"] = attrs

            # If the item is another group, recursively read it
            elif isinstance(item, h5py.Group):
                data[key] = self._read_group(item)
            else:
                # For other types (shouldn't happen in typical HDF5), just store the item
                data[key] = item

        # Process group-level attributes (not associated with any specific dataset)
        for key, value in group.attrs.items():
            # Store group attributes with a prefix "@" to distinguish them
            data[f"@{key}"] = _convert_bytes_to_str(value)

        # Return the constructed dictionary representing this group and its contents
        return data

    def get_structure(self) -> Dict[str, Any]:
        """
        Get the hierarchical structure of the .jh5 file.

        This method explores the entire HDF5 file structure, identifying datasets and groups,
        along with their properties like shape, dtype, and size. This is useful for understanding
        how the data is organized within the file.

        Returns:
            dict: Dictionary representing the file structure with keys as paths and values as
                  information about each node (dataset or group).
        """
        try:
            with h5py.File(self.filepath, "r") as f:
                return self._get_structure_recursive(f)
        except Exception as e:
            raise IOError(
                f"Error getting structure from file {self.filepath}: {str(e)}"
            )

    def _get_structure_recursive(self, obj, path="") -> Dict[str, Any]:
        """
        Recursively traverse an HDF5 object (either a group or dataset) to build a structure map.

        This helper method builds a dictionary that describes the hierarchical structure of
        the HDF5 file. It identifies each node (dataset or group) and collects relevant
        metadata such as shape, dtype, and size for datasets.
        Parameters:
            obj: HDF5 object (either h5py.Group or h5py.Dataset)
            path (str): Current path in the HDF5 hierarchy being explored

        Returns:
            dict: Structure map of the HDF5 object and its children
        """
        structure = {}

        # If the current object is an HDF5 group, explore its contents
        if isinstance(obj, h5py.Group):
            for key in obj.keys():
                # Construct the full path for the child item
                new_path = f"{path}/{key}" if path else key
                item = obj[key]
                # If the item is a dataset, record its properties
                if isinstance(item, h5py.Dataset):
                    structure[new_path] = {
                        "type": "dataset",
                        "shape": item.shape,
                        "dtype": str(item.dtype),
                        "size": item.size,
                    }
                    # Include attribute info if present
                    if len(item.attrs) > 0:
                        structure[new_path]["attributes"] = list(item.attrs.keys())
                else:
                    # If the item is a group, mark it and recursively explore it
                    structure[new_path] = {"type": "group"}
                    structure.update(self._get_structure_recursive(item, new_path))
        else:
            # If the object is a dataset, record its properties directly
            structure[path] = {
                "type": "dataset",
                "shape": obj.shape,
                "dtype": str(obj.dtype),
                "size": obj.size,
            }

        # Return the structure dictionary built for this part of the hierarchy
        return structure

    def list_datasets(self) -> List[str]:
        """
        List all datasets in the .jh5 file.

        This method iterates through the entire HDF5 file structure and collects the
        names (paths) of all datasets found. These paths can be used to access specific
        data arrays later.

        Returns:
            list: List of dataset names (full paths) in the file
        """
        datasets = []
        try:
            with h5py.File(self.filepath, "r") as f:
                self._list_datasets_recursive(f, datasets)
        except Exception as e:
            raise IOError(f"Error listing datasets from file {self.filepath}: {str(e)}")
        return datasets

    def _list_datasets_recursive(self, obj, datasets: List[str], path=""):
        """
        Helper method to recursively collect all dataset names from an HDF5 file.

        This function traverses the HDF5 hierarchy, appending the full path of every
        dataset encountered to the `datasets` list.

        Parameters:
            obj: HDF5 object (either h5py.Group or h5py.Dataset)
            datasets (list): List to append dataset paths to
            path (str): Current path in the hierarchy being explored
        """
        # If the current object is a group, iterate over its contents
        if isinstance(obj, h5py.Group):
            for key in obj.keys():
                # Construct the full path for the child item
                new_path = f"{path}/{key}" if path else key
                item = obj[key]
                # If the item is a dataset, add its path to the list
                if isinstance(item, h5py.Dataset):
                    datasets.append(new_path)
                else:
                    # If it's a group, continue exploring recursively
                    self._list_datasets_recursive(item, datasets, new_path)

    def get_dataset_info(self, dataset_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific dataset.

        This method retrieves comprehensive details about a given dataset within the .jh5 file,
        including its shape, data type, size, attributes, and a sample of its data for preview.

        Parameters:
        -----------
        dataset_path : str
            The full path to the dataset within the HDF5 file (e.g., '/group/dataset_name').
        Returns:
        --------
        dict
            A dictionary containing the following keys:
            - 'path': The path to the dataset.
            - 'shape': The shape of the dataset.
            - 'dtype': The data type of the dataset.
            - 'size': The total number of elements in the dataset.
            - 'attributes': A dictionary of the dataset's attributes, with special handling for bytes.
            - 'max_shape': Maximum shape if resizable, otherwise None.
            - 'sample': A sample of the dataset's data (up to 100 elements), for preview purposes.
        Raises:
        -------
        KeyError
            If the specified dataset path does not exist in the file.
        IOError
            If there is an error accessing the file or retrieving the dataset information.
        """
        try:
            with h5py.File(self.filepath, "r") as f:
                # Check if the requested dataset exists in the file
                if dataset_path in f:
                    dataset = f[dataset_path]
                    # Retrieve attributes of the dataset
                    attrs = dict(dataset.attrs) if len(dataset.attrs) > 0 else {}
                    # Convert attributes, handling potential bytes or nested strings
                    attrs = {
                        key: str2dict(_convert_bytes_to_str(value))
                        for key, value in attrs.items()
                    }

                    # Prepare the information dictionary with basic properties
                    info = {
                        "path": dataset_path,
                        "shape": dataset.shape,
                        "dtype": str(dataset.dtype),
                        "size": dataset.size,
                        "attributes": attrs,
                        "max_shape": (
                            dataset.maxshape if hasattr(dataset, "maxshape") else None
                        ),
                    }

                    # Attempt to extract a sample of the dataset for preview
                    try:
                        # Ensure the dataset has data
                        if dataset.size > 0:
                            # Handle 1D datasets
                            if len(dataset.shape) == 1:
                                info["sample"] = dataset[: min(10, len(dataset))]
                            # Handle 2D datasets
                            elif len(dataset.shape) == 2:
                                # Take up to 10x10 slice
                                info["sample"] = dataset[
                                    : min(10, dataset.shape[0]),
                                    : min(10, dataset.shape[1]),
                                ]
                            # Handle higher dimensional datasets
                            else:
                                # Flatten and take a sample of up to 100 elements
                                info["sample"] = dataset.flat[
                                    : min(100, dataset.size)
                                ].reshape(-1)
                    except Exception:
                        # If sampling fails, continue without a sample
                        pass

                    # Return the complete information dictionary
                    return info
                else:
                    # Raise an error if the dataset path is not found
                    raise KeyError(f"Dataset '{dataset_path}' not found in file")
        except Exception as e:
            # Re-raise any I/O errors encountered during file access
            raise IOError(f"Error getting dataset info for {dataset_path}: {str(e)}")

    def _get_signal_type(self, dataset_path: str) -> Union[str, None]:
        """
        Determine the Hyperspy signal type for a given dataset path in the .jh5 file.

        This method attempts to infer the appropriate Hyperspy signal type based on
        metadata stored within the dataset. It uses a predefined mapping (`_signal_type_mapping`)
        to translate raw data types (like 'EdsSpectrum') into Hyperspy-compatible signal types
        (like 'EDS_TEM'). If the metadata is missing or cannot be accessed, it returns None.

        Parameters:
        -----------
        dataset_path : str
            The full path to the dataset within the HDF5 file (e.g., '/group/dataset_name').

        Returns:
        --------
        str or None
            The corresponding Hyperspy signal type if available; otherwise, None.

        Example:
        --------
        >>> reader = JH5Reader("example.jh5")
        >>> signal_type = reader.get_signal_type("/EDS/Spectrum")
        >>> print(signal_type)
        'EDS_TEM'
        """
        try:
            # Check if the dataset exists and has the required metadata structure

            signal_type = unravel_dictionary(
                self.get_dataset_info(dataset_path),
                "attributes",
                "Metadata",
                "DataType",
                rtype=str,
            )
            hyperspy_signal_type = self._signal_type_mapping.get(signal_type, None)

        except Exception as e:
            # Log error if anything fails during the lookup process
            logger.error(
                f'Could not get hyperspy signal type from data object "{dataset_path}" due to error {e}'
            )
            hyperspy_signal_type = None

        return hyperspy_signal_type

    def _get_calibration(self, dataset_path: str) -> List:

        try:
            return unravel_dictionary(
                self.get_dataset_info(dataset_path),
                "attributes",
                "OptionalData",
                "Information",
                "MeasurementInformation",
                "CalibrationCoefficients",
                rtype=list,
            )
        except Exception as e:
            logger.error(
                f'Could not get calibration for data object "{dataset_path}" due to error {e}.'
            )
            return {}

    def _get_dimension_length(self, dataset_path: str) -> int:
        try:
            return unravel_dictionary(
                self.get_dataset_info(dataset_path),
                "attributes",
                "OptionalData",
                "Information",
                "DataInformation",
                "DimensionLength",
                rtype=int,
            )
        except Exception as e:
            logger.error(
                f'Could not get dimensions for data object "{dataset_path}" due to error {e}'
            )
            return None

    def _get_dimensions(self, dataset_path: str) -> list:
        try:
            return unravel_dictionary(
                self.get_dataset_info(dataset_path),
                "attributes",
                "OptionalData",
                "Information",
                "DataInformation",
                "Dimensions",
                rtype=list,
            )[::-1]
        except Exception as e:
            logger.error(
                f'Could not get dimensions for data object "{dataset_path}" due to error {e}'
            )
            return None

    def get_axes(self, dataset_path: str) -> list:
        calibration_coefficients = self._get_calibration(dataset_path)
        dimensions = self._get_dimensions(dataset_path)
        axes = []
        if dimensions is not None and calibration_coefficients is not None:
            for i, (dim, cal) in enumerate(
                zip(dimensions[::-1], calibration_coefficients)
            ):
                unit = cal.get("Unit", "")  # Get the unit as written by FEMTUS
                unit = self._units_mapping.get(
                    unit, unit
                )  # Convert the units to more sensible formatting. Needed to convert from e.g. "Nanometer" to "nm" and "KeV" to "keV"

                dict = {
                    "scale": cal.get("Scale", 1),
                    "units": unit,
                    "offset": cal.get("Offset", 0),
                    "name": f"Axis{i}",
                    "size": dim,
                }
                axes.append(dict)

                if dict["scale"] > 1.0 and dict["units"] == "keV":
                    logger.warning(
                        f'The scale of {dict["scale"]} {dict["units"]} seems to be too large for the given unit. Maybe this is a FEMTUS bug where the scale is given in KeV should really be given in eV? Consider changing the units when doing analysis!'
                    )
        return axes


def jh5_to_hspy(
    reader: JH5Reader, dataset_path: str = "0"
) -> Union[hs.signals.BaseSignal, hs.signals.Signal1D, hs.signals.Signal2D]:
    data = reader.read_file()
    try:
        array = data[dataset_path]
    except KeyError:
        raise KeyError(f"Dataset '{dataset_path}' not in the .jh5 file")

    array = np.squeeze(data[dataset_path])

    if len(array.shape) == 1:
        flip_axes = None
        navigation_axes = None
        signal_axes = [0]
    elif len(array.shape) == 2:
        flip_axes = None
        navigation_axes = None
        signal_axes = [0, 1]
    elif len(array.shape) == 3:
        flip_axes = [0, 2]
        navigation_axes = [0, 1]
        signal_axes = [2]
    elif len(array.shape) == 4:
        flip_axes = [0, 2]
        navigation_axes = [0, 1]
        signal_axes = [2, 3]
    else:
        flip_axes = None
        navigation_axes = None
        signal_axes = None

    if flip_axes is None:
        signal = hs.signals.BaseSignal(
            array, axes=reader.get_axes("0")[::-1], metadata={"JH5": data["0_attrs"]}
        )
    else:
        signal = hs.signals.BaseSignal(
            np.flip(array, axis=flip_axes),
            axes=reader.get_axes("0")[::-1],
            metadata={"JH5": data["0_attrs"]},
        )

    # Store metadata
    signal.original_metadata.set_item(
        "JH5", data.get(f"{dataset_path}_attrs", reader.get_dataset_info(dataset_path))
    )

    # Set title
    signal.metadata.set_item("General.title", f"{reader.filepath.stem}")

    try:
        signal = signal.transpose(
            navigation_axes=navigation_axes, signal_axes=signal_axes
        )
    except Exception as e:
        logger.error(
            f"Could not transpose signal {signal} of shape {signal.data.shape} using navigation_axes = {navigation_axes} and signal_axes={signal_axes} due to error {e}"
        )

    signal_type = reader._get_signal_type(dataset_path)
    if signal_type is not None:
        signal.set_signal_type(signal_type)

    return signal


def load_jh5(
    file_path: MyPath, dataset_path="0"
) -> Union[hs.signals.BaseSignal, hs.signals.Signal1D, hs.signals.Signal2D]:
    """Load a jh5 file into a hyperspy signal"""
    reader = JH5Reader(file_path)
    return jh5_to_hspy(reader, dataset_path=dataset_path)


def jh5_4DSTEM(file_path: MyPath) -> pxm.signals.ElectronDiffraction2D:

    file_path = MyPath(
        file_path
    )  # Path to 4DSTEM junk data file (but with useful metadata)

    logger.info(f"Loading {file_path} to extract metadata for 4DSTEM stack")
    if file_path.suffix == ".jh5":
        base_signal = load_jh5(
            file_path
        )  # Load the .jh5 file containing rubbish data but useful metadata
    else:

        base_signal = hs.load(
            file_path
        )  # Load a converted .jh5 file containing rubbish data but useful metadata.

    # Get the paths to the tif file
    try:
        tif_directory = (
            file_path.parent / file_path.stem
        )  # Get the path to the 4DSTEM data
        logger.debug(f'Attempting to load 4DSTEM tif stacks from "{tif_directory}"')
        tif_paths = tif_directory.glob("*.tif")  # Get the paths to all the tifs
        tif_paths = natsorted(tif_paths)  # Naturally sort the tifs
        logger.debug(f"Found {len(tif_paths)} tif files.")

        signal = hs.load(tif_paths, stack=True)  # Import the tifs as a stack
        logger.info(f"Loaded 4DSTEM signal with shape {signal.data.shape}.")
    except Exception as e:
        logger.error(
            f'Could not load .tif stacks (found ({len(tif_paths)}) from "{file_path}" due to error {e}.'
        )
        raise
    signal.metadata.add_dictionary(base_signal.metadata.as_dictionary())  # Add metadata
    signal.original_metadata.add_dictionary(
        base_signal.original_metadata.as_dictionary()
    )  # Add original metadata

    signal.set_signal_type("electron_diffraction")  # Set signal type

    calibrations = base_signal.metadata.OptionalData.Information.MeasurementInformation.CalibrationCoefficients[
        ::-1
    ]  # Remember to reverse the list!
    logger.warning(
        f"Using reversed order for the calibrations found in junk .jh5 file. It is probably not correct to reverse it, as it is probably just a matter of rolling the axes, but the result should be ok for equal step sizes and pixel sizes for the navigation and signal spaces."
    )
    for i, cal in enumerate(calibrations):
        signal.axes_manager[i].scale = cal["Scale"]
        signal.axes_manager[i].offset = cal["Offset"]
        signal.axes_manager[i].units = cal["Unit"]
        signal.axes_manager[i].name = f"Axes{i}"

    metadata_mapping = {
        "beam_energy": [
            "JH5",
            "OptionalData",
            "Information",
            "Tags",
            "HT",
            "AccelerationVoltage",
        ],
        "camera_length": [
            "JH5",
            "OptionalData",
            "Information",
            "Tags",
            "EOS",
            "CameraLength",
        ],
        "scan_rotation": [
            "JH5",
            "OptionalData",
            "Information",
            "Tags",
            "ScanGenerator",
            "ScanRotation",
        ],
        "exposure_time": [
            "JH5",
            "OptionalData",
            "Information",
            "Tags",
            "ScanGenerator",
            "ExposureTimeValue",
        ],
    }

    try:
        experimental_parameters = {
            parameter: round(
                unravel_dictionary(
                    base_signal.metadata.as_dictionary(), *metadata_mapping[parameter]
                ),
                2,
            )
            for parameter in metadata_mapping
        }
        signal.set_experimental_parameters(**experimental_parameters)
    except Exception as e:
        logger.error(
            f"Could not set experimental parameters due to error {e}. Ignoring this for now, but this should probably be checked manually"
        )

    signal.metadata.set_dictionary({"JH5_file": base_signal})
    logger.info(f'Added junk .jh5 file to metadata under key "JH5_file"')

    return signal


def load_JEOL_worksheet(file_path: MyPath) -> Dict:
    """Load JEOL worksheet file."""
    file_path = MyPath(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'File "{file_path.absolute()} was not found')
    if not file_path.suffix == ".jfw":
        raise ValueError(
            f'File "{file_path.absolute()} is not a JEOL worksheet file (.jfw)'
        )
    with file_path.open("r") as f:
        worksheet = json.load(f)

    title = worksheet.get("Title")
    elements = worksheet.get("ElementFileContents")
    files = [
        file_path.parent / title / element.get("RelativePath") for element in elements
    ]
    logger.debug(
        f'Found {len(files)} files for JEOL worksheet {file_path}:\n\t{"\n\t".join([str(file) for file in files])}'
    )

    signals = {}
    for file in files:
        if not file.exists():
            logger.warning(f'File "{file}" in JEOL worksheet {title} does not exist.')
        else:
            try:
                reader = JH5Reader(file)
                signals[file] = jh5_to_hspy(reader, "0")
            except Exception as e:
                logger.error(
                    f"Could not read file {file} with .jh5 filereader due to error {e}"
                )

    table = tabulate.tabulate(
        [[str(key), str(signals[key])] for key in signals], headers=["File", "Signal"]
    )
    logger.info(f"Loaded files\n{table}")
    return signals
