# NORTEMtools

Python tools to help and streamline analysis of TEM results at NORTEM

## Requirements

This package was made for pyxem 0.21.0. It might work with earlier or later versions, but it might also fail.

## Install

1. Download this package (if downloaded manually; move it somewhere logical. If downloaded with Git; continue).
2. Open a terminal (use a conda-terminal with the environments you want to use with this package) and navigate to the directory containing this package
3. In the terminal, activate the environment you want to use and install pip, then install this package in editable mode
```bash
conda activate <myenv> 
conda install pip
pip install --editable .
```

## Use

To use this package, simply import it in your scripts with `import NORTEMtools as nt` and access the submodules. See below for examples.

The code is structured into submodules created for each contributor. This is not the best way to do this in general, but it serves the purpose of this module best. The goal is to make it easier to reuse frequently used functions and utilities, and not to create new/added functionality. Significant code contributions should instead be made directly to e.g. `hyperspy` or `pyxem`.

### Calibration checks (Emil)

Sometimes it can be useful to see how changes in the calibration affects template matching results. While it is ALWAYS best to acquire calibration data during your session, or to calibrate using the data itself, it is sometimes necessary to see how small changes affects the results. The following example loads a dataset and a template bank and runs template matching with different calibration values.

```python
import hyperspy.api as hs
import pyxem as pxm
import pandas as pd
import NORTEMtools.Emil as nt #Import code from Emil
import pickle

data_path = nt.MyPath(r'<...>') #Path to data path
template_path = data_path.with_name(r'templates.pk') #Path to template bank saved as a pickle.in the same location as input data.

signal = hs.load('<...>')
with open('<templates.pkl>', 'rb') as f:
    simulations = pickle.load(f)

#To run calibration checks:
results = nt.calibration_check(signal, simulations, n=10, n_random=3, start=0.9, end=1.1, guess=0.01, npt=181)
df = nt.results2DataFrame(results) #Converts the template matching results into a dataframe
summary = nt.summarize_results(df) #Summarizes the data frame
```

## Contribute

To add your own code and contribute, you can create your own submodule. This way, your code does not interfere with the other code in the module, but you can still use the other submodules in your own code if you wish (note that this may make your code unstable). See the `/NORTEMtools/Example/` submodule for an example

### Create a branch

Before creating your submodule, please consider creating your own Branch of this repository first. This will make it much easier to maintain your code and also to make sure that your own contributions are stable.

### Create your submodule

To create your own submodule, create a new folder with the name you want for your submodule (preferably your own first name). You will then be able to access it through calls such as `from NORTEMtools.<name> import <...>`. In this new folder, you must put a file called `__init__.py` that imports the functionality you add. For example, if you create a file `utils.py` that contain the `foo` function, you will need to put `from .utils import foo` in your `__init__.py` file.

The following convention may be used as a template:

```
NORTEMtools/
|---`<YourName>/` <-- Add a folder with your name to contain your code
    |---`__init__.py` <-- Your import details goes here
    |---`utils.py` <-- Your code goes here
|---`__init__.py` <-- Add `from . import <YourName>` to the NORTEMtools `__init__.py` file
```
