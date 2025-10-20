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

To use this package, simply import it in your scripts with `import NORTEMtools as nt`. See below for examples

### Calibration checks
Sometimes it can be useful to see how changes in the calibration affects template matching results. While it is ALWAYS best to acquire calibration data during your session, or to calibrate using the data itself, it is sometimes necessary to see how small changes affects the results. The following example loads a dataset and a template bank and runs template matching with different calibration values.

```python
import hyperspy.api as hs
import pyxem as pxm
import pandas as pd
import NORTEMtools as nt
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