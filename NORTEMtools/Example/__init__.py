"""
Example submodule.

You can create submodules by creating a new folder with the name you want for your submodule. You will then be able to access it through calls such as `from NORTEMtools.<name> import <...>`
In this new folder, you must put a file called `__init__.py` (this file) that imports the functionality you add. For example, if you create a file `utils.py` that contain the `foo` function, you will need to put `from .utils import foo` in your `__init__.py` file.

The following convention may be used as a template:

NORTEMtools/
|---<YourName>/ <-- Add a folder with your name to contain your code
    |---`__init__.py` <-- Your import details goes here
    |---`utils.py` <-- Your code goes here
|---`__init__.py` <-- Add `from . import <YourName>` to the NORTEMtools `__init__.py` file
"""

from .utils import foo
