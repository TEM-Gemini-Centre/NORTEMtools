from setuptools import setup, find_packages


setup(
    name='NORTEMtools',
    version='0.0.0',
    description="streamlined python analysis at NORTEM",
    author='Emil Frang Christiansen',
    author_email='emil.christiansen@ntnu.no',
    license='MIT',
    url="",
    long_description=open("README.md").read(),
    keywords=[
        "data analysis",
        "diffraction",
        "microscopy",
        "electron diffraction",
        "electron microscopy",
    ],
    
    packages=find_packages(),
    install_requires=[
        "pyxem       >= 0.21.0",
        "seaborn",
        "pandas",
    ],
    python_requires=">=3.7",
)