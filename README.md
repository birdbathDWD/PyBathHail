## About PyBathSpectra
### Contents
The *PyBathHail* repository contains Python code and example files for retrieving the full hail size distributions from Doppler spectra recorded with DWD's operational C-band radar (vertically pointing) birdbath scan, as described by ????
### License
Distributed under the MIT License. See `LICENSE` for more information.
## Getting started
Experimental code. .....

Exact version numbers of the Python packages used for developing `pybathhail` modules are listed in `setup.py`. Other versions, especially much newer versions, can (and will) cause compatibility issues, particularly due to the many changes in `NumPy` and `pandas` between different version numbers. Therefore, it may be best to create a virtual environment with the exact versions of Python packages listed in 'setup.py' and then run the *PyBathHail* postprocessing analysis inside this Python environment. 
### Installation
The `pybathhail` Python package can be installed by downloading the *PyBathHail* repository and running `pip install .`, for example. But to preserve flexibility and avoid (some) compatibility issues, it is best to run the retrieval inside the downloaded repository structure, i.e., without prior installation of the Python package.
## Usage
All retrieval methods are collected in the `pybathhail` folder. The retrieval can be executed by running `hail_retrieval.py` with the desired retrieval settings selected in `hail_config.yaml`. The `input` folder is a collection of birdbath-scan example data; the `output` folder collects all numerical results as well as plots for visualizing the postprocessing steps.

2 birdbath-scan examples are included as `input`.
## Citing the code

## Acknowledegments
The work is supported by the German Research Foundation (DFG) 'PROM' priority program SPP-2115 (https://www2.meteo.uni-bonn.de/spp2115) and the German Meteorological Service (Deutscher Wetterdienst, DWD, https://www.dwd.de/DE/Home/home_node.html).
<!-- ## References -->
