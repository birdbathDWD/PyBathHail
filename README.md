## About PyBathHail
### Contents
The *PyBathHail* repository contains Python code and example files for retrieving the hail size distribution (HSD) and estimating the vertical wind speed from Doppler spectra recorded with DWD's operational C-band radar (vertically pointing) birdbath scan, as described by Gergely et al. (2025).
### License
Distributed under the MIT License. See `LICENSE` for more information.
## Getting started
Experimental code. To preserve flexibility, download the full *PyBathHail* repository (latest release), unzip, and run the hail retrieval from this directory. Also needs pre-installed `pybathspectra` package or downloaded `pybathspectra` directory and copied into working directory. 

Exact version numbers of the Python packages used for developing `pybathhail` modules are listed in `setup.py`. Other versions, especially much newer versions, can cause compatibility issues, particularly due to the many changes in `NumPy` and `pandas` between different version numbers. Therefore, it may be best to create a virtual environment with the exact versions of Python packages listed in 'setup.py' and then run the *PyBathHail* postprocessing analysis inside this Python environment. 
### Installation
The `pybathhail` Python package can be installed by downloading the *PyBathHail* repository and running `pip install .`, for example. But to preserve flexibility and avoid (some) compatibility issues, it is best to run the retrieval inside the downloaded repository structure, i.e., without prior installation.
## Usage
All retrieval and analysis methods are collected in the `pybathhail` directory. The retrieval can be executed by running `hail_retrieval.py` with the desired retrieval settings selected in `hail_config.yaml` (and radar signal postprocessing settings in `postprocessing_config.yaml`, if needed/desired). 

The `input` directory is a collection of birdbath-scan example data for 3 hailstorms, pre-calculated hail radar backscatter cross sections needed for the HSD retrieval, optional ICON atmospheric model outputs for 2 hailstorms at MHP radar (only needed if it is desired to alternatively anchor the slow hail edge to the maximum rain velocity instead of the default option, see `hail_config.yaml`), and the pre-determined hail analysis heights (with the `pybathspectra` toolkit); the `output` folder collects all numerical results as well as plots for visualizing the results (both for the hail retrievals of `pybathhail` and the radar signal postprocessing steps with `pybathspectra`).

3 birdbath-scan examples are included as `input`: 2 at MHP radar in southern Germany, 1 at FLD radar in central Germany (see cited article for details).
## Citing the code
Gergely, M., Ockenfuß, P., Seeger, F., Kneifel, S., Frech, M., 2025(?): Retrieval of the hail size distribution and vertical air motion from Doppler radar spectra, *J. Atmos. Oceanic Technol.*, submitted 2025, LINK_GOES_HERE
## Acknowledegments
The work is supported by the German Research Foundation (DFG) 'PROM' priority program SPP-2115 (https://www2.meteo.uni-bonn.de/spp2115) and the German Meteorological Service (Deutscher Wetterdienst, DWD, https://www.dwd.de/DE/Home/home_node.html).
<!-- ## References -->
