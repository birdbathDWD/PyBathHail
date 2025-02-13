"""
Retrieving hail size distributions from DWD birdbath-scan Doppler spectra
with the settings specified in the hail_settings_file.

Copyright (c) 2024 Mathias Gergely, DWD

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import yaml

from pybathhail import helper
from pybathhail.haildata import HailData


# Find directory of this script
this_directory = os.path.dirname(os.path.abspath(__file__))
# Path to config file containing the settings for hail retrieval
hail_settings_file = os.path.join(this_directory, 'hail_config.yaml')


###############################################################################
## Pre-settings, if desired: Python environment and working directory         #
###############################################################################
## Use (uninstalled) pybathhail modules from subfolder of this directory
#import sys
#if sys.path[0] == this_directory:
#    pass
#else:
#    sys.path.insert(0, this_directory)
## Explicitly set current working directory to resolve relative paths
## for plotting and saving retrieval results
#os.chdir(this_directory)


###############################################################################
# From here on, no interaction with code; set all options in config file(s)   #
###############################################################################

# Load current hail-retrieval settings from .yaml config file
with open(hail_settings_file, 'r') as config_file:
    hail_settings = yaml.safe_load(config_file)

# Split hail-retrieval settings into groups for different analysis steps
settings_groups = helper.split_hail_settings(hail_settings)
signal_set, hail_set, model_set, analysis_set = settings_groups

# Birdbath scan timestamp for hail retrieval
hail_time = hail_settings['birdbath_time'][0]

# Create hail data object where all results will be collected
hail_data = HailData()

# Produce/find postprocessing results and input directory for hail retrieval
hail_data.postprocessing(
    hail_time,
    signal_set,
    this_directory)

# Retrieve (binned) hail size distributions [mm-1 m3] and vertical wind [m s-1]
hail_data.retrieving(
    hail_time,
    hail_set,
    model_set)


###### Further analysis of retrieval results (for subset of hail heights)

# Fits to hail size distributions and hail characteristics, if selected
hail_data.analyzing(
    hail_time,
    analysis_set)


##### Plot and save results (after further analysis: subset of hail heights)

# Plot hail size distributions, fits, and hail characteristics
if hail_settings['plot_hail']:
    hail_data.plotting(
        hail_time,
        this_directory,
        hail_settings['plot_dir'])    

# Save retrieval results, if selected
if hail_settings['save_hail']:
    hail_data.saving(
        hail_time,
        this_directory,
        hail_settings['save_dir'])
