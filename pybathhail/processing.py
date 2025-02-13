"""
Find/run spectral postprocessing of DWD birdbath-scan Doppler spectra with
all settings specified in the hail_ and postprocessing_ yaml config files.

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
from pybathspectra import birdbathscan, reader


def postprocess(hail_timestamp, detailed_settings):
    """
    Run postprocessing routine for DWD birdbath scans (before hail retrieval).
    
    All functionality is contained in PyBathSpectra package. This is simply an
    interface to the postprocessing routine from within the PyBathHail package,
    using detailed_settings specified in the 'postprocessing_config.yaml' file.
    The hail_timestamp (from hail_config) has to match the birdbath_time (from
    the postprocessing_config in detailed_settings).
    """
    
    print('...running postprocessing routine for selected birdbath scan...')
    
    # Split settings into groups for different postprocessing steps
    settings_groups = reader.split_settings(detailed_settings)
    reader_settings, filter_settings, modes_settings = settings_groups
    
    # Birdbath scan timestamp
    # could be used to loop over multiple birdbath scans
    birdbath_timestamp = detailed_settings['birdbath_time'][0]
    # Check if birdbath-postprocessing and hail times in config yamls match
    if birdbath_timestamp != hail_timestamp:
        raise ValueError(
            'Birdbath time of postprocessing does not match hail retrieval.' \
            'Check postprocessing and hail config .yaml files.')
    
    # Load birdbath scan data (= radar output or DWD database files)
    birdbath_scan = birdbathscan.BirdBathScan()
    birdbath_scan.load(birdbath_timestamp, reader_settings)
    
    # Isolate weather signal in Doppler spectra
    # i.e. filter out clutter and background, if specified in settings
    birdbath_scan.isolate(filter_settings)
    
    # Multimodal analysis of isolated weather Doppler spectra
    birdbath_scan.analyze(modes_settings)
    
    # Plot radar outputs and postprocessing results, if selected
    if detailed_settings['plot_all']:
        birdbath_scan.plot(
            birdbath_timestamp, detailed_settings['rgb_scaling'],
            detailed_settings['plot_dir'])
    
    # Save postprocessing results, if selected
    if detailed_settings['save_results']:
        birdbath_scan.save(
            birdbath_timestamp, detailed_settings['results_dir'])


def results_dir(hail_timestamp, signal_settings, hail_retrieval_directory):
    """
    Return directory with postprocessing results.
    
    Depending on selected signal_settings, results are created
    (i.e. postprocessing routine is executed) and collected in specified
    directory, or the already existing results directory is returned.
    """
    
    print('finding/creating postprocessing results of birdbath scan...')
    
    # Check current working directory for resolving correct relative paths
    if os.getcwd() != hail_retrieval_directory:
        os.chdir(hail_retrieval_directory)
    
    # Path to postprocessing config file
    postprocess_file = signal_settings['signalprocessing_settings']
    postprocess_path = os.path.abspath(postprocess_file)
    # Detailed postprocessing settings from postprocessing config file
    with open(postprocess_path, 'r') as postprocess_config_file:
        postprocess_settings = yaml.safe_load(postprocess_config_file)
        
    # Run postprocessing routine, if desired/needed
    if signal_settings['signalprocessing']:
        postprocess(hail_timestamp, postprocess_settings)   
        
    # Directory of postprocessing results according to config file
    results_relative = postprocess_settings['results_dir']
    results_directory = os.path.abspath(results_relative)
    # Inputs directory
    inputs_relative = postprocess_settings['data_dir']
    inputs_dir = os.path.abspath(inputs_relative)
    
    return inputs_dir, results_directory
