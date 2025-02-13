"""
Functions to collect inputs needed for hail retrieval from DWD Doppler spectra.

Copyright (c) 2025 Mathias Gergely, DWD

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
import numpy as np
import datetime as dt

from pybathhail import optional


def load_txt_data(filename, dirname):
    """
    Load data stored as txt file(name) in the 'dirname' directory.
    """
    
    pathname = os.path.join(dirname, filename)
    data = np.loadtxt(pathname)
    
    return data


def collect_input(
        hail_timestamp, input_directory,
        postprocess_directory, hail_settings, model_settings):
    """
    Collect input data needed for retrieval of hail size distributions.
    
    Args:
        hail_timestamp (str):
            Time of birdbath scan during hail (yyyy-mm-dd HH:MM:SS).
            
        input_directory (str):
            Absolute path to directory where non-postprocessing inputs are
            stored.
        
        postprocess_directory (str):
            Absolute path of directory with postprocessing results for birdbath
            scan Doppler spectra.
        
        hail_settings (dict):
            Settings for (later) hail retrieval.
            
        model_settings (dict):
            Settings for getting ICON Meteogramm data if spectra are shifted
            based on corrected max rain fall velocity (in hail_settings). 
            
    Returns:
        retrieval_inputs (dict):
            Inputs needed for retrieving hail size distributions from birdbath
            scan reflectivity spectra that are the output of the PyBathSpectra
            postprocessing routine.
    """
    
    print('collecting/creating inputs needed for hail retrieval...')
    
    # Timestamp of birdbath scan for hail retrieval
    analysis_time = dt.datetime.strptime(hail_timestamp, '%Y-%m-%d %H:%M:%S')
    
    # All range bins [m above radar] of birdbath scan Doppler spectra
    heights_file = './heights.txt'
    heights_vector = load_txt_data(heights_file, postprocess_directory)
    # Full birdbath-scan reflectivity spectra after postprocessing
    spectra_pattern = './reflectivity_spectra_%Y%m%d_%H%M%S.txt'
    spectra_file = analysis_time.strftime(spectra_pattern)
    all_spectra = load_txt_data(spectra_file, postprocess_directory)
    # Velocity vectors [m/s] for all reflectivity spectra
    velocities_pattern = './velocity_vectors_%Y%m%d_%H%M%S.txt'
    velocities_file = analysis_time.strftime(velocities_pattern)
    velocity_vectors = load_txt_data(velocities_file, postprocess_directory)
    # Mode indices of individual precipitation modes from postprocessing
    modes_pattern = './mode_indices_%Y%m%d_%H%M%S.txt'
    modes_file = analysis_time.strftime(modes_pattern)
    modes = load_txt_data(modes_file, postprocess_directory)
    
    # Results from single-scattering calculations for modeled hailstones
    calc_dir = os.path.join(input_directory, hail_settings['radx_dir'])
    # Hail diameters [mm] used for scattering calculations
    hail_d_file = './hail_diameters_mm.txt'
    hail_d = load_txt_data(hail_d_file, calc_dir)
    # Corresponding radar backscatter cross sections [mm2]
    hail_x_file = './radx_hail_mm2.txt'
    hail_x = load_txt_data(hail_x_file, calc_dir)
    
    # Process model data, if needed
    if hail_settings['shift_type'] == 'max_rain':
        min_height = hail_settings['hail_minheight']
        max_height = hail_settings['hail_maxheight']
        hail_heights = (min_height, max_height)
        icon_dir = os.path.join(input_directory,
                                model_settings['icon_data_dir'])
        icon_data = optional.icon_model_data(
                model_settings['location'],
                analysis_time,
                hail_heights,
                model_settings['icon_unpack'],
                icon_dir)
    else:
        icon_data = None
    
    # Collect all relevant data for retrieving hail size distributions
    retrieval_inputs = dict(
        spectra_reflectivity=all_spectra,
        spectra_heights=heights_vector,
        spectra_velocities=velocity_vectors,
        mode_indices=modes,
        radx=hail_x,
        radx_diameters=hail_d,
        model_data=icon_data)
        
    return retrieval_inputs
