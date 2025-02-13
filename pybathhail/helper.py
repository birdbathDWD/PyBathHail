"""
Various utility functions needed for hail retrieval from DWD Doppler spectra.

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

import numpy as np
import datetime as dt
import pandas as pd


def split_hail_settings(all_settings):
    """
    Split hail-retrieval settings into groups for different use cases.
    
    Returns:
        Dictionaries of groups of settings to use as input for hail
        retrieval functions.
    """
    
    # For (post)processing module before hail retrieval
    signal_settings = dict(
        signalprocessing=all_settings['postprocessing'],
        signalprocessing_settings=all_settings['postprocessing_settings'])
    
    # Hail specifics and radar retrieval options
    hail_specifics = dict(
            radar_wavelength=all_settings['radar_wavelength'],
            hail_minheight=all_settings['hail_minheight'],
            hail_maxheight=all_settings['hail_maxheight'],
            vD_relation=all_settings['vD_relation'],
            hail_minsize=all_settings['hail_minsize'],
            shift_type=all_settings['shift_type'],
            constant_shift=all_settings['constant_shift'],
            noise_mode=all_settings['noise_mode'],
            elbow_range=all_settings['elbow_range'],
            radx_dir=all_settings['radx_dir'])
    
    # For ICON Meteogramm data, if needed for max rain velocity
    model_settings = dict(
        location=all_settings['location'],
        icon_data_dir=all_settings['icon_data_dir'],
        icon_unpack=all_settings['icon_unpack'])
    
    # Settings for further analysis of retrieved hail size distributions
    analysis_settings = dict(
        aspect_ratio=all_settings['aspect_ratio'],
        ice_fraction=all_settings['ice_fraction'],
        detailed_analysis=all_settings['detailed_analysis'],
        height_selection=all_settings['height_selection'],)
                 
    return signal_settings, hail_specifics, model_settings, analysis_settings


def hail_velocity_to_size(terminal_velocity, vD_relation='H20'):
    """
    Determine hail diameters [mm] from hail terminal fall velocities [m/s].
    
    Based on relationship by Heymsfield et al. (2020) 'H20', yielding slow fall
    speeds for given hail size, or Matson and Huggins (1980) 'MH80', yielding
    intermediate fall speeds, or Gokhale (1975) 'G75', giving the fastest
    fallspeeds for the same hail sizes.  
    """
    
    # Heymsfield et al. (2020)
    if vD_relation == 'H20':
        # Considers separate (small) graupel and (larger) hail
        # velocity = 7.6 * (diameters_mm / 10)**0.89  # diameters < 15 mm
        # velocity = 8.4 * (diameters_mm / 10)**0.67  # diameters > 15 mm
        # Here hail relationship extrapolated down to 5 mm
        prefactor = 8.39
        exponent = 0.67
    # Matson and Huggins (1980) uses equivalent volume diameter
    elif vD_relation == 'MH80':
        prefactor = 11.45
        exponent = 0.5
    # Gokhale (1975) book apparently uses spheroidal hailstones
    elif vD_relation == 'G75':
        prefactor = 15.0
        exponent = 0.5
    else:
        raise ValueError('Invalid vD_relation in hail_velocity_to_size().')
        
    # Calculate diameters in mm
    diameters_mm = 10 * (terminal_velocity / prefactor) ** (1 / exponent)
     
    return diameters_mm

 
def hail_size_to_velocity(diameters_in_mm, vD_relation='H20'):
    """
    Determine hail terminal fall velocities [m/s] from hail diameters [mm].
    
    Based on relationship by Heymsfield et al. (2020) 'H20', yielding slow fall
    speeds for given hail size, or Matson and Huggins (1980) 'MH80', yielding
    intermediate fall speeds, or Gokhale (1975) 'G75', giving the fastest
    fallspeeds for the same hail sizes.    
    """
    
    # Heymsfield et al. (2020)
    if vD_relation == 'H20':
        # Considers separate (small) graupel and (larger) hail
        # velocity = 7.6 * (diameters_mm / 10)**0.89  # diameters < 15 mm
        # velocity = 8.4 * (diameters_mm / 10)**0.67  # diameters > 15 mm
        # Here hail relationship extrapolated down to 5 mm
        prefactor = 8.39
        exponent = 0.67
    # Matson and Huggins (1980) uses equivalent volume diameter
    elif vD_relation == 'MH80':
        #velocity = 11.45 * (diameters_in_mm / 10)**0.5
        prefactor = 11.45
        exponent = 0.5
    # Gokhale (1975) book apparently uses spheroidal hailstones    
    elif vD_relation == 'G75':
        #velocity = 15.0 * (diameters_in_mm / 10)**0.5
        prefactor = 15.0
        exponent = 0.5
    else:
        raise ValueError('Invalid vD_relation in hail_velocity_to_size().')
        
    # Hail velocity-diameter relationship
    velocity = prefactor * (diameters_in_mm / 10)**exponent
        
    return velocity


def get_nearest_date(date_list, search_date, select=0):
    """
    Return the date_list date (1st, 2nd, ...) closest to search_date.
    
    Args:
        date_list (list):
            A list of datetime objects to search.
            
        search_date (datetime.datetime):
            The reference date.
            
        select (int):
            An integer specifying which date relative to the search date should
            be returned. "0": the date closest to the search date;
            "1": the second-closest date to the search date ...
    """
    
    dates = np.array(date_list)
    # Reformat numpy dates to datetime dates
    if type(dates[0]) == np.datetime64:
        dates = dates.astype(dt.datetime) * 1e-9
        dates_temp = list()
        for d in dates:
            dates_temp.append(dt.datetime.utcfromtimestamp(d))
        dates = np.array(dates_temp)
    if type(dates[0]) is pd.Timestamp:
        for num in range(len(dates)):
            #dates[num] = dates[num].to_pydatetime(tzinfo=None)
            dates[num] = dates[num].tz_localize(None)

    # Calculate absolute time deltas to the search date
    delta = dates - search_date
    delta = np.array([t.total_seconds() for t in delta])
    delta = np.abs(delta)
    # Create indexing to sort the time deltas
    idx = np.argsort(delta)
    # Select the k-th closest date
    out = np.array(date_list)[idx][select]
    # If the date type is numpy.datetime64 reformat it to a datetime.datetime
    if type(out) == np.datetime64:
        out = dt.datetime.utcfromtimestamp(np.int(out) * 1e-9)
        
    return out


def heights_for_analysis(height_interval, all_heights):
    """
    Only select previously determined hail heights for analysis/retrieval.
    
    Args:
        height_interval (tuple):
            Tuple of (minimum height above radar, maximum height above radar)
            in m.
            
        all_heights_dir (array):
            Array of all (previously processed) heights [m].
            
    Returns:
        selected_heights (arr):
            Array of valid heights for analysis out of all processed heights.
        
    """
    
    heights_vector = all_heights
    # Minimum and maximum hail heights [m above radar]
    min_height = height_interval[0]
    max_height = height_interval[-1]
    # Valid height range
    selected_heights = heights_vector[
        (heights_vector >= min_height) & (heights_vector <= max_height)]
        
    return selected_heights