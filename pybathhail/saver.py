"""
Function for saving (some) hail retrieval results from DWD birdbath scans.

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


def save_retrievals(detailed_results, time, save_path='./results/'):
    """
    Save retrieved hail size distributions and vertical wind speeds.
    
    Based on manually preselected hail heights for detailed analysis, i.e.,
    the parameter 'height_selection' in 'hail_config.yaml' settings file.
    """
    
    print('saving (some) retrieval results.')
    
    
    # Extract data from detailed results (for selected subset of hail heights)
    heights = detailed_results['detailed_heights']
    hsd_range = detailed_results['statistics']
    hail_diameters = hsd_range['hsd_diameter_selection']
    hail_hsds = hsd_range['hsd_selection']
    event_diameters = hsd_range['hsd_range_stats'][:,0]
    event_hsd = hsd_range['hsd_range_stats'][:,-1]
    vertical_wind = detailed_results['vertical_wind']
    # Datetime object as reasonable string for file names
    dateandtime = time.strftime('%Y%m%d_%H%M%S')
    
    # All hail heights [m] above radar (of selected subset) = hail event
    height_name = save_path + 'hail_heights_' + dateandtime + '.txt'
    np.savetxt(height_name, heights)
    # 2D arrays of hail diameters [mm] and hail HSDs [mm-1 m3] for hail heights
    diameter_name = save_path + 'hail_diameters_' + dateandtime + '.txt'
    hsd_name = save_path + 'hail_size_distributions_' + dateandtime + '.txt'
    np.savetxt(diameter_name, hail_diameters)
    np.savetxt(hsd_name, hail_hsds)
    # Vertical windspeeds [m s-1] for hail height levels
    wind_name = save_path + 'vertical_wind_' + dateandtime + '.txt'
    np.savetxt(wind_name, vertical_wind)
    d_name = save_path + 'event_hail_diameters_' + dateandtime + '.txt'
    e_name = save_path + 'event_hail_size_distribution_' + dateandtime + '.txt'
    # Sum(mary) of all hail heights, i.e. for hail event, normalized correctly
    np.savetxt(d_name, event_diameters)
    np.savetxt(e_name, event_hsd)
