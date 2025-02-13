"""
Optional functions for retrieving hail size distributions from Doppler spectra.

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

import gzip
import shutil
import datetime as dt
import numpy as np
import netCDF4 as nc

from pybathhail import helper


def rain_max_velocity(
        height, atmosphere_data,
        shift_type='constant', shift_value=0.0):
    """
    Find rain maximum terminal fall velocity [m/s] at given altitude.
    """
    
    # If spectra are not shifted; do not adjust theoretical max fall velocity
    if shift_type == 'no_shift':
        #velocity_shift = 0.0
        #velocity_shift_vector[hail_heights==height] = velocity_shift
        rain_vel_adjusted = 9.65
    # If spectra shifted by predetermined fixed offset
    elif shift_type == 'constant':
        #velocity_shift = constant_shift
        #velocity_shift_vector[hail_heights==height] = velocity_shift
        rain_vel_adjusted = 9.65 + shift_value
    # If spectra shifted by altitude-dependent offset
    elif shift_type in [
            'max_rain', 'min_hail_H20',
            'min_hail_MH80', 'min_hail_G75']:
        # Gunn and Kinzer (1949) environmental conditions, Atlas et al. (1973)
        # Maximum rain velocity
        rain_vel_max = 9.65
        # Detailed correction, see also Niu et al. 2010, etc.
        # Atlas et al. 1973, 0.4 to 0.5 according to Niu2010
        scaling_exponent = 0.5
        # Density in kg/m3 at 15 degrees Celsius
        density_standard_atmos = 1.23  # 1.2041 actually at 20 deg C?
        # Standard pressure [hPa or mbar] and temperature [K]
        standard_pressure = 1013.25
        standard_temperature = 293.15
        
        # Actual conditions derived from ICON model output
        ICON_output = atmosphere_data
        # Best data for selected height
        height_differences = np.abs(height - ICON_output[0,:]*1000)
        height_idx = np.where(height_differences == height_differences.min())
        # Pressure in mbar
        ICON_pressure = ICON_output[1,height_idx]
        # Temperature in K
        ICON_temperature = ICON_output[2,height_idx]
        # Velocity adjustment
        actual_pressure = ICON_pressure
        actual_temperature = ICON_temperature
        pressure_factor = actual_pressure / standard_pressure
        temp_factor = standard_temperature / actual_temperature
        density_air = density_standard_atmos * pressure_factor * temp_factor
        density_factor = density_air / density_standard_atmos
        rain_vel_adjusted = rain_vel_max * density_factor**(-scaling_exponent)  
    else:
        raise ValueError('Invalid shift_type entered for rain_max_velocity().')
        
    return rain_vel_adjusted

  
def icon_model_data(
        meteogram_location='Hohenpeissenberg',
        date_and_time=dt.datetime(2021, 4, 30, 15, 9, 41),
        hail_heights=(425.0, 1500.0), unpack=True, meteogram_dir='./'):
    """
    Extract ICON model data from ICON 'Meteogramm'.
    
    Needed for normalization of maximum rain terminal velocity in hail region;
    this is then used later for anchoring slow-falling edge of hail spectra
    to the normalized maximum rain terminal fall velocity, i.e.,
    calibrating the measured Doppler velocities if based on rain max velocity.
    NOT USED for more reasonable other anchoring method(s).
    """
    
    print('...extracting ICON Meteogramm data for max rain velocity...')
    
    # Read and transform model output
    # Time of data query (any year, month, day, hours, minutes, seconds)
    any_time = date_and_time
    # Meteogram location query
    location = meteogram_location
    # Directory where ICON .gz output files are located
    icon_dir = meteogram_dir
    # Date and time of ICON model output query for data query above
    # Corresponding ICON run (every 3 hours)
    any_time_h = any_time.hour + any_time.minute / 60
    icon_times_start = dt.datetime(
        any_time.year, any_time.month,
        any_time.day, 0, 0, 0)
    icon_times_h = np.arange(0, 24, 3)
    times_numbers = range(len(icon_times_h))
    icon_times_all = [
        icon_times_start + dt.timedelta(hours=3*run) for run in times_numbers
    ]
    timediffs_h = any_time_h - icon_times_h
    later_times = icon_times_h[timediffs_h>=0]
    icon_time_condition = np.argwhere(icon_times_h == later_times.max())[0][0]
    icon_time = icon_times_all[icon_time_condition]
    # Some formatting for later
    icontime_form = '%Y%m%d%H%M%S'
    # Specific forecast time (every 15 min), 1/4-hour timesteps for ICON model
    timesteps = 12
    steps = range(timesteps)
    forecast_times = [
        icon_time + dt.timedelta(hours=0.25*step) for step in steps
    ]
    request_time = helper.get_nearest_date(forecast_times, any_time, select=0)
    # Timestep to choose (see below), starting with 0
    plotstep = np.argwhere(np.array(forecast_times) == request_time)[0][0]
    # Data gzip file
    filename = (
        icon_dir + 'icon_d2_meteogram_'
        + icon_time.strftime(icontime_form) + '_nc'
    )
    # Decompress and save file if desired (unpack=1)
    if unpack:
        with gzip.open(filename + '.gz', 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    # Load data
    # All ICON-D2 meteogram data
    meteogram = nc.Dataset(filename)
    # data: time (109) x max_nlevs (66) x nvars (40) x nstations (31)
    # Find MOHp station index
    id_sel_station = -1
    for stat_ind, stat_name in enumerate(meteogram['station_name']):
        if ''.join(stat_name.compressed().astype('str')) == location:
            id_sel_station = stat_ind
            print(''.join(stat_name.compressed().astype(str)),
                  'station index = ', id_sel_station)
    if id_sel_station == -1:
        print('selected station ', location, ' is unknown')
        exit()
    MOHp_ind = id_sel_station
    MOHp_alltime_data = meteogram.variables['values'][:,:,:,MOHp_ind]
    MOHp_data = MOHp_alltime_data[:timesteps,:,:]
    # Forecast height levels [m], full and half levels depending on variables
    heights = meteogram.variables['heights'][:,:,MOHp_ind]
    
    # Select and calculate some thermodynamic properties for analysis
    # All layer center heights [km a.s.l.] and [km above radar]
    radar_height = 1.0062
    height_asl = heights / 1e3
    height_ar = height_asl - radar_height
    # Pressure [mbar] at all (full) heights (and half levels) and time steps
    pressure_ind = 0
    pressure = MOHp_data[:,:,pressure_ind] * 0.01
    # (Absolute) Temperature in K
    temp_K_ind = 1
    temp_K = MOHp_data[:,:,temp_K_ind]
    ## Horizontal wind speed and directional (u,v)-components [m/s]
    #wind_u_ind = 5
    #wind_v_ind = 6
    #wind_u = MOHp_data[:,:,wind_u_ind]  # zonal
    #wind_v = MOHp_data[:,:,wind_v_ind]  # meridional
    #wind_speed = (wind_u**2 + wind_v**2)**0.5
    ## Vertical wind [m/s] at half levels
    #wind_w_ind = 7
    #wind_w_half = MOHp_data[:,:,wind_w_ind]
    # Relative humidity [%]
    rel_hum_ind = 16
    rel_hum = MOHp_data[:,:,rel_hum_ind]
    
    # Collect results for determining maximum rain terminal fall velocity
    # Select height range for analysis [m], depends on hail analysis
    height_min = hail_heights[0]
    height_max = hail_heights[-1]
    # Pick results of correct height range and timestep
    height_ar_vector = height_ar[:,0]
    sel_condition = np.logical_and(
        height_ar_vector >= height_min/1000,
        height_ar_vector <= height_max/1000)
    # Final data needed for max rain fall velocity correction
    height_ar_sel = height_ar_vector[sel_condition]
    pressure_sel = pressure[plotstep,sel_condition]
    temp_K_sel = temp_K[plotstep,sel_condition] 
    rel_hum_sel = rel_hum[plotstep,sel_condition]
    # Summarize and reorder with increasing heights
    data_selection = np.array(
            [height_ar_sel, pressure_sel, temp_K_sel, rel_hum_sel])
    data_selection = data_selection[:,::-1]
        
    return data_selection


def maximum_hailsize(
        hailmode_width, minimum_hailsize=5, vD_relation='H20'):
    """
    Determine the maximum hailsize [mm] only from hail-mode widths.
    
    Derive max hailsize from 1-D array of total widths [m/s] of hail mode only
    (extracted from Doppler spectra, i.e. after only deriving maximum and
    minimum hail velocities, no shifting or power-to-reflectivity
    transformation necessary), based on Heymsfield et al. (2020) 'H20',
    yielding slow fall speeds for given hail size, or Matson and
    Huggins (1980) 'MH80', yielding intermediate fall speeds, or
    Gokhale (1975) 'G75', giving the fastest fallspeeds for the same hail size.
    The minimum hail size [mm] has to be given, which corresponds to the
    minimum hail velocity for the slow-falling edge of the hail mode.
    """
    
    if vD_relation == 'H20':
        # Heymsfield et al. (2020) hail-relationship parameters
        prefactor = 8.39
        exponent = 0.67
    elif vD_relation == 'MH80':
        # Matson and Huggins paper uses equivalent volume diameter
        prefactor = 11.45
        exponent = 0.5
    elif vD_relation == 'G75':
        # Gokhale book apparently uses spheroidal hailstones
        prefactor =15.0
        exponent = 0.5
    else:
        raise ValueError('Invalid vD_relation in maximum_hailsize().')
    
    # Calculate maximum diameter in cm
    ratio = hailmode_width / prefactor
    Dmin_term = (minimum_hailsize / 10) ** exponent
    max_diameter_cm = (ratio + Dmin_term) ** (1 / exponent)
    # Transform to mm
    max_diameter_mm = 10 * max_diameter_cm
        
    return max_diameter_mm
