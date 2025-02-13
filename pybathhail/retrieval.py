"""
Functions for retrieving hail size distributions from Doppler spectra.

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
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from kneed import KneeLocator

from pybathhail import helper, optional

    
def shift_spectra(
        heights_vector, doppler_velocities, spectral_reflectivities,
        mode_indices, icon_data, hail_heights=(425.0, 1500.0),
        shift_type='min_hail_H20', minimum_hailsize=5,
        constant_shift=0.0, height_idx=None):
    """
    Shift (linear) reflectivity spectra and corresponding Doppler velocities.
    
    Shift according to selected shift type to mitigate effects of vertical
    air motion on hail reflectivity spectra. This routine also 'retrieves' the
    vertical air motion (i.e. up- or downdraft speeds), which can strongly 
    depend on the selected 'shift_type'.
    
    Args:
        heights_vector (array):
            1-D array of all height levels in birdbath scan Doppler spectra.
            
        doppler velocities (array): 
            Array of Doppler velocities, without shifting Doppler spectra.
            
        spectral_reflectivities (array):
            Profile of the reflectivity spectra for doppler velocities input.
        
        mode_indices (array):
            Array of indices that indicate the start and end of individual
            precipitation modes in Doppler spectra identified previously
            (needed to 'cut out' hail reflectivity spectrum only from 
            full spectrum at each height level).
        
        icon_data (array):
            Data extracted from ICON Meteogramm for correcting rain max
            fall velocity (shape = (4, ?)). Only needed if 'shift_type' is 
            'max_rain' (see below).
            
            
        hail_heights (tuple):
            Tuple of pre-selected minimum and maximum heights (m above radar)
            where the hail analysis is to be performed.
            
        shift_type (str):
            Identifier for different methods for anchoring Doppler spectra,
            resulting in different velocity shifts. 'no_shift' to take spectra
            as they are, i.e. no correction for vertical air motions;
            'constant' applies a constant shift to the Doppler velocities
            as given by constant_shift; 'max_rain' anchors the fast-falling
            edge of the rain modes previously identified in the reflectivity
            spectra to the theoretical rain maximum fall velocity
            (accounting for the atmospheric state and thus needing ICON model
            data as input, which is only implemented for MHP radar currently),
            anchoring of rain spectra to max rain veloctiy can occur at each
            height level (height_idx=None) or for a single preselected height
            level in total column of reflectivity spectra given by its index of
            all heights (e.g. height_idx=3). Finally, probably the most useful
            (because of robustness) shifting method is to anchor the
            slow-falling edge of the hail modes (i.e. mode_indices[:,1]) to the
            hail fall velocity that corresponds to the smallest assumed hail
            size (default is 5 mm diameter, see 'minimum_hailsize' argument),
            based on one of three different pre-established hail fall-velocity-
            to-diameter relationships of Heymsfield et al. (2020) or Matson and
            Huggins (1980) or Gokhale (1975) as 'min_hail_H20',
            'min_hail_MH80', or 'min_hail_G75', respectively.
        
        minimum_hailsize (float):
            Assumed minimum diameter of hailstones in mm, generally 5 mm. Only
            relevant when using 'shift_type=min_hail_...' options,
            see also the description of 'shift_type' argument.
            
        constant_shift (float):
            Shift in m/s to apply to all Doppler velocities,
            only used for 'shift_type=constant'.
        
        height_idx (int):
            If Doppler spectra are to be anchored to the theoretical rain
            maximum fall velocity based only on a single height in the entire
            profile of reflectivity spectra, height_idx indicates the location
            of this height in the full profile. 'height_idx=None' results in
            finding the appropriate anchor point for every height level
            separately (this requires the presence of at least 2 precipitation
            modes). Only used for 'shift_type=max_rain'.
    
    Returns:
        hail_spectra_lin (array):
            Profile of reflectivity spectra of hail mode only, in linear units.
            Hail mode is defined as the fastest-falling precipitation mode.
        
        hail_spectra_dBZ (array):
            Profile of reflectivity spectra of hail mode only, in dBZ.
            Hail mode is defined as the fastest-falling precipitation mode.
            
        hail_vel_shifted (array):
            Shifted Doppler velocities for hail reflectivity spectra,
            same shape as hail_spectra.
        
        hail_levels (array):
            Valid hail_heights range picked from all heights. 
            Corresponding to hail_spectra.
            
        vertical_wind (array):
            Vertical air motion, i.e. vertical windspeeds in m/s, for all hail
            levels, inferred from velocity shifts for selected shift_type
            above (vertical_wind = -vel_shift). Updraft is positive,
            downdraft is negative.  
    """
    
    print('...shifting reflectivity spectra to correct for vertical wind...')
    
    # Only pick previously determined hail heights
    heights_valid = helper.heights_for_analysis(hail_heights, heights_vector)
    
    # Initialize results
    velocity_shift_vector = np.full_like(heights_valid, np.nan)
    hail_spectra_lin = np.full(
        (len(heights_valid), spectral_reflectivities.shape[1]), np.nan) 
    hail_vel_shifted = np.full_like(hail_spectra_lin, np.nan)
    
    
    for height in heights_valid:
        # Do NOT shift spectra at all
        if shift_type == 'no_shift':
            vel_shift = 0.0
            velocity_shift_vector[heights_valid==height] = vel_shift
        # Shift spectra by predetermined fixed offset
        elif shift_type == 'constant':
            vel_shift = constant_shift
            velocity_shift_vector[heights_valid==height] = vel_shift
        # Align fast-falling edge of rain mode with maximum rain velocity  
        elif shift_type == 'max_rain':
            if height_idx is not None:
                height_fixed = heights_vector[height_idx]
            else:
                height_fixed = height
            rain_velocity_adjusted = optional.rain_max_velocity(
                height_fixed,
                icon_data,
                shift_type=shift_type)
            h_sel = (heights_vector == height_fixed)
            rain_maxvel_idx = mode_indices[h_sel, 2]
            try:
                vel_shift = (
                    -rain_velocity_adjusted.ravel()
                    - doppler_velocities[h_sel, int(rain_maxvel_idx)]
                )
            except AttributeError:
                vel_shift = (
                    -rain_velocity_adjusted
                    - doppler_velocities[h_sel, int(rain_maxvel_idx)]
                )
            velocity_shift_vector[heights_valid==height] = vel_shift
        # Terminal fall velocity [m/s] according to H20 at smallest hail     
        elif shift_type == 'min_hail_H20':
            vD_min_H20 = helper.hail_size_to_velocity(minimum_hailsize,
                                                      vD_relation='H20')
            # Determine appropriate value to shift all hail velocities
            h_sel = (heights_vector == height)
            hail_minvel_idx = mode_indices[h_sel, 1]
            vel_shift = (
                -vD_min_H20 - doppler_velocities[h_sel, int(hail_minvel_idx)]
            )
            velocity_shift_vector[heights_valid==height] = vel_shift
        # Terminal fall velocity [m/s] according to MH80 at smallest hail   
        elif shift_type == 'min_hail_MH80':
            vD_min_MH80 = helper.hail_size_to_velocity(minimum_hailsize,
                                                       vD_relation='MH80')
            # Determine appropriate value to shift all hail velocities
            h_sel = (heights_vector == height)
            hail_minvel_idx = mode_indices[h_sel, 1]
            vel_shift = (
                -vD_min_MH80 - doppler_velocities[h_sel, int(hail_minvel_idx)]
            )
            velocity_shift_vector[heights_valid==height] = vel_shift
        # Terminal fall velocity [m/s] according to G75 at smallest hail    
        elif shift_type == 'min_hail_G75':
            vD_min_G75 = helper.hail_size_to_velocity(minimum_hailsize,
                                                      vD_relation='G75')
            # Determine appropriate value to shift all hail velocities
            h_sel = (heights_vector == height)
            hail_minvel_idx = mode_indices[h_sel, 1]
            vel_shift = (
                -vD_min_G75 - doppler_velocities[h_sel, int(hail_minvel_idx)]
            )
            velocity_shift_vector[heights_valid==height] = vel_shift
        else:
            raise ValueError('Invalid shift_type entered for shift_spectra().')
            
        # Only keep hail mode reflectivities and shift hail velocities
        valid_modes = mode_indices[heights_vector==height,:].ravel()
        valid_bins = valid_modes[1] + 1 - valid_modes[0]
        start_bin = int(valid_modes[0])
        end_bin = int(valid_modes[1]) + 1
        keep_hail_spectra = spectral_reflectivities[
            heights_vector==height, start_bin:end_bin].ravel()
        hail_vel_rep = doppler_velocities[
            heights_vector==height, start_bin:end_bin].ravel()
        # Final results, shifted hail modes only
        hv_sel = (heights_valid == height)
        hail_spectra_lin[hv_sel, :int(valid_bins)] = keep_hail_spectra
        hail_vel_shifted[hv_sel, :int(valid_bins)] = hail_vel_rep + vel_shift
        # Inferred vertical wind, i.e. vertical air motion
        vertical_wind = -velocity_shift_vector
    
    # Linear -> dBZ reflectivities
    hail_spectra_dBZ = 10 * np.log10(hail_spectra_lin)
    # Collect results as output
    shift_output = (
        hail_spectra_lin, hail_spectra_dBZ,
        hail_vel_shifted, heights_valid,
        vertical_wind,
    )
        
    return shift_output

    
def smooth_spectra(
        spectral_dBZ, heights_valid,
        window_length=51, filter_order=0, filter_mode='mirror'):
    """
    Filter reflectivity spectra by Savitzky-Golay filter.
    
    SG-Filter (0th order = moving average filter) applied to spectra to smooth
    noisy spectral reflectivity. This is only required/benefitial later on
    when fitting the fast falling edge of the reflectivity spectra to estimate
    the maximum hail velocity, but not needed for most of the other steps.
    """
    
    print('...smoothing reflectivity spectra...')
    
    # Initialize smoothed reflectivity spectra
    hail_reflectivities_smooth = np.full_like(spectral_dBZ, np.nan)
    # Parameters for Savitzky-Golay smoothing filter
    sg_window_length = window_length
    sg_order = filter_order
    sg_mode = filter_mode
    # Smooth reflectivity spectrum at every height
    for height in heights_valid:
        hv_sel = (heights_valid == height)
        unfiltered_data = spectral_dBZ[hv_sel,:]
        unfiltered_valid = unfiltered_data[~np.isnan(unfiltered_data)]
        length_valid = unfiltered_valid.size
        sg_filtered = savgol_filter(unfiltered_valid, sg_window_length,
                                    sg_order, mode=sg_mode)
        hail_reflectivities_smooth[hv_sel, :length_valid] = sg_filtered
        
    return hail_reflectivities_smooth

    
def spectra_elbows(
        fall_velocities, spectral_dBZ,
        heights_valid, kneedle_sensitivity=1.0,
        hail_curve='convex', curve_direction='increasing',
        kneedle_interp='polynomial', kneedle_online=True):
    """
    Find the knees/elbows, i.e. inflection points, of concave/convex curves.
    
    Find knees/elbows with the kneedle algorithm. Here, the inputs are profiles
    of hail reflectivity spectra with corresponding fall_velocities.
    This is required to estimate the maximum (most negative) hail fall velocity
    from the full hail reflectivity spectra. 
    """
    
    print('...finding elbow points in reflectivity spectra...')
    
    # Initialize elbow vector
    hail_velocity_elbows = np.zeros((spectral_dBZ.shape[0],))
    # Determine elbows at every height
    for height in heights_valid:
        velocities_all = fall_velocities[heights_valid==height,:]
        velocities_valid = velocities_all[~np.isnan(velocities_all)]
        reflectivities_all = spectral_dBZ[heights_valid==height,:]
        dBZ_valid = reflectivities_all[~np.isnan(reflectivities_all)]
        max_dBZ_idx = np.where(dBZ_valid==dBZ_valid.max())[0][0]
        dBZ_valid_final = dBZ_valid[0:max_dBZ_idx+1]
        velocities_valid_final = velocities_valid[0:max_dBZ_idx+1]
        kneedle = KneeLocator(
            velocities_valid_final, dBZ_valid_final,
            S=kneedle_sensitivity, curve=hail_curve,
            direction=curve_direction, online=kneedle_online,
            interp_method=kneedle_interp)
        hail_velocity_elbows[heights_valid==height] = kneedle.elbow
        
    return hail_velocity_elbows

    
def noise_reflectivity(
        fall_velocities, spectral_dBZ, spectra_elbows,
        heights_valid, noise_mode='midpoint'):
    """
    Estimate the noise levels beyond elbow points in Doppler spectra.
    
    Calculation is done for noise region for a given profile of hail
    reflectivity spectra, i.e. at fall velocities more negative than
    previously determined elbow or inflection point of each spectrum. 
    Select 'midpoint', 'elbow', or 'minimum' of fast-falling tail as
    noise level.
    """
    
    print('...estimating noise level of reflectivity spectra...')
    
    # Initialize vectors of reflectivity noise levels
    hail_noise_reflectivities = np.zeros(
            (spectral_dBZ.shape[0],))
    hail_noise_std = np.zeros_like(hail_noise_reflectivities)
    # Determine noise level
    for height in heights_valid:
        hv_sel = (heights_valid == height)
        velocities_all = fall_velocities[hv_sel,:]
        velocities_valid = velocities_all[~np.isnan(velocities_all)]
        reflect_all = spectral_dBZ[hv_sel,:]
        reflect_valid = reflect_all[~np.isnan(reflect_all)]
        faster_velocities = (velocities_valid < spectra_elbows[hv_sel])
        reflect_noise = reflect_valid[faster_velocities]
        # Noise level
        vel_sel = (velocities_valid == spectra_elbows[hv_sel])
        elbow_reflectivity = reflect_valid[vel_sel]
        noise_min_reflectivity = reflect_noise.min()
        # Noise is given by mean value between elbow and minimum beyond elbow
        if noise_mode == 'midpoint':
            noise_value = 0.5 * (elbow_reflectivity + noise_min_reflectivity)
        # Noise is given by reflectivity at elbow
        elif noise_mode == 'elbow':
            noise_value = elbow_reflectivity
        # Noise is given by reflectivity minimum beyond elbow
        elif noise_mode == 'minimum':
            noise_value = noise_min_reflectivity
        else:
            raise ValueError('Invalid noise_mode in noise_reflectivity().')
        noise_std = np.std(reflect_noise)
        
        # Collect results
        hail_noise_reflectivities[hv_sel] = noise_value
        hail_noise_std[hv_sel] = noise_std
        
    return hail_noise_reflectivities, hail_noise_std

    
def reflectivity_fit(
        fall_velocities, spectral_dBZ, spectra_elbows,
        heights_valid, elbow_velocity_range=2,
        anchored_at_elbow=True):
    """
    Fit line to reflectivity spectra at elbows.
    
    This line indicates the contiuation of hail reflectivity spectra without
    the impact of noise at fast-falling edge.
    """
    
    print('...fitting reflectivity spectra at elbows...')
    
    # Initialize array of linear fits to reflectivity spectra at elbow point
    elbow_fits = np.zeros((spectral_dBZ.shape[0],2))
    # CASE 1: fit going through elbow
    if anchored_at_elbow:
        # Function for fitting reflectivity spectra near elbows
        def fit_function(x, slope):
                return slope * x
        # Determine linear fits to reflectivity spectra anchored at elbows
        for height in heights_valid:
            hv_sel = (heights_valid == height)
            velocities_all = fall_velocities[hv_sel,:]
            vel_valid = velocities_all[~np.isnan(velocities_all)]
            v_slow = (vel_valid >= spectra_elbows[hv_sel])
            v_fast = (vel_valid <= spectra_elbows[hv_sel]+elbow_velocity_range)
            velocities_fitdata = vel_valid[v_slow & v_fast]
            reflect_all = spectral_dBZ[hv_sel,:]
            reflect_valid = reflect_all[~np.isnan(reflect_all)]
            reflect_fitdata = reflect_valid[v_slow & v_fast]
            # Re-center data w.r.t. elbow point
            velocities_centered = velocities_fitdata - velocities_fitdata[0]
            reflectivities_centered = reflect_fitdata - reflect_fitdata[0]
            # Fit the curve to the re-centered data -> optimal slope
            popt, pcov = curve_fit(
                fit_function, velocities_centered, reflectivities_centered)
            # Find intercept
            intercept = reflect_fitdata[0] - popt * velocities_fitdata[0]
            # Collect results
            elbow_fits[hv_sel,0] = popt
            elbow_fits[hv_sel,1] = intercept
    # CASE 2: fit NOT going through elbow
    else:
        # Determine linear fits to reflectivity spectra NOT anchored at elbows
        for height in heights_valid:
            hv_sel = (heights_valid == height)
            velocities_all = fall_velocities[hv_sel,:]
            vel_valid = velocities_all[~np.isnan(velocities_all)]
            v_slow = (vel_valid >= spectra_elbows[hv_sel])
            v_fast = (vel_valid <= spectra_elbows[hv_sel]+elbow_velocity_range)
            velocities_fitdata = vel_valid[v_slow & v_fast]
            reflect_all = spectral_dBZ[hv_sel,:]
            reflect_valid = reflect_all[~np.isnan(reflect_all)]
            reflect_fitdata = reflect_valid[v_slow & v_fast]
            # Linear fit to selected range of reflectivity spectra
            reflectivity_polfit = np.polyfit(
                    velocities_fitdata, reflect_fitdata, 1)
            # Collect results
            elbow_fits[hv_sel,:] = reflectivity_polfit
        
    return elbow_fits

    
def maximum_velocity(
        fall_velocities, reflectivity_fits,
        noise_levels, heights_valid):
    """
    Estimate maximum hail fall velocity for a profile of reflectivity spectra.
    
    For each reflectivity spectrum, the hail maximum fall velocity is assumed
    to be the most negative fall velocity just before the linear fit to the
    elbow point of the reflectivity spectrum dips below the noise level.
    """
    
    print('...finding fastest hail fall velocities (downward = negative)...')
    
    # Initialize vector of hail maximum fall velocity
    hail_max_velocities = np.zeros((fall_velocities.shape[0],))    
    # Determine most negative fall velocity before noise level
    for height in heights_valid:
        hv_sel = (heights_valid == height)
        velocities_all = fall_velocities[hv_sel,:]
        velocities_valid = velocities_all[~np.isnan(velocities_all)]
        # Linear fit to reflectivities
        fit_valid = reflectivity_fits[hv_sel,:].ravel()
        if fit_valid[0] < 0:
            # Exclude nonsensical fits to weirdly shaped spectra
            max_velocity = np.nan
        else:
            fit_reflect = fit_valid[0]*velocities_valid + fit_valid[1]
            # Find slowest fall velocity where reflectivity fit < noise level
            dip_idx = np.argwhere(fit_reflect < noise_levels[hv_sel]).max()
            # Max hail veloctiy = fastest velocity before dip below noise level
            max_velocity = velocities_valid[dip_idx+1]
        # Collect results
        hail_max_velocities[hv_sel] = max_velocity
        
    return hail_max_velocities

    
def replace_fast_reflectivities(
        fall_velocities, spectral_dBZ, spectra_elbows,
        reflectivity_fits, heights_valid):
    """
    Replace hail spectral reflectivitis at fast velocities beyond elobw points.
    
    Replace old values with previously determined linear reflectivity fit
    (to dBZ reflectivities) to mitigate effects of background noise/signal
    that affects these very wide hail Doppler spectra. 
    """
    
    print('...replacing spectra by fits at fast fall velocities...')
    
    # Initialize vector of hail spectral reflectivities to be replaced, in part
    hail_replaced_spectra = spectral_dBZ.copy()
    
    # Replace appropriate reflectivities
    for height in heights_valid:
        hv_sel = (heights_valid == height)
        velocities_all = fall_velocities[hv_sel,:]
        velocities_valid = velocities_all[~np.isnan(velocities_all)]
        reflectivities_all = spectral_dBZ[hv_sel,:]
        reflect_valid = reflectivities_all[~np.isnan(reflectivities_all)]
        # Linear fit to reflectivities
        fit_valid = reflectivity_fits[hv_sel,:].ravel()
        fit_reflect = fit_valid[0]*velocities_valid + fit_valid[1]
        # Replace reflectivities by linear fit
        vel_sel = (velocities_valid < spectra_elbows[hv_sel])
        reflect_valid[vel_sel] = fit_reflect[vel_sel]
        
        # Collect results
        hail_replaced_spectra[hv_sel,:reflect_valid.size] = reflect_valid
        
    return hail_replaced_spectra

    
def truncate_spectra(
        fall_velocities, spectral_dBZ, cut_velocities):
    """
    Cut off hail spectral reflectivitis (and fall velocities) beyond maxima.
    
    Maximum for cutting is given by previously determined hail maximum
    (most negative) fall velocity. 
    """
    
    print('...truncating spectra at fastest hail fall velocities...')
    
    # Initialize arrays of truncated velocities and spectral reflectivities
    hail_velocities = fall_velocities.copy()
    hail_reflectivities = spectral_dBZ.copy()
    truncated_velocities = np.full_like(hail_velocities, np.nan)
    truncated_dBZ = np.full_like(hail_reflectivities, np.nan)
    
    # Truncation
    cuts = cut_velocities.reshape((cut_velocities.size, 1))
    truncated = (hail_velocities < cuts)
    hail_velocities[truncated] = np.nan
    hail_reflectivities[truncated] = np.nan
    # Move valid data after truncation to front of array
    for spectrum in range(hail_velocities.shape[0]):
        velocities_all = hail_velocities[spectrum,:]
        velocities_valid = velocities_all[~np.isnan(velocities_all)]
        reflect_all = hail_reflectivities[spectrum,:]
        reflect_valid = reflect_all[~np.isnan(reflect_all)]
        # Move to front of spectra
        length = velocities_valid.size
        truncated_velocities[spectrum, :length] = velocities_valid
        truncated_dBZ[spectrum, :length] = reflect_valid
    
    return truncated_velocities, truncated_dBZ

    
def hail_size_distribution(
        fall_velocities, spectral_dBZ, heights_valid,
        radx_diameters, radx_xsections, radar_wavelength=53.1547,
        radar_factor=0.93, vD_mode='H20'):
    """
    Retrieve hail size distributions from hail reflectivity [dBZ] spectra.
    
    Based on pre-calculated radar (single-scattering backscatter) cross
    sections for a wide range of hail diameters, retrieve the (profile of) hail
    size distribution(s) [mm-1 m-3] for the given profiles of hail (linear)
    reflectivity spectra [mm6 m-3] at the used radar wavelength and
    previously retrieved hail diameters [mm] that correspond to the Doppler
    velocities of the spectral reflectivities. 
    The appropriate hail size resolutions, i.e. the hail size bins originally
    given by the spectral velocity resolution, are automatically incorporated
    in the calculations to transform reflectivities per spectral bin to
    correctly scaled size distributions per mm and m3.  
    """
    
    print('...retrieval of HSDs...')
    
    # Transform hail fall velocity to diameters with selected v-D relationship
    retrieved_diameters = helper.hail_velocity_to_size(
        -fall_velocities, vD_relation=vD_mode)
    # Corresponding reflectivities in LINEAR units
    spectral_reflectivities = 10 ** (0.1 * spectral_dBZ)
    
    # Initialize results array of hail size distributions
    hail_size_distribution = np.full_like(spectral_reflectivities, np.nan)
    # Perform retrieval for each spectrum, i.e. at each height bin
    for height in heights_valid:
        hv_sel = (heights_valid == height)
        # Interpolate radar X-sections to previously retrieved hail diameters
        sizes = retrieved_diameters[hv_sel,:].ravel()
        radx_interpolated = np.interp(sizes, radx_diameters, radx_xsections)
        # Reflectivities [mm6 m-3] for single hail particles pro mm pro m3
        # Zh = wavelength^4 / (pi^5 * |K|^2) Integral_D radx(D) N(D) dD
        prefactor = radar_wavelength**4 / (np.pi**5 * radar_factor)
        Zh_single_pro_mm = prefactor * radx_interpolated
        
        # Number of particles pro mm pro m3 for each spectral reflectivity bin
        # Hail size resolution for correct scaling
        size_resolution_final = np.full_like(sizes, np.nan)
        sizes_valid = sizes[~np.isnan(sizes)]
        size_resolution = np.abs(np.diff(sizes_valid))
        last_step = np.abs(np.diff(size_resolution)[-1])
        size_resolution_extend = np.append(
            size_resolution,
            size_resolution[-1] - last_step)
        length_extend = size_resolution_extend.size
        size_resolution_final[:length_extend] = size_resolution_extend
        # Scaling single-hailstone reflectivity factor 
        # <--> spectral reflectivity per size bin
        Zh_single_pro_resolution = Zh_single_pro_mm * size_resolution_final
        # Calculate normalized frequency
        hail_size_distribution[hv_sel,:] = (spectral_reflectivities[hv_sel,:] 
                                            / Zh_single_pro_resolution)
        
    return hail_size_distribution, retrieved_diameters


def retrieve_hsd(
        retrieval_inputs, retrieval_settings):
    """
    Retrieve hail size distribution and vertical wind for inputs and settings.
    
    Args:
        retrieval_inputs (dict):
            Dictionary of input data from postprocessing routine, scattering
            simulations, etc. needed for hail retrieval from birdbath scan 
            Doppler spectra.
        
        retrieval_settings (dict):
            Group of parameters that specify the options for the retrieval
            algorithm that have to be selected.
    """
    
    print('retrieving hail size distributions from input data...')
    
    # Extract individual settings required for separate retrieval steps
    wavelength = retrieval_settings['radar_wavelength']
    height_min = retrieval_settings['hail_minheight']
    height_max = retrieval_settings['hail_maxheight']
    vd_relation = retrieval_settings['vD_relation']
    size_min = retrieval_settings['hail_minsize']
    shift = retrieval_settings['shift_type']
    shift_const = retrieval_settings['constant_shift']
    noise = retrieval_settings['noise_mode']
    elbow_distance= retrieval_settings['elbow_range']
    # Analyzed range of original 'eye-balled' hail levels
    height_range = (height_min, height_max)
    
    # Input data for separate retrieval steps
    spectra_reflectivity = retrieval_inputs['spectra_reflectivity']
    spectra_heights = retrieval_inputs['spectra_heights']
    spectra_velocities = retrieval_inputs['spectra_velocities']
    mode_indices = retrieval_inputs['mode_indices']
    radx = retrieval_inputs['radx']
    radx_diameters = retrieval_inputs['radx_diameters']
    model_data = retrieval_inputs['model_data']
    
    # Correct reflectivity spectra for vertical wind
    shift_output = shift_spectra(
        spectra_heights, spectra_velocities, spectra_reflectivity,
        mode_indices, model_data, hail_heights=height_range,
        shift_type=shift, minimum_hailsize=size_min,
        constant_shift=shift_const, height_idx=None)
    (hail_spectra_lin, hail_spectra_dBZ,
     hail_vel_shifted, heights_valid, vertical_wind) = shift_output
    
    # Smooth spectra for more stable final retrievals
    hail_dBZ_smooth = smooth_spectra(
        hail_spectra_dBZ, heights_valid,
        window_length=51, filter_order=0, filter_mode='mirror')
    
    # Find elbow points to determine fast-falling edge of spectra
    elbows = spectra_elbows(
        hail_vel_shifted, hail_dBZ_smooth,
        heights_valid, kneedle_sensitivity=1.0,
        hail_curve='convex', curve_direction='increasing',
        kneedle_interp='polynomial', kneedle_online=True)
    
    # Estimate noise level at fast-falling edge of spectra
    noise_dBZ, noise_std = noise_reflectivity(
        hail_vel_shifted, hail_dBZ_smooth,
        elbows, heights_valid, noise_mode=noise)
    
    # Fit reflectivity spectra at elbows
    spectra_fits = reflectivity_fit(
        hail_vel_shifted, hail_dBZ_smooth,
        elbows, heights_valid,
        elbow_velocity_range=elbow_distance,
        anchored_at_elbow=True)
    
    # Find maximum hail velocities (= most negative velocity)
    hail_max_velocity = maximum_velocity(
        hail_vel_shifted, spectra_fits,
        noise_dBZ, heights_valid)
    
    # Replace fast edge of spectra to mitigate background-noise impact
    hail_dBZ_spectra_replaced = replace_fast_reflectivities(
        hail_vel_shifted, hail_dBZ_smooth, elbows,
        spectra_fits, heights_valid)
    
    # Truncate spectra at maximum hail velocities
    truncate_output = truncate_spectra(
        hail_vel_shifted, hail_dBZ_spectra_replaced,
        hail_max_velocity)
    (hail_velocities_cut, hail_dBZ_spectra_cut) = truncate_output
    
    # Retrieve hail size distributions [mm-1 m-3]
    hsd_scaled, hsd_diameters = hail_size_distribution(
        hail_velocities_cut, hail_dBZ_spectra_cut,
        heights_valid, radx_diameters, radx,
        radar_wavelength=wavelength,
        vD_mode=vd_relation)
    
    # Collect relevant results (in dictionary)
    retrieval_results = dict(
        hsd_frequencies=hsd_scaled,
        hsd_sizes=hsd_diameters,
        hsd_heights=heights_valid,
        vD_relation=vd_relation,
        hsd_dBZ_spectra=hail_dBZ_spectra_cut,
        hsd_velocities_spectra=hail_velocities_cut,
        vertical_windspeeds=vertical_wind)
        
    return retrieval_results
