"""
Functions for further analysis of retrieved hail size distributions.

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
from scipy import integrate

from pybathhail import helper, fitting, inputs


def fit_hsd(hsd_retrievals):
    """
    Fit exponential and gamma functions to retrieved hail size distributions.
    """
    
    print('fitting exp/gamma functions to retrieved full HSDs...')

    # Separate retrieval output for further analysis
    diameters = hsd_retrievals['hsd_sizes']
    binned_hsd = hsd_retrievals['hsd_frequencies']
    heights_valid = hsd_retrievals['hsd_heights']
    vD_method = hsd_retrievals['vD_relation']
    vertical_wind = hsd_retrievals['vertical_windspeeds']
    
    # Exponential fit as linear fit in log-space
    exp_logfit = fitting.exponential_fit(
        diameters, binned_hsd, fit_mode='log', vD_mode=vD_method)
    logfit_hsd, logfit_sizes, logfit_NO_Lambda = exp_logfit
    # Exponential fit in linear-space
    exp_linfit = fitting.exponential_fit(
        diameters, binned_hsd, fit_mode='lin', vD_mode=vD_method)
    linfit_hsd, linfit_sizes, linfit_NO_Lambda = exp_linfit
    # Gamma fit
    gamma_fit = fitting.gamma_fit(
        diameters, binned_hsd, vD_mode=vD_method)
    gammafit_hsd, gammafit_sizes, gammafit_NO_Lambda_mu = gamma_fit
    
    # Collect results as one output (similar to hsd_retrievals)
    logfit = dict(
        hsd_frequencies=logfit_hsd,
        hsd_sizes=logfit_sizes,
        hsd_heights=heights_valid,
        fit_N0_Lambda=logfit_NO_Lambda,
        vD_relation=vD_method,
        vertical_windspeeds=vertical_wind)
    linfit = dict(
        hsd_frequencies=linfit_hsd,
        hsd_sizes=linfit_sizes,
        hsd_heights=heights_valid,
        fit_N0_Lambda=linfit_NO_Lambda,
        vD_relation=vD_method,
        vertical_windspeeds=vertical_wind)
    gammafit = dict(
        hsd_frequencies=gammafit_hsd,
        hsd_sizes=gammafit_sizes,
        hsd_heights=heights_valid,
        fit_N0_Lambda=gammafit_NO_Lambda_mu,
        vD_relation=vD_method,
        vertical_windspeeds=vertical_wind)
    hsd_fits = dict(
        exp_logfits=logfit,
        exp_linfits=linfit,
        gamma_fits=gammafit)

    return hsd_fits


def hsd_characteristics(hsd_data, scattering_averages):
    """
    Calculate 14 characteristic hail properties from hail size distributions.
    
    From the retrieved HSDs, under scattering assumptions of average aspect
    ratio and ice volume fraction of spheroidal hailstones:
    Minimum hailstone diameter [mm], Maximum hailstone diameter [mm], Number
    density [1/m3], Hit rate [1/(s m2)], Atmospheric hail ice content [g/m3],
    Kinetic energy [J/m3], (Liquid equivalent) Hail rate [mm/h], Kinetic
    energy flux [W/m2], Average (= mean) diameter [mm], Median mass
    diameter [mm], Mass-weighted mean diameter [mm], Kinetic energy-weighted
    mean diameter [mm], Kinetic energy at energy-weighted mean diameter [J],
    Kinetic energy at maximum diameter [J].
    
    Only includes hail (terminal) fall velocity in its current implementation,
    and NO VERTICAL AIR MOTION!
    This is only relevant for the properties like hit rate or kinetic energy,
    where hailstone fall speeds enter the calculation.
    """

    # Split settings needed for calculating HSD characteristic properties
    aspect_ratio = scattering_averages['aspect_ratio']
    ice_fraction = scattering_averages['ice_fraction']
    # Separate input data for further analysis
    diameters = hsd_data['hsd_sizes']
    binned_hsd = hsd_data['hsd_frequencies']
    heights_valid = hsd_data['hsd_heights']
    vD_mode = hsd_data['vD_relation']
    # Calculate hail fall velocities from diameters
    velocities = helper.hail_size_to_velocity(diameters, vD_relation=vD_mode)
    
    # Initialize output array for all 13 properties:
    hsd_properties = np.full((binned_hsd.shape[0], 14), np.nan)
    
    # Loop over all height levels
    for height in heights_valid:
        hv_sel = (heights_valid == height)
        # Hail size distribution data for each height level
        hail_sizes = diameters[hv_sel,:]
        hail_velocities = velocities[hv_sel,:]
        hail_distribution = binned_hsd[hv_sel,:]
        # Only take useful data
        hail_velocities = hail_velocities[~np.isnan(hail_sizes)]
        hail_distribution = hail_distribution[~np.isnan(hail_sizes)]
        hail_sizes = hail_sizes[~np.isnan(hail_sizes)]
        # Sort correctly for integrations later on
        sort_idx = hail_sizes.argsort()
        hail_sizes = hail_sizes[sort_idx]
        hail_velocities = hail_velocities[sort_idx]
        hail_distribution = hail_distribution[sort_idx]
        
        # Some common terms needed for the calculations of hail properties
        # Ice density [g/cm3]
        ice_density = 0.917
        # Water density [g/cm3]
        water_density = 1.0
        # Hailstone density [g/cm3]
        hail_density = ice_density * ice_fraction
        # Hailstone masses [g]
        hail_volumes = np.pi / 6 * aspect_ratio * (hail_sizes / 10)**3
        hail_masses = hail_density * hail_volumes
        # Hailstone kinetic energies [J]
        hail_energies = 1e-3 * 0.5 * hail_masses * hail_velocities**2
        
        # 1) Minimum hailstone diameter [mm]
        minimum_diameter = hail_sizes.min()
        
        # 2) Maximum hailstone diameter [mm]
        maximum_diameter = hail_sizes.max()
        
        # 3) Number density [1/m3]
        number_density = integrate.simps(
            hail_distribution, x=hail_sizes, even='avg')
        
        # 4) Hit rate [1/(s m2)]
        hit_rate = integrate.simps(
            hail_velocities * hail_distribution, x=hail_sizes, even='avg')
        
        # 5) Atmospheric hail ice content [g/m3]
        ice_content = integrate.simps(
            hail_masses * hail_distribution, x=hail_sizes, even='avg')
        
        # 6) Kinetic energy [J/m3]
        kinetic_energy = integrate.simps(
            hail_energies * hail_distribution, x=hail_sizes, even='avg')
        
        # 7) (Liquid equivalent) Hail rate [mm/h]; with water density = 1 g/cm3
        # this is equivalent to mass flux in g / (h cm2) * mm/cm = kg / (h m2)
        integrand_hail_rate = (3.6 * hail_masses * hail_velocities
                               * hail_distribution / water_density)
        hail_rate = integrate.simps(
            integrand_hail_rate, x=hail_sizes, even='avg')
        
        # 8) Kinetic energy flux [W/m2]
        integrand_energy_flux = (hail_energies * hail_velocities
                                 * hail_distribution)
        energy_flux = integrate.simps(
            integrand_energy_flux, x=hail_sizes, even='avg')
        
        # 9) Average (= mean) diameter [mm]
        numerator_average = integrate.simps(
            hail_sizes * hail_distribution, x=hail_sizes, even='avg')
        average_diameter = numerator_average / number_density
        
        # 10) Median mass diameter [mm]
        half_mass = 0.5 * ice_content
        size_idx = 0
        partial_mass = 0
        while partial_mass < half_mass:
            this_size = hail_sizes[size_idx]
            size_idx += 1
            partial_mass = integrate.simps(
                hail_masses[:size_idx] * hail_distribution[:size_idx],
                x=hail_sizes[:size_idx], even='avg')
        median_mass_diameter = this_size
        
        # 11) Mass-weighted mean diameter [mm]
        numerator_mass = integrate.simps(
            hail_sizes * hail_masses * hail_distribution,
            x=hail_sizes, even='avg')
        mass_mean_diameter = numerator_mass / ice_content
        
        # 12) Kinetic energy-weighted mean diameter [mm]
        numerator_energy = integrate.simps(
            hail_sizes * hail_energies * hail_distribution,
            x=hail_sizes, even='avg')
        energy_mean_diameter = numerator_energy / kinetic_energy
        
        # 13) Kinetic energy at energy-weighted mean diameter [J]
        difference = np.abs(hail_sizes - energy_mean_diameter)
        ewmd_idx = np.argwhere(difference==difference.min()).max()
        ewmd_energy = hail_energies[ewmd_idx]
        
        # 14) Kinetic energy at maximum diameter [J]
        max_energy = hail_energies[hail_sizes==hail_sizes.max()][0]
    
        # Collect results
        property_vector = np.array(
            [minimum_diameter, maximum_diameter, number_density, hit_rate,
             ice_content, kinetic_energy, hail_rate, energy_flux,
             average_diameter, median_mass_diameter, mass_mean_diameter,
             energy_mean_diameter, ewmd_energy, max_energy])
        hsd_properties[hv_sel,:len(property_vector)] = property_vector
        
        
    # Give property names and collect in dictionary together with properties
    property_names = [
        'Min diameter [mm]', 'Max diameter [mm]', 'Number density [1/m3]',
        'Hit rate [1/(s m2)]', 'Atmospheric hail ice content [g/m3]',
        'Kinetic energy [J/m3]', '(Liquid equivalent) Hail rate [mm/h]',
        'Kinetic energy flux [W/m2]', 'Mean diameter [mm]',
        'Median mass diameter [mm]', 'Mass-weighted mean diameter [mm]',
        'E_kin-weighted mean diameter [mm]',
        'E_kin at E_kin-weighted mean diameter [J]',
        'E_kin at max diameter [J]',
    ]
    properties_output = dict(
        hsd_properties=hsd_properties,
        property_names=property_names)

        
    ## Some quick statistics 
    ## only makes sense if ALL data are reasonable, usually not the case
    #mean_props = hsd_properties.mean(axis=0)
    #std_props_percent = 100 * hsd_properties.std(axis=0) / mean_properties
        
    return properties_output


def get_properties(hsd_retrievals, hsd_fits, scattering_averages):
    """
    Calculate characteristic hail properties for retrieved HSDs and fits.
    """
    
    print('calculating characteristics of retrieved and fitted HSDs...')
    
    # Separate HSD data from different fits
    exp_logfits = hsd_fits['exp_logfits']
    exp_linfits = hsd_fits['exp_linfits']
    gamma_fits = hsd_fits['gamma_fits']
    
    # Calculate characteristic hail properties
    properties_retrievals = hsd_characteristics(
        hsd_retrievals, scattering_averages)
    properties_logfit = hsd_characteristics(
        exp_logfits, scattering_averages)
    properties_linfit = hsd_characteristics(
        exp_linfits, scattering_averages)
    properties_gammafit = hsd_characteristics(
        gamma_fits, scattering_averages)
    
    # Collect results in compact form
    hail_characteristics = dict(
        hsd_retrievals=properties_retrievals,
        exp_logfits=properties_logfit,
        exp_linfits=properties_linfit,
        gamma_fits=properties_gammafit)  
  
    return hail_characteristics


def hsd_stats(hsd_data, hail_properties, heights_subset):
    """
    Determine range of HSDs and calculate (some) statistics.
    
    Does not give correct output for full event HSD when option
    'shift_type: max_rain' is selected in 'hail_config.yaml' settings,
    because variable spectra shifts for 'max_rain' result in wrong
    normalization for the HSD sum of event. 
    This does not affect the original retrieval of all HSDs at individual
    hail heights (and corresponding hail characteristics), only the 
    combined (summed and normalized) HSD for hail event.
    """
    
    # Select only retrievals corresponding to selected heights_subset
    selection = np.isin(hsd_data['hsd_heights'], heights_subset)
    diameters = hsd_data['hsd_sizes'][selection,:]
    hsds = hsd_data['hsd_frequencies'][selection,:]
    
    # Unique hail diameters
    diameters_round = diameters.round(decimals=3)
    diameters_unique= np.unique(diameters_round)
    diameters_unique = diameters_unique[~np.isnan(diameters_unique)]
    # Compute 6 statistics for full hsd range at each hail diameter
    stats = 6
    range_stats = np.full((len(diameters_unique), stats+1), np.nan)
    for diameter in diameters_unique:
        range_idx = np.where(diameters_round==diameter)
        range_hsd = hsds[range_idx]
        range_min = np.nanmin(range_hsd)
        range_max = np.nanmax(range_hsd)
        range_mean = np.nanmean(range_hsd)
        range_median = np.nanmedian(range_hsd)
        range_std = np.nanstd(range_hsd)
        range_sum = np.nansum(range_hsd) / hsds.shape[0]
        # Collect results
        stats_vector = np.array(
            [diameter, range_min, range_max, range_mean,
             range_median, range_std, range_sum])
        d_sel = (diameters_unique == diameter)
        range_stats[d_sel, :len(stats_vector)] = stats_vector
    
    # Some simple statistics for corresponding hail properties
    props_stats = dict()
    for key in hail_properties:
        data = hail_properties[key]['hsd_properties']
        data_sel = data[selection,:]
        mean = np.nanmean(data_sel, axis=0)
        std = np.nanstd(data_sel, axis=0)
        std_percent = 100 * std / mean
        key_stats = dict(
            hail_mean=mean,
            hail_std=std,
            hail_std_percent=std_percent,
            property_names=hail_properties[key]['property_names']) 
        props_stats[key] = key_stats
    
    # Collect results in compact dictionary
    all_stats = dict(
        hsd_range_stats=range_stats,
        hsd_property_stats=props_stats,
        hsd_selection=hsds,
        hsd_diameter_selection=diameters)
    
    return all_stats


def hsd_fit_sum(range_statistics):
    """
    Fit exponential(s) to event sum of hail size distributions.
    """
    
    # Fit exponential (pol logfit and exp fit) to (sum of) HSD (bin) retrieval
    x_sum = range_statistics['hsd_range_stats'][:,0]
    y_sum = range_statistics['hsd_range_stats'][:,-1]
    # Weights and polyfit (in log space)
    weights = np.ones(x_sum.shape)
    y_sum_log = np.log(1.0 * y_sum)
    polfit_sum = np.polyfit(x_sum, y_sum_log, 1, w=weights)
    # Run linear exponential fit to data; first guess and fit
    pin = [np.exp(polfit_sum[1]), polfit_sum[0]]
    expfit_sum = fitting.fit_exp(x_sum, y_sum, pin, max_iterations=10000)
    # Recast linear exp fits in same form as log fits
    expfit_sum[0] = np.log(expfit_sum[0])
    expfit_sum = expfit_sum[::-1]
    # Calculate fitted HSDs (for plotting later)
    fitcalcs_polfit = np.exp(polfit_sum[0]*x_sum + polfit_sum[1])
    fitcalcs_expfit = np.exp(expfit_sum[0]*x_sum + expfit_sum[1])
    # Summarize N0s, Lambdas
    N0_Lambda_polfit = np.array([np.exp(polfit_sum[1]), polfit_sum[0]])
    N0_Lambda_expfit = np.array([np.exp(expfit_sum[1]), expfit_sum[0]])
    # Collect relevant results in one final dictionary
    logfits = dict(
        hsd_diameters=x_sum,
        hsd_frequency=fitcalcs_polfit,
        N0_Lambda=N0_Lambda_polfit)
    linfits = dict(
        hsd_diameters=x_sum,
        hsd_frequency=fitcalcs_expfit,
        N0_Lambda=N0_Lambda_expfit)
    hsd_event_fits = dict(
        exp_logfits=logfits,
        exp_linfits=linfits)

    return hsd_event_fits


def retrievals_comparison(fit_data, hail_properties, heights_subset):
    """
    Compare hail properties for bin-by-bin retrievals vs. exp/gammafits of HSD.
    """
    
    binned_vs_fits = dict()
    for key in fit_data:
        fits_props = hail_properties[key]['hsd_properties']
        binned_props = hail_properties['hsd_retrievals']['hsd_properties']
        # Select only data corresponding to selected heights_subset
        selection = np.isin(fit_data[key]['hsd_heights'], heights_subset)
        fits_sel = fits_props[selection,:]
        binned_sel = binned_props[selection,:]
        # Percent differences: bin-by-bin retrievals - fits to retrievals
        percent_diff = 100 * (binned_sel - fits_sel) / binned_sel
        # Statistics: bias, mean and median absolute percent differences
        bias = percent_diff.mean(axis=0)
        mean_absdiff = np.abs(percent_diff).mean(axis=0)
        median_absdiff = np.median(np.abs(percent_diff), axis=0)
        binned_vs_fits[key] = dict(
            bias=bias,
            mean_absdiff=mean_absdiff,
            median_absdiff=median_absdiff,
            property_names=hail_properties[key]['property_names'])
    
    return binned_vs_fits


def wind_selection(hsd_data, heights_subset):
    """
    Select subset of retrieved vertical windspeeds from hail heights.
    """
    
    selection = np.isin(hsd_data['hsd_heights'], heights_subset)
    wind_selected = hsd_data['vertical_windspeeds'][selection]
    
    return wind_selected


def fit_selection(fit_data, heights_subset):
    """
    Select subset of HSD fits (parameters) from hail heights.
    """
    
    detailed_parameters = dict()
    for key in fit_data:
        parameters = fit_data[key]['fit_N0_Lambda']
        # Select only data corresponding to selected heights_subset
        selection = np.isin(fit_data[key]['hsd_heights'], heights_subset)
        parameter_sel = parameters[selection,:]
        detailed_parameters[key] = parameter_sel
    
    return detailed_parameters


def spectra_selection(hsd_data, heights_subset):
    """
    Select subset of reflectivity spectra and velocities from hail heights.
    """
    
    selection = np.isin(hsd_data['hsd_heights'], heights_subset)
    spectra_selected = hsd_data['hsd_dBZ_spectra'][selection]
    velocities_selected = hsd_data['hsd_velocities_spectra'][selection]
    
    return spectra_selected, velocities_selected


def get_details(
        hail_timestamp, input_directory, hsd_retrievals,
        hsd_fits, hsd_properties, analysis_settings):
    """
    Detailed evaluation of hail retrievals and derived hail properties.
    """
    
    print('evaluating hail retrievals, characteristics, and fits...')
    
    # Timestamp of birdbath scan for detailed analysis of retrievals
    analysis_time = dt.datetime.strptime(hail_timestamp, '%Y-%m-%d %H:%M:%S')
    # Separate individual settings for detailed analysis
    heights_dir = analysis_settings['height_selection']
    # Subset of selected hail heights for detailed analysis
    selection_dir = os.path.join(input_directory, heights_dir)
    selection_pattern = './subset_%Y%m%d_%H%M%S.txt'
    selection_file = analysis_time.strftime(selection_pattern)
    analysis_heights = inputs.load_txt_data(selection_file, selection_dir)
    
    # Statistics for retrieved hail size distributions and hail properties
    stats = hsd_stats(
        hsd_retrievals,
        hsd_properties,
        analysis_heights)
    # Exp. fits to hail event sum (for this slected subset of hail heights)
    range_sum = hsd_fit_sum(stats)
    
    # Comparison of bin-by-bin retrievals and exp/gammafits for hail properties
    bin_vs_fits = retrievals_comparison(
        hsd_fits,
        hsd_properties,
        analysis_heights)
    
    # Inferred vertical wind, only at manually selected subset of hail heights
    wind_inferred = wind_selection(
        hsd_retrievals,
        analysis_heights)
    
    # HSD fit parameters, only at manually selected subset of hail heights
    fit_parameters = fit_selection(
        hsd_fits,
        analysis_heights)
    
    # Reflectivity spectra + velocities, right before retrievals
    spectra, velocities = spectra_selection(
        hsd_retrievals,
        analysis_heights)
    
    # Collect results in compact form
    hail_details = dict(
        statistics=stats,
        event_fits=range_sum,
        hsd_binned_vs_fits=bin_vs_fits,
        vertical_wind=wind_inferred,
        hsd_fits_parameters=fit_parameters,
        hail_dBZ_spectra=spectra,
        hail_velocity_spectra=velocities,
        detailed_heights=analysis_heights)
  
    return hail_details

