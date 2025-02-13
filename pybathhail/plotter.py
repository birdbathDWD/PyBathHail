"""
Functions for plotting hail retrieval results from DWD birdbath scans.

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

#import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


def plot_reflectivity_spectra(dBZ, velocities, time, plot_path='./plots/'):
    """
    Plot hail reflectivity spectra + fall velocities (used for HSD retrieval).
    
    Based on manually preselected hail heights for detailed analysis, i.e.,
    the parameter 'height_selection' in 'hail_config.yaml' settings file.
    """
    
    print('plotting reflectivity spectra used for hail retrievals...')
    
    # Assemble velocities and reflectivity spectra for plotting
    velocity_resolution = np.diff(velocities)[0,0]
    plot_velocities = -velocities.T
    # spectral reflectivity scaled in linear space (but plotted in dBZ / vel)
    Z_lin = 10 ** (0.1 * dBZ)
    plot_Z = Z_lin.T / velocity_resolution
    plot_dBZ = 10 * np.log10(plot_Z)
    # alternative with trash scaling in dB space
    #plot_dBZ = dBZ.T / velocity_resolution
    # Datetime object as reasonable string for figure name
    dateandtime = time.strftime('%Y%m%d_%H%M%S')
    
    # Figure
    fig = plt.figure(figsize=(16,6), dpi=None, tight_layout=True)
    figname = plot_path + 'hsd_reflectivity_spectra_' + dateandtime + '.png'
    ax1 = fig.add_subplot(111)
    ax1.plot(plot_velocities, plot_dBZ, '-', c='tab:grey')
    ax1.set_xlim([0, 40])
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.set_xlabel('Hail fall velocity [m s$^{-1}$]', fontsize=13)
    ax1.set_ylabel(r'Spectral reflectivity [dBZ m$^{-1}$ s]', fontsize=13)
    fig.savefig(figname)


def plot_hsd_retrievals(data, time, plot_path='./plots/'):
    """
    Plot range of retrieved hail size distributions + sum over hail event.
    
    Based on manually preselected hail heights for detailed analysis, i.e.,
    the parameter 'height_selection' in 'hail_config.yaml' settings file.
    """
    
    print('plotting retrieved hail size distribution ranges...')
    
    
    # Extract range statistics and individual hsds for plotting
    hsd_range_stats = data['hsd_range_stats']
    hail_diameters = data['hsd_diameter_selection']
    hail_hsds = data['hsd_selection']
    # Datetime object as reasonable string for figure name
    dateandtime = time.strftime('%Y%m%d_%H%M%S')
    
    # Figure
    fig = plt.figure(figsize=(8,5), dpi=None, tight_layout=True)
    figname = plot_path + 'hsd_range_' + dateandtime + '.png'
    ax1 = fig.add_subplot(111)
    ax1.fill_between(hsd_range_stats[:,0], hsd_range_stats[:,2],
                     hsd_range_stats[:,1], color='0.6', alpha=0.5,
                     label='range of all HSDs')  
    ax1.plot(hail_diameters[0,:], hail_hsds[0,:], '-', lw=2,
             c='tab:blue', label='lower height')    
    ax1.plot(hail_diameters[-3,:], hail_hsds[-3,:], '-', lw=2,
             c='tab:orange', label='higher height')
    ax1.plot(hsd_range_stats[:,0], hsd_range_stats[:,-1], '--', lw=4,
             c='k', label='HSD over all (hail) heights')  
    ax1.set_yscale('log')
    #ax1.set_xlim([0, 30])
    #ax1.set_ylim([1e-7, 10])
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.set_xlabel('Hailstone size [mm]', fontsize=15)
    ax1.set_ylabel(r'Hail size distribution [mm$^{-1}$ m$^{-3}$]', fontsize=15)
    ax1.legend(loc='lower left', numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=0.2, labelspacing=0.1, handlelength=3.0,
               framealpha=1, fontsize=14)
    fig.savefig(figname)


def plot_hail_properties(data, time, plot_path='./plots/'):
    """
    Plot summary (= mean) of hail characteristics (in table).
    
    For retrieved (binned) hail size distributions and function fits.
    Based on manually preselected hail heights for detailed analysis, i.e.,
    the parameter 'height_selection' in 'hail_config.yaml' settings file.
    """
    
    print('plotting mean of important hail characteristics...')
    
    # Extract hail properties for plotting (mean of hail-heights subset)
    all_props = data['hsd_property_stats']
    retrieved_mean = all_props['hsd_retrievals']['hail_mean']
    exp_logfit_mean = all_props['exp_logfits']['hail_mean']
    exp_linfit_mean = all_props['exp_linfits']['hail_mean']
    gammafit_mean = all_props['gamma_fits']['hail_mean']
    # Datetime object as reasonable string for figure name
    dateandtime = time.strftime('%Y%m%d_%H%M%S')
    
    # Figure
    fig = plt.figure(figsize=(15,3), dpi=None, tight_layout=False)
    figname = plot_path + 'hail_properties_mean_' + dateandtime + '.png'
    # Pre-plotting: build table
    table_data = np.array(
        [retrieved_mean, exp_logfit_mean, exp_linfit_mean, gammafit_mean])
    plot_idx = pd.Index(
        ['bin retrieval', 'exp. fit in log', 'exp. fit in lin', 'gamma fit'])
    plot_col = pd.Index(
        ['Dmin\n [mm]', 'Dmax\n [mm]', '# density\n [1/m3]',
         'Hit rate\n [1/(s m2)]', 'W\n [g/m3]', 'Ekin\n [J/m3]',
         'Hail rate\n [mm/h]', 'E_flux\n [W/m2]', 'Dmean\n [mm]',
         'Dmedian\n [mm]', 'Dmassmean\n [mm]', 'DEmean\n [mm]',
         'E(DEmean)\n [J]', 'E(Dmax)\n [J]'])
    df = pd.DataFrame(table_data, index=plot_idx, columns=plot_col)
    cell_values = np.around(df.values, 2)
    # Plot table
    ax1 = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    table = ax1.table(cellText=cell_values, rowLabels=df.index,
                      colLabels=df.columns, loc='center')
    #ax1.axis('off')
    table.scale(1, 2.8)
    #table.set_fontsize(20)
    fig.savefig(figname)


def plot_hsd_sum(retrieval_stats, event_fits, time, plot_path='./plots/'):
    """
    Plot sum of event's hail size distributions (bin retrieval and exp. fits).
    
    Based on manually preselected hail heights for detailed analysis, i.e.,
    the parameter 'height_selection' in 'hail_config.yaml' settings file.
    """
    
    print('plotting hail size distribution over event, binned and exp fits...')
    
    
    # Extract hsd sum and corresponding exponential fits 
    event_diameters = retrieval_stats['hsd_range_stats'][:,0]
    event_hsd = retrieval_stats['hsd_range_stats'][:,-1]
    event_diameters_fits = event_fits['exp_logfits']['hsd_diameters']
    event_hsd_logfit = event_fits['exp_logfits']['hsd_frequency']
    event_hsd_linfit = event_fits['exp_linfits']['hsd_frequency']
    event_logfit_Lambda = -event_fits['exp_logfits']['N0_Lambda'][1]
    event_linfit_Lambda = -event_fits['exp_linfits']['N0_Lambda'][1]
    logfit_Lam = str(np.around(event_logfit_Lambda, decimals=2))
    linfit_Lam = str(np.around(event_linfit_Lambda, decimals=2))
    # Datetime object as reasonable string for figure name
    dateandtime = time.strftime('%Y%m%d_%H%M%S')
    
    # Figure
    fig = plt.figure(figsize=(16,6), dpi=None, tight_layout=True)
    figname = plot_path + 'hsd_event_sum_' + dateandtime + '.png'
    ax1 = fig.add_subplot(111)
    
    ax1.plot(event_diameters, event_hsd, '--', lw=4, c='k',
             label='HSD bin sum over all (hail) heights')
    ax1.plot(event_diameters_fits, event_hsd_logfit, '-', lw=3, c='b',
             label='HSD sum log fit $\Lambda$ = ' + logfit_Lam + ' mm$^{-1}$')
    ax1.plot(event_diameters_fits, event_hsd_linfit, '-', lw=3, c='r',
             label=r'HSD sum exp fit $\Lambda$ = ' + linfit_Lam + ' mm$^{-1}$')
    ax1.set_yscale('log')
    ax1.set_xlim([0, 50])
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.set_xlabel('Hail diameter [mm]', fontsize=13)
    ax1.set_ylabel(r'Hail size distribution N(D) [mm$^{-1}$ m$^{-3}$]',
                   fontsize=13)
    ax1.legend(loc='upper right', numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=1.0, labelspacing=0.1, handlelength=3.0,
               framealpha=1, fontsize=13)
    fig.savefig(figname)


def plot_fit_differences(data, time, plot_path='./plots/'):
    """
    Plot bias in hail properties of function fits vs. (bin) retrievals.
    
    Plotted hail property bias [%] = mean(100 * (fit - binned) / binned).
    Based on manually preselected hail heights for detailed analysis, i.e.,
    the parameter 'height_selection' in 'hail_config.yaml' settings file.
    """
    
    print('plotting bias of hail properties: fits vs. full (binned)...')
    
    # Extract differences in hail characteristics for plotting
    logfit_bias = -data['exp_logfits']['bias']
    linfit_bias = -data['exp_linfits']['bias']
    gammafit_bias = -data['gamma_fits']['bias']
    # Datetime object as reasonable string for figure name
    dateandtime = time.strftime('%Y%m%d_%H%M%S')
    
    # Figure
    fig = plt.figure(figsize=(15,3), dpi=None, tight_layout=False)
    figname = plot_path + 'hail_fits_bias_' + dateandtime + '.png'
    # Pre-plotting: build table
    table_data = np.array([logfit_bias, linfit_bias, gammafit_bias])
    plot_idx = pd.Index(
        ['bias exp. log\n [%]', 'bias exp. lin\n [%]', 'bias gamma\n [%]'])
    plot_col = pd.Index(
        ['Dmin', 'Dmax', '# density', 'Hit rate', 'W', 'Ekin',
         'Hail rate', 'E_flux', 'Dmean', 'Dmedian', 'Dmassmean', 'DEmean',
         'E(DEmean)', 'E(Dmax)'])
    df = pd.DataFrame(table_data, index=plot_idx, columns=plot_col)
    cell_values = np.around(df.values, 1)
    cell_values[cell_values == -0.0] = 0
    scaled = plt.Normalize(-100, 100)
    color_values = cell_values
    colors = cm.coolwarm(scaled(color_values))
    # Plot table
    ax1 = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    table = ax1.table(cellText=cell_values, rowLabels=df.index,
                      colLabels=df.columns, loc='center', cellColours=colors)
    #ax1.axis('off')
    table.scale(1, 2.8)
    #table.set_fontsize(20)
    fig.savefig(figname)


def plot_expfit_differences(data, time, plot_path='./plots/'):
    """
    Plot differences in hail properties of linear and log exp. fits.
    
    In addition to bias, also plot mean and median absolute differences.
    Based on manually preselected hail heights for detailed analysis, i.e.,
    the parameter 'height_selection' in 'hail_config.yaml' settings file.
    """
    
    print('plotting exp. fits vs. full (binned) retrievals...')
    
    # Extract differences in hail characteristics for plotting
    logfit_bias = -data['exp_logfits']['bias']
    linfit_bias = -data['exp_linfits']['bias']
    logfit_meandiff = data['exp_logfits']['mean_absdiff']
    linfit_meandiff = data['exp_linfits']['mean_absdiff']
    logfit_mediandiff = data['exp_logfits']['median_absdiff']
    linfit_mediandiff = data['exp_linfits']['median_absdiff']
    # Datetime object as reasonable string for figure name
    dateandtime = time.strftime('%Y%m%d_%H%M%S')
    
    # Figure
    figname = plot_path + 'hail_expfits_differences_' + dateandtime + '.png'
    fig = plt.figure(figsize=(16,15), dpi=None, tight_layout=True)
    # Mean of absolute percent differences
    ax1 = fig.add_subplot(311)
    ax1.plot(logfit_meandiff, 'o', c='tab:blue',
             label='logfit mean of absolute differences')
    ax1.plot(linfit_meandiff, 'x', c='tab:orange',
             label='linfit mean of absolute differences')
    ax1.grid(linestyle=':', linewidth=0.5)
    #ax1.set_xlabel('14 hail properties', fontsize=13)
    ax1.set_ylabel('Difference [%]', fontsize=13)
    ax1.legend(loc='upper left', numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=1.0, labelspacing=0.1, handlelength=2.0,
               framealpha=1, fontsize=13)
    # Median of absolute percent differences
    ax2 = fig.add_subplot(312)
    ax2.plot(logfit_mediandiff, 'o', c='tab:blue',
             label='logfit median of absolute differences')
    ax2.plot(linfit_mediandiff, 'x', c='tab:orange',
             label='linfit median of absolute differences')
    ax2.grid(linestyle=':', linewidth=0.5)
    #ax2.set_xlabel('14 hail properties', fontsize=13)
    ax2.set_ylabel('Difference [%]', fontsize=13)
    ax2.legend(loc='upper left', numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=1.0, labelspacing=0.1, handlelength=2.0,
               framealpha=1, fontsize=13)
    # Bias (same as in bias plot)
    ax3 = fig.add_subplot(313)
    ax3.plot(logfit_bias, 'o', c='tab:blue', label='logfit bias')
    ax3.plot(linfit_bias, 'x', c='tab:orange', label='linfit bias')
    ax3.grid(linestyle=':', linewidth=0.5)
    ax3.set_xlabel('14 hail properties', fontsize=13)
    ax3.set_ylabel('Difference [%]', fontsize=13)
    ax3.legend(loc='upper left', numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=1.0, labelspacing=0.1, handlelength=2.0,
               framealpha=1, fontsize=13)
    fig.savefig(figname)


def plot_N0_Lambda(parameter_set, parameter_fits, time, plot_path='./plots/'):
    """
    Plot N0-to-Lambda hsd fit parameters and overall power-law relationship.
    
    Based on manually preselected hail heights for detailed analysis, i.e.,
    the parameter 'height_selection' in 'hail_config.yaml' settings file.
    """
    
    print('plotting powerlaw fit to N0(Lambda) hsd fit parameters..')
    
    # Extract N0s, Lambdas, and power-law fits
    exp_logfit_N0_Lambda = parameter_set['exp_logfits']
    exp_linfit_N0_Lambda = parameter_set['exp_linfits']
    gamma_fit_N0_Lambda_mu = parameter_set['gamma_fits']
    logfit_powerlaw = parameter_fits['N0_Lambda_relation']['exp_logfits']
    linfit_powerlaw = parameter_fits['N0_Lambda_relation']['exp_linfits']
    # Evaluate power-law relationships for plotting
    Lambdas = np.linspace(0,5,501)
    logfit_relation = logfit_powerlaw[0] * Lambdas**logfit_powerlaw[1]
    linfit_relation = linfit_powerlaw[0] * Lambdas**linfit_powerlaw[1]
    # Datetime object as reasonable string for figure name
    dateandtime = time.strftime('%Y%m%d_%H%M%S')
    
    # Figure
    fig = plt.figure(figsize=(16,6), dpi=None, tight_layout=True)
    figname = plot_path + 'hsd_fits_N0_Lambda_' + dateandtime + '.png'
    ax1 = fig.add_subplot(111)
    ax1.plot(-exp_logfit_N0_Lambda[:,1], exp_logfit_N0_Lambda[:,0],
             'o', c='tab:blue', label='HSD log polynomial fit')
    ax1.plot(-exp_linfit_N0_Lambda[:,1], exp_linfit_N0_Lambda[:,0],
             'o', c='tab:orange', label='HSD linear exponential fit')
    ax1.plot(-gamma_fit_N0_Lambda_mu[:,1], gamma_fit_N0_Lambda_mu[:,0],
             'o', c='tab:pink', label='HSD gamma fit')
    ax1.plot(Lambdas, logfit_relation, '-', c='tab:blue', lw=2,
             label='N0(Lambda) powerlaw logfit')
    ax1.plot(Lambdas, linfit_relation, '-', c='tab:orange', lw=2,
             label='N0(Lambda) powerlaw linfit')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1e-4)
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.set_xlabel('$\Lambda$ [mm$^{-1}$]', fontsize=13)
    ax1.set_ylabel('N$_0$ [mm$^{-1}$ m$^{-3}$]', fontsize=13)
    ax1.legend(loc='lower right', numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=1.0, labelspacing=0.1, handlelength=2.0,
               framealpha=1, fontsize=13)
    fig.savefig(figname)
    
    
def plot_Lambda_Dmax(
        parameter_set, parameter_fits, retrieval_stats, time,
        plot_path='./plots/'):
    """
    Plot Lambda-to-Dmax hsd parameters and overall power-law relationship.
    
    Also plot product of Lambda * Dmax in second figure.
    Based on manually preselected hail heights for detailed analysis, i.e.,
    the parameter 'height_selection' in 'hail_config.yaml' settings file.
    """
    
    print('plotting powerlaw fit Lambda(Dmax) and products Lambda*Dmax..')
    
    # Extract Lambdas, retrieved Dmax (= same for fits), and power-law fits
    exp_logfit_N0_Lambda = parameter_set['exp_logfits']
    exp_linfit_N0_Lambda = parameter_set['exp_linfits']
    gamma_fit_N0_Lambda_mu = parameter_set['gamma_fits']
    retrieved_diameters = retrieval_stats['hsd_diameter_selection']
    Dmax = np.nanmax(retrieved_diameters, axis=1)
    logfit_powerlaw = parameter_fits['Lambda_Dmax_relation']['exp_logfits']
    linfit_powerlaw = parameter_fits['Lambda_Dmax_relation']['exp_linfits']
    # Evaluate power-law relationships for plotting
    Dmax_sim = np.linspace(5,50,46)
    logfit_relation = logfit_powerlaw[0] * Dmax_sim**logfit_powerlaw[1]
    linfit_relation = linfit_powerlaw[0] * Dmax_sim**linfit_powerlaw[1]
    # Product of (fitted) Lambda * (retrieved=fitted) Dmax; > 5 ???
    Lam_x_Dmax_logfit = -exp_logfit_N0_Lambda[:,1] * Dmax
    Lam_x_Dmax_linfit = -exp_linfit_N0_Lambda[:,1] * Dmax
    Lam_x_Dmax_gammafit = -gamma_fit_N0_Lambda_mu[:,1] * Dmax
    # Datetime object as reasonable string for figure name
    dateandtime = time.strftime('%Y%m%d_%H%M%S')
    
    # Figure 1: Lambda-Dmax relationship
    figname = plot_path + 'hsd_fits_Lambda_Dmax_' + dateandtime + '.png'
    fig = plt.figure(figsize=(16,6), dpi=None, tight_layout=True)
    ax1 = fig.add_subplot(111)
    ax1.plot(-exp_logfit_N0_Lambda[:,1], Dmax,
             'o', c='tab:blue', label='HSD log polynomial fit')
    ax1.plot(-exp_linfit_N0_Lambda[:,1], Dmax,
             'o', c='tab:orange', label='HSD linear exponential fit')
    ax1.plot(-gamma_fit_N0_Lambda_mu[:,1], Dmax,
             'o', c='tab:pink', label='HSD gamma fit')
    ax1.plot(logfit_relation, Dmax_sim, '-', c='tab:blue', lw=2,
             label='Lambda(Dmax) powerlaw logfit')
    ax1.plot(linfit_relation, Dmax_sim, '-', c='tab:orange', lw=2,
             label='Lambda(Dmax) powerlaw linfit')
    #ax1.set_xscale('log')
    #ax1.set_yscale('log')
    #ax1.set_xlim([0,5])
    ax1.set_ylim(bottom=0)
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.set_xlabel('$\Lambda$ [mm$^{-1}$]', fontsize=13)
    ax1.set_ylabel('D$_\mathrm{max}$ [mm]', fontsize=13)
    ax1.legend(loc='lower right', numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=1.0, labelspacing=0.1, handlelength=2.0,
               framealpha=1, fontsize=13)
    fig.savefig(figname)

    # Figure 2: Lambda * Dmax product
    figname2 = plot_path + 'hsd_fits_Lambda_x_Dmax_' + dateandtime + '.png'
    fig2 = plt.figure(figsize=(16,6), dpi=None, tight_layout=True)
    ax1 = fig2.add_subplot(111)
    ax1.plot(-exp_logfit_N0_Lambda[:,1], Lam_x_Dmax_logfit,
             'o', c='tab:blue', label='HSD log polynomial fit')
    ax1.plot(-exp_linfit_N0_Lambda[:,1], Lam_x_Dmax_linfit,
             'o', c='tab:orange', label='HSD linear exponential fit')
    ax1.plot(-gamma_fit_N0_Lambda_mu[:,1], Lam_x_Dmax_gammafit,
             'o', c='tab:pink', label='HSD gamma fit')
    ax1.axhline(y=5, ls='-', lw=5, c='tab:green')
    #ax1.set_xlim([0,5])
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.set_xlabel('$\Lambda$ [mm$^{-1}$]', fontsize=13)
    ax1.set_ylabel('$\Lambda$ * D$_\mathrm{max}$ [mm]', fontsize=13)
    ax1.legend(loc='lower right', numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=1.0, labelspacing=0.1, handlelength=2.0,
               framealpha=1, fontsize=13)
    fig2.savefig(figname2)        


def plot_wind_retrievals(windspeed, heights, time, plot_path='./plots/'):
    """
    Plot vertical wind profile, inferred from shifted hail Doppler spectra.
    
    Downdraft < 0, Updraft > 0.
    Based on manually preselected hail heights for detailed analysis, i.e.,
    the parameter 'height_selection' in 'hail_config.yaml' settings file.
    """
    
    print('plotting retrieved vertical wind speeds...')
    
    # Datetime object as reasonable string for figure name
    dateandtime = time.strftime('%Y%m%d_%H%M%S')
    
    # Figure
    fig = plt.figure(figsize=(8,6), dpi=None, tight_layout=True)
    figname = plot_path + 'vertical_wind_profile_' + dateandtime + '.png'
    ax1 = fig.add_subplot(111)
    ax1.plot(windspeed, heights/1000, 'o', c='tab:blue', label='vertical wind')
    #ax1.set_xlim([-10,5])
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.set_xlabel('Air motion [m s$^{-1}$]', fontsize=13)
    ax1.set_ylabel('Height above radar [km]', fontsize=13)
    ax1.legend(loc='upper right', numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=1.0, labelspacing=0.1, handlelength=2.0,
               framealpha=1, fontsize=13)
    fig.savefig(figname)
