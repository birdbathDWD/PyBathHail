"""
Functions for plotting hail retrieval results from DWD birdbath scans.

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

#import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from matplotlib import cm


def plot_reflectivity_spectra(dBZ, velocities, time, plot_path='./plots/'):
    """
    Plot hail reflectivity spectra + fall velocities (used for HSD retrieval).
    
    Based on manually preselected hail heights for detailed analysis, i.e.,
    the parameter 'height_selection' in 'hail_config.yaml' settings file.
    """
    
    print('plotting reflectivity spectra used for hail retrievals...')
    

    plot_velocities = -velocities.T
    ## spectral reflectivity scaled in linear space (but plotted in dBZ / vel)
    #velocity_resolution = np.diff(velocities)[0,0]
    #Z_lin = 10 ** (0.1 * dBZ)
    #plot_Z = Z_lin.T / velocity_resolution
    #plot_dBZ = 10 * np.log10(plot_Z)
    plot_dBZ = dBZ.T
    # Datetime object as reasonable string for figure name
    dateandtime = time.strftime('%Y%m%d_%H%M%S')
    
    # Figure
    fig = plt.figure(figsize=(16,6), dpi=None, tight_layout=True)
    figname = plot_path + 'hsd_truncated_reflectivity_spectra_' + dateandtime + '.png'
    ax1 = fig.add_subplot(111)
    ax1.plot(plot_velocities, plot_dBZ, '-', c='tab:grey')
    ax1.set_xlim([0, 40])
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.set_xlabel('(positive) Hail fall velocity (corrected + truncated spectra) [m s$^{-1}$]', fontsize=13)
    ax1.set_ylabel(r'Spectral reflectivity [dBZ per bin]', fontsize=13)
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
    ax1.set_xlabel('Hail size [mm]', fontsize=15)
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
         'Hit rate\n [1/(s m2)]', 'Ice content\n [g/m3]', 'Ekin\n [J/m3]',
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
    ax1.plot(event_diameters_fits, event_hsd_logfit, '--', lw=2, c='g',
             label='HSD sum log fit $\Lambda$ = ' + logfit_Lam + ' mm$^{-1}$')
    ax1.plot(event_diameters_fits, event_hsd_linfit, '-', lw=2, c='r',
             label=r'HSD sum exp fit $\Lambda$ = ' + linfit_Lam + ' mm$^{-1}$')
    ax1.set_yscale('log')
    ax1.set_xlim([0, 50])
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.set_xlabel('Hail size [mm]', fontsize=13)
    ax1.set_ylabel(r'Hail size distribution N(D) [mm$^{-1}$ m$^{-3}$]',
                   fontsize=13)
    ax1.legend(loc='upper right', numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=1.0, labelspacing=0.1, handlelength=3.0,
               framealpha=1, fontsize=13)
    fig.savefig(figname)


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
    ax1.plot(windspeed, heights/1000, 'o', c='tab:blue', label='vertical wind speed')
    #ax1.set_xlim([-10,5])
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.set_xlabel('Vertical air motion [m s$^{-1}$]', fontsize=13)
    ax1.set_ylabel('Height above radar [km]', fontsize=13)
    ax1.legend(loc='upper right', numpoints=1, ncol=1, borderpad=0.2,
               borderaxespad=1.0, labelspacing=0.1, handlelength=2.0,
               framealpha=1, fontsize=13)
    fig.savefig(figname)
