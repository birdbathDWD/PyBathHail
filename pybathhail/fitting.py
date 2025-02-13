"""
Fitting functions for hail size distributions and hail-radar relationships.

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

import numpy as np
from scipy.optimize import curve_fit


def fit_exp(xdata, ydata, pin, weights=None, max_iterations=800):
    """
    Fit exponential function to ydata for xdata.
    
    Args:
        xdata:
            Diameters, for example.
            
        ydata:
            Frequency of xdata, for example.
            
        pin:
            [N0, Lam] first guess.
            
        weights:
            Assumed weights (as absolute uncertainties) for parameter
            optimization, e.g. 1 everywhere or simply ydata values themselves.
            
        max_iterations:
            Maximum number of iterations for parameter optimization with
            Levenberg-Marquardt algorithm.

    Returns:
        Best fit parameters for exponential function y(x).
    """
    
    # Define exp function to be fitted
    def exp_func(x, N0, Lam):
        return N0 * np.exp(Lam*x)
    ## Initial guess (from previous tests)
    #pinit = [N0_in, Lam_in]
    # Fit
    paropt_exp, parcov_exp = curve_fit(
        exp_func, xdata, ydata, p0=pin, sigma=weights,
        absolute_sigma=True, maxfev=max_iterations)
    ## Evaluate fit for xfitvalues
    #yfit = exp_func(xfitvalues, paropt_exp[0], paropt_exp[1])
    
    return paropt_exp


def fit_gamma(xdata, ydata, pin, weights=None, max_iterations=800):
    """
    Fit Gamma function to ydata for xdata.
    
    Args:
        xdata:
            Diameters, for example.
            
        ydata:
            Frequency of xdata, for example.
            
        pin:
            [N0, Lam, mu] first guess.
            
        weights:
            Assumed weights (as absolute uncertainties) for parameter
            optimization, e.g. 1 everywhere or simply ydata values themselves.
            
        max_iterations:
            Maximum number of iterations for parameter optimization with
            Levenberg-Marquardt algorithm.

    Returns:
        Best fit parameters for Gamma function y(x).
    """
    
    # Define Gamma function to be fitted
    def gamma_function(x, N0, Lam, mu):
        return N0 * x**mu * np.exp(Lam*x)
    # Fit
    paropt_gamma, parcov_gamma = curve_fit(
        gamma_function, xdata, ydata, p0=pin, sigma=weights,
        absolute_sigma=True, maxfev=max_iterations)
    ## Evaluate fit for xfitvalues
    #yfit = gamma_function(
    #    xfitvalues, paropt_gamma[0], paropt_gamma[1], paropt_gamma[2])
    
    return paropt_gamma


def fit_powerlaw(xdata, ydata, pin, weights=None, max_iterations=800):
    """
    Fit Power-law function to ydata = a * xdata**b.
    
    Args:
        xdata:
            Diameters, for example.
            
        ydata:
            Frequency of xdata, for example.
            
        pin:
            [a, b] first guess.
            
        weights:
            Assumed weights (as absolute uncertainties) for parameter
            optimization, e.g. 1 everywhere or simply ydata values themselves.
            
        max_iterations:
            Maximum number of iterations for parameter optimization with
            Levenberg-Marquardt algorithm.

    Returns:
        Best fit parameters for power-law function y(x).
    """
    
    # Define power-law function to be fitted
    def power_function(x, a, b):
        return a * x**b
    # Fit
    paropt_power, parcov_power = curve_fit(
        power_function, xdata, ydata, p0=pin, sigma=weights,
        absolute_sigma=True, maxfev=max_iterations)
    
    return paropt_power


def exponential_fit(
        retrieved_diameters, retrieved_hsd,
        fit_mode='log', vD_mode='H20'):
    """
    Exponential fit to retrieved hail size distributions.
    
    Either as linear fit in log space or by exponential function in linear
    space.
    """
    
    # Initialize output arrays 
    hsd_exp_fit = np.full_like(retrieved_hsd, np.nan)
    N0_Lambda = np.full((retrieved_hsd.shape[0],2), np.nan)
    diameters_exp_fit = np.full_like(retrieved_diameters, np.nan)
    #velocities_exp_fit = np.full_like(retrieved_diameters, np.nan)
    
    # Fit for every height level
    for level in range(retrieved_hsd.shape[0]):
        # Retrieved hail size distribution data 
        x_hailsizes = retrieved_diameters[level,:]
        y_hailsizes = retrieved_hsd[level,:]
        #v_hailsizes = velocities[level,:]
        y_hailsizes = y_hailsizes[~np.isnan(x_hailsizes)][::-1]
        #v_hailsizes = v_hailsizes[~np.isnan(x_hailsizes)][::-1]
        x_hailsizes = x_hailsizes[~np.isnan(x_hailsizes)][::-1]
        
        if fit_mode == 'log':
            # Weights and polyfit (in log space)
            weights = np.ones(x_hailsizes.shape)  # y_hailsizes**(-1)
            y_hailsizes_log = np.log(1.0 * y_hailsizes)
            exp_fit = np.polyfit(x_hailsizes, y_hailsizes_log, 1, w=weights)
            #corr = np.corrcoef(x_hailsizes, y_hailsizes_log)
        elif fit_mode == 'lin':
            # Polyfit in logspace as first guess
            weights = np.ones(x_hailsizes.shape)  # y_hailsizes**(-1)
            y_hailsizes_log = np.log(1.0 * y_hailsizes)
            polfit = np.polyfit(x_hailsizes, y_hailsizes_log, 1, w=weights)
            # Run linear exponential fit to data; first guess and fit
            pin = [np.exp(polfit[1]), polfit[0]]
            exp_fit = fit_exp(
                x_hailsizes, y_hailsizes, pin, max_iterations=10000)
            # Recast linear exp fits in same form as log fits
            exp_fit[0] = np.log(exp_fit[0])
            exp_fit = exp_fit[::-1]
        else:
            raise ValueError('Invalid fit_mode entered for exponential_fit().')
        
        # Collect results
        fitcalcs = np.exp(exp_fit[0]*x_hailsizes + exp_fit[1])
        hsd_exp_fit[level, :x_hailsizes.size] = fitcalcs
        N0_Lambda[level,:] = np.exp(exp_fit[1]), exp_fit[0]
        diameters_exp_fit[level, :x_hailsizes.size] = x_hailsizes
        #velocities_exp_fit[level, :x_hailsizes.size] = v_hailsizes
        
    return hsd_exp_fit, diameters_exp_fit, N0_Lambda


def gamma_fit(
        retrieved_diameters, retrieved_hsd, vD_mode='H20'):
    """
    Gamma fit to retrieved hail size distributions.  
    """
    
    # Initialize output arrays 
    hsd_gamma_fit = np.full_like(retrieved_hsd, np.nan)
    N0_Lam_mu = np.full((retrieved_hsd.shape[0], 3), np.nan)
    diameters_gamma_fit = np.full_like(retrieved_diameters, np.nan)
    #velocities_gamma_fit = np.full_like(retrieved_diameters, np.nan)
    
    # Fit for every height level
    for level in range(retrieved_hsd.shape[0]):
        # Retrieved hail size distribution data 
        x_hailsizes = retrieved_diameters[level,:]
        y_hailsizes = retrieved_hsd[level,:]
        #v_hailsizes = velocities[level,:]
        y_hailsizes = y_hailsizes[~np.isnan(x_hailsizes)][::-1]
        #v_hailsizes = v_hailsizes[~np.isnan(x_hailsizes)][::-1]
        x_hailsizes = x_hailsizes[~np.isnan(x_hailsizes)][::-1]
            
        # Polyfit in logspace as first guess
        weights = np.ones(x_hailsizes.shape)  # y_hailsizes**(-1)
        y_hailsizes_log = np.log(1.0 * y_hailsizes)
        polfit = np.polyfit(x_hailsizes, y_hailsizes_log, 1, w=weights)
        # Run Gamma fit to data; first guess and fit
        pin = [np.exp(polfit[1]), polfit[0], 0]
        gamma_fit = fit_gamma(
            x_hailsizes, y_hailsizes, pin, max_iterations=20000)
        
        # Calculate series of fitted values for plotting
        fitcalcs_gamma = (gamma_fit[0] * x_hailsizes**gamma_fit[2]
                          * np.exp(gamma_fit[1]*x_hailsizes))
        hsd_gamma_fit[level, :x_hailsizes.size] = fitcalcs_gamma
        N0_Lam_mu[level,:] = gamma_fit[0], gamma_fit[1], gamma_fit[2]
        diameters_gamma_fit[level, :x_hailsizes.size] = x_hailsizes
        #velocities_gamma_fit[hv_sel, :x_hailsizes.size] = v_hailsizes
        
    return hsd_gamma_fit, diameters_gamma_fit, N0_Lam_mu

    
def power_fit(x, y, fit_mode='loglog'):
    """
    Power-law fit y = a * x**b.
    
    For hail size distribution exponential-fit parameters. See Grieser and
    Hill (2019), e.g. N0(Lambda) or Lambda(Dmax). Fit either power law in
    linear space or 1-D polynomial in log-log space. 
    """
    
    if fit_mode == 'loglog':
        # Weights and polyfit (in log space)
        weights = np.ones(x.shape)
        x_log = np.log(1.0 * x)
        y_log = np.log(1.0 * y)
        power_log_fit = np.polyfit(x_log, y_log, 1, w=weights)
        #corr = np.corrcoef(x_hailsizes, y_hailsizes_log)
        power_fit = np.zeros_like(power_log_fit)
        # Recast loglog power fits in same form as linear power fits below
        power_fit[0] = np.exp(power_log_fit[-1])
        power_fit[-1] = power_log_fit[0]  
    elif fit_mode == 'lin':
        # Polyfit in log-logspace as first guess
        weights = np.ones(x.shape)
        x_log = np.log(1.0 * x)
        y_log = np.log(1.0 * y)
        polfit = np.polyfit(x_log, y_log, 1, w=weights)
        # Run linear power-law fit to data; first guess and fit
        pin = [np.exp(polfit[-1]), polfit[0]]
        power_fit = fit_powerlaw(x, y, pin, max_iterations=5000)
    else:
        raise ValueError('Invalid fit_mode entered for power_fit().')
        
    return power_fit
