"""
Class for retrieving hail size distributions from DWD birdbath Doppler spectra.

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
import datetime as dt

from pybathhail import (processing, inputs, retrieval, application,
                        plotter, saver)


class HailData:
    def __init__(self):
        self.inputs = dict()
        self.retrievals = dict()
        self.hsd_fits = dict()
        self.hail_properties = dict()
        self.hail_details = dict()
        
    def postprocessing(
            self, hail_timestamp, signal_settings, working_directory):
        """
        Get postprocessing results as input for hail retrieval.
        """
        
        processed = processing.results_dir(
            hail_timestamp,
            signal_settings,
            working_directory)
        
        self.inputs['inputs_dir'] = processed[0]
        self.inputs['postprocess_dir'] = processed[1]
    
    def retrieving(
            self, hail_timestamp, hail_settings, model_settings):
        """
        Retrieve hail size distributions and vertical wind speeds.
        """
        
        retrieval_input = inputs.collect_input(
            hail_timestamp,
            self.inputs['inputs_dir'],
            self.inputs['postprocess_dir'],
            hail_settings,
            model_settings)
        retrieved = retrieval.retrieve_hsd(
            retrieval_input,
            hail_settings)
        
        self.retrievals = retrieved
    
    def analyzing(
            self, hail_timestamp, analysis_settings):
        """
        Fits to hail size distributions and get important hail characteristics.
        """
        
        # Fit exp/gamma functions to retrieved (binned) hail size distributions
        fits = application.fit_hsd(
            self.retrievals)
        # Calculate important hail characteristics for HSD retrievals and fits
        properties = application.get_properties(
            self.retrievals,
            fits,
            analysis_settings)
        # Detailed analysis for manually pre-selected subset of hail heights
        if analysis_settings['detailed_analysis']:
            details = application.get_details(
                hail_timestamp,
                self.inputs['inputs_dir'],
                self.retrievals,
                fits,
                properties,
                analysis_settings)
        
        self.hsd_fits = fits
        self.hail_properties = properties
        self.hail_details = details

    def plotting(
            self, hail_timestamp, working_directory, plot_directory):
        """
        Visualize retrieved hail size distributions, fits, and characteristics.
        """
        
        # Time as datetime object
        hail_dt = dt.datetime.strptime(hail_timestamp, '%Y-%m-%d %H:%M:%S')
        # Full absolute path of directory for saving plots
        plot_dir = os.path.join(working_directory, plot_directory)
        # Create plotting directory if it does not exist
        os.makedirs(plot_dir, exist_ok=True)
        
        # Hail spectral reflectivities used for subsequent retrievals
        plotter.plot_reflectivity_spectra(
            self.hail_details['hail_dBZ_spectra'],
            self.hail_details['hail_velocity_spectra'],
            hail_dt,
            plot_dir)
        # Retrieved hail size distribution (hsd) range and event-sum for subset
        plotter.plot_hsd_retrievals(
            self.hail_details['statistics'],
            hail_dt,
            plot_dir)
        # Mean of hsd characteristics for (bin) retrieval and function fits
        plotter.plot_hail_properties(
            self.hail_details['statistics'],
            hail_dt,
            plot_dir)
        # Sum of event's hail size distributions (over subset)
        plotter.plot_hsd_sum(
            self.hail_details['statistics'],
            self.hail_details['event_fits'],
            hail_dt,
            plot_dir)
        # Vertical wind inferred from shifting hail Doppler spectra
        plotter.plot_wind_retrievals(
            self.hail_details['vertical_wind'],
            self.hail_details['detailed_heights'],
            hail_dt,
            plot_dir)
     
    def saving(
            self, hail_timestamp, working_directory, save_directory):
        """
        Save retrieved hail size distributions and vertical windspeeds.
        """
        
        # Time as datetime object
        hail_dt = dt.datetime.strptime(hail_timestamp, '%Y-%m-%d %H:%M:%S')
        # Full absolute path of directory for saving results
        save_dir = os.path.join(working_directory, save_directory)
        # Create directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)
        
        saver.save_retrievals(
            self.hail_details,
            hail_dt,
            save_dir)
