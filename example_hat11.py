"""
Experiment with Kepler 17
"""
#from __future__ import absolute_import, print_function
from glob import glob

from friedrich.lightcurve import (LightCurve, generate_lc_depth,
                                  hat11_params_morris)
from friedrich.fitting import peak_finder, summed_gaussians, gaussian

import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.console import ProgressBar

# Settings:
plots = True
light_curve_paths = glob('/Users/bmmorris/data/hat11/*slc.fits')
depth = 0.00343
hat11_params = hat11_params_morris()

# Construct light curve object from the raw data
whole_lc = LightCurve.from_raw_fits(light_curve_paths, name='Kepler17')
transits = LightCurve(**whole_lc.mask_out_of_transit(hat11_params)
                      ).get_transit_light_curves(hat11_params)

delta_chi2 = {}

with ProgressBar(len(transits)) as bar:
    for i, lc in enumerate(transits):
        # Remove linear out-of-transit trend from transit
        lc.remove_linear_baseline(hat11_params)

        # Subtract out a transit model
        transit_model = generate_lc_depth(lc.times_jd, depth, hat11_params)
        residuals = lc.fluxes - transit_model

        # Find peaks in the light curve residuals
        best_fit_params = peak_finder(lc.times.jd, residuals, lc.errors,
                                      hat11_params, n_peaks=2)
        best_fit_gaussian_model = summed_gaussians(lc.times.jd, best_fit_params)

        # Measure delta chi^2
        chi2_transit = np.sum((lc.fluxes - transit_model)**2 /
                              lc.errors**2)/len(lc.fluxes)

        if best_fit_params is not None:
            split_input_parameters = np.split(np.array(best_fit_params),
                                              len(best_fit_params)/3)
            delta_chi2[i] = []
            for amplitude, t0, sigma in split_input_parameters:
                model_i = gaussian(lc.times.jd, amplitude, t0, sigma)
                chi2_bumps = np.sum((lc.fluxes - transit_model - model_i)**2 /
                                    lc.errors**2)/len(lc.fluxes)
                delta_chi2[i].append(np.abs(chi2_transit - chi2_bumps))

            if plots:
                fig, ax = plt.subplots(3, 1, figsize=(8, 14), sharex=True)

                ax[0].errorbar(lc.times.jd, lc.fluxes, lc.errors, fmt='.',
                               color='k')
                ax[0].plot(lc.times.jd, transit_model, 'r')
                ax[0].set(ylabel='Flux')

                ax[1].axhline(0, color='gray', ls='--')
                ax[1].errorbar(lc.times.jd, lc.fluxes - transit_model,
                               fmt='.', color='k')
                ax[1].plot(lc.times.jd, best_fit_gaussian_model, color='r')
                ax[1].set_ylabel('Transit Residuals')

                ax[2].axhline(0, color='gray', ls='--')
                ax[2].errorbar(lc.times.jd, lc.fluxes - transit_model -
                               best_fit_gaussian_model, fmt='.', color='k')
                ax[2].set_ylabel('Gaussian Residuals')
                ax[2].set_title(r'$Delta \chi^2$ = '+'{0}'
                                .format(delta_chi2[i]))

                #fig.tight_layout()
                fig.savefig('plots/{0:03d}.png'.format(i))
                #plt.show()
                plt.close()

        bar.update()

all_delta_chi2 = np.concatenate(list(delta_chi2.values())).ravel()

fig, ax = plt.subplots(1,figsize=(12, 6))
ax.plot(np.log10(all_delta_chi2), '.')
plt.show()
