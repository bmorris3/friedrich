"""
Experiment with Kepler 17
"""
from __future__ import absolute_import, print_function
from glob import glob
import os

from friedrich.lightcurve import (LightCurve, generate_lc_depth,
                                  hat11_params_morris)
from friedrich.fitting import peak_finder, summed_gaussians, run_emcee_seeded

import matplotlib.pyplot as plt
from corner import corner

# Settings:
plots = True

if os.path.exists('/Users/bmmorris/data/hat11/'):
    # on laptop:
    light_curve_paths = glob('/Users/bmmorris/data/hat11/*slc.fits')
elif os.path.exists('/usr/lusers/bmmorris/data/hat11/'):
    # on Hyak
    light_curve_paths = glob('/usr/lusers/bmmorris/data/hat11/*slc.fits')
else:
    raise ValueError('No input files found.')

depth = 0.00343
hat11_params = hat11_params_morris()

# Construct light curve object from the raw data
whole_lc = LightCurve.from_raw_fits(light_curve_paths, name='HAT11')
transits = LightCurve(**whole_lc.mask_out_of_transit(hat11_params)
                      ).get_transit_light_curves(hat11_params)


#lc = transits[33]
lc = transits[24]
lc.remove_linear_baseline(hat11_params)

# Subtract out a transit model
transit_model = generate_lc_depth(lc.times_jd, depth, hat11_params)
residuals = lc.fluxes - transit_model

# Find peaks in the light curve residuals
best_fit_spot_params = peak_finder(lc.times.jd, residuals, lc.errors,
                                   hat11_params, n_peaks=4, plots=False,
                                   verbose=True)
best_fit_gaussian_model = summed_gaussians(lc.times.jd, best_fit_spot_params)

sampler, samples = run_emcee_seeded(lc.times.jd, lc.fluxes, lc.errors,
                                    hat11_params, best_fit_spot_params,
                                    n_steps=10000, n_walkers=100, n_threads=32,
                                    output_path=os.path.abspath('~/data/chains.txt'),
                                    burnin=0.7, n_extra_spots=1)

corner(samples)
plt.savefig('tmp.png')
plt.show()
# if best_fit_params is not None:
#     split_input_parameters = np.split(np.array(best_fit_params),
#                                       len(best_fit_params)/3)
#     for amplitude, t0, sigma in split_input_parameters:
#         model_i = gaussian(lc.times.jd, amplitude, t0, sigma)
#         chi2_bumps = np.sum((lc.fluxes - transit_model - model_i)**2 /
#                             lc.errors**2)/len(lc.fluxes)
#
#     if plots:
#         fig, ax = plt.subplots(3, 1, figsize=(8, 14), sharex=True)
#
#         errorbar_props = dict(fmt='.', color='k', capsize=0, ecolor='gray')
#
#         ax[0].errorbar(lc.times.jd, lc.fluxes, lc.errors, **errorbar_props)
#         ax[0].plot(lc.times.jd, transit_model, 'r')
#         ax[0].set(ylabel='Flux')
#
#         ax[1].axhline(0, color='gray', ls='--')
#         ax[1].errorbar(lc.times.jd, lc.fluxes - transit_model, lc.errors,
#                        **errorbar_props)
#         ax[1].plot(lc.times.jd, best_fit_gaussian_model, color='r')
#         ax[1].set_ylabel('Transit Residuals')
#
#         ax[2].axhline(0, color='gray', ls='--')
#         ax[2].errorbar(lc.times.jd, lc.fluxes - transit_model -
#                        best_fit_gaussian_model, lc.errors, **errorbar_props)
#         ax[2].set_ylabel('Gaussian Residuals')
#
#         #fig.tight_layout()
#         plt.show()
#         plt.close()
#
