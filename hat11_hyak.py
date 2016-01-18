"""
Experiment with HAT-P-11
"""
from __future__ import absolute_import, print_function
from glob import glob
import os
import sys

from friedrich.lightcurve import (LightCurve, generate_lc_depth,
                                  hat11_params_morris)
from friedrich.fitting import peak_finder, summed_gaussians, run_emcee_seeded

# Settings:
if os.path.exists('/Users/bmmorris/data/hat11/'):
    # on laptop:
    light_curve_paths = glob('/Users/bmmorris/data/hat11/*slc.fits')
    output_dir = os.path.abspath('/Users/bmmorris/data')
elif os.path.exists('/usr/lusers/bmmorris/data/hat11/'):
    # on Hyak
    light_curve_paths = glob('/usr/lusers/bmmorris/data/hat11/*slc.fits')
    output_dir = os.path.abspath('/gscratch/stf/bmmorris/friedrich/')

else:
    raise ValueError('No input files found.')

depth = 0.00343
hat11_params = hat11_params_morris()

# Construct light curve object from the raw data
whole_lc = LightCurve.from_raw_fits(light_curve_paths, name='HAT11')
transits = LightCurve(**whole_lc.mask_out_of_transit(hat11_params)
                      ).get_transit_light_curves(hat11_params)

# Read from commmand line argument
transit_number = int(sys.argv[1])

lc = transits[transit_number]
lc.remove_linear_baseline(hat11_params)

# Subtract out a transit model
transit_model = generate_lc_depth(lc.times_jd, depth, hat11_params)
residuals = lc.fluxes - transit_model

# Find peaks in the light curve residuals
best_fit_spot_params = peak_finder(lc.times.jd, residuals, lc.errors,
                                   hat11_params, n_peaks=4, plots=False,
                                   verbose=True)
best_fit_gaussian_model = summed_gaussians(lc.times.jd,
                                           best_fit_spot_params)

# If spots are detected:
if best_fit_spot_params is not None:
    output_path = os.path.join(output_dir,
                               'chains{0:03d}.hdf5'.format(transit_number))
    sampler = run_emcee_seeded(lc, hat11_params, best_fit_spot_params,
                               n_steps=5000, n_walkers=200, n_threads=32,
                               output_path=output_path, burnin=0.5,
                               n_extra_spots=1)

