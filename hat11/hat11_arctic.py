"""
Experiment with HAT-P-11
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from glob import glob
import os

# Import dev version of friedrich:
import sys
# sys.path.insert(0, '../')
sys.path.insert(0, '/usr/lusers/bmmorris/git/friedrich/')
from friedrich.lightcurve import (LightCurve, hat11_params_morris,
                                  generate_lc_depth)
from friedrich.fitting import peak_finder, summed_gaussians, run_emcee_seeded

depth = 0.00343
hat11_params = hat11_params_morris()

# Construct light curve object from the raw data

import numpy as np
path = os.path.expanduser('~/git/friedrich/hat11_20171030_detrended_1min.txt')
t, f = np.loadtxt(path, unpack=True)

lc = LightCurve(times=t, fluxes=f)

transit_model = generate_lc_depth(lc.times_jd, hat11_params.rp**2, hat11_params)
residuals = lc.fluxes - transit_model

# Find peaks in the light curve residuals
best_fit_spot_params = peak_finder(lc.times.jd, residuals, lc.errors,
                                   hat11_params, n_peaks=4, plots=False,
                                   verbose=True)
best_fit_gaussian_model = summed_gaussians(lc.times.jd,
                                           best_fit_spot_params)
output_dir = '.'
transit_number = 20171030

# If spots are detected:
if best_fit_spot_params is not None:
    output_path = os.path.join(output_dir,
                               'chains{0:03d}.hdf5'.format(transit_number))
    sampler = run_emcee_seeded(lc, hat11_params, best_fit_spot_params,
                               n_steps=15000, n_walkers=150,
                               output_path=output_path, burnin=0.6,
                               n_extra_spots=0)

