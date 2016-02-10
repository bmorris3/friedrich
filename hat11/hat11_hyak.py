"""
Experiment with HAT-P-11
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from glob import glob
import os

# Import dev version of friedrich:
import sys
#sys.path.insert(0, '../')
sys.path.insert(0, '/usr/lusers/bmmorris/git/friedrich/')
from friedrich.lightcurve import (LightCurve, hat11_params_morris,
                                  generate_lc_depth)
from friedrich.fitting import peak_finder, summed_gaussians, run_emcee_seeded
from scipy.ndimage import gaussian_filter
import numpy as np

# Settings:
if os.path.exists('/Users/bmmorris/data/hat11/'):
    # on laptop:
    light_curve_paths = glob('/Users/bmmorris/data/hat11/*slc.fits')
    output_dir = os.path.abspath('/Users/bmmorris/data')
elif os.path.exists('/usr/lusers/bmmorris/data/hat11/'):
    # on Hyak
    light_curve_paths = glob('/usr/lusers/bmmorris/data/hat11/*slc.fits')
    output_dir = os.path.abspath('/gscratch/stf/bmmorris/friedrich/hat11')
elif os.path.exists('/local/tmp/hat11'):
    # on mist
    light_curve_paths = glob('/local/tmp/hat11/*slc.fits')
    output_dir = os.path.abspath('./')
else:
    raise ValueError('No input files found.')

depth = 0.00343
hat11_params = hat11_params_morris()

# Construct light curve object from the raw data
whole_lc = LightCurve.from_raw_fits(light_curve_paths, name='HAT11')
transits = LightCurve(**whole_lc.mask_out_of_transit(hat11_params,
                                                     oot_duration_fraction=0.5)
                      ).get_transit_light_curves(hat11_params)

# Compute maxes for each quarter
available_quarters = whole_lc.get_available_quarters()
quarters = [whole_lc.get_quarter(q) for q in whole_lc.get_available_quarters()]

quarterly_maxes = {}
set_upper_limit = 4e10
for i, quarter_number, lc in zip(range(len(available_quarters)),
                                 available_quarters, quarters):
    fluxes = lc.fluxes[lc.fluxes < set_upper_limit]
    smoothed_fluxes = gaussian_filter(fluxes, sigma=20)
    quarterly_maxes[quarter_number] = np.max(smoothed_fluxes)

# Read from commmand line argument
transit_number = int(sys.argv[1])

lc = transits[transit_number]
lc.delete_outliers()
lc.subtract_polynomial_baseline(order=2, params=hat11_params)
lc.fluxes += quarterly_maxes[lc.quarters[0]]
lc.fluxes /= quarterly_maxes[lc.quarters[0]]
lc.errors /= quarterly_maxes[lc.quarters[0]]

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
                               n_steps=10000, n_walkers=200, n_threads=16,
                               output_path=output_path, burnin=0.5,
                               n_extra_spots=1)

