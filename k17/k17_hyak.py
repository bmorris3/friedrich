"""
Experiment with Kepler-17
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from glob import glob
import os

# Import dev version of friedrich:
import sys
# sys.path.insert(0, '../')
sys.path.insert(0, '/usr/lusers/bmmorris/git/friedrich/')
from friedrich.lightcurve import (LightCurve, k17_params_morris,
                                  generate_lc_depth)
from friedrich.fitting import peak_finder, summed_gaussians, run_emcee_seeded
from scipy.ndimage import gaussian_filter
import numpy as np

# Settings:
if os.path.exists('/Users/bmmorris/data/kepler17/'):
    # on laptop:
    light_curve_paths = glob('/Users/bmmorris/data/kepler17/*slc.fits')
    output_dir = os.path.abspath('/Users/bmmorris/data')
elif os.path.exists('/usr/lusers/bmmorris/data/kepler17/'):
    # on Hyak
    light_curve_paths = glob('/usr/lusers/bmmorris/data/kepler17/*slc.fits')
    output_dir = os.path.abspath('/gscratch/stf/bmmorris/friedrich/k17')
elif os.path.exists('/local/tmp/kepler17'):
    # on mist
    light_curve_paths = glob('/local/tmp/kepler17/*slc.fits')
    output_dir = os.path.abspath('./')

else:
    raise ValueError('No input files found.')

transit_params = k17_params_morris()
depth = transit_params.rp**2

# Construct light curve object from the raw data
whole_lc = LightCurve.from_raw_fits(light_curve_paths, name='K17')
transits = LightCurve(**whole_lc.mask_out_of_transit(transit_params,
                                                     oot_duration_fraction=0.5)
                      ).get_transit_light_curves(transit_params)
transits = [transit for transit in transits if len(transit.fluxes) > 225]
# Compute maxes for each quarter
available_quarters = whole_lc.get_available_quarters()
quarters = [whole_lc.get_quarter(q) for q in whole_lc.get_available_quarters()]

quarterly_maxes = {}
set_upper_limit = 3e4 
for i, quarter_number, lc in zip(range(len(available_quarters)),
                                 available_quarters, quarters):
    fluxes = lc.fluxes[lc.fluxes < set_upper_limit]
    smoothed_fluxes = gaussian_filter(fluxes, sigma=700)
    quarterly_maxes[quarter_number] = np.max(smoothed_fluxes)

# Read from command line argument
transit_number = int(sys.argv[1])
print('Transit number: {0}'.format(transit_number))
lc = transits[transit_number]
lc.delete_outliers()
lc.subtract_polynomial_baseline(order=2, params=transit_params)
lc.fluxes += quarterly_maxes[lc.quarters[0]]
lc.fluxes /= quarterly_maxes[lc.quarters[0]]
lc.errors /= quarterly_maxes[lc.quarters[0]]

lc_output_path = os.path.join(output_dir,
                              'lc{0:03d}.txt'.format(transit_number))
np.savetxt(lc_output_path, np.vstack([lc.times.jd, lc.fluxes, lc.errors]).T)

# Subtract out a transit model
transit_model = generate_lc_depth(lc.times_jd, depth, transit_params)
residuals = lc.fluxes - transit_model

# Find peaks in the light curve residuals
best_fit_spot_params = peak_finder(lc.times.jd, residuals, lc.errors,
                                   transit_params, n_peaks=6, plots=False,
                                   verbose=True)
# This first spot model might miss the broadest spots
best_fit_gaussian_model = summed_gaussians(lc.times.jd,
                                           best_fit_spot_params)

# # Run again on the residuals to pick up big spots
# best_fit_spot_params = peak_finder(lc.times.jd, residuals - best_fit_gaussian_model, lc.errors,
#                                    transit_params, n_peaks=4, plots=True,
#                                    verbose=True, broaden_gaussian_factor=2)

# If spots are detected:
if best_fit_spot_params is not None:
    output_path = os.path.join(output_dir,
                               'chains{0:03d}.hdf5'.format(transit_number))
    sampler = run_emcee_seeded(lc, transit_params, best_fit_spot_params,
                               n_steps=15000, n_walkers=150,
                               output_path=output_path, burnin=0.6,
                               n_extra_spots=0)
