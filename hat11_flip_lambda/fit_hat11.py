from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Import dev version of friedrich:
import sys, os
sys.path.insert(0, '../')

import numpy as np
import matplotlib.pyplot as plt
from friedrich.analysis import Transit, Spot, Measurement, MCMCResults
from friedrich.lightcurve import hat11_params_morris_experiment
from glob import glob

from fit_utils import run_emcee, model

if any(map(lambda x: 'mist' in x, os.uname())):
    mt_wilson_paths = glob('/local/tmp/Mt_Wilson_Tilt/*/sspot??.dat')
    archive_paths = sorted(glob('/local/tmp/friedrich/hat11_flip_lambda/chains???.hdf5'))
elif any(map(lambda x: 'hyak' in x, os.uname())):
    mt_wilson_paths = glob('/gscratch/stf/bmmorris/Mt_Wilson_Tilt/*/sspot??.dat')
    archive_paths = sorted(glob('/gscratch/stf/bmmorris/hat11_flip_lambda/chains???.hdf5'))
else:
    mt_wilson_paths = glob('/astro/store/scratch/tmp/bmmorris/friedrich/Mt_Wilson_Tilt/*/sspot??.dat')
    archive_paths = sorted(glob('/astro/store/scratch/tmp/bmmorris/friedrich/hat11_flip_lambda/chains???.hdf5'))

dropbox_path = '/astro/users/bmmorris/Dropbox/Apps/ShareLaTeX/STSP_HAT-P-11/figures/'

transit_params = hat11_params_morris_experiment()
labels = "a1, mean_latitude, lnv1, lnv2, new_i_s".split(', ')

transits = []
all_times = []
for archive_path in archive_paths:
    m = MCMCResults(archive_path, transit_params)
    all_times.extend(m.lc.times.jd)
    spots = m.get_spots()
    transits.append(Transit(spots))

from friedrich.orientation import times_to_occulted_lat_lon
from friedrich.lightcurve import hat11_params_morris_flip_lambda

all_lats = []
all_lons = []
all_amps = []
all_lats_errors = []
all_spot_times = []
all_BICs = []
for transit in transits:
    for spot in transit.spots:
        latitude, longitude = times_to_occulted_lat_lon(np.array([spot.t0.value]),
                                                        transit_params)
        all_lats.append(latitude)
        all_lons.append(longitude)
        all_amps.append(spot.amplitude.value)
        all_spot_times.append(spot.t0.value)
        all_BICs.append(spot.delta_BIC)
        all_lats_errors.append(np.mean([spot.amplitude.upper, spot.amplitude.lower]))
all_lats = np.array(all_lats)
all_lats_errors = np.array(all_lats_errors)
all_lons = np.array(all_lons)
all_amps = np.array(all_amps)
all_spot_times = np.array(all_spot_times)
all_BICs = np.array(all_BICs)

ignore_high_latitudes = ((all_lats > np.radians(-40)) & (all_lats < np.radians(50)))
significance_cutoff = np.atleast_2d(all_BICs > 10).T
significant_latitudes = np.degrees(all_lats[significance_cutoff & ignore_high_latitudes])
significant_times = np.atleast_2d(all_spot_times).T[significance_cutoff & ignore_high_latitudes]
significant_amps = np.atleast_2d(all_amps).T[significance_cutoff & ignore_high_latitudes]
significant_latitudes_errors = np.ones_like(significant_latitudes) * 6


initp_hat11 = np.array([0.3, 19, 3.5, 3.5, 180+80])

#times_mid_point = significant_times.ptp()/2 + significant_times.min()
# times_mid_point = significant_times[len(significant_times)//2]
# first_half_lats = significant_latitudes[significant_times < times_mid_point]
# first_half_lats_errs = significant_latitudes_errors[significant_times < times_mid_point]
# second_half_lats = significant_latitudes[significant_times > times_mid_point]
# second_half_lats_errs = significant_latitudes_errors[significant_times > times_mid_point]

samples_hat11_1, bestp_hat11_1 = run_emcee(initp_hat11, significant_latitudes,
                                           significant_latitudes_errors,
                                           #n_steps=10, burnin=5)
                                           n_steps=2000, burnin=1000)

# samples_hat11_2, bestp_hat11_2 = run_emcee(initp_hat11, second_half_lats,
#                                            second_half_lats_errs,
#                                            #n_steps=10, burnin=5)
#                                            n_steps=2000, burnin=1000)


hat11_1_errors_upper = np.diff(np.percentile(samples_hat11_1, [50, 84], axis=0), axis=0).T
hat11_1_errors_lower = np.diff(np.percentile(samples_hat11_1, [50, 16], axis=0), axis=0).T

# hat11_2_errors_upper = np.diff(np.percentile(samples_hat11_2, [50, 84], axis=0), axis=0).T
# hat11_2_errors_lower = np.diff(np.percentile(samples_hat11_2, [50, 16], axis=0), axis=0).T

np.save('fit_outputs/hat11_1_bestps.npy', bestp_hat11_1)
np.save('fit_outputs/hat11_1_error_upper.npy', hat11_1_errors_upper)
np.save('fit_outputs/hat11_1_error_lower.npy', hat11_1_errors_lower)

# np.save('fit_outputs/hat11_2_bestps.npy', bestp_hat11_2)
# np.save('fit_outputs/hat11_2_error_upper.npy', hat11_2_errors_upper)
# np.save('fit_outputs/hat11_2_error_lower.npy', hat11_2_errors_lower)

import corner
test_lats = np.linspace(-60, 60, 500)

plt.figure()
plt.hist(significant_latitudes, 30, normed=True,
     alpha=0.5, histtype='stepfilled', color='k')
plt.plot(test_lats, model(bestp_hat11_1, test_lats), ls='--', lw=2)
plt.legend(loc='upper left')
plt.xlabel('Latitude $\ell$ [deg]', fontsize=15)
plt.ylabel('$p(\ell)$', fontsize=18)
plt.savefig('plots/hat11_1_hist.png', bbox_inches='tight')
plt.close()

fig = corner.corner(samples_hat11_1, labels=labels)
fig.savefig('plots/hat11_1_triangle.png', bbox_inches='tight')
#plt.show()
plt.close()


# plt.figure()
# plt.hist(second_half_lats, 30, normed=True,
#      alpha=0.5, histtype='stepfilled', color='k')
# plt.plot(test_lats, model(bestp_hat11_2, test_lats), ls='--', lw=2)
# plt.legend(loc='upper left')
# plt.xlabel('Latitude $\ell$ [deg]', fontsize=15)
# plt.ylabel('$p(\ell)$', fontsize=18)
# plt.savefig('plots/hat11_2_hist.png', bbox_inches='tight')
# plt.close()
#
# fig = corner.corner(samples_hat11_2, labels=labels)
# fig.savefig('plots/hat11_2_triangle.png', bbox_inches='tight')
# #plt.show()
# plt.close()