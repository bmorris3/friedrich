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
i = int(sys.argv[1])

################################################################################
# Solar spot data

#paths = glob('/local/tmp/Mt_Wilson_Tilt/*/sspot??.dat')
# paths = glob('/Users/bmmorris/data/Mt_Wilson_Tilt/*/sspot??.dat')

from astropy.time import Time
import astropy.units as u
from astropy.table import Table
import corner
import datetime

test_lats = np.linspace(-60, 60, 500)
year_bins = np.arange(1917, 1986, 4)

def split_interval(string, n, cast_to_type=float):
    return [cast_to_type(string[i:i+n]) for i in range(0, len(string), n)]

all_years_array = []

header = ("jd n_spots_leading n_spots_following n_spots_day_1 n_spots_day_2 "
          "rotation_rate latitude_drift latitude_day_1 latitude_day_2 longitude_day_1 "
          "longitude_day_2 area_day_1 area_day_2 group_latitude_day_1 group_longitude_day_1 "
          "group_area_day_1 group_area_day_2 polarity_day_1 polarity_change tilt_day_1 tilt_day_2 "
          "group_rotation_rate group_latitude_drift").split()

for path in mt_wilson_paths:
    f = open(path).read().splitlines()

    n_rows = len(f) // 3
    n_columns = 23#18
    yearly_array = np.zeros((n_rows, n_columns))

    for j in range(n_rows):
        # First five ints specify time, afterwards specify sunspot data
        int_list = split_interval(f[0+j*3][:18], 2, int)
        month, day, year_minus_1900, hour, minute = int_list[:5]
        year = year_minus_1900 + 1900
        jd = Time("{year:d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
                  .format(**locals())).jd
        row = [jd] + int_list[5:] + split_interval(f[1+j*3], 7) + split_interval(f[2+j*3][1:], 7)
        yearly_array[j, :] = row

    all_years_array.append(yearly_array)

table = Table(np.vstack(all_years_array), names=header)


from fit_utils import run_emcee, model
initp_sun = np.array([0.5, 16, 3.5, 3.5, 180+80])


#for i in range(len(year_bins)):
start = Time(datetime.datetime(year_bins[i], 1, 1))
end = start + 4*u.year
in_time_bin = (table['jd'] > start.jd) & (table['jd'] < end.jd)

solar_lats = table['latitude_day_1'][in_time_bin]
solar_lats_error = np.ones_like(table['latitude_day_1'][in_time_bin])

print('begin mcmc {0}'.format(i))
samples_i, bestp_i = run_emcee(initp_sun, solar_lats,
                               solar_lats_error,
                               n_steps=2000, burnin=1000)

plt.figure()
plt.hist(solar_lats, 30, normed=True,
     alpha=0.5, histtype='stepfilled', color='k')
plt.plot(test_lats, model(bestp_i, test_lats), ls='--', lw=2)
plt.legend(loc='upper left')
plt.xlabel('Latitude $\ell$ [deg]', fontsize=15)
plt.ylabel('$p(\ell)$', fontsize=18)
plt.savefig('plots/{0:03d}_hist.png'.format(i), bbox_inches='tight')
plt.close()

fig = corner.corner(samples_i, labels=labels)
fig.savefig('plots/{0:03d}_triangle.png'.format(i), bbox_inches='tight')
#plt.show()
plt.close()

mt_wilson_bestps = np.median(samples_i, axis=0)
mt_wilson_error_upper = np.diff(np.percentile(samples_i, [50, 84], axis=0), axis=0)
mt_wilson_error_lower = np.diff(np.percentile(samples_i, [50, 16], axis=0), axis=0)

np.save('fit_outputs/mt_wilson_bestps_{0}.npy'.format(i), mt_wilson_bestps)
np.save('fit_outputs/mt_wilson_error_upper_{0}.npy'.format(i), mt_wilson_error_upper)
np.save('fit_outputs/mt_wilson_error_lower_{0}.npy'.format(i), mt_wilson_error_lower)
