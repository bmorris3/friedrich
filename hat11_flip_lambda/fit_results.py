import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.table import Table
from astropy.time import Time
from astropy.io import ascii
import datetime
import os

mtwilson_bestp_paths = sorted(glob('fit_outputs/mt_wilson_bestps_*'))
mtwilson_error_upper_paths = sorted(glob('fit_outputs/mt_wilson_error_upper*'))
mtwilson_error_lower_paths = sorted(glob('fit_outputs/mt_wilson_error_upper*'))

hat11_bestp_paths = sorted(glob('fit_outputs/hat11_?_bestps.npy'))
hat11_error_upper_paths = sorted(glob('fit_outputs/hat11_?_error_upper.npy'))
hat11_error_lower_paths = sorted(glob('fit_outputs/hat11_?_error_lower.npy'))

year_bins = np.arange(1917, 1986, 4)
spot_table_path = 'mt_wilson_spot_table.txt'

if not os.path.exists(spot_table_path):
    def split_interval(string, n, cast_to_type=float):
        return [cast_to_type(string[i:i+n]) for i in range(0, len(string), n)]

    mt_wilson_paths = glob('/local/tmp/Mt_Wilson_Tilt/*/sspot??.dat')
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

        for i in range(n_rows):
            # First five ints specify time, afterwards specify sunspot data
            int_list = split_interval(f[0+i*3][:18], 2, int)
            month, day, year_minus_1900, hour, minute = int_list[:5]
            year = year_minus_1900 + 1900
            jd = Time("{year:d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
                      .format(**locals())).jd
            row = [jd] + int_list[5:] + split_interval(f[1+i*3], 7) + split_interval(f[2+i*3][1:], 7)
            yearly_array[i, :] = row

        all_years_array.append(yearly_array)

    table = Table(np.vstack(all_years_array), names=header)
    ascii.write(table, spot_table_path)
else:
    table = ascii.read(spot_table_path)


year_bin_edges = Time(map(lambda i: datetime.datetime(i, 1, 1),
                          year_bins.tolist() + [year_bins[-1] + 2])).jd

n_spots_per_bin, edges = np.histogram(table['jd'], bins=year_bin_edges)
color_bar_max = n_spots_per_bin.max() // 1000 * 1000

def plot_color_gradients(fig, ax):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack(10*[gradient])

    ax.imshow(gradient.T, cmap=plt.get_cmap('viridis_r'), aspect=1.0,
              origin='lower')
    ax.set_xticks([])
    nticks = 6
    #datespace = np.linspace(all_spot_times.min(), all_spot_times.max(), nticks)
    xspace = np.linspace(0, 256, nticks, dtype=int)[::-1]
    #xtickdates = [t.date() for t in Time(datespace, format='jd').datetime]
    xtickdates = np.linspace(0, color_bar_max, nticks, dtype=int)[::-1]
    ax.set_yticks(xspace)
    ax.set_yticklabels(xtickdates)
    ax.yaxis.tick_right()


fig = plt.figure(figsize=(12, 5))
ax0 = plt.subplot2grid((1, 1), (0, 0))
divider = make_axes_locatable(ax0)
ax1 = divider.append_axes("right", size="100%", pad=0)
ax2 = divider.append_axes("right", size="20%", pad=0.1)
ax = [ax0, ax1, ax2]

def lnvar_to_std(bestp, error_lower, error_upper):
    mean_std = np.mean(np.exp(bestp[2:4])**0.5, axis=0)
    mean_std_lower_quad_sum = np.sqrt(error_lower[0, 2]**2 + error_lower[0, 3]**2)/2
    # mean_std_lower = 2/mean_std * mean_std_lower_quad_sum
    mean_std_lower = 0.5 * np.sqrt(np.exp(np.mean(bestp[2:4]))) * mean_std_lower_quad_sum

    mean_std_upper_quad_sum = np.sqrt(error_upper[0, 2]**2 + error_upper[0, 3]**2)/2
    # mean_std_upper = 2/mean_std * mean_std_upper_quad_sum
    mean_std_upper = 0.5 * np.sqrt(np.exp(np.mean(bestp[2:4]))) * mean_std_upper_quad_sum

    return mean_std, mean_std_upper, mean_std_lower


for i, bestp_path, lower_path, upper_path in zip(range(len(mtwilson_bestp_paths)),
                                                 mtwilson_bestp_paths,
                                                 mtwilson_error_lower_paths,
                                                 mtwilson_error_upper_paths):
    bestp = np.load(bestp_path)
    error_lower = np.load(lower_path)
    error_upper = np.load(upper_path)

    # Plot mean standard dev vs mean latitude
    mean_std, mean_std_upper, mean_std_lower = lnvar_to_std(bestp, error_lower, error_upper)
    mean_std_err = np.vstack([mean_std_lower, mean_std_upper])

    mean_latitude = bestp[1]
    mean_latitude_err = np.vstack([error_lower[0, 1], error_upper[0, 1]])

    solar_kwargs = dict(fmt='o', markersize=7,
                        label='Sun' if i==0 else None,
                        color=plt.cm.viridis_r(n_spots_per_bin[i] /
                                               float(n_spots_per_bin.max())))

    ax[0].errorbar(mean_std, mean_latitude, xerr=mean_std_err,
                   yerr=mean_latitude_err, **solar_kwargs)

    # Plot asymmetry vs mean latitude
    asymmetry = abs(0.5 - bestp[0])
    asymmetry_err = np.vstack([error_lower[0, 0], error_upper[0, 0]])
    ax[1].errorbar(asymmetry, mean_latitude, xerr=asymmetry_err,
                   yerr=mean_latitude_err, **solar_kwargs)

half_labels = [r'1$^{\mathrm{st}}$ Half', r'2$^{\mathrm{nd}}$ Half']
for i, bestp_path, lower_path, upper_path in zip(range(len(hat11_bestp_paths)),
                                                 hat11_bestp_paths,
                                                 hat11_error_lower_paths,
                                                 hat11_error_upper_paths):
    bestp = np.load(bestp_path)
    error_lower = np.load(lower_path).T
    error_upper = np.load(upper_path).T
    
    # Plot mean standard dev vs mean latitude
    mean_std, mean_std_upper, mean_std_lower = lnvar_to_std(bestp, error_lower, error_upper)
    mean_std_err = [[abs(mean_std_lower)], [mean_std_upper]]

    mean_latitude = bestp[1]
    mean_latitude_err = [[abs(error_lower[0, 1])], [error_upper[0, 1]]]

    hat11_kwargs = dict(color='r', fmt='s', markersize=9,
                        label='HAT-P-11' if i==0 else None)

    ax[0].errorbar(mean_std, mean_latitude, xerr=mean_std_err,
                   yerr=mean_latitude_err, **hat11_kwargs)

    ax[0].annotate(half_labels[i], xy=(mean_std+0.1, mean_latitude+0.2), ha='left', va='bottom')

    # Plot asymmetry vs mean latitude
    asymmetry = abs(0.5 - bestp[0])
    asymmetry_err = [[abs(error_lower[0, 0])], [error_upper[0, 0]]]

    ax[1].errorbar(asymmetry, mean_latitude, xerr=asymmetry_err,
                   yerr=mean_latitude_err, **hat11_kwargs)

    ax[1].annotate(half_labels[i], xy=(asymmetry+0.005, mean_latitude+0.2), ha='left', va='bottom')

plot_color_gradients(fig, ax[2])

ax[0].legend(loc='lower right', numpoints=1)
ax[0].set_xlabel(r'Standard Deviation [deg]')
ax[0].set_ylabel('Mean latitude [deg]')

ax[1].set_xlabel(r'Asymmetry')
ax[1].yaxis.tick_right()

ax[0].set_ylim([5, 25])
ax[1].set_ylim([5, 25])
ax[1].set_xlim([-0.05, 0.5])
ax[0].set_xlim([2, 12])

gridcolor = '#bfbfbf'
ax[0].grid(color=gridcolor)
ax[1].grid(color=gridcolor)

ax[2].yaxis.tick_right()
ax[2].set_ylabel('Number of Spots')
ax[2].yaxis.set_label_position("right")

dropbox_path = '/astro/users/bmmorris/Dropbox/Apps/ShareLaTeX/STSP_HAT-P-11/'
fig.savefig(os.path.join(dropbox_path, 'figures', 'sun_vs_hat11.png'),
            bbox_inches='tight', dpi=600)
plt.show()



