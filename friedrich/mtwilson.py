# Licensed under the MIT License - see LICENSE.rst
"""
Methods for parsing the Mt Wilson "Tilt" sunspot catalog [1] [2]

References
----------
.. [1] http://www.ngdc.noaa.gov/stp/solar/sunspotregionsdata.html
.. [2] Howard, R., Gilman, P.I., & Gilman, P.A. 1984, ApJ, 283, 373
       http://adsabs.harvard.edu/abs/1984ApJ...283..373H
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from glob import glob

import numpy as np

from astropy.table import Table
from astropy.time import Time
from astropy.io import ascii

from .stsp import STSP

__all__ = ['parse_mwo_group_spot', 'get_simulated_solar_transit']


def split_interval(string, n, cast_to_type=float):
    return [cast_to_type(string[i:i+n]) for i in range(0, len(string), n)]


def parse_mwo_group_spot(mwo_dir, table_file_name='mwo_spot_table.txt'):
    """
    Parameters
    ----------
    mwo_dir : str
        Directory containing the Mt Wilson Tilt catalog

    Returns
    -------
    spot_table : `~astropy.table.Table`
        Table of spots
    """
    table_path = os.path.join(mwo_dir, table_file_name)

    if not os.path.exists(table_path):
        #paths = glob('/local/tmp/Mt_Wilson_Tilt/*/gspot??.dat')
        paths = glob(os.path.join(mwo_dir, '*/gspot??.dat'))

        all_years_array = []

        header = ("jd n_spots_leading n_spots_following n_spots_day_1 n_spots_day_2 "
                  "rotation_rate latitude_drift area_weighted_latitude_day_1 area_weighted_longitude_day_1 "
                  "area_weighted_longitude_day_2 area_day_1 area_day_2 tilt_day_1 delta_polarity_separation "
                  "area_weighted_longitude_day_1_leading area_weighted_longitude_day_1_following "
                  "area_weighted_latitude_day_1_leading area_weighted_latitude_day_1_following "
                  "area_leading area_following area_weighted_longitude_day_2_leading "
                  "area_weighted_longitude_day_2_following delta_tilt").split()

        for path in paths:
            f = open(path).read().splitlines()

            n_rows = len(f) // 3
            n_columns = 23
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
        ascii.write(table, table_path)

    else:
        table = ascii.read(table_path)

    return table


class STSPSpot(object):
    def __init__(self, r, theta, phi):
        self.r = r
        self.theta = theta
        self.phi = phi


def spot_list_to_spot_params(spot_list):
    spot_params = []
    for spot in spot_list:
        spot_params.extend([spot.r, spot.theta, spot.phi])
    return spot_params


def find_overlapping_spots(spot_list):
    # starting from MCMCResults.find_overlapping_spots
    overlapping_pairs = []
    spots_with_overlap = []
    for i in range(len(spot_list)):
        for j in range(len(spot_list)):
            if i < j:
                sep = np.arccos(np.cos(spot_list[i].theta) * np.cos(spot_list[j].theta) +
                                np.sin(spot_list[i].theta) * np.sin(spot_list[j].theta) *
                                np.cos(spot_list[i].phi - spot_list[j].phi))
                if sep < 1.01 * (spot_list[i].r + spot_list[j].r):
                    overlapping_pairs.append((i, j))

                    if i not in spots_with_overlap:
                        spots_with_overlap.append(i)
                    if j not in spots_with_overlap:
                        spots_with_overlap.append(j)

    spots_without_overlap = [spot for i, spot in enumerate(spot_list)
                             if i not in spots_with_overlap]
    return overlapping_pairs, spots_with_overlap, spots_without_overlap


def get_spot_params_between_dates(start_time, end_time, mwo_dir,
                                  penumbra_to_umbra_area_ratio=5.0):

    table = parse_mwo_group_spot(mwo_dir=mwo_dir)

    time_range = (table['jd'] < end_time.jd) & (table['jd'] > start_time.jd)
    spots_within_range = table[time_range]

    area = penumbra_to_umbra_area_ratio * 1e-6 * spots_within_range['area_day_1'].data
    radii = np.sqrt(2 * area)

    phis = np.radians(spots_within_range['area_weighted_longitude_day_1'].data)
    phis[phis < 0] += 2*np.pi

    thetas = np.radians(90 - spots_within_range['area_weighted_latitude_day_1'].data)

    spots = [STSPSpot(r, theta, phi) for r, theta, phi in zip(radii, thetas, phis)]

    overlapping_pairs, spots_with_overlap, spots_without_overlap = find_overlapping_spots(spots)
    while len(overlapping_pairs) > 0:

        for spot_pair in overlapping_pairs:
            spot_a = spots[spot_pair[0]]
            spot_b = spots[spot_pair[1]]
            mean_theta = 0.5 * (spot_a.theta + spot_b.theta)
            mean_phi = 0.5 * (spot_a.phi + spot_b.phi)
            total_area = np.pi * (spot_a.r**2 + spot_b.r**2)
            total_r = np.sqrt(total_area/np.pi)

            new_spot = STSPSpot(total_r, mean_theta, mean_phi)
            spots_without_overlap.append(new_spot)

        overlapping_pairs, spots_with_overlap, spots_without_overlap = find_overlapping_spots(spots_without_overlap)

    spot_params = spot_list_to_spot_params(spots_without_overlap)

    if len(spot_params) > 0:
        spot_params = np.array(spot_params)
    else:
        spot_params = np.array([0.00001, 3*np.pi/2, np.pi/2])

    return spot_params


def get_simulated_solar_transit(spot_start_time, spot_end_time, light_curve,
                                transit_params, mwo_dir, stsp_executable):

    spot_params = get_spot_params_between_dates(spot_start_time, spot_end_time, mwo_dir)

    s = STSP(light_curve, transit_params, spot_params)
    stsp_times, stsp_fluxes = s.stsp_lc(stsp_exec=stsp_executable)

    return stsp_times, stsp_fluxes

