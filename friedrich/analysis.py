# Licensed under the MIT License - see LICENSE.rst
"""
Methods for analyzing results from `friedrich.fitting.run_emcee_seeded`.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from .fitting import spotted_transit_model, spotted_transit_model_individuals, generate_lc
from .storage import read_results_archive
from .lightcurve import TransitLightCurve
from .orientation import (planet_position_cartesian, observer_view_to_stellar_view,
                          cartesian_to_spherical, spherical_to_latlon,
                          project_planet_to_stellar_surface)
from .orientation import (true_anomaly, plot_lat_lon_gridlines, observer_view_to_stellar_view,
                          unit_circle, cartesian_to_spherical, spherical_to_cartesian,
                          get_lat_lon_grid, times_to_occulted_lat_lon)

from astroML.plotting import plot_tissot_ellipse

import matplotlib.pyplot as plt
try:
    from corner import corner
except ImportError:
    pass
import numpy as np
import os


class MCMCResults(object):
    """
    Visualize results from `friedrich.fitting.run_emcee_seeded`
    """
    def __init__(self, archive_path, transit_params):
        """
        Parameters
        ----------
        archive_path : str
            Path to HDF5 archive written by
            `friedrich.storage.create_results_archive`
        """
        results = read_results_archive(archive_path)
        self.lnprob, self.best_params, self.chains, lc_matrix = results

        self.lc = TransitLightCurve(times=lc_matrix[0, :],
                                    fluxes=lc_matrix[1, :],
                                    errors=lc_matrix[2, :])

        self.index = archive_path.split(os.sep)[-1].split('.')[0]
        self.transit_params = transit_params

    def plot_lnprob(self):
        """
        Plot the log-probability of the chains as a function of step number.
        """
        plt.figure()
        plt.title('$\log \,p$')
        plt.plot(self.lnprob)
        plt.xlabel('Step')
        plt.ylabel('$\log \,p$')

    def plot_corner(self, skip_every=100):
        """
        Make a corner plot using `~corner.corner`. Rather than plotting all steps
        of the chain, skip some of the points.

        Parameters
        ----------
        skip_every : int (optional)
            Skip every `skip_every` steps when making the corner plot. Default
            is 100.

        """
        labels = ['depth']
        for i in range(self.chains.shape[0]):
            labels.extend(['$a_{0}$'.format(i), '$t_{{0,{0}}}$'.format(i),
                           '$\sigma_{0}$'.format(i)])

        corner(self.chains[::skip_every, :], labels=labels)

    def plot_max_lnp_lc(self):
        """
        Plot the maximum likelihood transit+spots model over the data.

        Parameters
        ----------
        self.transit_params : `~batman.TransitParams`
            Transit light curve parameters

        """
        model = spotted_transit_model(self.best_params, self.lc.times.jd,
                                      self.transit_params)
        individual_models = spotted_transit_model_individuals(self.best_params,
                                                              self.lc.times.jd,
                                                              self.transit_params)

        errorbar_props = dict(fmt='.', color='k', capsize=0, ecolor='gray')

        fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        min_jd_int = int(self.lc.times.jd.min())
        ax[0].errorbar(self.lc.times.jd - min_jd_int, self.lc.fluxes,
                       self.lc.errors, **errorbar_props)
        ax[0].plot(self.lc.times.jd - min_jd_int, model, 'r', lw='3')

        for individual_model in individual_models:
            ax[0].plot(self.lc.times.jd - min_jd_int, individual_model, 'b')

        ax[1].axhline(0, color='k', ls='--')
        ax[1].errorbar(self.lc.times.jd - min_jd_int, self.lc.fluxes - model,
                       self.lc.errors, **errorbar_props)

        ax[1].set_xlabel('JD - {0}'.format(min_jd_int))
        ax[1].set_ylabel('Residuals')
        ax[0].set_ylabel('Flux')


    def plot_lat_lon(self):
        # times = self.chains[:, 2::3]
        # for i in range(times.shape[1]):
        #     time = times[:, i]


        # plt.plot(self.chains[:, 2::3])


        time = self.chains[:, 2]
        #corner(self.chains[:, 2::3])
        X, Y, Z = planet_position_cartesian(time, self.transit_params)
        spot_x, spot_y, spot_z = project_planet_to_stellar_surface(X, Y)
        spot_x_s, spot_y_s, spot_z_s = observer_view_to_stellar_view(spot_x, spot_y, spot_z, self.transit_params, time)
        spot_r, spot_theta, spot_phi = cartesian_to_spherical(spot_x_s, spot_y_s, spot_z_s)
        latitude, longitude = spherical_to_latlon(spot_r, spot_theta, spot_phi)

        plt.figure()
        plt.plot(longitude, latitude, 'k.', alpha=0.01)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # plt.figure()
        # n, bins = np.histogram(latitude[~np.isnan(latitude)], 200)
        # bincenters = 0.5*(bins[1:] + bins[:-1])
        # plt.plot(bincenters, n, ls='steps')
        # plt.title('lat')
        #
        # plt.figure()
        # n, bins = np.histogram(longitude[~np.isnan(longitude)], 200)
        # bincenters = 0.5*(bins[1:] + bins[:-1])
        # plt.plot(bincenters, n, ls='steps')
        # plt.title('lon')



    def plot_star(self):

        thetas = np.linspace(0, 2*np.pi, 10000)
        #self.transit_params.inc = 87.0

        t0 = self.lc.times.jd.mean()
        spot_times = self.chains[::500, 2::3].T
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        ax[0].plot(*unit_circle(thetas))

        # Plot gridlines
        [lat_x, lat_y, lat_z], [lon_x, lon_y, lon_z] = get_lat_lon_grid(31, self.transit_params)

        plot_lat_lon_gridlines(ax[0], [lat_x, lat_y, lat_z],
                               plot_x_axis=0, plot_y_axis=1, plot_color_axis=2)
        plot_lat_lon_gridlines(ax[0], [lon_x, lon_y, lon_z],
                               plot_x_axis=0, plot_y_axis=1, plot_color_axis=2)

        for t in spot_times:
            #times = t
            times = np.linspace(t0 - 0.07, t0 + 0.07, 1000)
            #times = np.linspace(self.transit_params.t0 - 0.07, self.transit_params.t0 + self.transit_params.per, 100)

            plot_pos_times = t #np.linspace(self.transit_params.t0 - 0.07,
                             #            self.transit_params.t0 + 0.07, 40)

            model_lc = generate_lc(times, self.transit_params)
            ax[1].plot(times, model_lc)
            ax[1].plot(plot_pos_times, generate_lc(plot_pos_times, self.transit_params),
                          'ro')

            X, Y, Z = planet_position_cartesian(plot_pos_times, self.transit_params)
            spot_x, spot_y, spot_z = project_planet_to_stellar_surface(X, Y)
            spot_x_s, spot_y_s, spot_z_s = observer_view_to_stellar_view(spot_x, spot_y, spot_z, self.transit_params, t)
            spot_r, spot_theta, spot_phi = cartesian_to_spherical(spot_x_s, spot_y_s, spot_z_s)

            cmap = plt.cm.winter
            for i, x, y, z in zip(range(len(X)), X, Y, Z):
                circle = plt.Circle((x, y), radius=self.transit_params.rp, alpha=1,
                                    color=cmap(float(i)/len(X)))#'k')
                c = ax[0].add_patch(circle)
                c.set_zorder(20)

            ax[0].scatter(spot_x, spot_y, color='g')

        ax[0].set(xlabel='$x / R_s$', ylabel='$y / R_s$', title="Observer view",
                     xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])

        ax[0].set_aspect('equal')

    def plot_star_projected(self):
        # projections: ['Hammer', 'Aitoff', 'Mollweide', 'Lambert']
        projection = 'Hammer'

        # Plot the built-in projections
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(211, projection=projection.lower())
        ax2 = plt.subplot(212)

        # plot latitude/longitude grid
        ax.xaxis.set_major_locator(plt.FixedLocator(np.pi / 3
                                                    * np.linspace(-2, 2, 5)))
        ax.xaxis.set_minor_locator(plt.FixedLocator(np.pi / 6
                                                    * np.linspace(-5, 5, 11)))
        ax.yaxis.set_major_locator(plt.FixedLocator(np.pi / 6
                                                    * np.linspace(-2, 2, 5)))
        ax.yaxis.set_minor_locator(plt.FixedLocator(np.pi / 12
                                                    * np.linspace(-5, 5, 11)))

        ax.grid(True, which='minor', color='gray', ls=':')
        ax.set_title(self.index)

        # plot transit path
        in_transit_times = self.lc.mask_out_of_transit(self.transit_params, oot_duration_fraction=0)['times'].jd
        # transit_chord_X, transit_chord_Y, transit_chord_Z = planet_position_cartesian(in_transit_times, self.transit_params)
        # transit_chord_x, transit_chord_y, transit_chord_z = project_planet_to_stellar_surface(transit_chord_X, transit_chord_Y)
        # transit_chord_x_s, transit_chord_y_s, transit_chord_z_s = observer_view_to_stellar_view(transit_chord_x,
        #                                                                                         transit_chord_y,
        #                                                                                         transit_chord_z,
        #                                                                                         self.transit_params,
        #                                                                                         in_transit_times)
        # transit_chord_r, transit_chord_theta, transit_chord_phi = cartesian_to_spherical(transit_chord_x_s,
        #                                                                                  transit_chord_y_s,
        #                                                                                  transit_chord_z_s)

        latitude, longitude = times_to_occulted_lat_lon(in_transit_times, self.transit_params)

        ax.scatter(longitude, latitude, color='k', s=0.7, alpha=0.5)


        # plot tissot ellipses for samples from the gaussian spots
        skip_every = 20000  # plots 50 ellipses per spot
        times = self.chains[::skip_every, 2::3].T
        amplitudes = self.chains[::skip_every, 1::3].T
        n_spots = times.shape[0]
        colors = ['b', 'g', 'r']
        while n_spots > len(colors):
            colors.extend(colors)

        for i, time, amplitude in zip(range(len(times)), times, amplitudes):
            X, Y, Z = planet_position_cartesian(time, self.transit_params)
            spot_x, spot_y, spot_z = project_planet_to_stellar_surface(X, Y)
            spot_x_s, spot_y_s, spot_z_s = observer_view_to_stellar_view(spot_x, spot_y, spot_z, self.transit_params, time)
            spot_r, spot_theta, spot_phi = cartesian_to_spherical(spot_x_s, spot_y_s, spot_z_s)

            longitude = spot_theta
            latitude = np.pi/2 - spot_phi

            alpha = np.median(amplitude)/self.transit_params.rp**2/25
            radius = 2*self.transit_params.rp  # from s=r*theta
            plot_tissot_ellipse(longitude, latitude, radius, ax=ax, linewidth=0,
                                fc=colors[i], alpha=alpha)

        # plot transit+spots model
        model = spotted_transit_model(self.best_params, self.lc.times.jd,
                                      self.transit_params)
        individual_models = spotted_transit_model_individuals(self.best_params,
                                                              self.lc.times.jd,
                                                              self.transit_params)

        errorbar_props = dict(fmt='.', color='k', capsize=0, ecolor='gray')


        min_jd_int = int(self.lc.times.jd.min())
        ax2.errorbar(self.lc.times.jd - min_jd_int, self.lc.fluxes,
                       self.lc.errors, **errorbar_props)
        ax2.plot(self.lc.times.jd - min_jd_int, model, 'r', lw='3')

        for individual_model in individual_models:
            ax2.plot(self.lc.times.jd - min_jd_int, individual_model, 'b')

        ax2.set_xlabel('JD - {0}'.format(min_jd_int))
        ax2.set_ylabel('Flux')


class Measurement(object):
    """
    Store a measurement from posterior distribution function with
    upper and lower errorbars
    """
    def __init__(self, value, upper, lower):
        self.value = value
        self.upper = upper
        self.lower = lower

    def __repr__(self):
        return "{0.value} +{0.upper} -{0.lower}".format(self)


class Spot(object):
    """
    Store collection of Measurements for spot parameters
    """
    def __init__(self, amplitude, t0, sigma):
        self.amplitude = amplitude
        self.sigma = sigma
        self.t0 = t0


class Transit(object):
    """
    Store a collection of spots
    """
    def __init__(self, spot_list):
        self.spots = spot_list


