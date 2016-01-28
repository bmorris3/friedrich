# Licensed under the MIT License - see LICENSE.rst
"""
Methods for analyzing results from `friedrich.fitting.run_emcee_seeded`.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from .fitting import spotted_transit_model, spotted_transit_model_individuals, generate_lc
from .storage import read_results_archive
from .lightcurve import TransitLightCurve, hat11_params_morris
from .orientation import (planet_position_cartesian, observer_view_to_stellar_view,
                          cartesian_to_spherical, spherical_to_latlon,
                          project_planet_to_stellar_surface)
from .orientation import (true_anomaly, plot_lat_lon_gridlines, observer_view_to_stellar_view,
                          unit_circle, cartesian_to_spherical, spherical_to_cartesian,
                          get_lat_lon_grid)

from astroML.plotting import plot_tissot_ellipse

import matplotlib.pyplot as plt
from corner import corner
import numpy as np

class MCMCResults(object):
    """
    Visualize results from `friedrich.fitting.run_emcee_seeded`
    """
    def __init__(self, archive_path):
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

    def plot_max_lnp_lc(self, transit_params):
        """
        Plot the maximum likelihood transit+spots model over the data.

        Parameters
        ----------
        transit_params : `~batman.TransitParams`
            Transit light curve parameters

        """
        model = spotted_transit_model(self.best_params, self.lc.times.jd,
                                      transit_params)
        individual_models = spotted_transit_model_individuals(self.best_params,
                                                              self.lc.times.jd,
                                                              transit_params)

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
        transit_params = hat11_params_morris()
        X, Y, Z = planet_position_cartesian(time, transit_params)
        spot_x, spot_y, spot_z = project_planet_to_stellar_surface(X, Y)
        spot_x_s, spot_y_s, spot_z_s = observer_view_to_stellar_view(spot_x, spot_y, spot_z, transit_params)
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
        transit_params = hat11_params_morris() # pysyzygy_example()  #  #
        #transit_params.inc = 87.0

        t0 = self.lc.times.jd.mean()
        spot_times = self.chains[::500, 2::3].T
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        ax[0].plot(*unit_circle(thetas))

        # Plot gridlines
        [lat_x, lat_y, lat_z], [lon_x, lon_y, lon_z] = get_lat_lon_grid(31, transit_params)

        plot_lat_lon_gridlines(ax[0], [lat_x, lat_y, lat_z],
                               plot_x_axis=0, plot_y_axis=1, plot_color_axis=2)
        plot_lat_lon_gridlines(ax[0], [lon_x, lon_y, lon_z],
                               plot_x_axis=0, plot_y_axis=1, plot_color_axis=2)

        for t in spot_times:
            #times = t
            times = np.linspace(t0 - 0.07, t0 + 0.07, 1000)
            #times = np.linspace(transit_params.t0 - 0.07, transit_params.t0 + transit_params.per, 100)

            plot_pos_times = t #np.linspace(transit_params.t0 - 0.07,
                             #            transit_params.t0 + 0.07, 40)

            model_lc = generate_lc(times, transit_params)
            ax[1].plot(times, model_lc)
            ax[1].plot(plot_pos_times, generate_lc(plot_pos_times, transit_params),
                          'ro')


            X, Y, Z = planet_position_cartesian(plot_pos_times, transit_params)
            spot_x, spot_y, spot_z = project_planet_to_stellar_surface(X, Y)
            spot_x_s, spot_y_s, spot_z_s = observer_view_to_stellar_view(spot_x, spot_y, spot_z, transit_params)
            spot_r, spot_theta, spot_phi = cartesian_to_spherical(spot_x_s, spot_y_s, spot_z_s)

            cmap = plt.cm.winter
            for i, x, y, z in zip(range(len(X)), X, Y, Z):
                circle = plt.Circle((x, y), radius=transit_params.rp, alpha=1,
                                    color=cmap(float(i)/len(X)))#'k')
                c = ax[0].add_patch(circle)
                c.set_zorder(20)

            ax[0].scatter(spot_x, spot_y, color='g')

        ax[0].set(xlabel='$x / R_s$', ylabel='$y / R_s$', title="Observer view",
                     xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])

        ax[0].set_aspect('equal')


    #
    # def plot_star(self):
    #
    #     thetas = np.linspace(0, 2*np.pi, 10000)
    #     transit_params = hat11_params_morris() # pysyzygy_example()  #  #
    #     #transit_params.inc = 87.0
    #
    #     t0 = self.lc.times.jd.mean()
    #     spot_times = self.chains[::500, 2::3].T
    #     fig, ax = plt.subplots(2, 3, figsize=(14, 10))
    #
    #
    #     ax[0, 0].plot(*unit_circle(thetas))
    #     ax[0, 1].plot(*unit_circle(thetas), color='b')
    #     ax[1, 1].plot(*unit_circle(thetas), color='b')
    #     ax[0, 2].plot(*unit_circle(thetas), color='b')
    #     ax[1, 2].plot(*unit_circle(thetas), color='b')
    #
    #
    #     # Plot gridlines
    #     [lat_x, lat_y, lat_z], [lon_x, lon_y, lon_z] = get_lat_lon_grid(31, transit_params)
    #
    #     plot_lat_lon_gridlines(ax[0, 0], [lat_x, lat_y, lat_z],
    #                            plot_x_axis=0, plot_y_axis=1, plot_color_axis=2)
    #     plot_lat_lon_gridlines(ax[0, 0], [lon_x, lon_y, lon_z],
    #                            plot_x_axis=0, plot_y_axis=1, plot_color_axis=2)
    #
    #     plot_lat_lon_gridlines(ax[0, 1], [lat_x, lat_y, -lat_z],
    #                            plot_x_axis=0, plot_y_axis=2, plot_color_axis=1)
    #     plot_lat_lon_gridlines(ax[0, 1], [lon_x, lon_y, -lon_z],
    #                            plot_x_axis=0, plot_y_axis=2, plot_color_axis=1)
    #
    #     plot_lat_lon_gridlines(ax[1, 1], [lat_x, lat_y, lat_z],
    #                            plot_x_axis=0, plot_y_axis=2, plot_color_axis=1,
    #                            flip_sign=-1)
    #     plot_lat_lon_gridlines(ax[1, 1], [lon_x, lon_y, lon_z],
    #                            plot_x_axis=0, plot_y_axis=2, plot_color_axis=1,
    #                            flip_sign=-1)
    #
    #     [lat_x, lat_y, lat_z], [lon_x, lon_y, lon_z] = get_lat_lon_grid(31, transit_params, transit_view=False)
    #
    #     plot_lat_lon_gridlines(ax[0, 2], [lat_x, lat_y, lat_z],
    #                            plot_x_axis=0, plot_y_axis=2, plot_color_axis=1)#,
    #                            #flip_sign=-1)
    #     plot_lat_lon_gridlines(ax[0, 2], [lon_x, lon_y, lon_z],
    #                            plot_x_axis=0, plot_y_axis=2, plot_color_axis=1)#,
    #                            #flip_sign=-1)
    #
    #     plot_lat_lon_gridlines(ax[1, 2], [lat_x, lat_y, lat_z],
    #                            plot_x_axis=0, plot_y_axis=2, plot_color_axis=1)#,
    #                            #flip_sign=-1)
    #     plot_lat_lon_gridlines(ax[1, 2], [lon_x, lon_y, lon_z],
    #                            plot_x_axis=0, plot_y_axis=2, plot_color_axis=1)#,
    #                           # flip_sign=-1)
    #
    #     for t in spot_times:
    #         #times = t
    #         times = np.linspace(t0 - 0.07, t0 + 0.07, 1000)
    #         #times = np.linspace(transit_params.t0 - 0.07, transit_params.t0 + transit_params.per, 100)
    #
    #         plot_pos_times = t #np.linspace(transit_params.t0 - 0.07,
    #                          #            transit_params.t0 + 0.07, 40)
    #
    #         model_lc = generate_lc(times, transit_params)
    #         ax[1, 0].plot(times, model_lc)
    #         ax[1, 0].plot(plot_pos_times, generate_lc(plot_pos_times, transit_params),
    #                       'ro')
    #
    #
    #         X, Y, Z = planet_position_cartesian(plot_pos_times, transit_params)
    #         spot_x, spot_y, spot_z = project_planet_to_stellar_surface(X, Y)
    #         spot_x_s, spot_y_s, spot_z_s = observer_view_to_stellar_view(spot_x, spot_y, spot_z, transit_params)
    #         spot_r, spot_theta, spot_phi = cartesian_to_spherical(spot_x_s, spot_y_s, spot_z_s)
    #
    #         cmap = plt.cm.winter
    #         for i, x, y, z in zip(range(len(X)), X, Y, Z):
    #             circle = plt.Circle((x, y), radius=transit_params.rp, alpha=1,
    #                                 color=cmap(float(i)/len(X)))#'k')
    #             c = ax[0, 0].add_patch(circle)
    #             c.set_zorder(20)
    #
    #        # Plot projected onto stellar surface
    #         # spot_x = X
    #         # spot_y = Y
    #         # spot_z = np.sqrt(1 - X**2 - Y**2)
    #
    #         ax[0, 0].scatter(spot_x, spot_y, color='g')
    #         ax[0, 1].scatter(spot_x[spot_y > 0], -spot_z[spot_y > 0], color='g')
    #         ax[1, 1].scatter(spot_x[spot_y < 0], spot_z[spot_y < 0], color='g')
    #
    #
    #
    #         print('from bmm:\ntheta={0}\nphi={1}'.format(np.median(spot_theta), np.median(spot_phi)))
    #         print('for stsp\nr={0}\nphi={1}\ntheta={2}'.format(np.median(spot_r),
    #                                                            np.median(spot_theta) + 2*np.pi,
    #                                                            np.pi/2 - np.median(spot_phi)))
    #         latitude, longitude = spherical_to_latlon(spot_r, spot_theta, spot_phi)
    #
    #         ax[0, 2].scatter(-spot_z_s[spot_x_s > 0], spot_y_s[spot_x_s > 0], color='g')
    #         ax[1, 2].scatter(-spot_z_s[spot_x_s < 0], spot_y_s[spot_x_s < 0], color='g')
    #
    #     # Orbit view:
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(111)
    #     # ax.plot(*unit_circle(thetas))
    #     # for i, x, y, z in zip(range(len(X)), X, Y, Z):
    #     #     circle = plt.Circle((x, y), radius=transit_params.rp, alpha=1,
    #     #                         color=cmap(float(i)/len(X)))#'k')
    #     #     c = ax.add_patch(circle)
    #     #     c.set_zorder(20)
    #     # ax.set_aspect('equal')
    #     #
    #     # ax.set(xlabel='$x / R_s$', ylabel='$y / R_s$', title="Observer view")
    #
    #     ax[0, 0].set(xlabel='$x / R_s$', ylabel='$y / R_s$', title="Observer view",
    #                  xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])
    #
    #     ax[0, 1].set(xlabel='$x / R_s$', ylabel='$-z / R_s$',
    #                  xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], title="Top orbit view")
    #
    #     ax[1, 1].set(xlabel='$x / R_s$', ylabel='$z / R_s$',
    #                  xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], title="Bottom orbit view")
    #
    #     ax[0, 2].set(xlabel='$-z^\\prime / R_s$', ylabel='$y^\\prime / R_s$',
    #                  xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], title="Top view")
    #
    #     ax[1, 2].set(xlabel='$-z^\\prime / R_s$', ylabel='$y^\\prime / R_s$',
    #                  xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], title="Bottom view")
    #
    #     ax[0, 0].set_aspect('equal')
    #     ax[0, 1].set_aspect('equal')
    #     ax[0, 2].set_aspect('equal')
    #     ax[1, 1].set_aspect('equal')
    #     ax[1, 2].set_aspect('equal')


    def plot_star_projected(self):

#        from astroML.plotting import setup_text_plots
#         setup_text_plots(fontsize=14, usetex=True)

        #------------------------------------------------------------
        # generate a latitude/longitude grid
        #circ_long = np.linspace(-np.pi, np.pi, 13)[1:-1]
        #circ_lat = np.linspace(-np.pi / 2, np.pi / 2, 7)[1:-1]
        radius = 10 * np.pi / 180.

        #------------------------------------------------------------
        # Plot the built-in projections
        plt.figure(figsize=(10, 8))
        plt.subplots_adjust(hspace=0, wspace=0.12,
                            left=0.08, right=0.95,
                            bottom=0.05, top=1.0)

        #for (i, projection) in enumerate(['Hammer', 'Aitoff', 'Mollweide', 'Lambert']):
        i = 0
        projection = 'Hammer'
        ax = plt.subplot(111, projection=projection.lower())

        ax.xaxis.set_major_locator(plt.FixedLocator(np.pi / 3
                                                    * np.linspace(-2, 2, 5)))
        ax.xaxis.set_minor_locator(plt.FixedLocator(np.pi / 6
                                                    * np.linspace(-5, 5, 11)))
        ax.yaxis.set_major_locator(plt.FixedLocator(np.pi / 6
                                                    * np.linspace(-2, 2, 5)))
        ax.yaxis.set_minor_locator(plt.FixedLocator(np.pi / 12
                                                    * np.linspace(-5, 5, 11)))

        ax.grid(True, which='minor')

        # plot_tissot_ellipse(circ_long[:, None], circ_lat, radius,
        #                     ax=ax, fc='k', alpha=0.3, linewidth=0)
        ax.set_title('%s projection' % projection)

        transit_params = hat11_params_morris() # pysyzygy_example()  #  #
        #transit_params.inc = 87.0

        t0 = self.lc.times.jd.mean()
#        t = self.chains[::3, 2][::500]
        times = self.chains[::500, 2::3].T
        n_spots = times.shape[0]
        colors = ['b', 'g', 'r']
        while n_spots > len(colors):
            colors.extend(colors)
        for i, time in enumerate(times):
            X, Y, Z = planet_position_cartesian(time, transit_params)
            spot_x, spot_y, spot_z = project_planet_to_stellar_surface(X, Y)
            spot_x_s, spot_y_s, spot_z_s = observer_view_to_stellar_view(spot_x, spot_y, spot_z, transit_params)
            spot_r, spot_theta, spot_phi = cartesian_to_spherical(spot_x_s, spot_y_s, spot_z_s)

            longitude = spot_theta
            latitude = np.pi/2 - spot_phi

            plot_tissot_ellipse(longitude, latitude, radius,
                                ax=ax, fc=colors[i], alpha=0.01, linewidth=0)

