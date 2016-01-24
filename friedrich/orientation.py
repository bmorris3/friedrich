
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
import batman
from astropy.coordinates import SphericalRepresentation, CartesianRepresentation
import astropy.units as u
from astropy.constants import G

def impact_parameter(transit_params):
    """
    Calculate impact parameter of transit from other transit parameters. From
    Winn 2010, Eqn 7 [1]_

    Parameters
    ----------
    transit_params : `~batman.TransitParams`
        Transit light curve parameters

    Returns
    -------
    b : float
        Impact parameter

    .. [1] http://adsabs.harvard.edu/abs/2010arXiv1001.2010W
    """
    e = transit_params.ecc  # eccentricity
    w = transit_params.w    # long. of pericenter [deg]
    a_on_Rs = transit_params.a  # a/R_s
    i = transit_params.inc  # inclination [deg]

    b = (a_on_Rs * np.cos(np.radians(i)) *
         (1 - e**2) / (1 + e*np.sin(np.radians(w))))
    return b


def r_orbit(true_anomaly, transit_params):
    """
    Calculate the sky-projected distance from the star to the planet from
    Murray & Correia 2010 Eqn. 19 [1]_

    Parameters
    ----------
    true_anomaly : `numpy.ndarray`
        True anomaly [radians]
    transit_params : `~batman.TransitParams`
        Transit light curve parameters

    Returns
    -------
    r : float
        Sky-projected distance from stellar center to planet center

    .. [1] http://arxiv.org/abs/1009.1738
    """
    a_on_Rs = transit_params.a  # a/R_s
    e = transit_params.ecc  # eccentricity

    r = a_on_Rs * (1 - e**2) / (1 + e*np.cos(true_anomaly))
    return r


def planet_position_cartesian(times, transit_params):
    """
    Calculate planet position in X, Y sky plane cartesian coordinates from
    times (via the true anomaly) using Winn 2011, Eqn. 2-4 [1]_. Do not return
    planet positions when behind the star.

    Parameters
    ----------
    times : `numpy.ndarray`
        Times in JD
    transit_params : `~batman.TransitParams`
        Transit light curve parameters

    Returns
    -------
    X : `numpy.ndarray`
        X sky-plane position
    Y : `numpy.ndarray`
        Y sky-plane position
    .. [1] http://arxiv.org/abs/1001.2010
    """
    f = true_anomaly(times, transit_params)
    r = r_orbit(f, transit_params)
    w = np.radians(transit_params.w) # This pi offset was required by Rodrigo also
    i = np.radians(transit_params.inc)

    X = -r*np.cos(w + f)
    Y = -r*np.sin(w + f) * np.cos(i)
    Z = r*np.sin(w + f) * np.sin(i)

    # Mask out planet positions when behind star
    planet_between_star_and_observer = Z > 0
    planet_behind_star = (X**2 + Y**2 < 1) & ~planet_between_star_and_observer

    return (X[~planet_behind_star], Y[~planet_behind_star],
            Z[~planet_behind_star])


def true_anomaly(times, transit_params):
    """
    Convert time array to array of true anomalies.

    Parameters
    ----------
    times : `numpy.ndarray`
        Times [JD]

    Returns
    -------
    f : `numpy.ndarray`
        True anomalies at `times`

    Notes
    -----
    """
    m = batman.TransitModel(transit_params, times)
    f = m.get_true_anomaly()
    return f


def unit_circle(theta):
    """
    Make a unit circle for plotting in cartesian coords.

    Parameters
    ----------
    theta : `numpy.ndarray`
        True longitude [radians]

    Returns
    -------
    x : `numpy.ndarray`
        x position
    y : `numpy.ndarray`
        y position
    """
    x = np.cos(theta)
    y = np.sin(theta)
    return x, y


def R_x(x, y, z, alpha):
    """
    Rotation matrix for rotation about the x axis

    Parameters
    ----------
    x : `np.ndarray` with dims (1, N)
    y : `np.ndarray` with dims (1, N)
    z : `np.ndarray` with dims (1, N)
    alpha : float
        angle [radians] to rotate about the `x` axis counterclockwise

    Returns
    -------
    x2 : `np.ndarray` with dims (1, N)
        Rotated x positions
    y2 : `np.ndarray` with dims (1, N)
        Rotated y positions
    z2 : `np.ndarray` with dims (1, N)
        Rotated z positions
    """
    original_shape = x.shape
    xyz = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    r_x = np.array([[1, 0, 0],
                    [0, np.cos(alpha), np.sin(alpha)],
                    [0, -np.sin(alpha), np.cos(alpha)]])
    new_xyz = np.dot(r_x, xyz)
    x2, y2, z2 = np.vsplit(new_xyz, 3)
    x2.resize(original_shape)
    y2.resize(original_shape)
    z2.resize(original_shape)
    return x2, y2, z2


def R_y(x, y, z, alpha=0):
    """
    Rotation matrix for rotation about the y axis

    Parameters
    ----------
    x : `np.ndarray` with dims (1, N)
    y : `np.ndarray` with dims (1, N)
    z : `np.ndarray` with dims (1, N)
    alpha : float
        angle [radians] to rotate about the `y` axis counterclockwise

    Returns
    -------
    x2 : `np.ndarray` with dims (1, N)
        Rotated x positions
    y2 : `np.ndarray` with dims (1, N)
        Rotated y positions
    z2 : `np.ndarray` with dims (1, N)
        Rotated z positions
    """
    original_shape = x.shape
    xyz = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    r_y = np.array([[np.cos(alpha), 0, -np.sin(alpha)],
                    [0, 1, 0],
                    [np.sin(alpha), 0, np.cos(alpha)]])
    new_xyz = np.dot(r_y, xyz)
    x2, y2, z2 = np.vsplit(new_xyz, 3)
    x2.resize(original_shape)
    y2.resize(original_shape)
    z2.resize(original_shape)
    return x2, y2, z2


def R_z(x, y, z, alpha=0):
    """
    Rotation matrix for rotation about the z axis

    Parameters
    ----------
    x : `np.ndarray` with dims (1, N)
    y : `np.ndarray` with dims (1, N)
    z : `np.ndarray` with dims (1, N)
    alpha : float
        angle [radians] to rotate about the `z` axis counterclockwise

    Returns
    -------
    x2 : `np.ndarray` with dims (1, N)
        Rotated x positions
    y2 : `np.ndarray` with dims (1, N)
        Rotated y positions
    z2 : `np.ndarray` with dims (1, N)
        Rotated z positions
    """
    original_shape = x.shape
    xyz = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    r_z = np.array([[np.cos(alpha), np.sin(alpha), 0],
                    [-np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1]])
    new_xyz = np.dot(r_z, xyz)
    x2, y2, z2 = np.vsplit(new_xyz, 3)
    x2.resize(original_shape)
    y2.resize(original_shape)
    z2.resize(original_shape)
    return x2, y2, z2


#def sky_cartesian_to_spherical_polar(X, Y, Z):
#    return R_z(*R_x(X, Y, Z, alpha=np.pi/2), alpha=np.pi/2)


def planet_pos_to_stellar_surf(X, Y, Z):
    """
    Project planet position in cartesian coords to projected stellar
    surface position (for observer at infinity) in polar coords

    Returns
    -------

    """

    # First check if planet is on the stellar surface:
    #if (X**2 + Y**2) >= 1:
    #    return np.nan, np.nan, np.nan

    # Project planet position onto x-y sky plane, solve for z on stellar surface
    z_s_surface = np.sqrt(1 - X**2 - Y**2)

    # Convert (x,y,z)_s stellar surface position from cartesian -> polar

    x_r, y_r, z_r = R_z(*R_x(X, Y, z_s_surface, alpha=np.pi/2), alpha=np.pi/2)

    r = 1.0
    theta = np.arctan2(y_r, x_r)
    phi = np.arccos(z_r/r)

    return r, theta, phi


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z/r)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    """

    Parameters
    ----------
    r
    theta
    phi

    Returns
    -------

    """
    x = r*np.cos(theta) * np.sin(phi)
    y = r*np.sin(theta) * np.sin(phi)
    z = r*np.cos(phi)
    return x, y, z


def pysyzygy_example():
    """
    Example for comparison with pysyzygy [1]_

    Returns
    -------
    params : `~batman.TransitParams`
        Transit parameters for example

    .. [1] https://github.com/rodluger/pysyzygy
    """
    eccentricity = 0.7
    omega = 350.0 # Needed to make plot like rodrigo

    params = batman.TransitParams()
    params.t0 = 0   # time of inferior conjunction
    params.per = 1        # orbital period
    params.rp = 0.1       # planet radius (in units of stellar radii)
    #b = 0.127                      # impact parameter
    dur = 0.05                   # transit duration
    params.inc = 65.0     # orbital inclination (in degrees)

    P = params.per*u.day
    rho_s = 1.0*u.g/u.cm**3
    a = (rho_s*G*P**2 / (3 * np.pi))**(1./3)

    params.ecc = eccentricity      # eccentricity
    params.w = omega               # longitude of periastron (in degrees)
    params.a = float(a)          # semi-major axis (in units of stellar radii)

    params.u = [0.5636, 0.1502]    # limb darkening coefficients
    params.limb_dark = "quadratic" # limb darkening model

    # Required by some friedrich methods below but not by batman:
    params.duration = dur
    params.lam = 106.0            # Sanchis-Ojeda & Winn 2011 (soln 1)
    params.inc_stellar = 80.0     # Sanchis-Ojeda & Winn 2011 (soln 1)

    # params.lam = 121.0            # Sanchis-Ojeda & Winn 2011 (soln 2)
    # params.inc_stellar = 168.0    # Sanchis-Ojeda & Winn 2011 (soln 2)
    return params


def plot_lat_lon_gridlines(axis, lat_lon, plot_x_axis, plot_y_axis, plot_color_axis, flip_sign=1):
    #[lat_lon_x, lat_lon_y, lat_lon_z] = lat_lon

    scale_color = ((lat_lon[plot_color_axis] - lat_lon[plot_color_axis].min()) /
                   lat_lon[plot_color_axis].ptp())
    colors = plt.cm.winter(scale_color.ravel())
    #ax[0, 0].plot(lat_x.ravel(), lat_y.ravel(), ls='-',  color=colors)
    #ax[0, 0].plot(lon_x.ravel(), lon_y.ravel(), ls='-',  color=colors)

    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    x, y = lat_lon[plot_x_axis].ravel(), lat_lon[plot_y_axis].ravel()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cmap = ListedColormap(['none', 'silver'][::flip_sign])
    norm = BoundaryNorm([-1, 0, 1], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)

    lc.set_array(lat_lon[plot_color_axis].ravel())
    lc.set_linewidth(1)
    axis.add_collection(lc)


def get_lat_lon_grid(n_points, transit_params):
    #n_points = 31#11
    print("lat grid spacing: {0} deg".format(180./(n_points-1)))
    pi = np.pi

    # Longitudes
    lon_x, lon_y, lon_z = spherical_to_cartesian(r=np.ones(n_points),
                                                 theta=np.linspace(0, 2*np.pi, n_points),
                                                 phi=np.linspace(0, np.pi, n_points)[:, np.newaxis])

    # Latitudes
    lat_x, lat_y, lat_z = spherical_to_cartesian(r=np.ones(n_points),
                                                 theta=np.linspace(0, 2*np.pi, n_points)[:, np.newaxis],
                                                 phi=(np.linspace(0, np.pi, n_points) + np.zeros((n_points, n_points))))
    # Flip 90 deg to show gridlines for aligned system
    # lon_x, lon_y, lon_z = R_y(*R_x(lon_x, lon_y, lon_z, alpha=np.pi/2), alpha=np.pi/2)
    # lat_x, lat_y, lat_z = R_y(*R_x(lat_x, lat_y, lat_z, alpha=np.pi/2), alpha=np.pi/2)

    # Set i_s (Fabrycky & Winn 2009)
    i_star = np.radians(transit_params.inc_stellar)    # i_s in Fabrycky & Winn (2009)
    lam_star = np.radians(transit_params.lam)          # lambda in Fabrycky & Winn (2009)

    # lon_x, lon_y, lon_z = R_z(*R_x(lon_x, lon_y, lon_z,
    #                                alpha=i_star - np.pi/2),
    #                           alpha=lam_star)
    # lat_x, lat_y, lat_z = R_z(*R_x(lat_x, lat_y, lat_z,
    #                                alpha=i_star),
    #                           alpha=lam_star)

    lon_x, lon_y, lon_z = R_z(*R_x(lon_x, lon_y, lon_z,
                                   alpha=i_star),
                              alpha=lam_star)
    lat_x, lat_y, lat_z = R_z(*R_x(lat_x, lat_y, lat_z,
                                   alpha=i_star),
                              alpha=lam_star)

    return [lat_x, lat_y, lat_z], [lon_x, lon_y, lon_z]

if __name__ == '__main__':
    from lightcurve import hat11_params_morris
    from fitting import generate_lc
    thetas = np.linspace(0, 2*np.pi, 10000)
    transit_params = hat11_params_morris()  # pysyzygy_example() #
    #transit_params.inc = 87.0

    times = np.linspace(transit_params.t0 - 0.07, transit_params.t0 + 0.07, 1000)
    #times = np.linspace(transit_params.t0 - 0.07, transit_params.t0 + transit_params.per, 100)

    plot_pos_times = np.linspace(transit_params.t0 - 0.07, transit_params.t0 + 0.07, 20)

    X, Y, Z = planet_position_cartesian(plot_pos_times, transit_params)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].plot(*unit_circle(thetas))
    ax[0, 1].plot(*unit_circle(thetas), 'b')
    ax[1, 1].plot(*unit_circle(thetas), 'b')

    model_lc = generate_lc(times, transit_params)
    ax[1, 0].plot(times, model_lc)
    ax[1, 0].plot(plot_pos_times, generate_lc(plot_pos_times, transit_params),
                  'r.')
    ax[1, 0].set(xlim=[times.min(), times.max()], ylim=[model_lc.min()*0.999,
                                                        model_lc.max()*1.001],
                 xlabel='Time [JD]', ylabel='Flux')

    # Plot gridlines
    [lat_x, lat_y, lat_z], [lon_x, lon_y, lon_z] = get_lat_lon_grid(31, transit_params)

    plot_lat_lon_gridlines(ax[0, 0], [lat_x, lat_y, lat_z],
                           plot_x_axis=0, plot_y_axis=1, plot_color_axis=2)
    plot_lat_lon_gridlines(ax[0, 0], [lon_x, lon_y, lon_z],
                           plot_x_axis=0, plot_y_axis=1, plot_color_axis=2)

    plot_lat_lon_gridlines(ax[0, 1], [lat_x, lat_y, -lat_z],
                           plot_x_axis=0, plot_y_axis=2, plot_color_axis=1)
    plot_lat_lon_gridlines(ax[0, 1], [lon_x, lon_y, -lon_z],
                           plot_x_axis=0, plot_y_axis=2, plot_color_axis=1)

    plot_lat_lon_gridlines(ax[1, 1], [lat_x, lat_y, lat_z],
                           plot_x_axis=0, plot_y_axis=2, plot_color_axis=1,
                           flip_sign=-1)
    plot_lat_lon_gridlines(ax[1, 1], [lon_x, lon_y, lon_z],
                           plot_x_axis=0, plot_y_axis=2, plot_color_axis=1,
                           flip_sign=-1)

    cmap = plt.cm.winter
    for i, x, y, z in zip(range(len(X)), X, Y, Z):
        circle = plt.Circle((x, y), radius=transit_params.rp, alpha=1,
                            color=cmap(float(i)/len(X)))#'k')
        c = ax[0, 0].add_patch(circle)
        c.set_zorder(20)

    ax[0, 0].set(xlabel='$x / R_s$', ylabel='$y / R_s$', title="Observer view",
                 xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])

    ax[0, 1].set(xlabel='$x / R_s$', ylabel='$-z / R_s$',
                 xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], title="Top view")

    ax[1, 1].set(xlabel='$x / R_s$', ylabel='$z / R_s$',
                 xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], title="Bottom view")

    ax[0, 0].set_aspect('equal')

    # Plot projected onto stellar surface
    spot_x = X
    spot_y = Y
    spot_z = np.sqrt(1 - X**2 - Y**2)

    spot_r, spot_theta, spot_phi = cartesian_to_spherical(spot_x, spot_y, spot_z)
    print(np.degrees(spot_r), np.degrees(spot_theta), np.degrees(spot_phi))

    ax[0, 0].scatter(spot_x, spot_y, color='g')
    ax[0, 1].scatter(spot_x[spot_y > 0], -spot_z[spot_y > 0], color='g')
    ax[1, 1].scatter(spot_x[spot_y < 0], spot_z[spot_y < 0], color='g')

    # Orbit view:
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(*unit_circle(thetas))
    # for i, x, y, z in zip(range(len(X)), X, Y, Z):
    #     circle = plt.Circle((x, y), radius=transit_params.rp, alpha=1,
    #                         color=cmap(float(i)/len(X)))#'k')
    #     c = ax.add_patch(circle)
    #     c.set_zorder(20)
    # ax.set_aspect('equal')
    #
    # ax.set(xlabel='$x / R_s$', ylabel='$y / R_s$', title="Observer view")

    plt.show()
