
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
import batman
from astropy.coordinates import SphericalRepresentation, CartesianRepresentation
import astropy.units as u


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


def planet_position_f_to_xyz(true_anomaly, transit_params):
    """
    Calculate planet position in X, Y sky plane cartesian coordinates from
    the true anomaly using Winn 2011, Eqn. 2-4 [1]_. Do not return planet
    positions when behind the star.

    Parameters
    ----------
    true_anomaly : `numpy.ndarray`
        True anomaly [radians]
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
    r = r_orbit(true_anomaly, transit_params)
    w = np.radians(transit_params.w)
    i = np.radians(transit_params.inc)

    X = -r*np.cos(w + true_anomaly)
    Y = -r*np.sin(w + true_anomaly) * np.cos(i)
    Z = r*np.sin(w + true_anomaly) * np.sin(i)

    # Mask out planet positions when behind star
    planet_between_star_and_observer = Z > 0
    planet_behind_star = (X**2 + Y**2 < 1) & ~planet_between_star_and_observer

    return (X[~planet_behind_star], Y[~planet_behind_star],
            Z[~planet_behind_star])


def time_to_f(times, transit_params):
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


def planet_position_t_to_xyz(times, transit_params):
    """
    Convert times to planet position in cartesian coordinates

    Parameters
    ----------
    times : `numpy.ndarray`
        Times [JD]
    transit_params : `~batman.TransitParams`
        Transit light curve parameters
    Returns
    -------
    X : `numpy.ndarray`
        X sky-plane position
    Y : `numpy.ndarray`
        Y sky-plane position
    """
    return planet_position_f_to_xyz(time_to_f(times, transit_params),
                                    transit_params)


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


def planet_pos_to_stellar_surf(X, Y, Z):
    """
    Project planet position in cartesian coords to projected stellar
    surface position (for observer at infinity) in polar coords

    Returns
    -------

    """

    # First check if planet is on the stellar surface:
    if (X**2 + Y**2) >= 1:
        return np.nan, np.nan, np.nan

    # Project planet position onto x-y sky plane, solve for z on stellar surface
    z_s_surface = np.sqrt(1 - X**2 - Y**2)

    # Convert (x,y,z)_s stellar surface position from cartesian -> polar

    x_r, y_r, z_r = R_z(*R_x(X, Y, z_s_surface, alpha=np.pi/2), alpha=np.pi/2)

    r = 1.0
    theta = np.arctan2(y_r, x_r)
    phi = np.arccos(z_r/r)

    return r, theta, phi


def stellar_surf_to_xyz(r, theta, phi):
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

if __name__ == '__main__':
    from lightcurve import hat11_params_morris
    from fitting import generate_lc
    thetas = np.linspace(0, 2*np.pi, 10000)
    transit_params = hat11_params_morris()
    transit_params.inc = 90.0

    #times = np.linspace(hat11_params.t0 - 0.02, hat11_params.t0 + 0.02, 1000)
    times = np.linspace(transit_params.t0 - 0.1, transit_params.t0 + 0.1, 1000)
    plot_pos_times = np.array([transit_params.t0,
                               transit_params.t0 - 0.25*transit_params.duration,
                               transit_params.t0 - 0.5*transit_params.duration])

    X, Y, Z = planet_position_t_to_xyz(plot_pos_times, transit_params)

    fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    ax[0].plot(*unit_circle(thetas))

    model_lc = generate_lc(times, transit_params)
    ax[1].plot(times, model_lc)
    ax[1].plot(plot_pos_times, generate_lc(plot_pos_times, transit_params), 'r.')
    ax[1].set(xlim=[times.min(), times.max()], ylim=[model_lc.min()*0.99,
                                                     model_lc.max()*1.01],
              xlabel='Time [JD]', ylabel='Flux')


    # Plot gridlines
    n_gridlines = 9
    print("lat grid spacing: {0} deg".format(180./(n_gridlines-1)))
    n_points = 35
    pi = np.pi

    latitude_lines = SphericalRepresentation(np.linspace(0, 2*pi, n_points)[:, np.newaxis]*u.rad,
                                             np.linspace(-pi/2, pi/2, n_gridlines).T*u.rad,
                                             np.ones((n_points, 1))
                                             ).to_cartesian()

    longitude_lines = SphericalRepresentation(np.linspace(0, 2*pi, n_gridlines)[:, np.newaxis]*u.rad,
                                              np.linspace(-pi/2, pi/2, n_points).T*u.rad,
                                              np.ones((n_gridlines, 1))
                                              ).to_cartesian()
    # Get arrays from quantity
    lat_x = latitude_lines.x.value
    lat_y = latitude_lines.y.value
    lat_z = latitude_lines.z.value
    lon_x = longitude_lines.x.value
    lon_y = longitude_lines.y.value
    lon_z = longitude_lines.z.value

    lat_x, lat_y, lat_z = R_x(lat_x, lat_y, lat_z, -np.pi/2)
    lon_x, lon_y, lon_z = R_x(lon_x, lon_y, lon_z, -np.pi/2)

    ax[0].plot(lat_x, lat_y, ls=':', color='silver')
    ax[0].plot(lon_x.T, lon_y.T, ls=':', color='silver')

    for x, y, z in zip(X, Y, Z):
        circle = plt.Circle((x, y), radius=transit_params.rp, alpha=1,
                            color='k')
        c = ax[0].add_patch(circle)
        c.set_zorder(20)
        print("x={0}, y={1}, z={2}".format(x, y, z))
        print("r={0}, theta={1}, phi={2}".format(
                *planet_pos_to_stellar_surf(x, y, z)))

    # ax.plot(X_t0, Y_t0, 'rs')

    ax[0].set(xlabel='$x / R_s$', ylabel='$y / R_s$',
              xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])
    ax[0].set_aspect('equal')

    spot_x, spot_y, spot_z = stellar_surf_to_xyz(r=1, theta=0, phi=np.pi/4)
    ax[0].scatter(spot_x, spot_y, color='g')


    plt.show()
