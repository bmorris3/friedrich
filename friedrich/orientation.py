
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
import batman
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
    w = np.radians(transit_params.w)
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
    sort = np.argsort(times)
    unsort = np.argsort(sort)
    times_sorted = np.array(times)[sort]

    m = batman.TransitModel(transit_params, times_sorted)
    f = m.get_true_anomaly()[unsort]
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
    Rotate the X-axis counter-clockwise by ``alpha``  when looking towards the
    origin

    Parameters
    ----------
    x : `np.ndarray` with dims (1, N)
    y : `np.ndarray` with dims (1, N)
    z : `np.ndarray` with dims (1, N)
    alpha : float
        angle [radians] to rotate

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
    Rotate the Y-axis counter-clockwise by ``alpha``  when looking towards the
    origin

    Parameters
    ----------
    x : `np.ndarray` with dims (1, N)
    y : `np.ndarray` with dims (1, N)
    z : `np.ndarray` with dims (1, N)
    alpha : float
        angle [radians] to rotate

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
    Rotate the Z-axis counter-clockwise by ``alpha``  when looking towards the
    origin

    Parameters
    ----------
    x : `np.ndarray` with dims (1, N)
    y : `np.ndarray` with dims (1, N)
    z : `np.ndarray` with dims (1, N)
    alpha : float
        angle [radians] to rotate

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
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

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


def spherical_to_latlon(r, theta, phi):
    latitude = 90 - np.degrees(phi)
    longitude = np.degrees(theta)
    return latitude, longitude


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
    params.lam = 0.0            # Sanchis-Ojeda & Winn 2011 (soln 1)
    params.inc_stellar = 0.0     # Sanchis-Ojeda & Winn 2011 (soln 1)

    # params.lam = 121.0            # Sanchis-Ojeda & Winn 2011 (soln 2)
    # params.inc_stellar = 168.0    # Sanchis-Ojeda & Winn 2011 (soln 2)
    return params


def plot_lat_lon_gridlines(axis, lat_lon, plot_x_axis, plot_y_axis, plot_color_axis, flip_sign=1):
    #[lat_lon_x, lat_lon_y, lat_lon_z] = lat_lon

    scale_color = ((lat_lon[plot_color_axis] - lat_lon[plot_color_axis].min()) /
                   lat_lon[plot_color_axis].ptp())
    colors = plt.cm.winter(scale_color.ravel())

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


def project_planet_to_stellar_surface(x, y):
    projected_z = np.sqrt(1 - x**2 - y**2)
    return x, y, projected_z


def observer_view_to_stellar_view(x, y, z, transit_params, times,
                                  stellar_t0=0.0, rotate_star=False):
    # rotate_star: added on Oct 14, 2016 for STSP simulations
    """
    First rotate to remove lambda (rotation about z-axis). Then rotate to
    remove i_s (rotation about x-axis). Then rotate one last time to remove
    stellar rotation with time (rotation about z-axis again).
    """

    i_star = np.radians(transit_params.inc_stellar)
    lam_star = np.radians(transit_params.lam)
    per_rot = transit_params.per_rot
    t_mean = np.mean(times)

    if not rotate_star:
        x_p, y_p, z_p = R_z(*R_x(*R_z(x, y, z, alpha=-lam_star), alpha=-i_star),
                            alpha=-np.pi/2)#(-2*np.pi/per_rot *
                                  # (np.abs(t_mean - transit_params.t0) % per_rot)))
    else:
        print('degrees: ', 2 * np.pi / per_rot * (np.abs(t_mean - transit_params.t0) % per_rot))
        x_p, y_p, z_p = R_z(*R_x(*R_z(x, y, z, alpha=-lam_star), alpha=-i_star),
                            #alpha=(-2*np.pi/per_rot *
                            alpha=(-np.pi/2 + 2 * np.pi / per_rot *
                                   (np.abs(t_mean - transit_params.t0) % per_rot)))

    return x_p, y_p, z_p


def get_lat_lon_grid(n_points, transit_params, transit_view=True):
    """

    Parameters
    ----------
    n_points
    transit_params
    transit_view : bool
        If true, show star as viewed from positive Z axis (transit view), else
        show star as viewed with its rotation axis in the +Z direction (standard
        spherical polar coords)

    Returns
    -------

    """
    #n_points = 31#11
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

    if transit_view:
        lon_x, lon_y, lon_z = R_z(*R_x(lon_x, lon_y, lon_z,
                                       alpha=i_star),
                                  alpha=lam_star)
        lat_x, lat_y, lat_z = R_z(*R_x(lat_x, lat_y, lat_z,
                                       alpha=i_star),
                                  alpha=lam_star)

    return [lat_x, lat_y, lat_z], [lon_x, lon_y, lon_z]


def times_to_occulted_lat_lon(times, transit_params, rotate_star=False):
    """Only works for single time inputs at the moment"""
    # rotate_star: added on Oct 14, 2016 for STSP simulations

    X, Y, Z = planet_position_cartesian(times, transit_params)
    spot_x, spot_y, spot_z = project_planet_to_stellar_surface(X, Y)

    spot_x_s, spot_y_s, spot_z_s = observer_view_to_stellar_view(spot_x, spot_y,
                                                                 spot_z,
                                                                 transit_params,
                                                                 times, rotate_star=rotate_star)
    spot_r, spot_theta, spot_phi = cartesian_to_spherical(spot_x_s,
                                                          spot_y_s, spot_z_s)
    longitude = spot_phi
    latitude = np.pi/2 - spot_theta

    return latitude, longitude


