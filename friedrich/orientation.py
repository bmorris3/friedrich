
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import matplotlib.pyplot as plt


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


def polar_to_cartesian_coords(true_anomaly, transit_params):
    """
    Winn 2011, Eqn. 2-4 [1]_

    Parameters
    ----------
    true_anomaly : `numpy.ndarray`
        True anomaly [radians]
    transit_params : `~batman.TransitParams`
        Transit light curve parameters

    Returns
    -------

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
    planet_behind_star = (X**2 + Y**2 < 1) & -planet_between_star_and_observer

    return X[-planet_behind_star], Y[-planet_behind_star]


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

if __name__ == '__main__':
    from lightcurve import hat11_params_morris
    thetas = np.linspace(0, 2*np.pi, 1000)
    transit_params = hat11_params_morris()
    r = r_orbit(thetas, transit_params)

    X, Y = polar_to_cartesian_coords(thetas, hat11_params_morris())

    X_t0, Y_t0 = polar_to_cartesian_coords([np.pi/2 -
                                            np.radians(transit_params.w)],
                                           hat11_params_morris())

    xc, yc = unit_circle(thetas)
    print(xc)#, yc_upper, yc_lower)
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(xc, yc)
    ax.plot(X, Y, 'bo')
    ax.plot(X_t0, Y_t0, 'ro')
    plt.xlabel('$x / R_s$')
    plt.ylabel('$y / R_s$')
    ax.set_aspect('equal')
    plt.show()
