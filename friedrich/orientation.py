
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
import batman

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


def planet_position_f_to_xy(true_anomaly, transit_params):
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
    planet_behind_star = (X**2 + Y**2 < 1) & -planet_between_star_and_observer

    return X[-planet_behind_star], Y[-planet_behind_star]


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


def planet_position_t_to_xy(times, transit_params):
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
    return planet_position_f_to_xy(time_to_f(times, transit_params),
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

if __name__ == '__main__':
    from lightcurve import hat11_params_morris
    from fitting import generate_lc
    thetas = np.linspace(0, 2*np.pi, 10000)
    transit_params = hat11_params_morris()


    hat11_params = hat11_params_morris()
    #times = np.linspace(hat11_params.t0 - 0.02, hat11_params.t0 + 0.02, 1000)
    times = np.linspace(hat11_params.t0 - 0.1, hat11_params.t0 + 0.1, 1000)
    plot_pos_times = np.array([hat11_params.t0 - 0.5*hat11_params.duration,
                               hat11_params.t0 + 0.5*hat11_params.duration,
                               hat11_params.t0])

    X, Y = planet_position_t_to_xy(plot_pos_times, hat11_params)

    fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    ax[0].plot(*unit_circle(thetas))

    for x, y in zip(X, Y):
        circle = plt.Circle((x, y), radius=transit_params.rp, alpha=1,
                            color='k')
        ax[0].add_patch(circle)

    # ax.plot(X_t0, Y_t0, 'rs')

    ax[0].set(xlabel='$x / R_s$', ylabel='$y / R_s$',
              xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])
    ax[0].set_aspect('equal')

    model_lc = generate_lc(times, hat11_params)
    ax[1].plot(times, model_lc)
    ax[1].plot(plot_pos_times, generate_lc(plot_pos_times, hat11_params), 'r.')
    ax[1].set(xlim=[times.min(), times.max()], ylim=[model_lc.min()*0.99,
                                                     model_lc.max()*1.01],
              xlabel='Time [JD]', ylabel='Flux')
    plt.show()