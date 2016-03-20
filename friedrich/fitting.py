# Licensed under the MIT License - see LICENSE.rst
"""
Methods for fitting transit light curves, spot occultations, or both, using
`scipy` minimizers and `emcee`.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import emcee
from scipy import optimize, signal
import matplotlib.pyplot as plt
import batman
from copy import deepcopy
from emcee.utils import MPIPool
import sys


def gaussian(times, amplitude, t0, sigma):
    """
    Gaussian function.

    Parameters
    ----------
    times : `numpy.ndarray`
        Times
    amplitude : float
        Amplitude of gaussian (not normalized)
    t0 : float
        Central time in units of `times`
    sigma : float
        Gaussian width.

    Returns
    -------
    y : `numpy.ndarray`
        Gaussian evaluated at `times`
    """
    return amplitude * np.exp(-0.5*(times - t0)**2/sigma**2)


def peak_finder_chi2(theta, x, y, yerr):
    """
    Chi^2 model given parameters `theta` and data {`x`, `y`, `yerr`}
    Parameters
    ----------
    theta : list
        Trial parameters
    x : `numpy.ndarray`
        Times [JD]
    y : `numpy.ndarray`
        Fluxes
    yerr : `numpy.ndarray`
        Uncertainties on fluxes

    Returns
    -------
    chi2 : float
        Chi^2 of the model
    """
    model = summed_gaussians(x, theta)
    return np.sum((y-model)**2/yerr**2)


def peak_finder(times, residuals, errors, transit_params, n_peaks=4,
                plots=False, verbose=False, skip_priors=False):
    """
    Find peaks in the residuals from a fit to a transit light curve, which
    correspond to starspot occultations.

    Parameters
    ----------
    times : `numpy.ndarray`
        Times [JD]
    residuals : `numpy.ndarray`
        Fluxes
    errors : `numpy.ndarray`
        Uncertainties on residuals
    transit_params : `~batman.TransitParams`
        Transit light curve parameters
    n_peaks : bool (optional)
        Number of peaks to search for. If more than `n_peaks` are found, return
        only the `n_peaks` largest amplitude peaks.
    plots : bool (optional)
        Show diagnostic plots
    verbose : bool (optional)
        Warn if no peaks are found

    Returns
    -------
    result_in_transit : list or `None`
        List of all spot parameters in [amp, t0, sig, amp, t0, sig, ...] order
        for spots detected.

    Notes
    -----
    Review of minimizers tried for `peak_finder`:

    `~scipy.optimize.fmin` gets amplitudes right, but doesn't vary sigmas much.
    For this reason, it tends to do a better job of finding nearby, semi-
    overlapping spots.

    `~scipy.optimize.fmin_powell` varies amplitudes and sigmas lots, but
    as a result, sometimes two nearby spots are fit with one wide gaussian.
    """
    # http://stackoverflow.com/a/25666951
    # Convolve residuals with a gaussian, find relative maxima
    n_points_kernel = 100
    window = signal.general_gaussian(n_points_kernel+1, p=1, sig=3)
    filtered = signal.fftconvolve(window, residuals)
    filtered = (np.max(residuals) / np.max(filtered)) * filtered
    filtered = np.roll(filtered, int(-n_points_kernel/2))[:len(residuals)]

    maxes = signal.argrelmax(filtered)[0]

    # Only take maxima, not minima
    maxes = maxes[filtered[maxes] > 0]

    lower_t_bound, upper_t_bound = get_in_transit_bounds(times, transit_params)
    maxes_in_transit = maxes[(times[maxes] < upper_t_bound) &
                             (times[maxes] > lower_t_bound)]

    # Only take the `n_peaks` highest peaks
    if len(maxes_in_transit) > n_peaks:
        highest_maxes_in_transit = maxes_in_transit[np.argsort(filtered[maxes_in_transit])][-n_peaks:]
    else:
        highest_maxes_in_transit = maxes_in_transit

    # plt.plot(times, filtered)
    # plt.plot(times, residuals, '.')
    # plt.plot(times[maxes_in_transit], filtered[maxes_in_transit], 'ro')
    # [plt.axvline(times[m], color='k') for m in maxes]
    # [plt.axvline(times[m], color='m') for m in maxes_in_transit]
    # if len(maxes_in_transit) > n_peaks:
    #     [plt.axvline(times[m], color='b') for m in highest_maxes_in_transit]
    # plt.axvline(upper_t_bound, color='r')
    # plt.axvline(lower_t_bound, color='r')
    # plt.show()

    if len(maxes_in_transit) == 0:
        if verbose:
            print('no maxes found')
        return None

    peak_times = times[highest_maxes_in_transit]
    peak_amplitudes = residuals[highest_maxes_in_transit]
    peak_sigmas = np.zeros(len(peak_times)) + 2./60/24  # 3 min
    input_parameters = np.vstack([peak_amplitudes, peak_times,
                                  peak_sigmas]).T.ravel()

    result = optimize.fmin_powell(peak_finder_chi2, input_parameters,
                                  disp=False, args=(times, residuals, errors),
                                  xtol=0.00001, ftol=0.00001)

    # if np.all(result == input_parameters):
    #     print('oh no!, fmin didnt produce a fit')

    # Only use gaussians that occur in transit (fmin fit is unbounded in time)
    # and amplitude is positive:
    split_result = np.split(result, len(input_parameters)/3)
    result_in_transit = []
    for amplitude, t0, sigma in split_result:
        depth = transit_params.rp**2

        trial_params = np.array([amplitude, t0, sigma])
        if not np.isinf(lnprior(trial_params, residuals, lower_t_bound,
                                upper_t_bound, transit_params, skip_priors)):
            result_in_transit.extend([amplitude, t0, np.abs(sigma)])
    result_in_transit = np.array(result_in_transit)

    if len(result_in_transit) == 0:
        return None

    if plots:
        fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        ax[0].errorbar(times, residuals, fmt='.', color='k')
        [ax[0].axvline(t) for t in result_in_transit[1::3]]
        ax[0].plot(times, summed_gaussians(times, input_parameters), 'r')
        ax[0].axhline(0, color='k', ls='--')
        ax[0].set_ylabel('Transit Residuals')

        ax[1].errorbar(times, residuals, fmt='.', color='k')
        ax[1].plot(times, summed_gaussians(times, result_in_transit), 'r')
        ax[1].axhline(0, color='k', ls='--')
        ax[1].set_ylabel('Residuals')

        ax[2].errorbar(times,
                       residuals - summed_gaussians(times, result_in_transit),
                       fmt='.', color='k')
        #ax[1].errorbar(times, gaussian_model, fmt='.', color='r')
        ax[2].axhline(0, color='k', ls='--')
        ax[2].set_ylabel('Residuals')

        for axis in ax:
            axis.axvline(upper_t_bound, color='r')
            axis.axvline(lower_t_bound, color='r')

        fig.tight_layout()
        plt.show()

    return result_in_transit


def generate_lc(times, transit_params):
    """
    Make a transit light curve.

    Parameters
    ----------
    times : `numpy.ndarray`
        Times in JD
    transit_params : `~batman.TransitParams`
        Transit light curve parameters

    Returns
    -------
    model_flux : `numpy.ndarray`
        Fluxes from model transit light curve
    """
    exp_time = 1./60/24  # 1 minute cadence -> [days]

    m = batman.TransitModel(transit_params, times, supersample_factor=7,
                            exp_time=exp_time)
    model_flux = m.light_curve(transit_params)
    return model_flux


def summed_gaussians(times, spot_parameters):
    """
    Take a list of gaussian input parameters (3 parameters per gaussian), make
    a model of the sum of all of those gaussians.

    Parameters
    ----------
    times : `numpy.ndarray`
        Times in JD
    spot_parameters : list
        List of all spot parameters in [amp, t0, sig, amp, t0, sig, ...] order

    Returns
    -------
    model : `numpy.ndarray`
        Sum of gaussians
    """
    model = np.zeros(len(times), dtype=np.float128)

    if spot_parameters is not None and len(spot_parameters) % 3 == 0:
        split_input_parameters = np.split(np.array(spot_parameters),
                                          len(spot_parameters)/3)
        for amplitude, t0, sigma in split_input_parameters:
            model += gaussian(times, amplitude, t0, sigma)

    return model


def get_in_transit_bounds(times, params, duration_fraction=0.9):
    """
    Approximate the boundaries of "in-transit" for tranist occuring
    during times `times`.

    Parameters
    ----------
    times : `numpy.ndarray`
        Times in JD
    params : `~batman.TransitParams`
        Transit light curve parameters
    duration_fraction : float
        Fraction of the full transit duration to consider "in-transit"

    Returns
    -------
    lower_t_bound : float
        Earliest in-transit time [JD]
    upper_t_bound : float
        Latest in-transit time [JD]
    """
    phased = (times - params.t0) % params.per
    near_transit = ((phased < params.duration*(0.5*duration_fraction)) |
                    (phased > params.per -
                     params.duration*(0.5*duration_fraction)))
    if np.count_nonzero(near_transit) == 0:
        near_transit = 0
    return times[near_transit].min(), times[near_transit].max()


def lnprior(theta, y, lower_t_bound, upper_t_bound, transit_params,
            skip_priors):
    """
    Log prior for `emcee` runs.

    Parameters
    ----------
    theta : list
        Fitting parameters
    y : `numpy.ndarray`
        Fluxes
    lower_t_bound : float
        Earliest in-transit time [JD]
    upper_t_bound : float
        Latest in-transit time [JD]
    skip_priors : bool
        Should the priors be skipped?

    Returns
    -------
    lnpr : float
        Log-prior for trial parameters `theta`
    """
    spot_params = theta

    amplitudes = spot_params[::3]
    t0s = spot_params[1::3]
    sigmas = spot_params[2::3]
    depth = transit_params.rp**2

    min_sigma = 1.5/60/24
    max_sigma = 6.0e-3  # upper_t_bound - lower_t_bound
    t0_ok = ((lower_t_bound < t0s) & (t0s < upper_t_bound)).all()
    sigma_ok = ((min_sigma < sigmas) & (sigmas < max_sigma)).all()
    if not skip_priors:
        amplitude_ok = ((0 <= amplitudes) & (amplitudes < depth)).all()
    else:
        amplitude_ok = (amplitudes >= 0).all()

    if amplitude_ok and t0_ok and sigma_ok:
        return 0.0
    return -np.inf


def lnlike(theta, x, y, yerr, transit_params, skip_priors=False):
    """
    Log-likelihood of data given model.

    Parameters
    ----------
    theta : list
        Trial parameters
    x : `numpy.ndarray`
        Times in JD
    y : `numpy.ndarray`
        Fluxes
    yerr : `numpy.ndarray`
        Uncertainties on fluxes
    transit_params : `~batman.TransitParams`
        Transit light curve parameters

    Returns
    -------
    lnp : float
        Log-likelihood of data given model, i.e. ln( P(x | theta) )
    """
    model = spotted_transit_model(theta, x, transit_params, skip_priors)
    return -0.5*np.sum((y-model)**2/yerr**2)


def lnprob(theta, x, y, yerr, lower_t_bound, upper_t_bound, transit_params,
           skip_priors):
    """
    Log probability.

    Parameters
    ----------
    theta : list
        Trial parameters
    x : `numpy.ndarray`
        Times in JD
    y : `numpy.ndarray`
        Fluxes
    yerr : `numpy.ndarray`
        Uncertainties on fluxes
    lower_t_bound : float
        Earliest in-transit time [JD]
    upper_t_bound : float
        Latest in-transit time [JD]
    transit_params : `~batman.TransitParams`
        Transit light curve parameters
    Returns
    -------

    """
    lp = lnprior(theta, y, lower_t_bound, upper_t_bound, transit_params,
                 skip_priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, transit_params, skip_priors)


def spotted_transit_model(theta, times, transit_params, skip_priors=False):
    """
    Compute sum of spot model and transit model

    Parameters
    ----------
    theta : list
        Trial parameters
    times : `numpy.ndarray`
        Times in JD
    transit_params : `~batman.TransitParams`
        Transit light curve parameters

    Returns
    -------
    f : `numpy.ndarray`
        Model fluxes
    """

    spot_params = theta

    # Set depth according to input parameters, compute transit model
    lower_t_bound, upper_t_bound = get_in_transit_bounds(times, transit_params,
                                                         duration_fraction=1.0)
    transit_model = generate_lc(times, transit_params)
    spot_model = summed_gaussians(times, spot_params)

    # Sum the models only where planet is in transit
    transit_plus_spot_model = transit_model
    in_transit_times = (times < upper_t_bound) & (times > lower_t_bound)
    transit_plus_spot_model[in_transit_times] += spot_model[in_transit_times]

    if not skip_priors:
        # Force all model fluxes <=1
        transit_plus_spot_model[transit_plus_spot_model > 1] = 1.0
    return transit_plus_spot_model


def spotted_transit_model_individuals(theta, times, transit_params):
    """
    Compute sum of each spot model and the transit model individually,
    return a list of each.

    Parameters
    ----------
    theta : list
        Trial parameters
    times : `numpy.ndarray`
        Times in JD
    transit_params : `~batman.TransitParams`
        Transit light curve parameters

    Returns
    -------
    f_list : list
        List of model fluxes
    """
    spot_params = theta

    split_spot_params = np.split(spot_params, len(spot_params)/3)

    return [spotted_transit_model(spot_params, times, transit_params)
            for spot_params in split_spot_params]

def run_emcee_seeded(light_curve, transit_params, spot_parameters, n_steps,
                     n_walkers, output_path, burnin=0.7,
                     n_extra_spots=1, skip_priors=False):
    """
    Fit for transit depth and spot parameters given initial guess informed by
    results from `peak_finder`

    Parameters
    ----------
    light_curve : `friedrich.lightcurve.TransitLightCurve`
        Light curve to fit
    transit_params : `~batman.TransitParams`
        Transit light curve parameters
    spot_parameters : list
        List of all spot parameters in [amp, t0, sig, amp, t0, sig, ...] order
    n_steps : int
        Number of MCMC steps to take
    n_walkers : int
        Number of MCMC walkers to initialize (must be even, more than twice the
        number of free params in fit)
    output_path : str
        Path to HDF5 archive output for storing results
    burnin : float
        Fraction of total number of steps to save to output (will truncate
        the first `burnin` of the light curve)
    n_extra_spots : int
        Add `n_extra_spots` extra spots to the fit to soak up spots not
        predicted by `peak_finder`
    skip_priors : bool
        Should a prior be applied to the depth parameter?

    Returns
    -------
    sampler : `emcee.EnsembleSampler`
        Sampler object returned by `emcee`
    """

    times = light_curve.times.jd
    fluxes = light_curve.fluxes
    errors = light_curve.errors

    lower_t_bound, upper_t_bound = get_in_transit_bounds(times, transit_params)
    amps = spot_parameters[::3]
    init_depth = transit_params.rp**2

    extra_spot_params = [0.1*np.min(amps), np.mean(times),
                         0.05*(upper_t_bound-lower_t_bound)]
    fit_params = np.concatenate([spot_parameters,
                                 n_extra_spots*extra_spot_params])

    ndim, nwalkers = len(fit_params), n_walkers
    pos = []

    while len(pos) < nwalkers:
        realization = fit_params + 1e-5*np.random.randn(ndim)

        if not np.isinf(lnprior(realization, fluxes, lower_t_bound,
                                upper_t_bound, transit_params, skip_priors)):
            pos.append(realization)

    print('Begin MCMC...')

    pool = MPIPool(loadbalance=True)
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(times, fluxes, errors, lower_t_bound,
                                          upper_t_bound, transit_params,
                                          skip_priors),
                                    pool=pool)
    sampler.run_mcmc(pos, n_steps)
    print('Finished MCMC...')
    pool.close()

    burnin_len = int(burnin*n_steps)

    from .storage import create_results_archive

    create_results_archive(output_path, light_curve, sampler, burnin_len, ndim)

    return sampler
