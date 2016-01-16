
from __future__ import absolute_import, print_function
import numpy as np
import emcee
from scipy import optimize, signal
import matplotlib.pyplot as plt
import batman
from copy import deepcopy

def gaussian(times, amplitude, t0, sigma):
    return amplitude * np.exp(-0.5*(times - t0)**2/sigma**2)


def generate_lc(times, transit_params):
    exp_time = 1./60/24  # 1 minute cadence -> [days]

    m = batman.TransitModel(transit_params, times, supersample_factor=7,
                            exp_time=exp_time)
    model_flux = m.light_curve(transit_params)
    return model_flux


def summed_gaussians(times, spot_parameters):
    """
    Take a list of gaussian input parameters (3 parameters per gaussian), make a model of the
    sum of all of those gaussians.
    """
    model = np.zeros(len(times), dtype=np.float128)

    if spot_parameters is not None and len(spot_parameters) % 3 == 0:
        split_input_parameters = np.split(spot_parameters,
                                          len(spot_parameters)/3)
        for amplitude, t0, sigma in split_input_parameters:
            model += gaussian(times, amplitude, t0, sigma)

    return model


def get_in_transit_bounds(x, params, duration_fraction=0.7):
    phased = (x - params.t0) % params.per
    near_transit = ((phased < params.duration*(0.5*duration_fraction)) |
                    (phased > params.per -
                     params.duration*(0.5*duration_fraction)))
    if np.count_nonzero(near_transit) == 0:
        near_transit = 0
    return x[near_transit].min(), x[near_transit].max()


def lnprior(theta, y, lower_t_bound, upper_t_bound):
    depth, spot_params = theta[0], theta[1:]

    amplitudes = spot_params[::3]
    t0s = spot_params[1::3]
    sigmas = spot_params[2::3]

    if (((0 <= amplitudes) & (amplitudes < 0.2)).all() and
        ((lower_t_bound < t0s) & (t0s < upper_t_bound)).all() and
        ((0.5/60/24 < sigmas) &
             (sigmas < upper_t_bound - lower_t_bound)).all()):
        return 0.0
    return -np.inf


def lnlike(theta, x, y, yerr, transit_params):
    model = spotted_transit_model(theta, x, transit_params)
    return -0.5*np.sum((y-model)**2/yerr**2)


def lnprob(theta, x, y, yerr, lower_t_bound, upper_t_bound):
    lp = lnprior(theta, y, lower_t_bound, upper_t_bound)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


def spotted_transit_model(theta, times, transit_params):
    """
    Compute sum of spot model and transit model

    Parameters
    ----------
    theta
    times
    transit_params

    Returns
    -------

    """
    depth, spot_params = theta[0], theta[1:]

    # Copy initial transit parameters
    transit_params_tmp = deepcopy(transit_params)
    # Set depth according to input parameters, comput transit model
    transit_params_tmp.rp = depth**0.5
    transit_model = generate_lc(times, transit_params_tmp)
    spot_model = summed_gaussians(times, spot_params)
    return transit_model + spot_model

def run_emcee_seeded(times, residuals, errors, transit_params,
                     spot_parameters, burnin=0.7):
    """Fit depth + spots given initial guess from `peak_finder`"""

    lower_t_bound, upper_t_bound = get_in_transit_bounds(times, transit_params)

    init_depth = transit_params.rp**2
    fit_params = np.concatenate([[init_depth], spot_parameters])

    ndim, nwalkers = len(fit_params), 10*len(fit_params)
    pos = []

    while len(pos) < nwalkers:
        realization = fit_params + 1e-3*np.random.randn(ndim)
        if lnprior(realization, residuals, lower_t_bound, upper_t_bound) == 0.0:
            pos.append(realization)

    pool = emcee.interruptible_pool.InterruptiblePool(processes=4)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(times, residuals, errors,
                                          lower_t_bound, upper_t_bound),
                                    pool=pool)
    n_steps = 25000
    sampler.run_mcmc(pos, n_steps)
    burnin_len = int(burnin*n_steps)
    samples = sampler.chain[:, burnin_len:, :].reshape((-1, ndim))
    return sampler, samples


def peak_finder_chi2(theta, x, y, yerr):
    model = summed_gaussians(x, theta)
    return np.sum((y-model)**2/yerr**2)


def peak_finder(times, residuals, errors, transit_params, n_peaks=4, plots=False,
                verbose=False):
    """

    Parameters
    ----------
    times
    residuals
    errors
    transit_params
    n_peaks
    plots
    verbose

    Returns
    -------


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
    peak_sigmas = np.zeros(len(highest_maxes_in_transit)) + 2./60/24  # 3 min
    input_parameters = np.vstack([peak_amplitudes, peak_times,
                                  peak_sigmas]).T.ravel()

    result = optimize.fmin_powell(peak_finder_chi2, input_parameters, disp=False,
                                  args=(times, residuals, errors),
                                  xtol=0.00001, ftol=0.00001)

    # if np.all(result == input_parameters):
    #     print 'oh no!, fmin didnt produce a fit')
    print(result)
    # Only use gaussians that occur in transit (fmin fit is unbounded in time)
    split_result = np.split(result, len(input_parameters)/3)
    result_in_transit = []
    for amplitude, t0, sigma in split_result:
        if (t0 < upper_t_bound) and (t0 > lower_t_bound):
            result_in_transit.extend([amplitude, t0, sigma])

    if plots:
        fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
        ax[0].errorbar(times, residuals, fmt='.', color='k')
        [ax[0].axvline(t) for t in times[maxes_in_transit]]
        ax[0].plot(times, summed_gaussians(times, input_parameters), 'r')
        ax[0].axhline(0, color='k', ls='--')
        ax[0].set_ylabel('Transit Residuals')
    
        ax[1].errorbar(times, residuals, fmt='.', color='k')
        ax[1].plot(times, summed_gaussians(times, result), 'r')
        ax[1].axhline(0, color='k', ls='--')
        ax[1].set_ylabel('Residuals')
    
        ax[2].errorbar(times, residuals - summed_gaussians(times, result), fmt='.', color='k')
        #ax[1].errorbar(times, gaussian_model, fmt='.', color='r')
        ax[2].axhline(0, color='k', ls='--')
        ax[2].set_ylabel('Residuals')

        fig.tight_layout()
        plt.show()
    return result