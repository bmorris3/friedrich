from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Import dev version of friedrich:
import sys, os
sys.path.insert(0, '../')

import numpy as np
import matplotlib.pyplot as plt
from friedrich.analysis import Transit, Spot, Measurement, MCMCResults
from friedrich.lightcurve import hat11_params_morris_experiment
from glob import glob
transit_params = hat11_params_morris_experiment()
import emcee

def gaussian(x, mean, lnvar, amp):
    var = np.exp(lnvar)
    return amp/np.sqrt(2*np.pi*var) * np.exp(-0.5 * (x - mean)**2 / var)

def lnlikelihood_gaussian(x, yerr, mean, lnvar, amp):
    var = np.exp(lnvar) + yerr**2
#     return  -0.5 * (x - mean)**2 / var - 0.5 * np.log(2*np.pi*var) + np.log(amp)
    return  -0.5 * ((x - mean)**2 / var + np.log(var)) + np.log(amp)

def lnlikelihood_sum_gaussians(parameters, x, yerr):
    a1, mean_latitude, lnv1, lnv2, new_i_s = parameters
    a2 = 1 - a1
    delta_i_s = transit_params.inc_stellar - new_i_s
    l1 = -mean_latitude - delta_i_s
    l2 = mean_latitude - delta_i_s

    ln_likes = (lnlikelihood_gaussian(x, yerr, l1, lnv1, a1),
                lnlikelihood_gaussian(x, yerr, l2, lnv2, a2))
    return np.sum(np.logaddexp.reduce(ln_likes)), ln_likes

def minimize_this(parameters, x, yerr):
    return -1*lnlikelihood_sum_gaussians(parameters, x, yerr)

def model(parameters, x):
    a1, mean_latitude, lnv1, lnv2, new_i_s = parameters
    a2 = 1 - a1
    delta_i_s = transit_params.inc_stellar - new_i_s
    l1 = -mean_latitude - delta_i_s
    l2 = mean_latitude - delta_i_s
    return (gaussian(x, l1, lnv1, a1) + gaussian(x, l2, lnv2, a2))

def model_components(parameters, x):
    a1, mean_latitude, lnv1, lnv2, new_i_s = parameters
    a2 = 1 - a1
    delta_i_s = transit_params.inc_stellar - new_i_s
    l1 = -mean_latitude - delta_i_s
    l2 = mean_latitude - delta_i_s
    return np.array([gaussian(x, l1, lnv1, a1), gaussian(x, l2, lnv2, a2)]).T

def lnprior(parameters):
    a1, mean_latitude, lnv1, lnv2, new_i_s = parameters
    v1, v2 = np.exp([lnv1, lnv2])
    # 0 < new_i_s < 90
    #if (2 < mean_latitude and 180+70 < new_i_s < 180+90 and 1 < v1 < 15**2 and 1 < v2 < 15**2 and 0 < a1 < 1):
    if (2 < mean_latitude and 180+70 < new_i_s < 180+90 and 1 < v1 < 25**2 and 1 < v2 < 25**2 and 0 < a1 < 1):
        return 0.0
    return -np.inf

def lnprob(parameters, x, yerr):
    lp = lnprior(parameters)
    if not np.isfinite(lp):
        return -np.inf, None
    lnlike, blobs = lnlikelihood_sum_gaussians(parameters, x, yerr)
    return lp + lnlike, blobs

def run_emcee(initp, y, error, n_steps, burnin, threads=4):

    ndim, nwalkers = len(initp), 4*len(initp)
    p0 = [np.array(initp) + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(y, error), threads=threads)

    #pos = sampler.run_mcmc(p0, 500)[0]
    samples = sampler.run_mcmc(p0, n_steps)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    bestp = np.median(samples, axis=0)
    return samples, bestp
