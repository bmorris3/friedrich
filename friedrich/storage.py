# Licensed under the MIT License - see LICENSE.rst
"""
Methods for making or retrieving archived results from
`friedrich.fitting.run_emcee_seeded`.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import h5py
import numpy as np

default_compression = 'lzf'


def create_results_archive(archive_path, light_curve, sampler, burnin_len, ndim,
                           compression=default_compression):
    """
    Create an HDF5 archive of the results from
    `friedrich.fitting.run_emcee_seeded`.

    Parameters
    ----------
    archive_path : str
        Path to archive to create
    light_curve : `friedrich.lightcurve.TransitLightCurve`
        Light curve input to `friedrich.fitting.run_emcee_seeded`
    sampler : `emcee.EnsembleSampler`
        Sampler instance returned by `emcee`
    burnin_len : int
        Number of MCMC steps to skip over when saving resutls
    ndim : int
        Number of dimensions in the fit
    compression : str (optional)
        Type of compression to use on long outputs. Default is "lzf".

    """
    lnprob = sampler.lnprobability[:, burnin_len:].T
    best_params = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
    samples = sampler.chain[:, burnin_len:, :].reshape((-1, ndim))

    with h5py.File(archive_path, 'w') as f:
        # Save the log-probability (compressed)
        lnprob_dset = f.create_dataset('lnprob', dtype=lnprob.dtype,
                                       shape=lnprob.shape,
                                       compression=compression)
        lnprob_dset[:] = lnprob

        # Save the small array of parameters that maximize lnprob
        best_params_dset = f.create_dataset('best_params',
                                            dtype=best_params.dtype,
                                            shape=best_params.shape)
        best_params_dset[:] = best_params

        # Save the full chains (compressed)
        samples_dset = f.create_dataset('samples', dtype=samples.dtype,
                                        shape=samples.shape,
                                        compression=compression)
        samples_dset[:] = samples

        # Save the light curve
        lc_matrix = np.vstack([light_curve.times.jd, light_curve.fluxes,
                               light_curve.errors])
        lc_dset = f.create_dataset('lightcurve', dtype=lc_matrix.dtype,
                                   shape=lc_matrix.shape)
        lc_dset[:] = lc_matrix


def read_results_archive(archive_path, compression=default_compression):
    """
    Read in an HDF5 archive of the results from
    `friedrich.fitting.run_emcee_seeded`.

    Parameters
    ----------
    archive_path : str
        Path to archive to read

    Returns
    -------
    lnprob : `numpy.ndarray`
        Log-probability of each walker for each iteration after burn-in
    best_params : `numpy.ndarray`
        Maximum probability parameters
    samples : `numpy.ndarray`
        Trial parameter positions at each step, for each walker
    lc_matrix : `numpy.ndarray`
        (3, N) matrix of light curve data including [times, fluxes, errors].
    """
    with h5py.File(archive_path, 'r') as f:

        lnprob = f['lnprob'][:]
        best_params = f['best_params'][:]
        samples = f['samples'][:]
        lc_matrix = f['lightcurve'][:]

    return lnprob, best_params, samples, lc_matrix
