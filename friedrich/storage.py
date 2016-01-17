
from __future__ import absolute_import, print_function

import h5py
import numpy as np

default_compression = 'lzf'


def create_results_archive(archive_path, light_curve, sampler, burnin_len, ndim,
                           compression=default_compression):
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

        # Update attributes on `holograms` with metadata
        #f['holograms'].attrs.update(metadata)


def read_results_archive(archive_path, compression=default_compression):
    with h5py.File(archive_path, 'r') as f:

        lnprob = f['lnprob'][:]
        best_params = f['best_params'][:]
        samples = f['samples'][:]
        lc_matrix = f['lightcurve'][:]

    return lnprob, best_params, samples, lc_matrix