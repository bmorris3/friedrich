
from __future__ import absolute_import, print_function

from .fitting import spotted_transit_model, spotted_transit_model_individuals
from .storage import read_results_archive
from .lightcurve import TransitLightCurve

import matplotlib.pyplot as plt
from corner import corner


class MCMCResults(object):
    def __init__(self, archive_path):

        results = read_results_archive(archive_path)
        self.lnprob, self.best_params, self.chains, lc_matrix = results

        self.lc = TransitLightCurve(times=lc_matrix[0, :],
                                    fluxes=lc_matrix[1, :],
                                    errors=lc_matrix[2, :])

    def plot_lnprob(self):
        plt.title('$\log \,p$')
        plt.plot(self.lnprob)
        plt.xlabel('Step')
        plt.ylabel('$\log \,p$')

    def plot_corner(self):
        labels = ['depth']
        for i in range(self.chains.shape[0]):
            labels.extend(['$a_{0}$'.format(i), '$t_{{0,{0}}}$'.format(i),
                           '$\sigma_{0}$'.format(i)])
        corner(self.chains, labels=labels)

    def plot_max_lnp_lc(self, transit_params):
        model = spotted_transit_model(self.best_params, self.lc.times.jd,
                                      transit_params)
        individual_models = spotted_transit_model_individuals(self.best_params,
                                                              self.lc.times.jd,
                                                              transit_params)

        errorbar_props = dict(fmt='.', color='k', capsize=0, ecolor='gray')

        fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        min_jd_int = int(self.lc.times.jd.min())
        ax[0].errorbar(self.lc.times.jd - min_jd_int, self.lc.fluxes,
                       self.lc.errors, **errorbar_props)
        ax[0].plot(self.lc.times.jd - min_jd_int, model, 'r', lw='3')

        for individual_model in individual_models:
            ax[0].plot(self.lc.times.jd - min_jd_int, individual_model, 'b')

        ax[1].axhline(0, color='k', ls='--')
        ax[1].errorbar(self.lc.times.jd - min_jd_int, self.lc.fluxes - model,
                       self.lc.errors, **errorbar_props)

        ax[1].set_xlabel('JD - {0}'.format(min_jd_int))
        ax[1].set_ylabel('Residuals')
        ax[0].set_ylabel('Flux')

