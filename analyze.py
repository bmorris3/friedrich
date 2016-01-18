
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from friedrich.analysis import MCMCResults
from friedrich.lightcurve import hat11_params_morris
import matplotlib.pyplot as plt
import sys

archive_path = ('/gscratch/stf/bmmorris/friedrich/chains{0:03d}.hdf5'
                .format(int(sys.argv[1])))

m = MCMCResults(archive_path)
m.plot_corner()
m.plot_lnprob()
m.plot_max_lnp_lc(hat11_params_morris())
plt.show()
