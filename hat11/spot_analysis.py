
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Import dev version of friedrich:
import sys
sys.path.insert(0, '../')
from friedrich.analysis import MCMCResults
from friedrich.lightcurve import hat11_params_morris
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

archive_paths = sorted(glob('/local/tmp/friedrich/hat11/chains???.hdf5'))

#archive_paths = ['/local/tmp/friedrich/hat11/chains043.hdf5']

n_significant_spots = []
indices = []
for archive_path in archive_paths:
    m = MCMCResults(archive_path, hat11_params_morris())
    delta_bics = m.get_spots_delta_chi2(plots=False)
    indices.append(m.index)
    n_significant_spots.append(np.count_nonzero(delta_bics > 10))
    #plt.savefig('spots/{0}.png'.format(m.index))
    #plt.close()
indices = [int(i[6:]) for i in indices]
np.savetxt('nspots.txt', np.vstack([indices, n_significant_spots]).T)
plt.show()
