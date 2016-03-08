
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Import dev version of friedrich:
import sys
sys.path.insert(0, '../')
from friedrich.analysis import MCMCResults
from friedrich.lightcurve import hat11_params_morris
from glob import glob
import matplotlib.pyplot as plt

#archive_path = ('/gscratch/stf/bmmorris/friedrich/chains{0:03d}.hdf5'
#                .format(int(sys.argv[1])))
# archive_path = ('/media/PASSPORT/friedrich/chains{0:03d}.hdf5'
#                 .format(int(sys.argv[1])))
#
# m = MCMCResults(archive_path)
# # m.plot_corner()
# # m.plot_lnprob()
# # m.plot_max_lnp_lc(hat11_params_morris())
# # m.plot_lat_lon()
# # m.plot_star()
# m.plot_star_projected()
# plt.savefig('tmp/{0:03d}.png'.format(int(sys.argv[1])))
# #plt.show()

archive_paths = sorted(glob('/local/tmp/friedrich/hat11/chains???.hdf5'))
#archive_paths = ['/astro/users/bmmorris/Desktop/chains000.hdf5']
for archive_path in archive_paths:
    m = MCMCResults(archive_path, hat11_params_morris())
    #m.max_lnp_theta_phi()
    m.plot_lnprob()
    m.plot_corner()
    #m.plot_star_projected()
    transit_number = m.index.split('chains')[1]
    #plt.savefig('tmp/{0:03d}.png'.format(int(transit_number)))
    plt.show()
    #plt.close()
#plt.show()
